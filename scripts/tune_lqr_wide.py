"""Revision: bracket the LQR weight search and score the chosen designs on held-out scenarios.

    uv run python scripts/tune_lqr_wide.py [--out results/tmlr/lqr_wide.json]

The original search (``tune_lqr.py``) found both optima on the edge of its grid in all
five multipliers. This search starts from that grid extended one step beyond every
edge where an optimum sat, and keeps extending an edge while an optimum sits on it and
the last extension improved that optimum's return by at least ``MIN_GAIN`` (so it stops
when the optimum is interior or the return stops improving). Steps are factors of 2
(4 for the asymmetric-morphing axis, as in the original grid).

Three optima are tracked, all scored by the PPO reward with the actuator-limit action
scale, on tuning scenarios disjoint from evaluation (seed 7, 96 scenarios):

* ``nominal_constrained``: nominal training distribution, every frozen design point
  stable with 60 ms delay (the paper's "GS LQR");
* ``nominal_unconstrained``: the same without the delay requirement ("no-margin LQR");
* ``dr_constrained``: the domain-randomized training distribution (model error, delay
  0-60 ms, noise) under the same 60 ms requirement ("gs_dr").

The chosen designs (and the original ones) are then scored on held-out scenarios
(seed 1007, 256 per distribution) that played no part in the choice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tune_lqr import DELAY_REQUIREMENT_STEPS, GRID, SEED, episode_return
from tune_lqr import N as TUNE_N

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.data import Table
from morphjax.rl import ACT_LIMITS, PPOConfig, training_scenario
from morphjax.sim import DT, rollout

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_SEED, HELDOUT_N = 1007, 256
N = TUNE_N
MIN_GAIN = 0.05  # return units; an extension that gains less counts as "stopped improving"
MAX_ROUNDS = 5
BATCH = 16  # candidates simulated together
STEP = {"phi": 2.0, "integ": 2.0, "beta": 2.0, "eta_a": 4.0, "effort": 2.0}
OBJECTIVES = ("nominal_constrained", "nominal_unconstrained", "dr_constrained")


def scenarios(seed: int, n: int, dr: bool):
    cfg = PPOConfig(domain_randomization=dr)
    return jax.vmap(lambda k: training_scenario(k, cfg))(
        jax.random.split(jax.random.PRNGKey(seed), n)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmlr" / "lqr_wide.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the script")
    args = ap.parse_args()
    global N, HELDOUT_N, MAX_ROUNDS, GRID
    if args.smoke:
        N, HELDOUT_N, MAX_ROUNDS = 4, 4, 2
        GRID = {k: v[-2:] for k, v in GRID.items()}
    t0 = time.time()
    setup = pipeline.build()
    k_lon = design.longitudinal_gains(setup.grid)
    base = make_schedule(setup.grid, k_lon, setup.lat_gains["gs"])
    shp = setup.grid.residual.shape
    tune = {"nominal": scenarios(SEED, N, False), "dr": scenarios(SEED, N, True)}
    print(f"tuning on {N} scenarios per distribution; grid {GRID}", flush=True)

    @jax.jit
    def run_batch(k_vals, scen):
        """Rollouts of a batch of lateral gain tables on one scenario batch."""

        def one(kv):
            sched = base._replace(k_lat=Table(base.k_lat.grids, kv))
            return jax.vmap(lambda s: rollout(setup.plant, sched, lambda x, p, i: lateral_law(
                sched, x, p, i), s))(scen)  # fmt: skip

        return jax.vmap(one)(k_vals)

    cache: dict[tuple, dict] = {}

    def score(ws: list[tuple], gains: dict, dist: str) -> None:
        """Mean return and failure count of each candidate on one tuning distribution."""
        scen = tune[dist]
        for i in range(0, len(ws), BATCH):
            chunk = ws[i : i + BATCH]
            pad = chunk + [chunk[-1]] * (BATCH - len(chunk))  # fixed shape: one compile
            kv = jnp.asarray(np.stack([gains[w].reshape(*shp, -1) for w in pad]))
            traj = run_batch(kv, scen)
            for j, w in enumerate(chunk):
                tj = jax.tree_util.tree_map(lambda x, j=j: x[j], traj)
                ret = episode_return(tj, scen, ACT_LIMITS)
                cache[w][f"{dist}_return"] = float(ret.mean())
                cache[w][f"{dist}_failures"] = int(np.asarray(tj.failed).any(axis=1).sum())

    def evaluate(ws: list[tuple]) -> None:
        todo = [w for w in ws if w not in cache]
        gains = {w: design.lateral_gains_gs(setup.grid, design.LatWeights(*w)) for w in todo}
        for w in todo:
            ok = design.lateral_delay_margin_ok(setup.grid, gains[w], DT, DELAY_REQUIREMENT_STEPS)
            cache[w] = {"weights": dict(zip(GRID, w, strict=True)), "delay_margin_ok": ok,
                        "dr_return": None, "dr_failures": None}  # fmt: skip
        score(todo, gains, "nominal")
        # gs_dr is constrained, so only designs meeting the 60 ms requirement fly DR.
        score([w for w in todo if cache[w]["delay_margin_ok"]], gains, "dr")
        print(f"[{time.time() - t0:6.0f}s] {len(cache)} candidates scored", flush=True)

    def best(objective: str) -> dict:
        dist = "dr" if objective.startswith("dr") else "nominal"
        rows = [r for r in cache.values() if r["delay_margin_ok"] or "unconstrained" in objective]
        return max(rows, key=lambda r: (r[f"{dist}_failures"] == 0, r[f"{dist}_return"]))

    # Round 0: the original grid, extended one step beyond each edge where an original
    # optimum sits (from results/lqr_weights.json).
    orig = json.loads((ROOT / "results" / "lqr_weights.json").read_text(encoding="utf-8"))
    axes = {k: sorted(v) for k, v in GRID.items()}
    for opt in [] if args.smoke else (orig["weights"], orig["unconstrained_best"]["weights"]):
        for k, v in opt.items():
            if v == min(axes[k]):
                axes[k] = sorted({*axes[k], min(axes[k]) / STEP[k]})
            if v == max(axes[k]):
                axes[k] = sorted({*axes[k], max(axes[k]) * STEP[k]})
    rounds = []
    prev_best: dict[str, float] = {}
    for rnd in range(MAX_ROUNDS):
        evaluate([tuple(w) for w in itertools.product(*(axes[k] for k in GRID))])
        opts = {o: best(o) for o in OBJECTIVES}
        rec = {"axes": {k: list(v) for k, v in axes.items()}, "candidates": len(cache),
               "optima": opts}  # fmt: skip
        rounds.append(rec)
        print(json.dumps({o: (r["weights"], r["nominal_return"], r["dr_return"])
                          for o, r in opts.items()}), flush=True)  # fmt: skip
        grew = False
        for o, r in opts.items():
            dist = "dr" if o.startswith("dr") else "nominal"
            gain = r[f"{dist}_return"] - prev_best.get(o, -np.inf)
            prev_best[o] = r[f"{dist}_return"]
            if rnd > 0 and gain < MIN_GAIN:
                continue  # this optimum stopped improving
            for k, v in r["weights"].items():
                if v == min(axes[k]):
                    axes[k] = sorted({*axes[k], v / STEP[k]})
                    grew = True
                elif v == max(axes[k]):
                    axes[k] = sorted({*axes[k], v * STEP[k]})
                    grew = True
        if not grew:
            break

    final = rounds[-1]["optima"]
    axes = rounds[-1]["axes"]  # the axes actually evaluated
    edges = {o: {k: ("low edge" if v == min(axes[k]) else "high edge" if v == max(axes[k])
                     else "interior") for k, v in r["weights"].items()}
             for o, r in final.items()}  # fmt: skip

    # Held-out scores of the chosen designs and of the original ones.
    held = {"nominal": scenarios(HELDOUT_SEED, HELDOUT_N, False),
            "dr": scenarios(HELDOUT_SEED, HELDOUT_N, True)}  # fmt: skip
    chosen = {o: tuple(r["weights"].values()) for o, r in final.items()}
    chosen["original_constrained"] = tuple(orig["weights"][k] for k in GRID)
    chosen["original_unconstrained"] = tuple(orig["unconstrained_best"]["weights"][k] for k in GRID)
    heldout = {}

    @jax.jit
    def run_one(kv, scen):
        sched = base._replace(k_lat=Table(base.k_lat.grids, kv))
        return jax.vmap(lambda s: rollout(setup.plant, sched, lambda x, p, i: lateral_law(
            sched, x, p, i), s))(scen)  # fmt: skip

    for name, w in chosen.items():
        g = design.lateral_gains_gs(setup.grid, design.LatWeights(*w))
        kv = jnp.asarray(g.reshape(*shp, -1))
        row = {"weights": dict(zip(GRID, w, strict=True))}
        for dist, scen in held.items():
            traj = run_one(kv, scen)
            ret = episode_return(traj, scen, ACT_LIMITS)
            row[f"{dist}_return"] = float(ret.mean())
            row[f"{dist}_return_se"] = float(ret.std(ddof=1) / np.sqrt(ret.size))
            row[f"{dist}_failures"] = int(np.asarray(traj.failed).any(axis=1).sum())
        row["tuning"] = cache.get(w)
        row["linear_delay_margin_ms"] = 1000 * DT * design.max_stable_delay(setup.grid, g, DT)
        heldout[name] = row

    def top(objective: str, k: int = 5):
        dist = "dr" if objective.startswith("dr") else "nominal"
        rows = [r for r in cache.values() if r["delay_margin_ok"] or "unconstrained" in objective]
        return sorted(rows, key=lambda r: -r[f"{dist}_return"])[:k]

    out = {
        "description": __doc__,
        "action_scale": ACT_LIMITS,
        "tuning": {"seed": SEED, "n": N},
        "heldout": {"seed": HELDOUT_SEED, "n": HELDOUT_N, "rows": heldout},
        "min_gain": MIN_GAIN,
        "rounds": rounds,
        "final_axes": axes,
        "optima_position": edges,
        "top5": {o: top(o) for o in OBJECTIVES},
        # pipeline.load_weights-compatible entries for the evaluation
        "gs_wide": final["nominal_constrained"],
        "gs_hg_wide": final["nominal_unconstrained"],
        "gs_dr": final["dr_constrained"],
        "candidates": list(cache.values()),
        "wall_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"optima_position": edges, "heldout": {k: {kk: v[kk] for kk in (
        "nominal_return", "dr_return", "nominal_failures", "dr_failures")} for k, v in
        heldout.items()}}, indent=1))  # fmt: skip
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
