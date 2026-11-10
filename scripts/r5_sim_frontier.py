# ruff: noqa: E501
"""Round 5: one simulated delay-tolerance rule for every controller, and the LQR frontier.

    uv run python scripts/r5_sim_frontier.py lqr        # 1,395 LQR candidates
    uv run python scripts/r5_sim_frontier.py learners [--policies DIR ...]

Scenarios: the 256 nominal (frozen-shape) evaluation scenarios (seed 1000), split into a
selection set (first 64) and a held-out set (last 192); each is flown with a command
delay of 0, 20, 40, 60, 80 ms (the same scenarios, only the delay changed).

Simulated delay tolerance (cumulative): the largest d such that for every d' <= d no run
loses control and
* rule "rel" (primary): RMSE(d') <= 2 x RMSE(0) (scale-free, so accurate and coarse
  designs are held to the same relative standard);
* rule "abs": RMSE(d') <= 1 deg.
-1 means even d = 0 fails (loss of control). Delays are evaluated in order and stop at
the first failure.

``lqr``: every candidate of ``r3/frontier.json`` is scored on the selection set; for each
tolerance level the most accurate candidate at that level (selection RMSE at 0 ms) is the
frontier design, and the frontier designs (both rules), the frozen-rule frontier designs
(135 envelope points, 63 design points, all 325 points of the check grid) and gs / gs_hg /
gs_hg_wide are then flown on the held-out set (all delays) and scored for return on
held-out tuning-distribution scenarios (seed 1007, 256, actuator-limit action scale).
``learners``: every controller of the evaluation (per seed) flies the held-out set under
the same rule; gaps to the frontier at the controller's tolerance level get 95 %
bootstrap intervals (scenarios, and seeds for learners), paired on the 192 scenarios.
Writes ``r5/sim_frontier_lqr.json`` (+ ``.npz`` with per-scenario arrays) and
``r5/sim_frontier_learners.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from evaluate import controllers, scenarios
from r3_frontier import cumulative_delay, zoh
from tune_lqr import GRID, episode_return

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.data import Table
from morphjax.dynamics import W
from morphjax.rl import ACT_LIMITS, PPOConfig, training_scenario
from morphjax.sim import rollout

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
R5 = TM / "r5"
N_SEL, BATCH, DELAYS, B = 64, 16, (0, 1, 2, 3, 4), 2000
RULES = {"rel": lambda r, r0: r <= 2.0 * r0, "abs": lambda r, r0: r <= 1.0}


def nominal_split():
    sc = scenarios(256, seed=1000, morph_rate=0.0)
    sel = jax.tree_util.tree_map(lambda x: x[:N_SEL], sc)
    held = jax.tree_util.tree_map(lambda x: x[N_SEL:], sc)
    return sel, held


def with_delay(sc, d):
    return sc._replace(delay=jnp.full_like(sc.delay, d))


def per_scenario(traj, sc):
    """(per-scenario RMSE with NaN for lost runs, lost flags)."""
    xw, failed = np.asarray(traj.xw), np.asarray(traj.failed).any(axis=-1)
    err = np.rad2deg(xw[..., W.PHI] - np.asarray(sc.phi_cmd))
    r = np.sqrt(np.mean(err**2, axis=-1))
    return np.where(failed, np.nan, r), failed


def tolerance(stats):
    """stats: list over delays of (mean RMSE, any lost); returns per-rule tolerance."""
    out = {}
    for name, ok in RULES.items():
        t = -1
        r0 = stats[0][0] if stats else np.nan
        for d, (r, lost) in enumerate(stats):
            if lost or not ok(r, r0):
                break
            t = d
        out[name] = t
    return out


def lqr(args) -> None:
    t0 = time.time()
    setup = pipeline.build()
    k_lon = design.longitudinal_gains(setup.grid)
    base = make_schedule(setup.grid, k_lon, setup.lat_gains["gs"])
    shp = setup.grid.residual.shape
    sel, held = nominal_split()
    fr = json.loads((TM / "r3" / "frontier.json").read_text(encoding="utf-8"))
    cands = [tuple(r["weights"][k] for k in GRID) for r in fr["candidates"]]
    frozen = {tuple(r["weights"][k] for k in GRID): r for r in fr["candidates"]}

    @jax.jit
    def run_batch(kv, sc):
        def one(k):
            sched = base._replace(k_lat=Table(base.k_lat.grids, k))
            return jax.vmap(lambda s: rollout(setup.plant, sched, partial(lateral_law, sched), s))(
                sc
            )

        return jax.vmap(one)(kv)

    @jax.jit
    def run_one(k, sc):
        sched = base._replace(k_lat=Table(base.k_lat.grids, k))
        return jax.vmap(lambda s: rollout(setup.plant, sched, partial(lateral_law, sched), s))(sc)

    # frozen rule at all 325 points as well
    from morphjax.verify import dense_grid, interpolate_gains

    dense, _ = dense_grid(setup)
    all_idx = list(np.ndindex(*dense.residual.shape))
    all_models = [zoh(*design.lateral_model(dense.A[i], dense.B[i])) for i in all_idx]
    rows = []
    for i in range(0, len(cands), BATCH):
        chunk = cands[i : i + BATCH]
        gains = [design.lateral_gains_gs(setup.grid, design.LatWeights(*w)) for w in chunk]
        kv = jnp.asarray(
            np.stack([g.reshape(*shp, -1) for g in gains + gains[-1:] * (BATCH - len(chunk))])
        )
        stats = [[] for _ in chunk]
        alive = np.ones(len(chunk), bool)
        for d in DELAYS:
            if not alive.any():
                break
            tr = run_batch(kv, with_delay(sel, d))
            for j in range(len(chunk)):
                if not alive[j]:
                    continue
                tj = jax.tree_util.tree_map(lambda x, j=j: x[j], tr)
                r, lost = per_scenario(tj, sel)
                stats[j].append(
                    (float(np.nanmean(r)) if (~lost).any() else np.inf, bool(lost.any()))
                )
                tol = tolerance(stats[j])
                alive[j] = any(t == d for t in tol.values())
        for j, w in enumerate(chunk):
            dg = interpolate_gains(setup.grid, gains[j], dense)
            fz = frozen[w]
            rows.append({"weights": dict(zip(GRID, w, strict=True)), "selection": stats[j],
                         "tolerance": tolerance(stats[j]),
                         "frozen_envelope": fz["envelope_delay"], "frozen_design_points": fz["design_points_delay"],
                         "frozen_all_325": cumulative_delay(all_models, [dg[k] for k in all_idx])})  # fmt: skip
        print(f"[{time.time() - t0:6.0f}s] {len(rows)}/{len(cands)}", flush=True)

    def front(key_fn):
        out = []
        for t in range(-1, 5):
            ok = [
                r
                for r in rows
                if key_fn(r) >= t and r["selection"] and np.isfinite(r["selection"][0][0])
            ]
            if ok:
                b = min(ok, key=lambda r: r["selection"][0][0])
                out.append(
                    {"level": t, "weights": b["weights"], "selection_rmse0": b["selection"][0][0]}
                )
        return out

    fronts = {
        "sim_rel": front(lambda r: r["tolerance"]["rel"]),
        "sim_abs": front(lambda r: r["tolerance"]["abs"]),
        "frozen_envelope_135": front(lambda r: r["frozen_envelope"]),
        "frozen_design_points_63": front(lambda r: r["frozen_design_points"]),
        "frozen_all_325": front(lambda r: r["frozen_all_325"]),
    }
    legacy = json.loads((ROOT / "results" / "lqr_weights.json").read_text(encoding="utf-8"))
    wide = json.loads((TM / "lqr_wide.json").read_text(encoding="utf-8"))
    designs = {}
    for fname, f in fronts.items():
        for p in f:
            designs.setdefault(tuple(p["weights"][k] for k in GRID), []).append(
                f"{fname}@{p['level']}"
            )
    for name, w in (("gs", legacy["weights"]), ("gs_hg", legacy["unconstrained_best"]["weights"]),
                    ("gs_hg_wide", wide["gs_hg_wide"]["weights"])):  # fmt: skip
        designs.setdefault(tuple(w[k] for k in GRID), []).append(name)
    tcfg = PPOConfig()
    tun = jax.vmap(lambda k: training_scenario(k, tcfg))(
        jax.random.split(jax.random.PRNGKey(1007), 256)
    )
    held_rows, arrays = [], {}
    for idx, (w, labels) in enumerate(designs.items()):
        g = design.lateral_gains_gs(setup.grid, design.LatWeights(*w))
        kv = jnp.asarray(g.reshape(*shp, -1))
        st = []
        for d in DELAYS:
            tr = run_one(kv, with_delay(held, d))
            r, lost = per_scenario(tr, held)
            arrays[f"d{idx}/delay{d}/rmse"] = r
            st.append(
                (
                    float(np.nanmean(r)) if (~lost).any() else np.inf,
                    bool(lost.any()),
                    float(lost.mean()),
                )
            )
        trr = run_one(kv, tun)
        ret = episode_return(trr, tun, ACT_LIMITS)
        rb = np.random.default_rng(0)
        boots = [ret[rb.integers(0, ret.size, ret.size)].mean() for _ in range(B)]
        held_rows.append({"id": idx, "weights": dict(zip(GRID, w, strict=True)), "labels": labels,
                          "heldout_by_delay": [{"delay_ms": 20 * d, "rmse": s[0] if np.isfinite(s[0]) else None,
                                                "loc_share": s[2]} for d, s in zip(DELAYS, st, strict=True)],
                          "heldout_tolerance": tolerance([(s[0], s[1]) for s in st]),
                          "heldout_tuning_return": float(ret.mean()),
                          "heldout_tuning_return_ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]})  # fmt: skip
        print(
            labels,
            held_rows[-1]["heldout_tolerance"],
            held_rows[-1]["heldout_by_delay"][0],
            flush=True,
        )
    res = {"description": __doc__, "rules": {"rel": "no LOC and RMSE(d) <= 2 RMSE(0)", "abs": "no LOC and RMSE(d) <= 1 deg"},
           "candidates": rows, "fronts": fronts, "heldout_designs": held_rows, "wall_seconds": time.time() - t0}  # fmt: skip
    R5.mkdir(parents=True, exist_ok=True)
    (R5 / "sim_frontier_lqr.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(R5 / "sim_frontier_lqr.npz", **arrays)


def learners(args) -> None:
    t0 = time.time()
    setup = pipeline.build_revision(ROOT / "results" / "lqr_weights.json", TM / "lqr_wide.json",
                                    lpv_tuned_path=TM / "r3" / "lpv_tuned.json")  # fmt: skip
    ctrls = {}
    for d in args.policies:
        ctrls.update(controllers(setup, d))
    if args.only:
        ctrls = {k: v for k, v in ctrls.items() if k in args.only.split(",")}
    _, held = nominal_split()
    lq = json.loads((R5 / "sim_frontier_lqr.json").read_text(encoding="utf-8"))
    arr = np.load(R5 / "sim_frontier_lqr.npz")
    front_ids = {}
    for rule in RULES:
        for p in lq["fronts"][f"sim_{rule}"]:
            w = p["weights"]
            hid = next(h["id"] for h in lq["heldout_designs"] if h["weights"] == w)
            front_ids[(rule, p["level"])] = hid
    out = {}
    if (R5 / "sim_frontier_learners.json").exists() and args.append:
        out = json.loads((R5 / "sim_frontier_learners.json").read_text(encoding="utf-8"))
    rb = np.random.default_rng(1)
    for name, (lon, lats, factory, params) in ctrls.items():
        if factory is None:
            fns = [
                jax.jit(jax.vmap(lambda s, lat=lat, lon=lon: rollout(setup.plant, lon, lat, s)))
                for lat in lats
            ]
        else:
            g = jax.jit(
                jax.vmap(
                    lambda p, s, lon=lon, fac=factory: rollout(setup.plant, lon, fac(p), s),
                    in_axes=(None, 0),
                )
            )
            fns = [lambda s, p=p, g=g: g(p, s) for p in params]
        seeds = []
        r0 = []
        for f in fns:
            st = []
            for d in DELAYS:
                r, lost = per_scenario(f(with_delay(held, d)), held)
                if d == 0:
                    r0.append(r)
                st.append(
                    (
                        float(np.nanmean(r)) if (~lost).any() else np.inf,
                        bool(lost.any()),
                        float(lost.mean()),
                    )
                )
                if all(t < d for t in tolerance([(s[0], s[1]) for s in st]).values()):
                    break
            seeds.append({"by_delay": [{"delay_ms": 20 * d, "rmse": s[0] if np.isfinite(s[0]) else None, "loc_share": s[2]}
                                       for d, s in enumerate(st)],
                          "tolerance": tolerance([(s[0], s[1]) for s in st])})  # fmt: skip
        r0 = np.stack(r0)  # seeds x 192
        row = {"seeds": seeds, "rmse0": float(np.nanmean(r0))}
        for rule in RULES:
            tols = [s["tolerance"][rule] for s in seeds]
            row[f"tolerance_{rule}_per_seed"] = tols
            lvl = int(np.median(tols))
            hid = front_ids.get((rule, lvl))
            if hid is None:
                continue
            fr = arr[f"d{hid}/delay0/rmse"]

            def gap(a, b):
                return float(np.nanmean(a) - np.nanmean(b))

            est = gap(r0, fr)
            boots = []
            for _ in range(B):
                si = rb.integers(0, r0.shape[0], r0.shape[0])
                sc = rb.integers(0, r0.shape[1], r0.shape[1])
                boots.append(gap(r0[si][:, sc], fr[sc]))
            row[f"gap_to_frontier_{rule}"] = {"level": lvl, "frontier_rmse": float(np.nanmean(fr)), "gap_deg": est,
                                              "ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]}  # fmt: skip
        out[name] = row
        print(
            f"[{time.time() - t0:6.0f}s] {name}: rmse0 {row['rmse0']:.3f} rel {row['tolerance_rel_per_seed']} abs {row['tolerance_abs_per_seed']}",
            flush=True,
        )
    (R5 / "sim_frontier_learners.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["lqr", "learners"])
    ap.add_argument(
        "--policies", type=Path, nargs="+", default=[TM / "policies", TM / "r3" / "policies"]
    )
    ap.add_argument("--only", default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    lqr(args) if args.part == "lqr" else learners(args)


if __name__ == "__main__":
    main()
