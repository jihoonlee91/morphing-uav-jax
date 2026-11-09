"""Round 3: accuracy-delay frontier of the gain-scheduled LQR, with every learned policy overlaid.

    uv run python scripts/r3_frontier.py [--learned-only]

For each LQR weight candidate (the 768 of ``tune_lqr.GRID``, a random sample of the
bracketed grid of ``tune_lqr_wide.py`` and its optima):

* nominal roll RMSE on the first 64 of the 256 scenarios of ``evaluate.py``'s nominal
  (frozen-shape) condition (seed 1000; the same scenarios for every controller);
* the cumulative delay margin: the largest d in 0..4 samples such that the frozen ZOH
  loop is stable for every d' <= d, at all 135 envelope points of the 325-point check
  grid, and separately at the 63 design points (-1: unstable without delay).

Learned controllers get the same two numbers from the revision evaluation
(``per_scenario.npz``, same 64 scenarios) and the audit (``audit_points.npz``,
linearization at trim, and at the closed-loop equilibrium), per seed and pooled.
Writes ``results/tmlr/r3/frontier.json``.
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
from evaluate import scenarios
from tune_lqr import GRID

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.data import Table
from morphjax.dynamics import W
from morphjax.sim import DT, rollout
from morphjax.verify import dense_grid, interpolate_gains

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
N_SCEN, BATCH, N_WIDE = 64, 16, 640
MAX_D = 4


def zoh(a, b):
    import scipy.linalg

    n, m = b.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = a, b
    e = scipy.linalg.expm(blk * DT)
    return e[:n, :n], e[:n, n:]


def cumulative_delay(models, gains) -> int:
    """Largest d with a stable delayed loop for every d' <= d at every point; -1 if none."""
    best = MAX_D
    for (ad, bd), k in zip(models, gains, strict=True):
        n, m = bd.shape
        for d in range(best + 1):
            size = n + max(d, 1) * m
            big = np.zeros((size, size))
            big[:n, :n] = ad
            if d == 0:
                big[:n, :n] += bd @ k
            else:
                big[:n, n + (d - 1) * m : n + d * m] = bd
                big[n : n + m, :n] = k
                for j in range(1, d):
                    big[n + j * m : n + (j + 1) * m, n + (j - 1) * m : n + j * m] = np.eye(m)
            if np.abs(np.linalg.eigvals(big)).max() >= 1.0:
                best = d - 1
                break
        if best < 0:
            return -1
    return best


def pareto(rows, key):
    """For each margin level, the most accurate candidate with at least that margin."""
    front = []
    for d in range(-1, MAX_D + 1):
        ok = [r for r in rows if r[key] >= d and r["rmse"] is not None]
        if ok:
            b = min(ok, key=lambda r: r["rmse"])
            front.append({"min_delay_steps": d, "rmse": b["rmse"], "weights": b["weights"]})
    return front


def learned_rows(metrics_dir: Path, audit_dir: Path, on_grid, envelope) -> dict:
    ps = np.load(metrics_dir / "per_scenario.npz")
    pts = np.load(audit_dir / "audit_points.npz")
    out = {}
    names = sorted(
        {k.split("/")[2] for k in ps.files if k.startswith("table/nominal/")} - {"_scenario"}
    )
    for name in names:
        rm = np.asarray(ps[f"table/nominal/{name}/roll_rmse_deg"])[:, :N_SCEN]
        row = {"rmse_per_seed": [float(np.nanmean(x)) for x in rm], "rmse": float(np.nanmean(rm))}
        for mode in ("x0", "eq"):
            keys = sorted(
                k
                for k in pts.files
                if k.startswith(f"{name}/") and k.endswith(f"/{mode}/max_delay_steps")
            )
            if not keys:
                if mode == "x0" and f"{name}/max_delay_steps" in pts.files:
                    keys = [f"{name}/max_delay_steps"]  # model-based design
                else:
                    continue
            env = [int(np.asarray(pts[k])[envelope].min()) for k in keys]
            grd = [int(np.asarray(pts[k])[on_grid].min()) for k in keys]
            row[f"envelope_delay_{mode}"] = env
            row[f"design_points_delay_{mode}"] = grd
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned-only", action="store_true", help="refresh the learned overlay only")
    ap.add_argument("--metrics", type=Path, default=TM)
    ap.add_argument("--audit", type=Path, default=TM)
    ap.add_argument("--out", type=Path, default=TM / "r3" / "frontier.json")
    args = ap.parse_args()
    t0 = time.time()
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    dense, on_grid = dense_grid(setup)
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    envelope = (
        (pts[..., 0] >= 70) & (pts[..., 0] <= 110) & (pts[..., 1] >= 500) & (pts[..., 1] <= 9500)
    )

    if args.learned_only and args.out.exists():
        res = json.loads(args.out.read_text(encoding="utf-8"))
        res["learned"] = learned_rows(args.metrics, args.audit, on_grid, envelope)
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return

    env_idx = [idx for idx in np.ndindex(*envelope.shape) if envelope[idx]]
    env_models = [zoh(*design.lateral_model(dense.A[i], dense.B[i])) for i in env_idx]
    grid_idx = list(np.ndindex(*setup.grid.residual.shape))
    grid_models = [zoh(*design.lateral_model(setup.grid.A[i], setup.grid.B[i])) for i in grid_idx]

    orig = [tuple(w) for w in itertools.product(*GRID.values())]
    wide = json.loads((TM / "lqr_wide.json").read_text(encoding="utf-8"))
    wide_axes = wide["final_axes"]
    rng = np.random.default_rng(0)
    pool = list(itertools.product(*(wide_axes[k] for k in GRID)))
    sample = [pool[i] for i in rng.choice(len(pool), N_WIDE, replace=False)]
    extra = [tuple(r["weights"][k] for k in GRID) for r in wide["heldout"]["rows"].values()]
    cands = list(dict.fromkeys(orig + sample + extra))
    origin = {w: ("original" if w in set(orig) else "wide") for w in cands}

    k_lon = design.longitudinal_gains(setup.grid)
    base = make_schedule(setup.grid, k_lon, setup.lat_gains["gs"])
    shp = setup.grid.residual.shape
    scen = jax.tree_util.tree_map(lambda x: x[:N_SCEN], scenarios(256, seed=1000, morph_rate=0.0))

    @jax.jit
    def run_batch(kv):
        def one(k):
            sched = base._replace(k_lat=Table(base.k_lat.grids, k))
            return jax.vmap(
                lambda s: rollout(
                    setup.plant, sched, lambda x, p, i: lateral_law(sched, x, p, i), s
                )
            )(scen)

        return jax.vmap(one)(kv)

    legacy = json.loads((ROOT / "results" / "lqr_weights.json").read_text(encoding="utf-8"))
    legacy_ret = {tuple(r["weights"][k] for k in GRID): r for r in legacy["candidates"]}
    rows = []
    for i in range(0, len(cands), BATCH):
        chunk = cands[i : i + BATCH]
        gains = [design.lateral_gains_gs(setup.grid, design.LatWeights(*w)) for w in chunk]
        pad = gains + gains[-1:] * (BATCH - len(chunk))
        tr = run_batch(jnp.asarray(np.stack([g.reshape(*shp, -1) for g in pad])))
        xw, failed = np.asarray(tr.xw), np.asarray(tr.failed).any(axis=-1)
        err = np.rad2deg(xw[..., W.PHI] - np.asarray(scen.phi_cmd)[None])
        rm = np.sqrt(np.mean(err**2, axis=-1))
        for j, (w, g) in enumerate(zip(chunk, gains, strict=True)):
            dg = interpolate_gains(setup.grid, g, dense)
            ok = ~failed[j]
            leg = legacy_ret.get(w)
            rows.append({
                "weights": dict(zip(GRID, w, strict=True)),
                "set": origin[w],
                "rmse": float(rm[j][ok].mean()) if ok.any() else None,
                "lost": int(failed[j].sum()),
                "envelope_delay": cumulative_delay(env_models, [dg[i] for i in env_idx]),
                "design_points_delay": cumulative_delay(grid_models, [g[i] for i in grid_idx]),
                "tuning_return_legacy": leg["mean_return"] if leg else None,
                "tuning_failures_legacy": leg["failures"] if leg else None,
                "noncumulative_60ms_ok_legacy": leg["delay_margin_ok"] if leg else None,
            })  # fmt: skip
        print(f"[{time.time() - t0:5.0f}s] {len(rows)}/{len(cands)}", flush=True)

    ori = [r for r in rows if r["set"] == "original"]
    qual = [r for r in ori if r["design_points_delay"] >= 3]
    best = max(qual, key=lambda r: (r["tuning_failures_legacy"] == 0, r["tuning_return_legacy"]))
    chosen = legacy["weights"]
    res = {
        "description": __doc__,
        "scenarios": "first 64 of the 256 nominal (frozen-shape) evaluation scenarios, seed 1000",
        "n_candidates": len(rows),
        "candidates": rows,
        "pareto_envelope": pareto(rows, "envelope_delay"),
        "pareto_design_points": pareto(rows, "design_points_delay"),
        "cumulative_rule_original_768": {
            "qualify_all_d_le_3_at_63_design_points": len(qual),
            "qualify_noncumulative_d3_legacy": sum(
                bool(r["noncumulative_60ms_ok_legacy"]) for r in ori
            ),
            "best_under_cumulative_rule": best["weights"],
            "best_return_legacy": best["tuning_return_legacy"],
            "chosen_gs_weights": chosen,
            "chosen_is_still_optimum": best["weights"] == chosen,
        },
        "learned": learned_rows(args.metrics, args.audit, on_grid, envelope),
        "wall_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: res[k]
                for k in ("pareto_envelope", "pareto_design_points", "cumulative_rule_original_768")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
