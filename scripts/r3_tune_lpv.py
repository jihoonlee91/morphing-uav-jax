# ruff: noqa: E501
"""Round 3: tune the LPV design's Bryson multipliers like the LQR's.

    uv run python scripts/r3_tune_lpv.py

The paper's LPV law reuses the weights tuned for the GS LQR. Here the common-Lyapunov
guaranteed-cost synthesis (same decay rate 0.1, pole disk 30 rad/s, trace objective)
is solved for a 3^5 grid of the same five multipliers around those weights; each design
is scored like ``tune_lqr.py`` (PPO reward, actuator-limit action scale, tuning
scenarios seed 7, 96 per distribution) under the same requirement, applied
cumulatively: every frozen design point stable for every delay of 0-60 ms. The chosen
design is then scored on the held-out scenarios (seed 1007, 256 per distribution).
Rows are appended to ``results/tmlr/r3/lpv_tuning_rows.jsonl`` as they finish (the run
can be resumed); the result is ``results/tmlr/r3/lpv_tuned.json``.
"""

from __future__ import annotations

import itertools
import json
import time
import warnings
from functools import partial
from pathlib import Path

import jax
import numpy as np
from tune_lqr import GRID, SEED, N, episode_return

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.rl import ACT_LIMITS, PPOConfig, training_scenario
from morphjax.sim import DT, rollout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tmlr" / "r3"
LPV_GRID = {
    "phi": [0.5, 0.125, 0.03125],
    "integ": [0.125, 0.03125, 0.0078125],
    "beta": [0.5, 0.125, 0.03125],
    "eta_a": [1.0, 4.0, 16.0],
    "effort": [0.125, 0.03125, 0.0078125],
}
HELDOUT_SEED, HELDOUT_N = 1007, 256


def scen(seed, n, dr):
    cfg = PPOConfig(domain_randomization=dr)
    return jax.vmap(lambda k: training_scenario(k, cfg))(
        jax.random.split(jax.random.PRNGKey(seed), n)
    )


def main() -> None:
    t0 = time.time()
    setup = pipeline.build()
    k_lon = design.longitudinal_gains(setup.grid)
    tune = scen(SEED, N, False)

    @jax.jit
    def run(sched, s):
        return jax.vmap(lambda x: rollout(setup.plant, sched, partial(lateral_law, sched), x))(s)

    rows_path = OUT / "lpv_tuning_rows.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[tuple(r["weights"][k] for k in GRID)] = r
    cands = [tuple(w) for w in itertools.product(*(LPV_GRID[k] for k in GRID))]
    with rows_path.open("a", encoding="utf-8") as fh:
        for w in cands:
            if w in done:
                continue
            row = {"weights": dict(zip(GRID, w, strict=True))}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    lpv = design.lateral_gains_lpv(setup.grid, design.LatWeights(*w))
                g = lpv.gains
                row["feasible"] = True
                row["cumulative_60ms_ok"] = all(
                    design.lateral_delay_margin_ok(setup.grid, g, DT, d) for d in range(4)
                )
                tr = run(make_schedule(setup.grid, k_lon, g), tune)
                ret = episode_return(tr, tune, ACT_LIMITS)
                row["return"] = float(ret.mean())
                row["failures"] = int(np.asarray(tr.failed).any(axis=1).sum())
            except Exception as e:  # infeasible LMI or failed certificate
                row["feasible"] = False
                row["error"] = str(e)[:200]
            done[w] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"[{time.time() - t0:6.0f}s] {len(done)}/{len(cands)} {row.get('return')} {row.get('cumulative_60ms_ok')}",
                flush=True,
            )

    rows = list(done.values())
    ok = [r for r in rows if r.get("feasible") and r["cumulative_60ms_ok"] and r["failures"] == 0]
    best = max(ok, key=lambda r: r["return"]) if ok else None
    orig = setup.weights._asdict()
    orig_row = done.get(tuple(orig[k] for k in GRID))
    res = {
        "description": __doc__,
        "grid": LPV_GRID,
        "candidates": len(rows),
        "feasible": sum(bool(r.get("feasible")) for r in rows),
        "feasible_and_60ms": len(ok),
        "original_lpv_weights": orig,
        "original_row": orig_row,
        "lpv_tuned": best,
        "top5": sorted(ok, key=lambda r: -r["return"])[:5],
    }
    if best is not None:
        w = design.LatWeights(**best["weights"])
        held = {}
        for name, ww in (("lpv_tuned", w), ("lpv_original", setup.weights)):
            g = design.lateral_gains_lpv(setup.grid, ww).gains
            sch = make_schedule(setup.grid, k_lon, g)
            for dist, dr in (("nominal", False), ("dr", True)):
                s = scen(HELDOUT_SEED, HELDOUT_N, dr)
                tr = run(sch, s)
                ret = episode_return(tr, s, ACT_LIMITS)
                held[f"{name}/{dist}_return"] = float(ret.mean())
                held[f"{name}/{dist}_failures"] = int(np.asarray(tr.failed).any(axis=1).sum())
            held[f"{name}/linear_delay_margin_ms"] = (
                1000 * DT * design.max_stable_delay(setup.grid, g, DT)
            )
        res["heldout"] = held
    res["wall_seconds"] = time.time() - t0
    (OUT / "lpv_tuned.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "candidates",
                    "feasible",
                    "feasible_and_60ms",
                    "lpv_tuned",
                    "original_row",
                )
            },
            indent=1,
        )
    )
    print(json.dumps(res.get("heldout"), indent=1))


if __name__ == "__main__":
    main()
