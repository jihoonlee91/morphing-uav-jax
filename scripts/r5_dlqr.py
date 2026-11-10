# ruff: noqa: E501
"""Round 5: delay-aware discrete gain-scheduled LQR, tuned like the other baselines.

    uv run python scripts/r5_dlqr.py

For d = 0, 1, 2, 3 samples (0-60 ms) the discrete LQR of ``design.lateral_gains_dlqr``
(exact ZOH model augmented with the d pending commands) is built for a 216-point subgrid
of the five Bryson multipliers of ``tune_lqr.GRID`` (phi 1, 1/4, 1/16; integ 1, 1/4,
1/16, 1/64; beta 1, 1/4, 1/16; eta_a 1, 4; effort 1/4, 1/16, 1/64) and scored like the
LQR (PPO reward, actuator-limit action scale, the LQR tuning scenarios, seed 7, 96) with
the command delay fixed at d, the condition the design assumes. Rule: no loss of
control on the tuning scenarios and the frozen discrete loop with d samples of delay
stable at all 63 design points (it is by construction; checked). Chosen designs are
scored on held-out scenarios (seed 1007, 256, delay d) and saved as
``r5/dlqr_gains_d{d}.npy`` (evaluated as ``dlqr{d}``). Writes ``r5/dlqr.json``.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tune_lqr import GRID, SEED, N, episode_return

from morphjax import design, pipeline
from morphjax.control import make_delayed_lateral, make_schedule
from morphjax.data import Table
from morphjax.rl import ACT_LIMITS, PPOConfig, training_scenario
from morphjax.sim import DT, rollout

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "tmlr" / "r5"
SUB = {"phi": [1.0, 0.25, 0.0625], "integ": [1.0, 0.25, 0.0625, 0.015625], "beta": [1.0, 0.25, 0.0625],
       "eta_a": [1.0, 4.0], "effort": [0.25, 0.0625, 0.015625]}  # fmt: skip
BATCH = 12


def scen(seed, n, d):
    cfg = PPOConfig(delay=d)
    return jax.vmap(lambda k: training_scenario(k, cfg))(
        jax.random.split(jax.random.PRNGKey(seed), n)
    )


def stable_at_design_delay(grid, gains, weights, d) -> bool:
    """Frozen discrete loop with d samples of delay, physical units, all design points."""
    for idx in np.ndindex(*grid.residual.shape):
        a, b = design.lateral_model(grid.A[idx], grid.B[idx])
        n, m = b.shape
        blk = np.zeros((n + m, n + m))
        blk[:n, :n], blk[:n, n:] = a, b
        import scipy.linalg

        e = scipy.linalg.expm(blk * DT)
        ad, bd = e[:n, :n], e[:n, n:]
        k = gains[idx]
        size = n + m * max(d, 1)
        big = np.zeros((size, size))
        big[:n, :n] = ad
        if d == 0:
            big[:n, :n] += bd @ k[:, :n]
        else:
            big[:n, n + (d - 1) * m : n + d * m] = bd
            big[n : n + m, :n] = k[:, :n]
            for j in range(d):
                big[n : n + m, n + j * m : n + (j + 1) * m] = k[:, n + j * m : n + (j + 1) * m]
            for j in range(1, d):
                big[n + j * m : n + (j + 1) * m, n + (j - 1) * m : n + j * m] = np.eye(m)
        if np.abs(np.linalg.eigvals(big)).max() >= 1.0:
            return False
    return True


def main() -> None:
    t0 = time.time()
    setup = pipeline.build()
    k_lon = design.longitudinal_gains(setup.grid)
    sched = make_schedule(setup.grid, k_lon, setup.lat_gains["gs"])
    axes = sched.k_lat.grids
    shp = setup.grid.residual.shape
    cands = [tuple(w) for w in itertools.product(*(SUB[k] for k in GRID))]
    res = {"description": __doc__, "subgrid": SUB}
    for d in (0, 1, 2, 3):

        @jax.jit
        def run_batch(kv, sc, d=d):
            def one(k):
                lat = make_delayed_lateral(sched, Table(axes, k), d)
                return jax.vmap(lambda s: rollout(setup.plant, sched, lat, s))(sc)

            return jax.vmap(one)(kv)

        tune = scen(SEED, N, d)
        rows = []
        for i in range(0, len(cands), BATCH):
            chunk = cands[i : i + BATCH]
            gains = [design.lateral_gains_dlqr(setup.grid, design.LatWeights(*w), d) for w in chunk]
            pad = gains + gains[-1:] * (BATCH - len(chunk))
            tr = run_batch(jnp.asarray(np.stack([g.reshape(*shp, -1) for g in pad])), tune)
            for j, w in enumerate(chunk):
                tj = jax.tree_util.tree_map(lambda x, j=j: x[j], tr)
                ret = episode_return(tj, tune, ACT_LIMITS)
                rows.append({"weights": dict(zip(GRID, w, strict=True)), "return": float(ret.mean()),
                             "failures": int(np.asarray(tj.failed).any(axis=1).sum())})  # fmt: skip
            print(f"[{time.time() - t0:6.0f}s] d={d} {len(rows)}/{len(cands)}", flush=True)
        ok = sorted((r for r in rows if r["failures"] == 0), key=lambda r: -r["return"])
        best = None
        for r in ok[:10]:  # the design-delay check is cheap but not free
            g = design.lateral_gains_dlqr(setup.grid, design.LatWeights(**r["weights"]), d)
            if stable_at_design_delay(setup.grid, g, None, d):
                best, gbest = r, g
                break
        held = scen(1007, 256, d)
        tr = jax.tree_util.tree_map(
            lambda x: x[0],
            run_batch(jnp.asarray(np.stack([gbest.reshape(*shp, -1)] * BATCH)), held),
        )
        ret = episode_return(tr, held, ACT_LIMITS)
        np.save(R5 / f"dlqr_gains_d{d}.npy", gbest)
        res[f"d{d}"] = {"best": best, "heldout_return": float(ret.mean()),
                        "heldout_return_se": float(ret.std(ddof=1) / np.sqrt(ret.size)),
                        "heldout_failures": int(np.asarray(tr.failed).any(axis=1).sum()),
                        "top5": ok[:5], "candidates": len(rows)}  # fmt: skip
        print(d, best, res[f"d{d}"]["heldout_return"], flush=True)
    res["wall_seconds"] = time.time() - t0
    (R5 / "dlqr.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
