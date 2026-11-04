"""Throughput of the JAX model on this machine (CPU).

    uv run python scripts/benchmark.py

Times are measured after JIT compilation (the first call), median of 3.
Writes ``results/benchmark.json``.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import jax

from morphjax import design, diffsim, pipeline, rl
from morphjax.control import lateral_law
from morphjax.scenarios import sample_scenario
from morphjax.sim import rollout

ROOT = Path(__file__).resolve().parents[1]
STEPS = 500


def timed(fn, *args, repeats: int = 3) -> tuple[float, float]:
    """(compile + first run, median steady-state run) in seconds."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        runs.append(time.perf_counter() - t0)
    return first, statistics.median(runs)


def main() -> None:
    setup = pipeline.build()
    plant, sched = setup.plant, setup.schedules["gs"]
    out: dict = {
        "machine": {
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "jax": jax.__version__,
            "device": str(jax.devices()[0].device_kind),
        }
    }

    first, steady = timed(lambda: design.build_grid(plant).A)
    out["trim_and_linearize_63_points_s"] = {"first": first, "steady": steady}
    print(f"trim + linearize, 63 points: {steady:.2f} s (first call {first:.1f} s)")

    lat = partial(lateral_law, sched)
    run = jax.jit(jax.vmap(lambda s: rollout(plant, sched, lat, s)))
    rows = []
    for n in (1, 16, 64, 256, 1024):
        scen = jax.vmap(lambda k: sample_scenario(k, STEPS, 25.0))(
            jax.random.split(jax.random.PRNGKey(0), n)
        )
        _, t = timed(run, scen)
        rows.append({"envs": n, "seconds": t, "env_steps_per_s": n * STEPS / t})
        print(f"closed-loop rollout, {n:5d} envs x {STEPS} steps: {n * STEPS / t:>10,.0f} steps/s")
    out["rollout"] = rows

    key = jax.random.PRNGKey(1)
    cfg = rl.PPOConfig()
    init, update = rl.make_train(plant, sched, cfg)
    p, o = init(key)
    _, t = timed(lambda: update(p, o, key)[2])
    out["ppo_update_s"] = {"seconds": t, "env_steps": cfg.num_envs * cfg.steps}
    print(f"PPO update ({cfg.num_envs * cfg.steps:,} env steps): {t:.2f} s")

    dcfg = diffsim.DiffSimConfig()
    init, update = diffsim.make_train(plant, sched, dcfg)
    p, o = init(key)
    _, t = timed(lambda: update(p, o, key)[2])
    out["diffsim_iteration_s"] = {"seconds": t, "env_steps": dcfg.num_envs * dcfg.episode}
    n_steps = dcfg.num_envs * dcfg.episode
    print(f"diff-sim iteration ({n_steps:,} env steps, with gradients): {t:.2f} s")

    path = ROOT / "results" / "benchmark.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {path}  ({out['machine']})")


if __name__ == "__main__":
    main()
