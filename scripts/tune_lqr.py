"""Tune the lateral LQR weights for the same objective the PPO policy is trained on.

    uv run python scripts/tune_lqr.py

Grid search over Bryson-scale multipliers (roll error, its integral, sideslip,
asymmetric-morphing cost, overall control cost). Every candidate is the
gain-scheduled LQR, scored by mean episode return under the PPO reward on the
PPO training distribution (nominal model). Scenario seeds are disjoint from
the evaluation seeds.

On a nominal model, return keeps rising with loop gain until the sampled loop
goes unstable, so the search is constrained the way flight-control gains are
in practice: every frozen design point must stay stable with a 60 ms (3-sample)
transport delay, a sampled-data stand-in for the usual phase-margin rule.
The untuned defaults and the unconstrained optimum are recorded too.
Writes ``results/lqr_weights.json``, used by ``evaluate.py`` for all three
model-based designs.
"""

from __future__ import annotations

import itertools
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.dynamics import Cmd
from morphjax.rl import ACT_SCALE, PPOConfig, reward, training_scenario
from morphjax.sim import DT, rollout

ROOT = Path(__file__).resolve().parents[1]
SEED = 7
N = 96
DELAY_REQUIREMENT_STEPS = 3  # 60 ms at 50 Hz
# Multipliers on the default Bryson scales. Without the "effort" axis no
# candidate meets the delay requirement: the default control cost is so low
# that the actuator-state feedback loses stability with one sample of delay.
GRID = {
    "phi": [1.0, 0.5, 0.25, 0.125],
    "integ": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
    "beta": [1.0, 0.5, 0.25, 0.125],
    "eta_a": [1.0, 4.0],
    "effort": [0.25, 0.125, 0.0625, 0.03125],
}


def episode_return(traj, scen) -> np.ndarray:
    """PPO reward summed over each episode (actions ~ commands / ACT_SCALE)."""
    act = traj.cmd[..., jnp.array([Cmd.DA, Cmd.DR, Cmd.ETA_A])] / ACT_SCALE
    prev = jnp.concatenate([jnp.zeros_like(act[:, :1]), jnp.clip(act[:, :-1], -1, 1)], axis=1)
    r = jax.vmap(jax.vmap(reward))(traj.xw, scen.phi_cmd, act, prev)
    alive = ~traj.failed
    term = jnp.concatenate([jnp.zeros_like(alive[:, :1]), alive[:, :-1] & ~alive[:, 1:]], axis=1)
    return np.asarray(jnp.sum(jnp.where(alive, r, 0.0), axis=1) - 10.0 * jnp.sum(term, axis=1))


def main() -> None:
    t0 = time.time()
    setup = pipeline.build(verbose=True)
    cfg = PPOConfig()
    keys = jax.random.split(jax.random.PRNGKey(SEED), N)
    scen = jax.vmap(lambda k: training_scenario(k, cfg))(keys)
    k_lon = design.longitudinal_gains(setup.grid)

    @jax.jit
    def run(sched, scen):
        lat = partial(lateral_law, sched)
        return jax.vmap(lambda s: rollout(setup.plant, sched, lat, s))(scen)

    rows = []
    candidates = [design.LatWeights(**dict(zip(GRID, v, strict=True)))
                  for v in itertools.product(*GRID.values())]  # fmt: skip
    for w in [design.LatWeights(), *candidates]:  # row 0 = untuned defaults
        gains = design.lateral_gains_gs(setup.grid, w)
        margin_ok = design.lateral_delay_margin_ok(setup.grid, gains, DT, DELAY_REQUIREMENT_STEPS)
        traj = run(make_schedule(setup.grid, k_lon, gains), scen)
        ret = episode_return(traj, scen)
        rows.append({"weights": w._asdict(), "mean_return": float(ret.mean()),
                     "failures": int(np.asarray(traj.failed).any(axis=1).sum()),
                     "delay_margin_ok": margin_ok})  # fmt: skip
        print(f"[{time.time() - t0:5.0f}s] {w}  return {ret.mean():7.2f}  "
              f"failures {rows[-1]['failures']}  60ms-stable {margin_ok}")  # fmt: skip

    def score(r):
        return (r["failures"] == 0, r["mean_return"])

    default, searched = rows[0], rows[1:]
    best = max((r for r in searched if r["delay_margin_ok"]), key=score)
    unconstrained = max(searched, key=score)
    out = {
        "weights": best["weights"],
        "mean_return": best["mean_return"],
        "delay_requirement_ms": DELAY_REQUIREMENT_STEPS * DT * 1000,
        "unconstrained_best": unconstrained,
        "default": default,
        "n_scenarios": N,
        "seed": SEED,
        "candidates": searched,
    }
    path = ROOT / "results" / "lqr_weights.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"best {best['weights']} return {best['mean_return']:.2f} "
          f"(untuned {default['mean_return']:.2f}, "
          f"no delay requirement {unconstrained['mean_return']:.2f}); wrote {path}")  # fmt: skip


if __name__ == "__main__":
    main()
