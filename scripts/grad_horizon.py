"""How the simulator gradient grows with the differentiated horizon.

    uv run python scripts/grad_horizon.py              # ~6 min
    uv run python scripts/grad_horizon.py --plot-only  # redraw from the JSON

For horizons from 0.2 s to 8 s, computes the norm of d(return)/d(policy
weights) through the closed-loop simulator, averaged over a fixed batch of
training scenarios, for the untrained and the trained diff-sim policy. Long
horizons through lightly damped dynamics are where backpropagation through
time usually explodes; this is why training differentiates 2 s windows.
Writes ``results/grad_horizon.json`` and ``results/figures/grad_horizon.{png,pdf}``.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import diffsim, pipeline
from morphjax.rl import PPOConfig, init_params, load_params, training_scenario
from morphjax.sim import DT, initial_carry

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (10, 25, 50, 100, 200, 400)  # steps at 50 Hz
N = 16


def main() -> None:
    path = ROOT / "results" / "grad_horizon.json"
    if "--plot-only" in sys.argv:
        plot(json.loads(path.read_text(encoding="utf-8")))
        return
    setup = pipeline.build()
    plant, lon = setup.plant, setup.schedules["gs"]
    cfg = PPOConfig(steps=max(HORIZONS))
    scen = jax.vmap(lambda k: training_scenario(k, cfg))(
        jax.random.split(jax.random.PRNGKey(11), N)
    )
    carry0 = jax.vmap(partial(initial_carry, plant))(scen)

    def grad_norm(params, horizon: int):
        def ret(p):
            step = jax.checkpoint(partial(diffsim._step, plant, lon, p))

            def body(state, t):
                carry, prev = state
                carry, prev, r = jax.vmap(step, in_axes=(0, 0, 0, None))(carry, prev, scen, t)
                return (carry, prev), r

            _, r = jax.lax.scan(body, (carry0, jnp.zeros((N, 3))), jnp.arange(horizon))
            return jnp.mean(jnp.sum(r, axis=0))

        value, g = jax.value_and_grad(ret)(params)
        norm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
        return float(value), float(norm)

    policies = {"untrained": init_params(jax.random.PRNGKey(0), PPOConfig())}
    trained = ROOT / "results" / "policies" / "diffsim_nominal_seed0.npz"
    if trained.exists():
        policies["trained (diff-sim seed 0)"] = load_params(trained, PPOConfig())

    out: dict = {"horizons_steps": list(HORIZONS), "dt": DT, "n_scenarios": N}
    out["policies"] = {}
    for label, params in policies.items():
        rows = [grad_norm(params, h) for h in HORIZONS]
        out["policies"][label] = {
            "return": [r[0] for r in rows],
            "grad_norm": [r[1] for r in rows],
        }
        print(
            label, [f"{h * DT:.1f}s: |g|={r[1]:.3g}" for h, r in zip(HORIZONS, rows, strict=True)]
        )
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    plot(out)


def plot(out: dict) -> None:
    from figures import COLORS, INK2, _save, plain_log_ticks, plt

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    secs = np.asarray(out["horizons_steps"]) * out["dt"]
    for (label, rows), color in zip(
        out["policies"].items(), [INK2, COLORS["diffsim"]], strict=False
    ):
        ax.plot(secs, rows["grad_norm"], marker="o", color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(secs, [f"{s:g}" for s in secs])
    plain_log_ticks(ax.yaxis)
    ax.set_xlabel("Differentiated horizon (s, log)")
    ax.set_ylabel("|d return / d weights| (log)")
    ax.set_title("Simulator gradient vs. horizon (training uses 2 s windows)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, ROOT / "results" / "figures" / "grad_horizon.png")


if __name__ == "__main__":
    main()
