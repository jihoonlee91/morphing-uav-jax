from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import diffsim
from morphjax.rl import PPOConfig, init_params, training_scenario
from morphjax.sim import initial_carry


def _short_return(setup, steps: int = 30, n: int = 2):
    """Mean return of a short deterministic rollout, as a function of the policy."""
    plant, lon = setup.plant, setup.schedules["gs"]
    cfg = PPOConfig(steps=steps)
    scen = jax.vmap(lambda k: training_scenario(k, cfg))(jax.random.split(jax.random.PRNGKey(3), n))
    carry0 = jax.vmap(partial(initial_carry, plant))(scen)

    def ret(params):
        step = partial(diffsim._step, plant, lon, params)

        def body(state, t):
            carry, prev = state
            carry, prev, r = jax.vmap(step, in_axes=(0, 0, 0, None))(carry, prev, scen, t)
            return (carry, prev), r

        _, r = jax.lax.scan(body, (carry0, jnp.zeros((n, 3))), jnp.arange(steps))
        return jnp.mean(jnp.sum(r, axis=0))

    return ret


def test_gradient_through_the_simulator_matches_finite_differences(setup):
    ret = _short_return(setup)
    params = init_params(jax.random.PRNGKey(0), PPOConfig())
    # Perturb only the policy's output bias: a smooth, low-dimensional direction.
    direction = jax.tree_util.tree_map(jnp.zeros_like, params)
    direction["pi"][-1]["b"] = jnp.array([1.0, -0.5, 0.25])
    g = jax.grad(ret)(params)
    analytic = sum(
        float(jnp.vdot(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(direction), strict=True
        )
    )
    eps = 1e-5

    def shifted(s):
        return jax.tree_util.tree_map(lambda p, d: p + s * d, params, direction)

    numeric = (float(ret(shifted(eps))) - float(ret(shifted(-eps)))) / (2 * eps)
    assert abs(analytic) > 1e-6
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3)


def test_one_training_iteration_runs(setup):
    cfg = diffsim.DiffSimConfig(num_envs=2, episode=40, window=20, iters=1)
    init_state, update = diffsim.make_train(setup.plant, setup.schedules["gs"], cfg)
    params, opt = init_state(jax.random.PRNGKey(0))
    new, _, stats = update(params, opt, jax.random.PRNGKey(1))
    assert all(np.isfinite(float(v)) for v in stats.values())
    old, upd = jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(new)
    assert (
        max(
            float(np.abs(np.asarray(a) - np.asarray(b)).max())
            for a, b in zip(old, upd, strict=True)
        )
        > 0
    )
