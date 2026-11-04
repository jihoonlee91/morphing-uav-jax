"""Policy optimization by backpropagating through the simulator.

The third way to get a controller from the same model. Model-based synthesis
uses the model's *linearizations*; PPO uses only *samples*; this uses the
model's *gradients*. Because the whole closed loop (aero tables, 6-DOF
dynamics, actuators, longitudinal hold) is written in JAX, the episode return
is differentiable with respect to the policy weights, and the policy can be
trained by plain gradient ascent on it (backpropagation through time,
sometimes called analytic policy gradients).

The policy has the same architecture, observation, and action interface as
the PPO policy, so the two are directly comparable and share
:func:`morphjax.rl.make_policy_lateral`. Gradients through a long horizon of a
lightly damped system can explode, so each update differentiates through a
fixed window (default 2 s) starting from states reached by the current policy,
with gradient clipping. ``jax.checkpoint`` on the step keeps memory linear in
the state, not in the intermediate activations.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from morphjax.control import Schedule
from morphjax.dynamics import Plant, body_to_wind
from morphjax.rl import (
    Params,
    PPOConfig,
    action_to_command,
    init_params,
    observation,
    policy_mean,
    reward,
    training_scenario,
)
from morphjax.sim import Carry, Scenario, closed_loop_step, initial_carry, measure


class DiffSimConfig(NamedTuple):
    num_envs: int = 64
    episode: int = 500  # 10 s scenarios, same distribution as PPO training
    window: int = 100  # steps differentiated per update (2 s)
    iters: int = 200
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    hidden: int = 128
    domain_randomization: bool = False
    seed: int = 0


def _step(plant: Plant, lon: Schedule, params: Params, carry: Carry, prev, scen: Scenario, t):
    """One deterministic closed-loop step; returns (carry, action, reward)."""
    xw_meas, key = measure(carry, scen)
    carry = carry._replace(key=key)
    phi_cmd, eta_cmd = scen.phi_cmd[t], scen.eta_s_cmd[t]
    obs, u0 = observation(lon, xw_meas, phi_cmd, carry.integ[2])
    act = policy_mean(params, obs)
    new, _ = closed_loop_step(
        plant, lon, carry, action_to_command(u0, act), phi_cmd, eta_cmd, scen, xw_meas
    )
    r = reward(body_to_wind(new.x), phi_cmd, act, prev)
    alive = ~carry.failed
    r = jnp.where(new.failed & alive, -10.0, jnp.where(alive, r, 0.0))
    return new, jnp.clip(act, -1.0, 1.0), r


def make_train(plant: Plant, lon: Schedule, cfg: DiffSimConfig):
    """Return ``(init_state, update)``; one update = one window per environment."""
    ppo_like = PPOConfig(
        hidden=cfg.hidden, steps=cfg.episode, domain_randomization=cfg.domain_randomization
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(optax.cosine_decay_schedule(cfg.lr, cfg.iters, alpha=0.1)),
    )
    windows = cfg.episode // cfg.window

    def init_state(key):
        params = init_params(key, ppo_like)
        return params, tx.init(params)

    def window_loss(params, carry, prev, scen, t0):
        step = jax.checkpoint(partial(_step, plant, lon, params))

        def body(state, t):
            carry, prev = state
            carry, prev, r = jax.vmap(step, in_axes=(0, 0, 0, None))(carry, prev, scen, t)
            return (carry, prev), r

        (carry, prev), rew = jax.lax.scan(body, (carry, prev), t0 + jnp.arange(cfg.window))
        return -jnp.mean(jnp.sum(rew, axis=0)), (carry, prev, jnp.mean(jnp.sum(rew, axis=0)))

    grad_fn = jax.value_and_grad(window_loss, has_aux=True)

    @jax.jit
    def update(params, opt_state, key):
        """Fresh scenarios; gradient steps on consecutive windows of the episode."""
        scen = jax.vmap(lambda k: training_scenario(k, ppo_like))(
            jax.random.split(key, cfg.num_envs)
        )
        carry = jax.lax.stop_gradient(jax.vmap(partial(initial_carry, plant))(scen))
        prev = jnp.zeros((cfg.num_envs, 3))

        def one_window(state, w):
            params, opt_state, carry, prev = state
            (_, (carry, prev, ret)), g = grad_fn(params, carry, prev, scen, w * cfg.window)
            upd, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            carry, prev = jax.lax.stop_gradient((carry, prev))
            return (params, opt_state, carry, prev), (ret, optax.tree.norm(g))

        (params, opt_state, carry, _), (rets, gnorm) = jax.lax.scan(
            one_window, (params, opt_state, carry, prev), jnp.arange(windows)
        )
        stats = {
            "return": jnp.sum(rets),
            "survival": 1.0 - jnp.mean(carry.failed),
            "grad_norm": jnp.mean(gnorm),
        }
        return params, opt_state, stats

    return init_state, update


def train(plant: Plant, lon: Schedule, cfg: DiffSimConfig, log: Callable[[str], None] = print):
    """Train a policy by gradient ascent through the simulator."""
    init_state, update = make_train(plant, lon, cfg)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    params, opt_state = init_state(k0)
    history: list[dict[str, float]] = []
    t0 = time.time()
    for i in range(cfg.iters):
        key, k = jax.random.split(key)
        params, opt_state, stats = update(params, opt_state, k)
        row = {k: float(v) for k, v in stats.items()}
        row["update"] = i
        row["env_steps"] = (i + 1) * cfg.num_envs * cfg.episode
        row["wall_s"] = time.time() - t0
        history.append(row)
        if i % 10 == 0 or i == cfg.iters - 1:
            log(
                f"iter {i:4d}  steps {row['env_steps']:>9,d}  return {row['return']:7.1f}  "
                f"survival {row['survival']:.2f}  |g| {row['grad_norm']:.2e}  "
                f"({row['wall_s']:.0f}s)"
            )
    return params, history
