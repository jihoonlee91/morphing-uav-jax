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

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from morphjax.control import Schedule
from morphjax.dynamics import Plant
from morphjax.rl import (
    ACT_LIMITS,
    Params,
    PPOConfig,
    deterministic_step,
    init_params,
    make_eval,
    run_training,
    training_scenario,
)
from morphjax.sim import Carry, Scenario, initial_carry


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
    act_scale: tuple[float, float, float] = ACT_LIMITS
    delay: int = -1  # fixed training delay (samples); see rl.PPOConfig
    history: int = 0  # pending commands in the observation; see rl.PPOConfig
    rate_weight: float = 0.05  # see rl.PPOConfig (round-3 options)
    delay_rand: int = -1
    mask_pending: bool = False
    linear: bool = False
    eval_every: int = 5
    eval_envs: int = 64
    # The cosine learning-rate decay spans every optimizer step (iters x windows).
    # Up to bbafe40 it spanned only ``iters`` while each iteration takes
    # episode // window = 5 steps, so the rate reached its floor at iteration 40;
    # True restores that schedule.
    legacy_lr_schedule: bool = False


def policy_interface(cfg: DiffSimConfig) -> PPOConfig:
    """The PPO-style config that fixes the network, observation, action and scenarios."""
    return PPOConfig(
        hidden=cfg.hidden, steps=cfg.episode, domain_randomization=cfg.domain_randomization,
        act_scale=cfg.act_scale, delay=cfg.delay, history=cfg.history, eval_envs=cfg.eval_envs,
        rate_weight=cfg.rate_weight, delay_rand=cfg.delay_rand, mask_pending=cfg.mask_pending,
        linear=cfg.linear,
    )  # fmt: skip


def _step(plant: Plant, lon: Schedule, params: Params, carry: Carry, prev, scen: Scenario, t,
          cfg: PPOConfig | None = None):  # fmt: skip
    """One deterministic closed-loop step; returns (carry, action, reward)."""
    return deterministic_step(plant, lon, params, carry, prev, scen, t, cfg)


def make_train(plant: Plant, lon: Schedule, cfg: DiffSimConfig):
    """Return ``(init_state, update)``; one update = one window per environment."""
    ppo_like = policy_interface(cfg)
    windows = cfg.episode // cfg.window
    decay_steps = cfg.iters if cfg.legacy_lr_schedule else cfg.iters * windows
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(optax.cosine_decay_schedule(cfg.lr, decay_steps, alpha=0.1)),
    )

    def init_state(key):
        params = init_params(key, ppo_like)
        return params, tx.init(params)

    def window_loss(params, carry, prev, scen, t0):
        step = jax.checkpoint(partial(_step, plant, lon, params, cfg=ppo_like))

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
        # Each window starts from the state the policy reached; the delay buffer is
        # part of that state, so a delayed loop is differentiated through it too.

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
    """Train by gradient ascent through the simulator; returns ``(params, history, timing)``."""
    init_state, update = make_train(plant, lon, cfg)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    params, opt_state = init_state(k0)
    evaluate = make_eval(plant, lon, policy_interface(cfg), cfg.episode)
    return run_training(update, params, opt_state, key, cfg.iters, cfg.num_envs * cfg.episode,
                        evaluate, cfg.eval_every, log, "iter")  # fmt: skip
