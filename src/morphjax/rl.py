"""PPO for the lateral-directional controller, written from scratch in JAX.

The learned policy replaces only the lateral law: it sees the same deviation
state and scheduling variables as the model-based controllers and outputs
aileron, rudder, and asymmetric-morphing commands around the scheduled trim.
The longitudinal hold is the shared gain-scheduled LQR.

Episodes have a fixed length and all environments start together, so a whole
batch of episodes is one jitted ``lax.scan``; an environment that departs is
frozen and its remaining steps are masked out of the loss.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from morphjax.airframe import DEG
from morphjax.control import Schedule, lateral_error_state, lateral_law
from morphjax.dynamics import Cmd, Plant, W, body_to_wind
from morphjax.scenarios import sample_scenario
from morphjax.sim import Carry, Scenario, closed_loop_step, initial_carry, measure

OBS_DIM = 12
ACT_DIM = 3
# Action a in [-1, 1] maps to u0 + scale * a. The default is the actuator range the
# model-based laws can use: +/-30 deg of aileron and rudder (the position limits of
# dynamics.SURFACE_LIMITS) and +/-12.5 % asymmetric span (the full span travel about the
# mid position). Policies committed up to bbafe40 were trained with +/-25 deg
# (LEGACY_ACT_SCALE); their JSON configs carry no "act_scale" and load with it.
ACT_LIMITS = (30.0, 30.0, 12.5)  # aileron deg, rudder deg, eta_a %
LEGACY_ACT_SCALE = (25.0, 25.0, 12.5)
ACT_SCALE = jnp.array(ACT_LIMITS)
LAT_CMD = jnp.array([Cmd.DA, Cmd.DR, Cmd.ETA_A])
OBS_SCALE = jnp.array(
    [2 * DEG, 40 * DEG, 20 * DEG, 10 * DEG, 30.0, 30.0, 12.5, 0.2,  # lateral deviation state
     12.5, 30.0, 5000.0, 45 * DEG]  # eta_s, V_syn, h, phi
)  # fmt: skip
OBS_SHIFT = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, -12.5, 90.0, 5000.0, 0.0])
Params = dict[str, Any]


class PPOConfig(NamedTuple):
    num_envs: int = 64
    steps: int = 500  # 10 s episodes at 50 Hz
    updates: int = 400
    epochs: int = 6
    minibatches: int = 16
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 128
    init_log_std: float = -1.0
    domain_randomization: bool = False
    seed: int = 0
    act_scale: tuple[float, float, float] = ACT_LIMITS
    # Fixed command delay (samples) in every training scenario; -1 keeps the
    # scenario's own (0 nominal, random 0-3 with domain randomization).
    delay: int = -1
    # Number of pending (already computed, not yet applied) commands appended
    # to the observation; with a fixed delay of d samples, d makes it Markov.
    history: int = 0
    # Deterministic (mean-action) evaluation on a fixed held-out batch of the
    # training distribution, every ``eval_every`` updates (0 = off).
    eval_every: int = 10
    eval_envs: int = 64


EVAL_SEED = 424242  # held-out evaluation batch during training (disjoint from training keys)


def obs_dim(cfg) -> int:
    return OBS_DIM + ACT_DIM * cfg.history


# ----------------------------------------------------------------------------- network


def _dense(key, n_in, n_out, scale):
    w = jax.random.orthogonal(key, max(n_in, n_out))[:n_in, :n_out] * scale
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key: jax.Array, cfg: PPOConfig) -> Params:
    k = jax.random.split(key, 6)
    h = cfg.hidden
    n = obs_dim(cfg)
    return {
        "pi": [_dense(k[0], n, h, math.sqrt(2)), _dense(k[1], h, h, math.sqrt(2)),
               _dense(k[2], h, ACT_DIM, 0.01)],
        "v": [_dense(k[3], n, h, math.sqrt(2)), _dense(k[4], h, h, math.sqrt(2)),
              _dense(k[5], h, 1, 1.0)],
        "log_std": jnp.full(ACT_DIM, cfg.init_log_std),
    }  # fmt: skip


def _mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def policy_mean(params: Params, obs: jnp.ndarray) -> jnp.ndarray:
    return _mlp(params["pi"], obs)


def value(params: Params, obs: jnp.ndarray) -> jnp.ndarray:
    return _mlp(params["v"], obs)[..., 0]


def log_prob(params: Params, obs, act):
    mu, log_std = policy_mean(params, obs), params["log_std"]
    z = (act - mu) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z * z - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


# ----------------------------------------------------------------------------- environment


def observation(
    sched: Schedule,
    xw,
    phi_cmd,
    integ_phi,
    buf: jnp.ndarray | None = None,
    history: int = 0,
    act_scale=ACT_LIMITS,
):
    """Normalized observation and the trim command ``u0``.

    With ``history = d > 0`` the ``d`` most recent commands in the delay buffer
    ``buf`` (``sim.Carry.buf``; row 0 is the command computed one sample ago) are
    appended as deviations from ``u0`` in action units. Under a ``d``-sample
    transport delay these are exactly the commands still to be applied.
    """
    dx, u0, rho = lateral_error_state(sched, xw, phi_cmd, integ_phi)
    raw = jnp.concatenate([dx, jnp.array([rho[2], rho[0], rho[1], xw[W.PHI]])])
    obs = (raw - OBS_SHIFT) / OBS_SCALE
    if history > 0:
        assert buf is not None, "a history policy needs the command buffer"
        pending = (buf[:history][:, LAT_CMD] - u0) / jnp.asarray(act_scale)
        obs = jnp.concatenate([obs, pending.reshape(-1)])
    return obs, u0


def action_to_command(u0: jnp.ndarray, act: jnp.ndarray, act_scale=ACT_LIMITS) -> jnp.ndarray:
    return u0 + jnp.asarray(act_scale) * jnp.clip(act, -1.0, 1.0)


# Residual mode: the policy adds at most 20 % of each actuator range
# (aileron, rudder in deg; eta_a in %) on top of a model-based law.
RESIDUAL_SCALE = 0.2 * jnp.array([30.0, 30.0, 12.5])


# (rho, dx, du) -> du: a filter applied to a residual, e.g. morphjax.safety.project.
ResidualFilter = Callable[[tuple, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def lateral_command(
    xw,
    phi_cmd,
    integ_phi,
    u0,
    act,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
    act_scale=ACT_LIMITS,
):
    """Aileron, rudder, and asymmetric-span commands for a policy action.

    Without ``base`` the policy commands around the scheduled trim with full
    authority. With ``base`` it is a bounded residual on top of that
    model-based law (e.g. the certified LPV design): ``u = u_base + r * a``,
    optionally passed through ``residual_filter`` after clipping.
    """
    if base is None:
        return action_to_command(u0, act, act_scale)
    du = RESIDUAL_SCALE * jnp.clip(act, -1.0, 1.0)
    if residual_filter is not None:
        dx, _, rho = lateral_error_state(base, xw, phi_cmd, integ_phi)
        du = residual_filter(rho, dx, du)
    return lateral_law(base, xw, phi_cmd, integ_phi) + du


def make_policy_lateral(
    sched: Schedule,
    params: Params,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
    cfg: PPOConfig | None = None,
) -> Callable:
    """Deterministic policy as a :data:`morphjax.sim.LateralFn`.

    ``cfg`` supplies the action scale and the observed command history (see
    :func:`load_policy`); a history policy is called with the delay buffer too.
    """
    cfg = cfg or PPOConfig()

    def lateral(xw, phi_cmd, integ_phi, buf=None):
        obs, u0 = observation(sched, xw, phi_cmd, integ_phi, buf, cfg.history, cfg.act_scale)
        act = policy_mean(params, obs)
        return lateral_command(
            xw, phi_cmd, integ_phi, u0, act, base, residual_filter, cfg.act_scale
        )

    lateral.uses_history = cfg.history > 0  # type: ignore[attr-defined]
    return lateral


def reward(xw_true, phi_cmd, act, prev_act):
    e_phi = (xw_true[W.PHI] - phi_cmd) / (5 * DEG)
    beta = xw_true[W.BETA] / (1.5 * DEG)
    p = xw_true[W.P] / (30 * DEG)
    shaped = 0.6 * jnp.exp(-(e_phi**2)) + 0.3 * jnp.exp(-(beta**2)) + 0.1 * jnp.exp(-(p**2))
    a = jnp.clip(act, -1.0, 1.0)
    return shaped - 0.05 * jnp.sum((a - prev_act) ** 2) - 0.01 * jnp.sum(a**2)


def training_scenario(key, cfg: PPOConfig) -> Scenario:
    """Training distribution: random flight condition and morphing rate.

    With ``domain_randomization`` the model errors, delay and noise of the
    robustness tests are also randomized during training.
    """
    k_rate, k_kind, k_s, k_d = jax.random.split(key, 4)
    kind = jax.random.randint(k_kind, (), 0, 5)
    log_rate = jax.random.uniform(k_rate, minval=0.0, maxval=math.log(60.0))
    rate = jnp.where(kind == 0, 0.0, jnp.where(kind == 1, jnp.inf, jnp.exp(log_rate)))
    if not cfg.domain_randomization:
        scen = sample_scenario(k_s, cfg.steps, rate)
    else:
        scen = sample_scenario(
            k_s, cfg.steps, rate,
            morph_aero=(0.6, 1.4), inertia=(0.8, 1.2), aileron_effect=(0.6, 1.0),
        )  # fmt: skip
        k1, k2 = jax.random.split(k_d)
        scen = scen._replace(
            delay=jax.random.randint(k1, (), 0, 4), noise=jax.random.uniform(k2, maxval=1.5)
        )
    delay = getattr(cfg, "delay", -1)
    if delay >= 0:
        scen = scen._replace(delay=jnp.asarray(delay))
    return scen


def deterministic_step(
    plant: Plant,
    lon: Schedule,
    params: Params,
    carry: Carry,
    prev,
    scen: Scenario,
    t,
    cfg: PPOConfig | None = None,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
):
    """One closed-loop step under the mean action; returns ``(carry, action, reward)``.

    The reward is the training reward, ``-10`` at a departure and ``0`` after it,
    so summing it over an episode gives the return the learners maximize.
    """
    cfg = cfg or PPOConfig()
    xw_meas, key = measure(carry, scen)
    carry = carry._replace(key=key)
    phi_cmd, eta_cmd = scen.phi_cmd[t], scen.eta_s_cmd[t]
    obs, u0 = observation(
        lon, xw_meas, phi_cmd, carry.integ[2], carry.buf, cfg.history, cfg.act_scale
    )
    act = policy_mean(params, obs)
    lat = lateral_command(
        xw_meas, phi_cmd, carry.integ[2], u0, act, base, residual_filter, cfg.act_scale
    )
    new, _ = closed_loop_step(plant, lon, carry, lat, phi_cmd, eta_cmd, scen, xw_meas)
    r = reward(body_to_wind(new.x), phi_cmd, act, prev)
    alive = ~carry.failed
    r = jnp.where(new.failed & alive, -10.0, jnp.where(alive, r, 0.0))
    return new, jnp.clip(act, -1.0, 1.0), r


def eval_scenarios(cfg: PPOConfig, steps: int) -> Scenario:
    """The fixed held-out batch of the training distribution used for learning curves."""
    keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED), cfg.eval_envs)
    c = cfg._replace(steps=steps)
    return jax.vmap(lambda k: training_scenario(k, c))(keys)


def make_eval(
    plant: Plant,
    lon: Schedule,
    cfg: PPOConfig,
    steps: int,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
):
    """Jitted ``params -> (mean return, survival)`` of the mean policy on the held-out batch."""
    scen = eval_scenarios(cfg, steps)

    @jax.jit
    def evaluate(params):
        carry0 = jax.vmap(partial(initial_carry, plant))(scen)
        step = partial(
            deterministic_step, plant, lon, params, cfg=cfg, base=base,
            residual_filter=residual_filter,
        )  # fmt: skip

        def body(state, t):
            carry, prev = state
            carry, prev, r = jax.vmap(step, in_axes=(0, 0, 0, None))(carry, prev, scen, t)
            return (carry, prev), r

        (carry, _), r = jax.lax.scan(
            body, (carry0, jnp.zeros((cfg.eval_envs, ACT_DIM))), jnp.arange(steps)
        )
        return jnp.mean(jnp.sum(r, axis=0)), 1.0 - jnp.mean(carry.failed)

    return evaluate


def run_training(update, params, opt_state, key, iters: int, steps_per_iter: int,
                 evaluate=None, eval_every: int = 0, log: Callable[[str], None] = print,
                 label: str = "update"):  # fmt: skip
    """Shared training loop of PPO and diff-sim, with compile time and evaluation separated.

    ``update`` is compiled ahead of time and timed on its own (``compile_s``), so
    ``wall_s`` / ``cpu_s`` in the history count training only: no JIT compilation
    and no held-out evaluation. Rows with ``eval_return`` carry the deterministic
    (mean-action) return on the fixed held-out batch after that update; update
    ``-1`` is the untrained policy.
    """
    t_c = time.time()
    compiled = update.lower(params, opt_state, jax.random.PRNGKey(0)).compile()
    compile_s = time.time() - t_c
    eval_compile_s = 0.0
    history: list[dict[str, float]] = []
    if evaluate is not None and eval_every:
        t_c = time.time()
        evaluate = evaluate.lower(params).compile()
        eval_compile_s = time.time() - t_c
        ret, surv = evaluate(params)
        history.append({"update": -1, "env_steps": 0, "wall_s": 0.0, "cpu_s": 0.0,
                        "eval_return": float(ret), "eval_survival": float(surv)})  # fmt: skip
    wall = cpu = 0.0
    for i in range(iters):
        key, k = jax.random.split(key)
        t0, c0 = time.time(), time.process_time()
        params, opt_state, stats = compiled(params, opt_state, k)
        row = {k: float(v) for k, v in stats.items()}  # blocks until the update is done
        wall += time.time() - t0
        cpu += time.process_time() - c0
        row.update(update=i, env_steps=(i + 1) * steps_per_iter, wall_s=wall, cpu_s=cpu)
        if evaluate is not None and eval_every and ((i + 1) % eval_every == 0 or i == iters - 1):
            ret, surv = evaluate(params)
            row["eval_return"], row["eval_survival"] = float(ret), float(surv)
        history.append(row)
        if i % 10 == 0 or i == iters - 1:
            extra = f"  eval {row['eval_return']:7.1f}" if "eval_return" in row else ""
            log(
                f"{label} {i:4d}  steps {row['env_steps']:>9,d}  return {row['return']:7.1f}"
                f"{extra}  survival {row['survival']:.2f}  ({wall:.0f}s)"
            )
    timing = {"compile_s": compile_s, "eval_compile_s": eval_compile_s, "train_wall_s": wall,
              "train_cpu_s": cpu}  # fmt: skip
    return params, history, timing


class Batch(NamedTuple):
    obs: jnp.ndarray
    act: jnp.ndarray
    logp: jnp.ndarray
    val: jnp.ndarray
    rew: jnp.ndarray
    valid: jnp.ndarray  # step happened before any departure
    term: jnp.ndarray  # departure happened at this step


def collect(
    plant: Plant,
    lon: Schedule,
    params: Params,
    key,
    cfg: PPOConfig,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
):
    """Run ``num_envs`` episodes in parallel; returns a time-major batch."""
    k_scen, k_act = jax.random.split(key)
    scen = jax.vmap(lambda k: training_scenario(k, cfg))(jax.random.split(k_scen, cfg.num_envs))
    carry0 = jax.vmap(partial(initial_carry, plant))(scen)

    def env_step(carry: Carry, prev_act, scen: Scenario, t, k):
        xw_meas, key = measure(carry, scen)
        carry = carry._replace(key=key)
        phi_cmd, eta_cmd = scen.phi_cmd[t], scen.eta_s_cmd[t]
        obs, u0 = observation(
            lon, xw_meas, phi_cmd, carry.integ[2], carry.buf, cfg.history, cfg.act_scale
        )
        mu = policy_mean(params, obs)
        act = mu + jnp.exp(params["log_std"]) * jax.random.normal(k, mu.shape)
        lat = lateral_command(
            xw_meas, phi_cmd, carry.integ[2], u0, act, base, residual_filter, cfg.act_scale
        )
        new, _ = closed_loop_step(plant, lon, carry, lat, phi_cmd, eta_cmd, scen, xw_meas)
        r = reward(body_to_wind(new.x), phi_cmd, act, prev_act)
        term = new.failed & ~carry.failed
        r = jnp.where(term, -10.0, r)
        out = (obs, act, log_prob(params, obs, act), value(params, obs), r, ~carry.failed, term)
        return new, jnp.clip(act, -1.0, 1.0), out

    def body(state, t):
        carry, prev = state
        keys = jax.random.split(jax.random.fold_in(k_act, t), cfg.num_envs)
        carry, prev, out = jax.vmap(env_step, in_axes=(0, 0, 0, None, 0))(
            carry, prev, scen, t, keys
        )
        return (carry, prev), out

    init = (carry0, jnp.zeros((cfg.num_envs, ACT_DIM)))
    (carry, _), out = jax.lax.scan(body, init, jnp.arange(cfg.steps))
    final_obs, _ = jax.vmap(
        lambda c, s: observation(
            lon, body_to_wind(c.x), s.phi_cmd[-1], c.integ[2], c.buf, cfg.history, cfg.act_scale
        )
    )(carry, scen)
    last_val = jnp.where(carry.failed, 0.0, value(params, final_obs))
    return Batch(*out), last_val


def gae(batch: Batch, last_val, gamma: float, lam: float):
    """Generalized advantage estimation over a time-major batch."""
    next_val = jnp.concatenate([batch.val[1:], last_val[None]], axis=0)
    next_nonterm = 1.0 - batch.term.astype(jnp.float64)
    delta = batch.rew + gamma * next_val * next_nonterm - batch.val

    def back(adv, inp):
        d, nt = inp
        adv = d + gamma * lam * nt * adv
        return adv, adv

    _, adv = jax.lax.scan(back, jnp.zeros_like(last_val), (delta, next_nonterm), reverse=True)
    return adv, adv + batch.val


def ppo_loss(params, cfg: PPOConfig, obs, act, logp_old, adv, ret, mask):
    logp = log_prob(params, obs, act)
    ratio = jnp.exp(logp - logp_old)
    # The mean is masked but the std is not: steps after a departure (whose
    # advantages are ~0) slightly shrink the std. Kept as-is so the committed
    # policies stay reproducible; a masked std is the cleaner choice.
    adv = (adv - jnp.sum(adv * mask) / jnp.sum(mask)) / (jnp.std(adv) + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
    vf = (value(params, obs) - ret) ** 2
    ent = jnp.sum(params["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    n = jnp.maximum(jnp.sum(mask), 1.0)
    loss = jnp.sum((pg + cfg.vf_coef * vf) * mask) / n - cfg.ent_coef * ent
    return loss, (jnp.sum(pg * mask) / n, jnp.sum(vf * mask) / n)


def make_train(
    plant: Plant,
    lon: Schedule,
    cfg: PPOConfig,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
):
    """Return ``(init_state, update)``; ``update`` is one jitted PPO iteration."""
    sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, cfg.updates * cfg.epochs * cfg.minibatches)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched, eps=1e-5))

    def init_state(key):
        params = init_params(key, cfg)
        return params, tx.init(params)

    @jax.jit
    def update(params, opt_state, key):
        k_col, k_perm = jax.random.split(key)
        batch, last_val = collect(plant, lon, params, k_col, cfg, base, residual_filter)
        adv, ret = gae(batch, last_val, cfg.gamma, cfg.lam)
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            (batch.obs, batch.act, batch.logp, adv, ret, batch.valid.astype(jnp.float64)),
        )
        n = flat[0].shape[0]
        mb = n // cfg.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, n)[: mb * cfg.minibatches].reshape(cfg.minibatches, mb)

            def minibatch(carry, idx):
                params, opt_state = carry
                data = jax.tree_util.tree_map(lambda x: x[idx], flat)
                (_, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, cfg, *data)
                upd, opt_state = tx.update(grads, opt_state, params)
                return (cast(Params, optax.apply_updates(params, upd)), opt_state), aux

            return jax.lax.scan(minibatch, (params, opt_state), perm)

        (params, opt_state), aux = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(k_perm, cfg.epochs)
        )
        valid = batch.valid.astype(jnp.float64)
        stats = {
            "return": jnp.mean(jnp.sum(batch.rew * valid, axis=0)),
            "survival": 1.0 - jnp.mean(jnp.any(batch.term, axis=0)),
            "pg_loss": jnp.mean(aux[0]),
            "v_loss": jnp.mean(aux[1]),
            "std": jnp.mean(jnp.exp(params["log_std"])),
        }
        return params, opt_state, stats

    return init_state, update


def train(
    plant: Plant,
    lon: Schedule,
    cfg: PPOConfig,
    log: Callable[[str], None] = print,
    base: Schedule | None = None,
    residual_filter: ResidualFilter | None = None,
):
    """Train a policy; returns ``(params, history, timing)`` (see :func:`run_training`)."""
    init_state, update = make_train(plant, lon, cfg, base, residual_filter)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    params, opt_state = init_state(k0)
    evaluate = make_eval(plant, lon, cfg, cfg.steps, base, residual_filter)
    return run_training(update, params, opt_state, key, cfg.updates, cfg.num_envs * cfg.steps,
                        evaluate, cfg.eval_every, log, "update")  # fmt: skip


def save_params(path, params: Params) -> None:
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez_compressed(path, *[np.asarray(x) for x in leaves])


def load_params(path, cfg: PPOConfig) -> Params:
    template = init_params(jax.random.PRNGKey(0), cfg)
    _, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path) as f:
        leaves = [jnp.asarray(f[f"arr_{i}"]) for i in range(len(f.files))]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def policy_config(meta_config: dict) -> PPOConfig:
    """The interface (network width, action scale, observed history) of a saved policy.

    Configs written before the revision have no ``act_scale`` and were trained
    with :data:`LEGACY_ACT_SCALE`.
    """
    return PPOConfig(
        hidden=int(meta_config.get("hidden", 128)),
        act_scale=tuple(meta_config.get("act_scale", LEGACY_ACT_SCALE)),
        history=int(meta_config.get("history", 0)),
        delay=int(meta_config.get("delay", -1)),
    )


def load_policy(path) -> tuple[Params, PPOConfig]:
    """Parameters and interface of ``<stem>.npz``, read with its ``<stem>.json``."""
    meta = Path(path).with_suffix(".json")
    conf = json.loads(meta.read_text(encoding="utf-8"))["config"] if meta.exists() else {}
    cfg = policy_config(conf)
    return load_params(path, cfg), cfg
