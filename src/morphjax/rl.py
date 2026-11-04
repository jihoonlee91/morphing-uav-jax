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

import math
import time
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from morphjax.airframe import DEG
from morphjax.control import Schedule, lateral_error_state, lateral_law
from morphjax.dynamics import Plant, W, body_to_wind
from morphjax.scenarios import sample_scenario
from morphjax.sim import Carry, Scenario, closed_loop_step, initial_carry, measure

OBS_DIM = 12
ACT_DIM = 3
ACT_SCALE = jnp.array([25.0, 25.0, 12.5])  # aileron deg, rudder deg, eta_a %
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


# ----------------------------------------------------------------------------- network


def _dense(key, n_in, n_out, scale):
    w = jax.random.orthogonal(key, max(n_in, n_out))[:n_in, :n_out] * scale
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key: jax.Array, cfg: PPOConfig) -> Params:
    k = jax.random.split(key, 6)
    h = cfg.hidden
    return {
        "pi": [_dense(k[0], OBS_DIM, h, math.sqrt(2)), _dense(k[1], h, h, math.sqrt(2)),
               _dense(k[2], h, ACT_DIM, 0.01)],
        "v": [_dense(k[3], OBS_DIM, h, math.sqrt(2)), _dense(k[4], h, h, math.sqrt(2)),
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


def observation(sched: Schedule, xw, phi_cmd, integ_phi):
    dx, u0, rho = lateral_error_state(sched, xw, phi_cmd, integ_phi)
    raw = jnp.concatenate([dx, jnp.array([rho[2], rho[0], rho[1], xw[W.PHI]])])
    return (raw - OBS_SHIFT) / OBS_SCALE, u0


def action_to_command(u0: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
    return u0 + ACT_SCALE * jnp.clip(act, -1.0, 1.0)


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
):
    """Aileron, rudder, and asymmetric-span commands for a policy action.

    Without ``base`` the policy commands around the scheduled trim with full
    authority. With ``base`` it is a bounded residual on top of that
    model-based law (e.g. the certified LPV design): ``u = u_base + r * a``,
    optionally passed through ``residual_filter`` after clipping.
    """
    if base is None:
        return action_to_command(u0, act)
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
) -> Callable:
    """Deterministic policy as a :data:`morphjax.sim.LateralFn`."""

    def lateral(xw, phi_cmd, integ_phi):
        obs, u0 = observation(sched, xw, phi_cmd, integ_phi)
        act = policy_mean(params, obs)
        return lateral_command(xw, phi_cmd, integ_phi, u0, act, base, residual_filter)

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
        return sample_scenario(k_s, cfg.steps, rate)
    scen = sample_scenario(
        k_s, cfg.steps, rate, morph_aero=(0.6, 1.4), inertia=(0.8, 1.2), aileron_effect=(0.6, 1.0)
    )
    k1, k2 = jax.random.split(k_d)
    return scen._replace(
        delay=jax.random.randint(k1, (), 0, 4), noise=jax.random.uniform(k2, maxval=1.5)
    )


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
        obs, u0 = observation(lon, xw_meas, phi_cmd, carry.integ[2])
        mu = policy_mean(params, obs)
        act = mu + jnp.exp(params["log_std"]) * jax.random.normal(k, mu.shape)
        lat = lateral_command(xw_meas, phi_cmd, carry.integ[2], u0, act, base, residual_filter)
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
        lambda c, s: observation(lon, body_to_wind(c.x), s.phi_cmd[-1], c.integ[2])
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
    """Train a policy; returns ``(params, history)``."""
    init_state, update = make_train(plant, lon, cfg, base, residual_filter)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    params, opt_state = init_state(k0)
    history: list[dict[str, float]] = []
    t0 = time.time()
    for i in range(cfg.updates):
        key, k = jax.random.split(key)
        params, opt_state, stats = update(params, opt_state, k)
        row = {k: float(v) for k, v in stats.items()}
        row["update"] = i
        row["env_steps"] = (i + 1) * cfg.num_envs * cfg.steps
        row["wall_s"] = time.time() - t0
        history.append(row)
        if i % 10 == 0 or i == cfg.updates - 1:
            log(
                f"update {i:4d}  steps {row['env_steps']:>9,d}  return {row['return']:7.1f}  "
                f"survival {row['survival']:.2f}  std {row['std']:.3f}  ({row['wall_s']:.0f}s)"
            )
    return params, history


def save_params(path, params: Params) -> None:
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez_compressed(path, *[np.asarray(x) for x in leaves])


def load_params(path, cfg: PPOConfig) -> Params:
    template = init_params(jax.random.PRNGKey(0), cfg)
    _, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path) as f:
        leaves = [jnp.asarray(f[f"arr_{i}"]) for i in range(len(f.files))]
    return jax.tree_util.tree_unflatten(treedef, leaves)
