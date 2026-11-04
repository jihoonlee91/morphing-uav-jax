"""Random test scenarios, generated in JAX so training and evaluation share one sampler.

A scenario is a flight condition, a roll-angle reference (random bank steps
shaped by a second-order reference model, as a pilot command filter would),
a symmetric-morphing schedule with a given rate limit, and optional model
errors, transport delay, and sensor noise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from morphjax.airframe import DEG
from morphjax.sim import DT, Scenario
from morphjax.trim import synthetic_to_true

ROLL_PERIOD = 3.0  # s between bank-angle steps
ROLL_MAX = 30.0  # deg
MORPH_PERIOD = 2.5  # s between symmetric-morphing targets
REF_WN, REF_ZETA = 2.5, 0.9  # roll reference model


def _hold(values: jnp.ndarray, period: float, steps: int) -> jnp.ndarray:
    idx = jnp.minimum((jnp.arange(steps) * DT / period).astype(jnp.int32), values.shape[0] - 1)
    return values[idx]


def roll_reference(key: jax.Array, steps: int) -> jnp.ndarray:
    """Filtered random bank-angle steps (rad), starting wings-level."""
    n = int(steps * DT / ROLL_PERIOD) + 2
    targets = jax.random.uniform(key, (n,), minval=-ROLL_MAX, maxval=ROLL_MAX) * DEG
    targets = targets.at[0].set(0.0)
    raw = _hold(targets, ROLL_PERIOD, steps)
    raw = jnp.where(jnp.arange(steps) * DT < 1.0, 0.0, raw)

    def filt(s, u):
        x, v = s
        a = REF_WN**2 * (u - x) - 2 * REF_ZETA * REF_WN * v
        v = v + DT * a
        x = x + DT * v
        return (x, v), x

    _, ref = jax.lax.scan(filt, (0.0, 0.0), raw)
    return ref


def morph_schedule(key: jax.Array, steps: int, rate: jnp.ndarray) -> jnp.ndarray:
    """Symmetric-span command (%) moving between alternating far-apart targets.

    ``rate`` is the command rate limit in %/s (0 = frozen shape, ``inf`` = steps;
    the 0.5 Hz morphing actuator still filters what the wing actually does).
    """
    k0, k1, k2 = jax.random.split(key, 3)
    n = int(steps * DT / MORPH_PERIOD) + 2
    u = jax.random.uniform(k0, (n,))
    parity = jax.random.bernoulli(k1)
    hi = (jnp.arange(n) % 2 == 0) ^ parity
    frac = jnp.where(hi, 0.6 + 0.4 * u, 0.4 * u)
    targets = -25.0 * frac
    start = -25.0 * jax.random.uniform(k2)
    raw = _hold(targets, MORPH_PERIOD, steps)
    raw = jnp.where(jnp.arange(steps) * DT < 1.0, start, raw)
    step = rate * DT

    def limit(prev, tgt):
        nxt = prev + jnp.clip(tgt - prev, -step, step)
        return nxt, nxt

    _, sched = jax.lax.scan(limit, start, raw)
    return sched


def sample_scenario(
    key: jax.Array,
    steps: int,
    morph_rate: jnp.ndarray | float,
    *,
    v_syn: tuple[float, float] = (70.0, 110.0),
    alt: tuple[float, float] = (500.0, 9500.0),
    morph_aero: tuple[float, float] = (1.0, 1.0),
    inertia: tuple[float, float] = (1.0, 1.0),
    aileron_effect: tuple[float, float] = (1.0, 1.0),
    delay: int = 0,
    noise: float = 0.0,
) -> Scenario:
    ks = jax.random.split(key, 8)
    h = jax.random.uniform(ks[0], minval=alt[0], maxval=alt[1])
    vs = jax.random.uniform(ks[1], minval=v_syn[0], maxval=v_syn[1])

    def u(k, lo_hi):
        return jax.random.uniform(k, minval=lo_hi[0], maxval=lo_hi[1])

    return Scenario(
        v_kt=synthetic_to_true(vs, h),
        h=h,
        phi_cmd=roll_reference(ks[2], steps),
        eta_s_cmd=morph_schedule(ks[3], steps, jnp.asarray(morph_rate, dtype=jnp.float64)),
        morph_aero=u(ks[4], morph_aero),
        inertia=u(ks[5], inertia),
        aileron_effect=u(ks[6], aileron_effect),
        delay=jnp.asarray(delay),
        noise=jnp.asarray(noise),
        key=ks[7],
    )
