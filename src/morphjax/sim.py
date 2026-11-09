"""Closed-loop simulation: one jitted ``lax.scan`` per scenario, ``vmap`` over scenarios.

The loop runs at 50 Hz: measure (optional noise), compute commands, delay
them (optional transport delay), and integrate the nonlinear plant with RK4.
A scenario that departs controlled flight is frozen and flagged as failed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from morphjax.airframe import DEG, KT_TO_FPS
from morphjax.control import Schedule, longitudinal_law
from morphjax.dynamics import Cmd, Plant, Uncertainty, W, body_to_wind, rk4_step, wind_to_body
from morphjax.trim import trim

DT = 0.02
MAX_DELAY = 4  # steps (80 ms)
NOISE_STD = jnp.array(
    [1.0, 0.2 * DEG, 0.2 * DEG, 0.5 * DEG, 0.5 * DEG, 0.5 * DEG, 0.2 * DEG, 0.2 * DEG, 0.0,
     0.0, 0.0, 3.0, 0, 0, 0, 0, 0, 0]
)  # fmt: skip
INTEG_LIMITS = jnp.array([30.0, 150.0, 0.3])  # int(V err) ft, int(h err) ft s, int(phi err) rad s

LateralFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


class Scenario(NamedTuple):
    v_kt: jnp.ndarray  # initial / commanded true airspeed
    h: jnp.ndarray  # initial / commanded altitude
    phi_cmd: jnp.ndarray  # (T,) roll-angle command, rad
    eta_s_cmd: jnp.ndarray  # (T,) symmetric span command, %
    morph_aero: jnp.ndarray  # model-error multipliers (1 = nominal)
    inertia: jnp.ndarray
    aileron_effect: jnp.ndarray
    delay: jnp.ndarray  # command transport delay, steps (0..MAX_DELAY)
    noise: jnp.ndarray  # sensor-noise multiplier on NOISE_STD
    key: jnp.ndarray


class Carry(NamedTuple):
    x: jnp.ndarray  # body-axis plant state
    integ: jnp.ndarray  # [int V err, int h err, int phi err]
    buf: jnp.ndarray  # (MAX_DELAY + 1, 6) command history
    failed: jnp.ndarray
    key: jnp.ndarray


class Trajectory(NamedTuple):
    xw: jnp.ndarray  # (T, 18)
    cmd: jnp.ndarray  # (T, 6) commands as computed (before delay)
    failed: jnp.ndarray  # (T,)


def departed(xw: jnp.ndarray) -> jnp.ndarray:
    """Loss of controlled flight (bank, sideslip, stall, speed, or NaN)."""
    bad = (
        (jnp.abs(xw[W.PHI]) > 85 * DEG)
        | (jnp.abs(xw[W.BETA]) > 25 * DEG)
        | (xw[W.ALPHA] < -10 * DEG)
        | (xw[W.ALPHA] > 25 * DEG)
        | (xw[W.V] < 40 * KT_TO_FPS)
    )
    return bad | ~jnp.all(jnp.isfinite(xw))


def initial_carry(plant: Plant, scen: Scenario) -> Carry:
    tp = trim(plant, scen.v_kt, scen.h, scen.eta_s_cmd[0])
    buf = jnp.broadcast_to(tp.cmd, (MAX_DELAY + 1, tp.cmd.shape[0]))
    return Carry(wind_to_body(tp.xw), jnp.zeros(3), buf, jnp.asarray(False), scen.key)


def closed_loop_step(
    plant: Plant,
    lon: Schedule,
    carry: Carry,
    lat_cmd: jnp.ndarray,
    phi_cmd: jnp.ndarray,
    eta_s_cmd: jnp.ndarray,
    scen: Scenario,
    xw_meas: jnp.ndarray,
) -> tuple[Carry, jnp.ndarray]:
    """Apply one control interval given the lateral command; returns the new carry."""
    lon_cmd = longitudinal_law(lon, xw_meas, scen.v_kt * KT_TO_FPS, scen.h, *carry.integ[:2])
    cmd = jnp.zeros(6)
    cmd = cmd.at[Cmd.DE].set(lon_cmd[0]).at[Cmd.THR].set(lon_cmd[1])
    cmd = cmd.at[Cmd.DA].set(lat_cmd[0]).at[Cmd.DR].set(lat_cmd[1]).at[Cmd.ETA_A].set(lat_cmd[2])
    cmd = cmd.at[Cmd.ETA_S].set(eta_s_cmd)

    buf = jnp.roll(carry.buf, 1, axis=0).at[0].set(cmd)
    applied = buf[jnp.clip(scen.delay, 0, MAX_DELAY)]
    unc = Uncertainty(scen.morph_aero, scen.inertia, scen.aileron_effect)
    x_next = rk4_step(plant, carry.x, applied, DT, unc)

    err = jnp.array(
        [
            xw_meas[W.V] - scen.v_kt * KT_TO_FPS,
            xw_meas[W.H] - scen.h,
            xw_meas[W.PHI] - phi_cmd,
        ]
    )
    integ = jnp.clip(carry.integ + DT * err, -INTEG_LIMITS, INTEG_LIMITS)
    failed = carry.failed | departed(body_to_wind(x_next))
    x_next = jnp.where(failed, carry.x, x_next)
    return Carry(x_next, integ, buf, failed, carry.key), cmd


def measure(carry: Carry, scen: Scenario) -> tuple[jnp.ndarray, jnp.ndarray]:
    key, sub = jax.random.split(carry.key)
    xw = body_to_wind(carry.x)
    noisy = xw + scen.noise * NOISE_STD * jax.random.normal(sub, xw.shape)
    return noisy, key


def rollout(plant: Plant, lon: Schedule, lateral: LateralFn, scen: Scenario) -> Trajectory:
    """Simulate one scenario for ``len(scen.phi_cmd)`` steps.

    A lateral function with ``uses_history = True`` (a policy that observes its
    pending commands) is also passed the command buffer ``carry.buf``.
    """
    with_buf = getattr(lateral, "uses_history", False)
    lateral_any: Callable[..., jnp.ndarray] = lateral

    def body(carry: Carry, inputs):
        phi_cmd, eta_s_cmd = inputs
        xw_meas, key = measure(carry, scen)
        carry = carry._replace(key=key)
        if with_buf:
            lat_cmd = lateral_any(xw_meas, phi_cmd, carry.integ[2], carry.buf, scen.delay)
        else:
            lat_cmd = lateral(xw_meas, phi_cmd, carry.integ[2])
        new, cmd = closed_loop_step(plant, lon, carry, lat_cmd, phi_cmd, eta_s_cmd, scen, xw_meas)
        return new, (body_to_wind(carry.x), cmd, carry.failed)

    _, (xw, cmd, failed) = jax.lax.scan(
        body, initial_carry(plant, scen), (scen.phi_cmd, scen.eta_s_cmd)
    )
    return Trajectory(xw, cmd, failed)
