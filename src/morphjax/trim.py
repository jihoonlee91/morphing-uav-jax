"""Steady, wings-level trim and Jacobian linearization with JAX autodiff.

Replaces Simulink ``trim``/``linmod`` from the original MATLAB workflow:
the trim is a damped Newton solve on exact Jacobians, and the linear model is
``jax.jacfwd`` of the wind-axis dynamics at the trim point.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from morphjax.airframe import KT_TO_FPS
from morphjax.dynamics import N_CMD, N_STATE, Cmd, Plant, W, wind_derivative

# Unknowns: alpha beta theta [rad], elevator aileron rudder [deg], throttle [%].
_SCALE_Z = jnp.array([0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 50.0])
# Residual scaling: V-dot, alpha-dot, beta-dot, p q r dots, h-dot.
_SCALE_R = jnp.array([1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])


class TrimPoint(NamedTuple):
    xw: jnp.ndarray  # wind-axis state
    cmd: jnp.ndarray  # actuator commands
    residual: jnp.ndarray  # max |scaled residual|
    converged: jnp.ndarray


def _assemble(z, airspeed_fps, h, eta_s):
    alpha, beta, theta, de, da, dr, thr = z
    xw = jnp.zeros(N_STATE)
    xw = xw.at[W.V].set(airspeed_fps).at[W.ALPHA].set(alpha).at[W.BETA].set(beta)
    xw = xw.at[W.THETA].set(theta).at[W.H].set(h)
    xw = xw.at[W.DE].set(de).at[W.DA].set(da).at[W.DR].set(dr)
    xw = xw.at[W.ETA_S].set(eta_s).at[W.THR].set(thr)
    cmd = jnp.zeros(N_CMD)
    cmd = cmd.at[Cmd.DE].set(de).at[Cmd.DA].set(da).at[Cmd.DR].set(dr)
    cmd = cmd.at[Cmd.ETA_S].set(eta_s).at[Cmd.THR].set(thr)
    return xw, cmd


def _residual(plant: Plant, z, airspeed_fps, h, eta_s):
    xw, cmd = _assemble(z, airspeed_fps, h, eta_s)
    d = wind_derivative(plant, xw, cmd)
    r = jnp.array([d[W.V], d[W.ALPHA], d[W.BETA], d[W.P], d[W.Q], d[W.R], d[W.H]])
    return r / _SCALE_R


@partial(jax.jit, static_argnames=("iters",))
def trim(plant: Plant, airspeed_kt, h, eta_s, iters: int = 40) -> TrimPoint:
    """Trim for straight, level, wings-level flight at ``(V [kt], h [ft], eta_s [%])``.

    With the wings held level, the small lateral asymmetry of the airframe
    (CG offset, table asymmetry) is absorbed by sideslip, aileron and rudder.
    """
    v = airspeed_kt * KT_TO_FPS
    res = partial(_residual, plant, airspeed_fps=v, h=h, eta_s=eta_s)
    z0 = jnp.array([0.08, 0.0, 0.08, 0.0, 0.0, 0.0, 50.0])

    def step(z, _):
        y = z / _SCALE_Z
        f = res(y * _SCALE_Z)
        jac = jax.jacfwd(lambda yy: res(yy * _SCALE_Z))(y)
        lam = 1e-6 * jnp.trace(jac.T @ jac) / 7
        dy = jnp.linalg.solve(jac.T @ jac + lam * jnp.eye(7), -jac.T @ f)
        # Damped step: never move alpha/theta by more than ~3 deg per iteration.
        dy = dy * jnp.minimum(1.0, 0.5 / (jnp.max(jnp.abs(dy)) + 1e-12))
        return (y + dy) * _SCALE_Z, None

    z, _ = jax.lax.scan(step, z0, None, length=iters)
    err = jnp.max(jnp.abs(res(z)))
    xw, cmd = _assemble(z, v, h, eta_s)
    return TrimPoint(xw, cmd, err, err < 1e-8)


@jax.jit
def linearize(plant: Plant, xw: jnp.ndarray, cmd: jnp.ndarray):
    """Continuous-time ``(A, B)`` of the wind-axis model at ``(xw, cmd)``."""
    A = jax.jacfwd(lambda s: wind_derivative(plant, s, cmd))(xw)
    B = jax.jacfwd(lambda c: wind_derivative(plant, xw, c))(cmd)
    return A, B


def synthetic_to_true(v_syn_kt, h, v0: float = 100.0, h0: float = 30000.0):
    """Invert the synthetic airspeed of Eq. (48): ``V_S = V0 - h0/(h0-h) (V0 - V_T)``."""
    return v0 - (h0 - h) / h0 * (v0 - v_syn_kt)


def true_to_synthetic(v_true_kt, h, v0: float = 100.0, h0: float = 30000.0):
    return v0 - h0 / (h0 - h) * (v0 - v_true_kt)
