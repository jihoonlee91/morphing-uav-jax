"""Six-degree-of-freedom equations of motion with actuator dynamics.

Flat Earth, constant gravity, quasi-static morphing (Lee et al. 2023, Sec. 2.1):
mass properties and aerodynamics follow the current wing shape, and the
unsteady effect of the shape change itself is not modelled.

State (18):  u v w [ft/s], p q r [rad/s], phi theta psi [rad], north east h [ft],
             elevator aileron rudder [deg], eta_l eta_r [%], throttle [%]
Command (6): elevator aileron rudder [deg], eta_s eta_a [%], throttle [%]

Differences from NASA GTM_DesignSim (documented in the README): no sensor
models, no landing gear, spoilers or flaps, a static engine map with a
first-order lag instead of the full JetCat P70 model, and actuators modelled
as first-order lags with position limits (no rate limits or dead zone).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from morphjax.aero import AeroModel, coefficients
from morphjax.airframe import DEG, Airframe, G, mass_properties
from morphjax.atmosphere import density, pressure_ratio

N_STATE = 18
N_CMD = 6


class Ix:
    """Named indices into the state vector."""

    U, V, W, P, Q, R, PHI, THETA, PSI, NORTH, EAST, H = range(12)
    DE, DA, DR, ETA_L, ETA_R, THR = range(12, 18)


class Cmd:
    """Named indices into the command vector."""

    DE, DA, DR, ETA_S, ETA_A, THR = range(6)


# Actuators: 5 Hz control surfaces, 0.5 Hz morphing (paper Sec. 3), engine lag.
OMEGA_SURFACE = 2 * jnp.pi * 5.0
OMEGA_MORPH = 2 * jnp.pi * 0.5
TAU_ENGINE = 0.5
SURFACE_LIMITS = jnp.array([[-30.0, 20.0], [-30.0, 30.0], [-30.0, 30.0]])

# JetCat P70 static map from AC_calcparams_T2.m: handle % -> thrust per engine (lbf).
_HANDLE = jnp.array(
    [0.0, 6, 12, 19, 24, 30, 33, 37, 42, 48, 54.5, 60, 66, 72, 84, 100], dtype=jnp.float64
)
_THRUST = jnp.array(
    [0.8776, 1.2515, 1.7358, 2.4243, 2.9855, 3.7211, 4.1103, 4.6478,
     5.3449, 6.2119, 7.1828, 8.0279, 8.9759, 9.9562, 12.0519, 15.3152]
)  # fmt: skip


class Uncertainty(NamedTuple):
    """Multiplicative model errors for robustness tests (1.0 = nominal)."""

    morph_aero: jnp.ndarray | float = 1.0
    inertia: jnp.ndarray | float = 1.0
    aileron_effect: jnp.ndarray | float = 1.0


class Plant(NamedTuple):
    aero: AeroModel
    airframe: Airframe = Airframe()
    # Optional actuator rate limits (round-3 sensitivity study only; None = the model of
    # the paper, which has none): (control surfaces deg/s, each wingtip %/s).
    rate_limits: tuple[float, float] | None = None


def engine_thrust(throttle: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Thrust of one engine (lbf), scaled by ambient pressure."""
    return jnp.interp(jnp.clip(throttle, 0.0, 100.0), _HANDLE, _THRUST) * pressure_ratio(h)


def morph_commands(eta_s: jnp.ndarray, eta_a: jnp.ndarray, saturate: bool = True):
    """Left/right span commands from symmetric/asymmetric parts, Eqs. (22)-(23)."""
    eta_l, eta_r = eta_s + eta_a, eta_s - eta_a
    if saturate:
        eta_l, eta_r = jnp.clip(eta_l, -25.0, 0.0), jnp.clip(eta_r, -25.0, 0.0)
    return eta_l, eta_r


def air_data(x: jnp.ndarray):
    """True airspeed (ft/s), angle of attack and sideslip (rad)."""
    u, v, w = x[Ix.U], x[Ix.V], x[Ix.W]
    vt = jnp.sqrt(u * u + v * v + w * w)
    alpha = jnp.arctan2(w, u)
    beta = jnp.arcsin(jnp.clip(v / jnp.maximum(vt, 1e-6), -1.0, 1.0))
    return vt, alpha, beta


def dcm_body_to_ned(phi, theta, psi) -> jnp.ndarray:
    cf, sf = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    return jnp.array(
        [
            [ct * cp, -cf * sp + sf * st * cp, sf * sp + cf * st * cp],
            [ct * sp, cf * cp + sf * st * sp, -sf * cp + cf * st * sp],
            [-st, sf * ct, cf * ct],
        ]
    )


def forces_moments(plant: Plant, x: jnp.ndarray, unc: Uncertainty = Uncertainty()):
    """Body-axis force (lbf) and moment about the CG (lbf ft), plus mass data."""
    af = plant.airframe
    vt, alpha, beta = air_data(x)
    pqr = x[Ix.P : Ix.R + 1]
    h = x[Ix.H]
    # A degraded aileron (e.g. partial actuator failure) scales its deflection.
    surfaces = x[Ix.DE : Ix.DR + 1] * jnp.array([1.0, 1.0, 1.0]).at[1].set(unc.aileron_effect)
    eta_l, eta_r = x[Ix.ETA_L], x[Ix.ETA_R]

    m, rcg, J = mass_properties(eta_l, eta_r, af)
    J = J * unc.inertia
    c = coefficients(
        plant.aero, alpha, beta, vt, pqr, surfaces, eta_l, eta_r, af.b, af.cbar, unc.morph_aero
    )
    qbar = 0.5 * density(h) * vt * vt
    f_aero = qbar * af.S * c[:3]
    m_aero = qbar * af.S * jnp.array([af.b, af.cbar, af.b]) * c[3:]
    m_aero = m_aero + jnp.cross(af.aero_ref - rcg, f_aero)

    t_eng = engine_thrust(x[Ix.THR], h)
    f_thr = jnp.array([2.0 * t_eng, 0.0, 0.0])
    m_thr = sum(
        (jnp.cross(jnp.asarray(pos) - rcg, jnp.array([t_eng, 0.0, 0.0])) for pos in af.engine_pos),
        start=jnp.zeros(3),
    )
    return f_aero + f_thr, m_aero + m_thr, m, J


def actuator_rates(
    x: jnp.ndarray, cmd: jnp.ndarray, saturate: bool = True, rate_limits=None
) -> jnp.ndarray:
    """Time derivatives of the six actuator states."""
    surf_cmd = cmd[Cmd.DE : Cmd.DR + 1]
    if saturate:
        surf_cmd = jnp.clip(surf_cmd, SURFACE_LIMITS[:, 0], SURFACE_LIMITS[:, 1])
    eta_l_c, eta_r_c = morph_commands(cmd[Cmd.ETA_S], cmd[Cmd.ETA_A], saturate)
    thr_c = jnp.clip(cmd[Cmd.THR], 0.0, 100.0) if saturate else cmd[Cmd.THR]
    rates = jnp.concatenate(
        [
            OMEGA_SURFACE * (surf_cmd - x[Ix.DE : Ix.DR + 1]),
            OMEGA_MORPH * (jnp.array([eta_l_c, eta_r_c]) - x[Ix.ETA_L : Ix.ETA_R + 1]),
            jnp.array([(thr_c - x[Ix.THR]) / TAU_ENGINE]),
        ]
    )
    if rate_limits is not None:
        surf, morph = rate_limits
        lim = jnp.array([surf, surf, surf, morph, morph, jnp.inf])
        rates = jnp.clip(rates, -lim, lim)
    return rates


def state_derivative(
    plant: Plant,
    x: jnp.ndarray,
    cmd: jnp.ndarray,
    unc: Uncertainty = Uncertainty(),
    saturate: bool = True,
) -> jnp.ndarray:
    """``dx/dt`` of the nonlinear model, Eqs. (1)-(14) of the paper."""
    force, moment, m, J = forces_moments(plant, x, unc)
    u, v, w, p, q, r, phi, theta, psi = x[:9]
    sf, cf, st, ct = jnp.sin(phi), jnp.cos(phi), jnp.sin(theta), jnp.cos(theta)

    gravity = G * jnp.array([-st, sf * ct, cf * ct])
    vel = jnp.array([u, v, w])
    omega = jnp.array([p, q, r])
    vel_dot = -jnp.cross(omega, vel) + force / m + gravity
    omega_dot = jnp.linalg.solve(J, moment - jnp.cross(omega, J @ omega))
    euler_dot = jnp.array(
        [
            p + st / ct * (q * sf + r * cf),
            q * cf - r * sf,
            (q * sf + r * cf) / ct,
        ]
    )
    ned_dot = dcm_body_to_ned(phi, theta, psi) @ vel
    pos_dot = jnp.array([ned_dot[0], ned_dot[1], -ned_dot[2]])
    return jnp.concatenate(
        [
            vel_dot,
            omega_dot,
            euler_dot,
            pos_dot,
            actuator_rates(x, cmd, saturate, plant.rate_limits),
        ]
    )


def rk4_step(plant: Plant, x, cmd, dt: float, unc: Uncertainty = Uncertainty()) -> jnp.ndarray:
    """One fixed-step RK4 update with zero-order-hold commands."""

    def f(s):
        return state_derivative(plant, s, cmd, unc)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Wind-axis coordinates, used for trim and linearization.
# xw = [V alpha beta p q r phi theta psi north east h  de da dr  eta_s eta_a thr]
# ---------------------------------------------------------------------------


def body_to_wind(x: jnp.ndarray) -> jnp.ndarray:
    vt, alpha, beta = air_data(x)
    eta_s = 0.5 * (x[Ix.ETA_L] + x[Ix.ETA_R])
    eta_a = 0.5 * (x[Ix.ETA_L] - x[Ix.ETA_R])
    return jnp.concatenate(
        [jnp.array([vt, alpha, beta]), x[3:15], jnp.array([eta_s, eta_a]), x[17:]]
    )


def wind_to_body(xw: jnp.ndarray) -> jnp.ndarray:
    vt, alpha, beta = xw[0], xw[1], xw[2]
    uvw = vt * jnp.array(
        [jnp.cos(alpha) * jnp.cos(beta), jnp.sin(beta), jnp.sin(alpha) * jnp.cos(beta)]
    )
    eta_s, eta_a = xw[15], xw[16]
    return jnp.concatenate([uvw, xw[3:15], jnp.array([eta_s + eta_a, eta_s - eta_a]), xw[17:]])


def wind_derivative(plant: Plant, xw: jnp.ndarray, cmd: jnp.ndarray) -> jnp.ndarray:
    """``d(xw)/dt`` without command saturation (the design model)."""
    xb = wind_to_body(xw)
    xb_dot = state_derivative(plant, xb, cmd, saturate=False)
    _, xw_dot = jax.jvp(body_to_wind, (xb,), (xb_dot,))
    return xw_dot


class W:
    """Named indices into the wind-axis state."""

    V, ALPHA, BETA, P, Q, R, PHI, THETA, PSI, NORTH, EAST, H = range(12)
    DE, DA, DR, ETA_S, ETA_A, THR = range(12, 18)


__all__ = [
    "DEG",
    "N_CMD",
    "N_STATE",
    "Cmd",
    "Ix",
    "Plant",
    "Uncertainty",
    "W",
    "air_data",
    "body_to_wind",
    "forces_moments",
    "rk4_step",
    "state_derivative",
    "wind_derivative",
    "wind_to_body",
]
