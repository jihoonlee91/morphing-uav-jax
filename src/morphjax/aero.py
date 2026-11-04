"""Aerodynamic coefficient build-up for the span-morphing GTM-T2.

Mirrors the ``GTM_T2/Aero`` subsystem of NASA GTM_DesignSim (undamaged case)
plus the morphing increments of Lee et al. (2023), Eqs. (41)-(43):

    C = C_basic(a, b) + dC_elevator + dC_aileron + dC_rudder
        + dC_rate (forced oscillation, Kalviste-blended with rotary balance)
        + dC_morph_left + dC_morph_right

Coefficients are body-axis ``[CX CY CZ Cl Cm Cn]``; moments are about the
25 % MAC reference point. Angles in the tables are in degrees.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from morphjax.data import AeroTables, Table
from morphjax.interp import interp, slice_table

# Mirror image of a coefficient vector across the plane of symmetry.
MIRROR = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
LON = jnp.array([0, 2, 4])  # CX CZ Cm
LAT = jnp.array([1, 3, 5])  # CY Cl Cn


class AeroModel(NamedTuple):
    """Tables pre-processed for fast, smooth evaluation."""

    basic: Table
    elevator: Table  # stabilizer fixed at 0 deg
    aileron: Table  # right aileron
    rudder: Table  # extended to positive deflections by symmetry
    roll_damp: Table
    pitch_damp: Table
    yaw_damp: Table
    rotary: Table
    wingtip: Table


def _mirror_rudder(t: Table) -> Table:
    """Extend the negative-only rudder table to positive deflections.

    GTM_DesignSim evaluates a positive deflection as the mirror image of the
    negative one: ``C(a, b, +d) = MIRROR * C(a, -b, -d)``. Building the full
    table once gives the same values and keeps the map piecewise-linear and
    differentiable at ``d = 0`` (an ``abs()`` there would zero the Jacobian).
    """
    alpha, beta, rud = t.grids
    assert bool(jnp.all(beta == -beta[::-1])), "beta grid must be symmetric"
    pos = rud[rud < 0][::-1] * -1.0  # (10, 30, 45)
    mirrored = t.data[:, ::-1, :, :][:, :, rud < 0, :][:, :, ::-1, :] * MIRROR
    grid = jnp.concatenate([rud, pos])
    data = jnp.concatenate([t.data, mirrored], axis=2)
    return Table((alpha, beta, grid), data)


def build_aero_model(tables: AeroTables) -> AeroModel:
    """Prepare raw NASA tables for :func:`coefficients`."""
    # dC3_sym (the optional symmetrising increment) is identically zero in this
    # database release, so it is omitted; the landing-gear-up increment is too.
    if float(jnp.abs(tables.sym.data).max()) != 0.0:
        raise ValueError("unexpected non-zero dC3_sym table")
    return AeroModel(
        basic=tables.basic,
        elevator=slice_table(tables.elevator, axis=2, value=0.0),
        aileron=tables.aileron,
        rudder=_mirror_rudder(tables.rudder),
        roll_damp=tables.roll_damp,
        pitch_damp=tables.pitch_damp,
        yaw_damp=tables.yaw_damp,
        rotary=tables.rotary,
        wingtip=tables.wingtip,
    )


def _kalviste(alpha: jnp.ndarray, beta: jnp.ndarray, pqr: jnp.ndarray):
    """Hybrid Kalviste split of body rates into steady spin + oscillation.

    Port of ``hykal.m`` (A. Murch, NASA). Returns ``(omega, pqr_osc)`` in rad/s.
    """
    p, q, r = pqr
    ca, sa, cb, sb = jnp.cos(alpha), jnp.sin(alpha), jnp.cos(beta), jnp.sin(beta)
    eps = 1e-12

    d1 = ca * cb
    ok1 = (jnp.abs(d1) > eps) & (jnp.abs(p) > eps)
    w1 = jnp.where(ok1, p / jnp.where(ok1, d1, 1.0), 0.0)
    r_osc1 = r - w1 * sa * cb
    ok1 = ok1 & ~(jnp.abs(r_osc1) > jnp.abs(r)) & ~(r_osc1 * r < 0)

    d2 = sa * cb
    ok2 = (jnp.abs(d2) > eps) & (jnp.abs(r) > eps)
    w2 = jnp.where(ok2, r / jnp.where(ok2, d2, 1.0), 0.0)
    p_osc2 = p - w2 * ca * cb
    ok2 = ok2 & ~(jnp.abs(p_osc2) > jnp.abs(p)) & ~(p_osc2 * p < 0)

    case1 = jnp.array([0.0, q - w1 * sb, r_osc1])
    case2 = jnp.array([p_osc2, q - w2 * sb, 0.0])
    omega = jnp.where(ok2, w2, jnp.where(ok1, w1, 0.0))
    pqr_osc = jnp.where(ok2, case2, jnp.where(ok1, case1, pqr))
    return omega, pqr_osc


def coefficients(
    model: AeroModel,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    airspeed: jnp.ndarray,
    pqr: jnp.ndarray,
    surfaces: jnp.ndarray,
    eta_l: jnp.ndarray,
    eta_r: jnp.ndarray,
    b: float,
    cbar: float,
    morph_scale: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Total body-axis coefficients.

    Args:
        alpha, beta: aerodynamic angles in radians.
        airspeed: true airspeed, ft/s.
        pqr: body rates, rad/s.
        surfaces: ``[elevator, aileron, rudder]`` in degrees. Elevator > 0 is
            trailing edge down; aileron > 0 rolls right (left surface down,
            right surface up); rudder > 0 is trailing edge left (nose left).
        eta_l, eta_r: span variation of each wing in percent, -25..0.
        morph_scale: multiplier on the morphing increments, used only to test
            controllers against model error.
    """
    a_deg, b_deg = jnp.rad2deg(alpha), jnp.rad2deg(beta)
    de, da, dr = surfaces

    c = interp(model.basic, a_deg, b_deg)
    c = c.at[LON].add(interp(model.elevator, a_deg, b_deg, de))
    right_ail = interp(model.aileron, a_deg, b_deg, -da)
    left_ail = MIRROR * interp(model.aileron, a_deg, -b_deg, da)
    c = c + right_ail + left_ail
    c = c + interp(model.rudder, a_deg, b_deg, dr)

    # Rate damping, Kalviste-blended with the rotary-balance data.
    v = jnp.maximum(airspeed, 1.0)
    omega, pqr_osc = _kalviste(alpha, beta, pqr)
    scale = jnp.array([b, cbar, b]) / (2.0 * v)
    p_hat, q_hat, r_hat = pqr_osc * scale
    w_hat = jnp.clip(omega * b / (2.0 * v), -0.5, 0.5)
    c = c + interp(model.rotary, a_deg, w_hat, b_deg, extrap="clip")
    c = c.at[LAT].add(interp(model.roll_damp, a_deg, p_hat, extrap="clip"))
    c = c.at[LON].add(interp(model.pitch_damp, a_deg, q_hat, extrap="clip"))
    c = c.at[LAT].add(interp(model.yaw_damp, a_deg, r_hat, extrap="clip"))

    # Span morphing, Eqs. (41)-(43): a retracted tip reuses the "wingtip off" data.
    left = -(eta_l / 25.0) * interp(model.wingtip, a_deg, b_deg, extrap="clip")
    right = -(eta_r / 25.0) * MIRROR * interp(model.wingtip, a_deg, -b_deg, extrap="clip")
    return c + morph_scale * (left + right)
