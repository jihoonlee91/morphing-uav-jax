"""GTM-T2 geometry, mass properties, and the span-morphing mass model.

Units follow the NASA model: feet, slugs, seconds, pounds-force. Positions are
in the Aircraft Reference System (ARS): x forward, y right, z down, origin
8.745 in ahead of the nose. Body axes are parallel to the ARS.

The morphing model is the one in Lee et al., *Sensors* 23(6):3075 (2023),
Eqs. (24)-(39): each wingtip telescopes by ``eta`` percent of the semispan
(``-25 <= eta <= 0``), and the mass properties of the 25 % "left wingtip off"
damage case are reinterpreted as a fully retracted tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

LB_TO_SLUG = 1.0 / 32.174
G = 32.174  # ft/s^2
KT_TO_FPS = 1.6878
DEG = math.pi / 180.0


@jax.tree_util.register_static
@dataclass(frozen=True)
class Airframe:
    """Constant airframe data (NASA ``AC_baseparams_T2.m``)."""

    weight_lb: float = 57.75
    S: float = 5.9018  # reference wing area, ft^2
    cbar: float = 0.9153  # mean aerodynamic chord, ft
    b: float = 6.8488  # span, ft
    le_mac: float = 4.5462  # leading edge of MAC, ft aft of the ARS origin
    sweep_deg: float = 28.43  # leading-edge sweep
    dihedral_deg: float = 5.0
    tip_x_uses_sweep: bool = True  # Eq. (24) as printed; False = original MATLAB code
    # Inertia about the nominal CG (slug ft^2), NASA sign convention.
    Ixx: float = 1.221
    Iyy: float = 4.655
    Izz: float = 5.587
    Ixz: float = 0.274
    Iyz: float = 0.0
    Ixy: float = 0.006
    # "Left wingtip (25 %) off" damage increments (DamageModels_T2.m, case 4).
    tip_dweight_lb: float = -0.81
    tip_dcg_in: tuple[float, float, float] = (0.148, 0.628, 0.032)
    tip_dinertia: tuple[float, float, float, float, float, float] = (
        -0.25821,  # dIxx
        -0.01727,  # dIyy
        -0.27400,  # dIzz
        -0.00295,  # dIxz
        -0.01346,  # dIyz
        -0.05998,  # dIxy
    )
    # Twin JetCat P70 engines (ARS, ft), thrust assumed along body x.
    engine_pos: tuple[tuple[float, float, float], ...] = (
        (-51.903 / 12, -14.20 / 12, -7.71 / 12),
        (-51.903 / 12, 14.20 / 12, -7.71 / 12),
    )

    @property
    def mass(self) -> float:
        return self.weight_lb * LB_TO_SLUG

    @property
    def cg_nominal(self) -> jnp.ndarray:
        """Nominal CG at 21.99 % MAC, gear up (ft, ARS)."""
        return jnp.array([-(self.le_mac + self.cbar * 0.2199), -0.1416 / 12, -0.9761])

    @property
    def aero_ref(self) -> jnp.ndarray:
        """Aerodynamic moment reference point, 25 % MAC (ft, ARS)."""
        return jnp.array([-(self.le_mac + self.cbar * 0.25), 0.0, -0.9401])

    @property
    def inertia_nominal(self) -> jnp.ndarray:
        return _inertia_matrix(self.Ixx, self.Iyy, self.Izz, self.Ixz, self.Iyz, self.Ixy)


def _inertia_matrix(ixx, iyy, izz, ixz, iyz, ixy) -> jnp.ndarray:
    return jnp.array(
        [
            [ixx, -ixy, -ixz],
            [-ixy, iyy, -iyz],
            [-ixz, -iyz, izz],
        ]
    )


def _steiner(m: float | jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Parallel-axis term ``-m [r]x^2 = m (|r|^2 I - r r^T)``."""
    return m * (jnp.dot(r, r) * jnp.eye(3) - jnp.outer(r, r))


def mass_properties(eta_l: jnp.ndarray, eta_r: jnp.ndarray, af: Airframe = Airframe()):
    """Return ``(mass, cg, J)`` for left/right span variations in percent.

    Mass is constant (the tips telescope, nothing is shed); the CG moves
    linearly and the inertia quadratically with the morphing parameters.
    """
    m = af.mass
    ml = -af.tip_dweight_lb * LB_TO_SLUG  # mass of one moving wingtip
    mr = ml
    mb = m - ml - mr
    lam, gam = af.sweep_deg * DEG, af.dihedral_deg * DEG

    rcm_bar = af.cg_nominal
    d_rcm_dmg = jnp.asarray(af.tip_dcg_in) / 12.0
    dIxx, dIyy, dIzz, dIxz, dIyz, dIxy = af.tip_dinertia
    dJ_dmg = _inertia_matrix(dIxx, dIyy, dIzz, dIxz, dIyz, dIxy)
    J_bar = af.inertia_nominal

    # Nominal component CGs, Eqs. (29)-(33).
    rl_bar = rcm_bar - (m / ml - 1.0) * d_rcm_dmg
    rr_bar = rl_bar * jnp.array([1.0, -1.0, 1.0])
    rb_bar = (m * rcm_bar - ml * rl_bar - mr * rr_bar) / mb

    # Wingtip inertia about its own CG: subtract the damaged configuration
    # (left tip removed) from Eq. (35).
    rcm_dmg = rcm_bar + d_rcm_dmg
    Jl_bar = (
        -dJ_dmg
        - _steiner(ml, rl_bar - rcm_bar)
        - _steiner(mb, rb_bar - rcm_bar)
        - _steiner(mr, rr_bar - rcm_bar)
        + _steiner(mb, rb_bar - rcm_dmg)
        + _steiner(mr, rr_bar - rcm_dmg)
    )
    mirror = jnp.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    Jr_bar = Jl_bar * mirror
    # Body inertia from Eq. (35). (The MATLAB code omitted the body's own
    # parallel-axis term here, which offsets J by ~0.1 %; with it, an
    # unmorphed wing returns the NASA inertia exactly.)
    Jb_bar = (
        J_bar
        - Jl_bar
        - _steiner(ml, rl_bar - rcm_bar)
        - _steiner(mb, rb_bar - rcm_bar)
        - Jr_bar
        - _steiner(mr, rr_bar - rcm_bar)
    )

    # Tip displacement, Eq. (24): the tip slides along the swept, dihedral wing.
    # The original MATLAB code used tan(dihedral) in the x-term; Eq. (24) as
    # printed uses tan(sweep). Both are available (see Airframe.tip_x_uses_sweep).
    x_slope = math.tan(lam) if af.tip_x_uses_sweep else math.tan(gam)
    arm = jnp.array([af.b * x_slope / 8, af.b / 8, af.b * math.tan(gam) / (8 * math.cos(lam))])
    drl = -(eta_l / 25.0) * arm
    drr = -(eta_r / 25.0) * arm * jnp.array([1.0, -1.0, 1.0])
    rcm = rcm_bar + (ml / m) * drl + (mr / m) * drr

    rl, rb, rr = rl_bar + drl, rb_bar, rr_bar + drr
    J = (
        Jl_bar
        + _steiner(ml, rl - rcm)
        + Jb_bar
        + _steiner(mb, rb - rcm)
        + Jr_bar
        + _steiner(mr, rr - rcm)
    )
    return jnp.asarray(m), rcm, J
