"""Linear checks of every controller, at and between the design points.

Gridded LPV and gain-scheduled designs are only synthesized at the 63 grid
points; at run time the gains are interpolated. This module re-trims and
re-linearizes the aircraft on a denser grid (by default 13 x 5 x 5 = 325
points, 262 of them off the design grid) and checks, for each controller and
each point:

* frozen-point closed-loop stability,
* the largest transport delay the frozen loop tolerates (sampled at 50 Hz),
* whether the LPV design's *common* Lyapunov matrix certifies the closed loop.

Learned policies are audited the same way: the policy's lateral command is
differentiated with ``jax.jacfwd`` with respect to the 8-state lateral
deviation at the trim point (wings level, zero command), giving a local gain
``K_pi(rho)`` that is checked exactly like an LQR gain.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from morphjax import design
from morphjax.pipeline import Setup
from morphjax.rl import ACT_SCALE, OBS_SCALE, OBS_SHIFT, Params, policy_mean
from morphjax.sim import DT

DENSE = (
    np.linspace(60.0, 120.0, 13),  # synthetic airspeed, kt
    np.linspace(0.0, 10000.0, 5),  # altitude, ft
    np.linspace(-25.0, 0.0, 5),  # symmetric span, %
)


class DesignCheck(NamedTuple):
    points: int
    off_grid: int
    stable: int  # frozen closed loop stable
    worst_real_eig: float
    min_delay_ms: float  # smallest tolerated delay over all points (-20 = unstable)
    certified: int  # points where the LPV common Lyapunov inequality holds
    on_grid_stable: int  # the same counts restricted to the 63 design points
    on_grid_certified: int
    min_delay_ms_on_grid: float | None  # None when no point lies on the design grid


class PointChecks(NamedTuple):
    """Per-point arrays on the dense grid (shape nv x nh x ne)."""

    max_real_eig: np.ndarray
    max_delay_steps: np.ndarray  # -1 = unstable without delay
    certified: np.ndarray


def interpolate_gains(coarse: design.DesignGrid, gains: np.ndarray, dense: design.DesignGrid):
    """Multilinear interpolation of a gain table, exactly as the runtime schedule does."""
    f = RegularGridInterpolator((coarse.v_syn, coarse.alt, coarse.eta_s), gains)
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), axis=-1)
    return f(pts.reshape(-1, 3)).reshape(*pts.shape[:3], *gains.shape[3:])


def policy_command(params: Params, dx: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """Unclipped policy command (deviation from trim) for lateral deviation ``dx`` at ``rho``.

    ``rho = (V_syn, h, eta_s)``. Uses the observation layout of
    :func:`morphjax.rl.observation`; at wings-level trim with zero roll command,
    the roll angle equals the roll-error state ``dx[3]``.
    """
    v_syn, h, eta_s = rho
    raw = jnp.concatenate([dx, jnp.array([eta_s, v_syn, h, dx[3]])])
    return ACT_SCALE * policy_mean(params, (raw - OBS_SHIFT) / OBS_SCALE)


def policy_gains(params: Params, grid: design.DesignGrid) -> tuple[np.ndarray, np.ndarray]:
    """Local gains ``K_pi = d(command)/d(deviation)`` of a learned policy at each grid point.

    Returns ``(gains, saturated)``: gains in the same units as the LQR gains
    (command per state, shape ``(..., 3, 8)``), and whether the policy's mean
    action at the trim point is already outside the action clip (there the
    true local gain is zero).
    """

    def command(dx, rho):
        return policy_command(params, dx, rho)

    pts = np.stack(np.meshgrid(grid.v_syn, grid.alt, grid.eta_s, indexing="ij"), axis=-1)
    rho = jnp.asarray(pts.reshape(-1, 3))
    zero = jnp.zeros(8)
    jac = jax.vmap(lambda r: jax.jacfwd(command)(zero, r))(rho)
    mean = jax.vmap(lambda r: command(zero, r) / ACT_SCALE)(rho)
    shape = pts.shape[:3]
    saturated = np.asarray(jnp.any(jnp.abs(mean) >= 1.0, axis=-1)).reshape(shape)
    return np.asarray(jac).reshape(*shape, 3, 8), saturated


def check_gains(
    setup: Setup, dense: design.DesignGrid, gains: np.ndarray, max_delay_steps: int = 4
) -> PointChecks:
    """Stability, delay margin, and LPV certificate of ``u = K x`` at every dense point."""
    xs, us = setup.weights.scales()
    p_common = np.linalg.inv(setup.lpv.X)
    shape = dense.residual.shape
    eig = np.zeros(shape)
    delay = np.zeros(shape, dtype=int)
    cert = np.zeros(shape, dtype=bool)
    for idx in np.ndindex(*shape):
        a, b = design.lateral_model(dense.A[idx], dense.B[idx])
        k = gains[idx]
        eig[idx] = float(np.real(np.linalg.eigvals(a + b @ k)).max())
        d = -1
        for steps in range(max_delay_steps + 1):
            if design.delayed_spectral_radius(a, b, k, DT, steps) >= 1.0:
                break
            d = steps
        delay[idx] = d
        an, bn = design._normalize(a, b, xs, us)
        acl = an + bn @ (np.diag(1 / us) @ k @ np.diag(xs))
        cert[idx] = np.linalg.eigvalsh(acl.T @ p_common + p_common @ acl).max() < 0
    return PointChecks(eig, delay, cert)


def summarize(checks: PointChecks, on_grid: np.ndarray) -> DesignCheck:
    stable = checks.max_real_eig < 0
    return DesignCheck(
        points=int(stable.size),
        off_grid=int((~on_grid).sum()),
        stable=int(stable.sum()),
        worst_real_eig=float(checks.max_real_eig.max()),
        min_delay_ms=float(1000 * DT * checks.max_delay_steps.min()),
        certified=int(checks.certified.sum()),
        on_grid_stable=int(stable[on_grid].sum()),
        on_grid_certified=int(checks.certified[on_grid].sum()),
        min_delay_ms_on_grid=(
            float(1000 * DT * checks.max_delay_steps[on_grid].min()) if on_grid.any() else None
        ),
    )


def dense_grid(setup: Setup, axes=DENSE) -> tuple[design.DesignGrid, np.ndarray]:
    dense = design.build_grid(setup.plant, *axes)
    on_grid = (
        np.isin(dense.v_syn, setup.grid.v_syn)[:, None, None]
        & np.isin(dense.alt, setup.grid.alt)[None, :, None]
        & np.isin(dense.eta_s, setup.grid.eta_s)[None, None, :]
    )
    return dense, on_grid


def check_designs(setup: Setup, axes=DENSE) -> dict[str, DesignCheck]:
    """Summaries for every model-based design in ``setup.lat_gains``."""
    dense, on_grid = dense_grid(setup, axes)
    return {
        name: summarize(check_gains(setup, dense, interpolate_gains(setup.grid, g, dense)), on_grid)
        for name, g in setup.lat_gains.items()
    }
