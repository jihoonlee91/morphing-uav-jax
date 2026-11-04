"""Check the model-based designs *between* their design points.

Gridded LPV and gain-scheduled designs are only synthesized at the 63 grid
points; at run time the gains are interpolated. This module re-trims and
re-linearizes the aircraft on a denser grid (by default 13 x 5 x 5 = 325
points, 262 of them off the design grid), applies the interpolated gains the
controller would actually use there, and checks three things per design:

* frozen-point closed-loop stability,
* the largest transport delay the frozen loop tolerates (sampled at 50 Hz),
* for the LPV design, whether the *common* Lyapunov matrix still certifies
  the interpolated closed loop, i.e. whether the guarantee extends past the grid.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from morphjax import design
from morphjax.pipeline import Setup
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
    certified: int | None  # LPV only: common Lyapunov inequality holds


def interpolate_gains(coarse: design.DesignGrid, gains: np.ndarray, dense: design.DesignGrid):
    """Multilinear interpolation of a gain table, exactly as the runtime schedule does."""
    f = RegularGridInterpolator((coarse.v_syn, coarse.alt, coarse.eta_s), gains)
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), axis=-1)
    return f(pts.reshape(-1, 3)).reshape(*pts.shape[:3], *gains.shape[3:])


def check_designs(setup: Setup, axes=DENSE, max_delay_steps: int = 4) -> dict[str, DesignCheck]:
    dense = design.build_grid(setup.plant, *axes)
    on_grid = (
        np.isin(dense.v_syn, setup.grid.v_syn)[:, None, None]
        & np.isin(dense.alt, setup.grid.alt)[None, :, None]
        & np.isin(dense.eta_s, setup.grid.eta_s)[None, None, :]
    )
    xs, us = setup.weights.scales()
    p_common = np.linalg.inv(setup.lpv.X)
    out = {}
    for name, coarse_gains in setup.lat_gains.items():
        gains = interpolate_gains(setup.grid, coarse_gains, dense)
        stable, certified, worst, min_delay = 0, 0, -np.inf, np.inf
        for idx in np.ndindex(*dense.residual.shape):
            a, b = design.lateral_model(dense.A[idx], dense.B[idx])
            k = gains[idx]
            eig = float(np.real(np.linalg.eigvals(a + b @ k)).max())
            worst = max(worst, eig)
            stable += eig < 0
            d = -1
            for steps in range(max_delay_steps + 1):
                if design.delayed_spectral_radius(a, b, k, DT, steps) >= 1.0:
                    break
                d = steps
            min_delay = min(min_delay, 1000 * DT * d)
            if name == "lpv":
                an, bn = design._normalize(a, b, xs, us)
                acl = an + bn @ (np.diag(1 / us) @ k @ np.diag(xs))
                certified += np.linalg.eigvalsh(acl.T @ p_common + p_common @ acl).max() < 0
        out[name] = DesignCheck(
            points=int(dense.residual.size),
            off_grid=int((~on_grid).sum()),
            stable=int(stable),
            worst_real_eig=worst,
            min_delay_ms=float(min_delay),
            certified=int(certified) if name == "lpv" else None,
        )
    return out
