"""Multilinear interpolation on rectilinear grids, written for ``jit``/``grad``.

Matches the Simulink ``PreLookup`` + ``Interpolation Using Prelookup`` pair
used by NASA GTM_DesignSim: linear interpolation inside the grid, and either
linear extrapolation or clipping outside it.
"""

from __future__ import annotations

import itertools
from typing import Literal

import jax.numpy as jnp

from morphjax.data import Table

Extrap = Literal["linear", "clip"]


def _bracket(grid: jnp.ndarray, x: jnp.ndarray, extrap: Extrap) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lower cell index and fractional position of ``x`` in ``grid``."""
    n = grid.shape[0]
    i = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, n - 2)
    lo, hi = grid[i], grid[i + 1]
    t = (x - lo) / (hi - lo)
    if extrap == "clip":
        # Not jnp.clip: its gradient is 0.5 at the bounds (max/min split ties),
        # which would halve every derivative at a breakpoint such as p = 0.
        t = jnp.where(t < 0.0, 0.0, jnp.where(t > 1.0, 1.0, t))
    return i, t


def interp(table: Table, *x: jnp.ndarray, extrap: Extrap = "linear") -> jnp.ndarray:
    """Evaluate ``table`` at the scalar query point ``x`` (one value per grid)."""
    if len(x) != len(table.grids):
        raise ValueError(f"expected {len(table.grids)} coordinates, got {len(x)}")
    brackets = [
        _bracket(g, jnp.asarray(xi, dtype=g.dtype), extrap)
        for g, xi in zip(table.grids, x, strict=True)
    ]
    out = jnp.zeros(table.data.shape[len(x) :], dtype=table.data.dtype)
    for corner in itertools.product((0, 1), repeat=len(x)):
        weight = jnp.asarray(1.0, dtype=table.data.dtype)
        index = []
        for (i, t), bit in zip(brackets, corner, strict=True):
            weight = weight * (t if bit else 1.0 - t)
            index.append(i + bit)
        out = out + weight * table.data[tuple(index)]
    return out


def slice_table(table: Table, axis: int, value: float) -> Table:
    """Fix one grid axis at an exact breakpoint (e.g. stabilizer at 0 deg)."""
    grid = table.grids[axis]
    matches = jnp.nonzero(grid == value, size=1, fill_value=-1)[0]
    k = int(matches[0])
    if k < 0:
        raise ValueError(f"{value} is not a breakpoint of axis {axis}")
    grids = tuple(g for j, g in enumerate(table.grids) if j != axis)
    return Table(grids, jnp.take(table.data, k, axis=axis))
