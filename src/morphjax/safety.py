"""Lyapunov projection: keep a learned residual inside the LPV certificate.

The certified LPV law ``u = K(rho) x`` has a common Lyapunov function
``V = x' P x`` (normalized coordinates, ``P = X^-1``) with, on the frozen
linear model at every design point,

    x' (A_cl' P + P A_cl + 2 lambda P) x <= -|x|^2 < 0,   A_cl = A + B K.

Adding a residual ``du`` changes ``V-dot`` by ``2 x' P B du``. Requiring
``V-dot <= -2 lambda V`` is a half-space in ``du``:

    a' du <= b,   a = 2 B' P x,   b = -x' (A_cl' P + P A_cl + 2 lambda P) x  (>= 0).

The filter clips the residual to its box first and then projects it onto the
half-space, so the inequality holds exactly on the (interpolated) frozen model
for any network output; the projection can push a component past the box, but
only along ``a``, i.e. in the direction that makes ``V`` decrease faster.

Two caveats are part of the result, not hidden: the guarantee is for the
frozen linear model without delay or saturation, and when the residual is
nonzero at the trim point the filtered law is not differentiable there, so a
single Jacobian at trim is not a meaningful linear audit. ``sampled_decrease``
checks the condition on the *true* linearization instead.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import design
from morphjax.control import Schedule
from morphjax.data import Table
from morphjax.interp import interp
from morphjax.pipeline import Setup
from morphjax.rl import RESIDUAL_SCALE, Params, make_policy_lateral


class SafetyFilter(NamedTuple):
    a_n: Table  # rho -> normalized lateral A (8 x 8, flattened)
    b_n: Table  # rho -> normalized lateral B (8 x 3)
    k_n: Table  # rho -> normalized LPV gain (3 x 8)
    p: jnp.ndarray  # common Lyapunov matrix X^-1
    xs: jnp.ndarray  # state scales (normalized = physical / xs)
    us: jnp.ndarray  # input scales
    decay: float  # lambda of the LPV synthesis


def make_safety_filter(setup: Setup) -> SafetyFilter:
    grid, (xs, us) = setup.grid, setup.weights.scales()
    shp = grid.residual.shape
    an = np.zeros((*shp, 8, 8))
    bn = np.zeros((*shp, 8, 3))
    kn = np.zeros((*shp, 3, 8))
    for idx in np.ndindex(*shp):
        an[idx], bn[idx] = design._normalize(
            *design.lateral_model(grid.A[idx], grid.B[idx]), xs, us
        )
        kn[idx] = np.diag(1.0 / us) @ setup.lat_gains["lpv"][idx] @ np.diag(xs)
    axes = (jnp.asarray(grid.v_syn), jnp.asarray(grid.alt), jnp.asarray(grid.eta_s))

    def tab(arr):
        return Table(axes, jnp.asarray(arr.reshape(*shp, -1)))

    return SafetyFilter(
        tab(an),
        tab(bn),
        tab(kn),
        jnp.asarray(np.linalg.inv(setup.lpv.X)),
        jnp.asarray(xs),
        jnp.asarray(us),
        setup.lpv.decay,
    )


def halfspace(f: SafetyFilter, rho, dx: jnp.ndarray):
    """``(a, b)`` of the constraint ``a' du_n <= b`` at ``rho`` for deviation ``dx``."""
    a_mat = interp(f.a_n, *rho).reshape(8, 8)
    b_mat = interp(f.b_n, *rho).reshape(8, 3)
    k_mat = interp(f.k_n, *rho).reshape(3, 8)
    x = dx / f.xs
    acl = a_mat + b_mat @ k_mat
    m = acl.T @ f.p + f.p @ acl + 2.0 * f.decay * f.p
    return 2.0 * b_mat.T @ f.p @ x, -x @ m @ x


def project(f: SafetyFilter, rho, dx: jnp.ndarray, du: jnp.ndarray) -> jnp.ndarray:
    """Residual ``du`` (physical units, already clipped) projected onto the half-space."""
    a, b = halfspace(f, rho, dx)
    du_n = du / f.us
    excess = a @ du_n - b
    # Not jnp.maximum: its gradient splits ties, and the boundary is reached at x = 0.
    step = jnp.where(excess > 0.0, excess, 0.0) / (a @ a + 1e-12)
    return (du_n - step * a) * f.us


def make_projected_residual_lateral(lon: Schedule, params: Params, base: Schedule, f: SafetyFilter):
    """LPV law + residual policy, with the clipped residual projected onto the certificate."""
    return make_policy_lateral(lon, params, base, residual_filter=partial(project, f))


def sampled_decrease(
    f: SafetyFilter,
    dense: design.DesignGrid,
    lpv_gains: np.ndarray,
    samples: int = 4096,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Check ``V-dot`` after filtering on the *true* linearization at each dense point.

    For random states and random (box-clipped) residuals, the filter projects
    using its interpolated model; ``V-dot`` is then evaluated with the true
    frozen model ``(A, B)`` at that point and the interpolated LPV gain. Returns
    per-point fractions where ``V-dot <= -2 lambda V`` and ``V-dot < 0`` hold,
    and the same for the unfiltered residual.
    """
    rng = np.random.default_rng(seed)
    shp = dense.residual.shape
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), axis=-1)
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    out = {k: np.zeros(shp) for k in ("filtered_decay", "filtered_neg", "raw_decay", "raw_neg")}
    proj = jax.jit(jax.vmap(project, in_axes=(None, None, 0, 0)))
    box = np.asarray(RESIDUAL_SCALE)
    for idx in np.ndindex(*shp):
        a_true, b_true = design._normalize(
            *design.lateral_model(dense.A[idx], dense.B[idx]), xs, us
        )
        k_n = np.diag(1.0 / us) @ lpv_gains[idx] @ np.diag(xs)
        acl = a_true + b_true @ k_n
        # Directions uniform on the sphere, sizes log-uniform over four decades:
        # a residual offends most near the trim point, where b ~ |x|^2 is small.
        direction = rng.normal(size=(samples, 8))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        x_n = direction * 10.0 ** rng.uniform(-3.0, 1.0, size=(samples, 1))
        du = rng.uniform(-1.0, 1.0, size=(samples, 3)) * box
        rho = tuple(jnp.asarray(v) for v in pts[idx])
        du_f = np.asarray(proj(f, rho, jnp.asarray(x_n * xs), jnp.asarray(du)))
        v = np.einsum("ni,ij,nj->n", x_n, p, x_n)
        base = np.einsum("ni,ij,nj->n", x_n, acl.T @ p + p @ acl, x_n)
        for tag, d in (("filtered", du_f), ("raw", du)):
            vdot = base + 2.0 * np.einsum("ni,ij,jk,nk->n", x_n, p, b_true, d / us)
            out[f"{tag}_decay"][idx] = np.mean(vdot <= -2.0 * f.decay * v + 1e-9)
            out[f"{tag}_neg"][idx] = np.mean(vdot < 0.0)
    return out
