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

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

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
    # Relative interpolation-error margin gamma (see estimate_margin): the
    # half-space is tightened to a'du <= (1 - gamma) b. 0 = plain projection.
    margin: float = 0.0


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
    """Residual ``du`` (physical units, already clipped) projected onto the half-space.

    With ``f.margin = gamma > 0`` the half-space is ``a' du <= (1 - gamma) b``,
    which keeps the decrease on the true model whenever the interpolation
    error satisfies the relative bound of :func:`estimate_margin`.
    """
    a, b = halfspace(f, rho, dx)
    du_n = du / f.us
    excess = a @ du_n - (1.0 - f.margin) * b
    # Not jnp.maximum: its gradient splits ties, and the boundary is reached at x = 0.
    step = jnp.where(excess > 0.0, excess, 0.0) / (a @ a + 1e-12)
    return (du_n - step * a) * f.us


def make_projected_residual_lateral(lon: Schedule, params: Params, base: Schedule, f: SafetyFilter):
    """LPV law + residual policy, with the clipped residual projected onto the certificate."""
    return make_policy_lateral(lon, params, base, residual_filter=partial(project, f))


def make_safe_residual_lateral(lon: Schedule, params: Params, base: Schedule, f: SafetyFilter):
    """LPV law + residual policy, projected with the interpolation-error margin."""
    return make_policy_lateral(lon, params, base, residual_filter=partial(project, f))


def _relative_error(f: SafetyFilter, grid: design.DesignGrid) -> tuple[np.ndarray, np.ndarray]:
    """Per point: the relative error gamma and the B mismatch, on an arbitrary grid.

    ``gamma = lambda_max(M_true - M_int, N)`` with ``N = -(M_int + 2 lambda P)``
    (a generalized eigenvalue), so that ``x'(M_true - M_int) x <= gamma x' N x``
    for every ``x``. ``|P (B_true - B_int)|`` is returned to confirm that the
    input matrix carries no interpolation error (commands drive first-order
    actuators, so ``B`` is the same everywhere).
    """
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    pts = np.stack(np.meshgrid(grid.v_syn, grid.alt, grid.eta_s, indexing="ij"), axis=-1)
    gamma = np.zeros(grid.residual.shape)
    b_err = np.zeros(grid.residual.shape)
    for idx in np.ndindex(*grid.residual.shape):
        a_t, b_t = design._normalize(*design.lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        rho = tuple(jnp.asarray(v) for v in pts[idx])
        a_i = np.asarray(interp(f.a_n, *rho)).reshape(8, 8)
        b_i = np.asarray(interp(f.b_n, *rho)).reshape(8, 3)
        k_i = np.asarray(interp(f.k_n, *rho)).reshape(3, 8)
        acl_t, acl_i = a_t + b_t @ k_i, a_i + b_i @ k_i
        m_i = acl_i.T @ p + p @ acl_i
        dm = (acl_t.T @ p + p @ acl_t) - m_i
        gamma[idx] = float(
            scipy.linalg.eigh(dm, -(m_i + 2.0 * f.decay * p), eigvals_only=True).max()
        )
        b_err[idx] = np.linalg.norm(p @ (b_t - b_i), 2)
    return gamma, b_err


def estimate_margin(
    setup: Setup, f: SafetyFilter, per_axis: int = 10, seed: int = 0, factor: float = 1.5
) -> tuple[SafetyFilter, dict]:
    """Bound the interpolation error on random flight conditions; return the margined filter.

    The bound comes from ``per_axis**3`` random conditions, on a random
    tensor grid independent of the 325-point grid the result is checked on.
    It is multiplied by ``factor``, because a sampled maximum underestimates
    the supremum over the continuous envelope.
    """
    rng = np.random.default_rng(seed)
    axes = (
        np.sort(rng.uniform(60.0, 120.0, per_axis)),
        np.sort(rng.uniform(0.0, 10000.0, per_axis)),
        np.sort(rng.uniform(-25.0, 0.0, per_axis)),
    )
    gamma, b_err = _relative_error(f, design.build_grid(setup.plant, *axes))
    margin = factor * max(float(gamma.max()), 0.0)
    if margin >= 1.0:
        raise RuntimeError(f"interpolation error too large for a margin: gamma = {margin:.3f}")
    info = {
        "estimation_points": int(gamma.size),
        "seed": seed,
        "factor": factor,
        "max_gamma": float(gamma.max()),
        "margin": margin,
        "max_input_matrix_error": float(b_err.max()),
    }
    return f._replace(margin=margin), info


Filter = Callable[[SafetyFilter, tuple, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def sampled_decrease(
    f: SafetyFilter,
    dense: design.DesignGrid,
    lpv_gains: np.ndarray,
    filters: dict[str, Filter | None],
    samples: int = 4096,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Check ``V-dot`` on the *true* linearization at each dense point.

    For random states (directions uniform on the sphere, sizes log-uniform
    over 1e-3..10, since a residual offends most near trim) and random box
    residuals, each filter acts with its interpolated model; ``V-dot`` is then
    evaluated with the true frozen model at that point and the interpolated
    LPV gain. ``filters`` maps a name to a filter, or to ``None`` for the
    unfiltered residual; the name ``"lpv"`` means no residual at all. Returns
    per-point fractions of samples where ``V-dot <= -2 lambda V`` ("decay")
    and ``V-dot < 0`` ("neg") hold.
    """
    rng = np.random.default_rng(seed)
    shp = dense.residual.shape
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), axis=-1)
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    out = {n: {"decay": np.zeros(shp), "neg": np.zeros(shp)} for n in filters}
    jitted = {
        n: jax.jit(jax.vmap(fn, in_axes=(None, None, 0, 0)))
        for n, fn in filters.items()
        if fn is not None
    }
    box = np.asarray(RESIDUAL_SCALE)
    for idx in np.ndindex(*shp):
        a_t, b_t = design._normalize(*design.lateral_model(dense.A[idx], dense.B[idx]), xs, us)
        acl = a_t + b_t @ (np.diag(1.0 / us) @ lpv_gains[idx] @ np.diag(xs))
        direction = rng.normal(size=(samples, 8))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        x_n = direction * 10.0 ** rng.uniform(-3.0, 1.0, size=(samples, 1))
        du = rng.uniform(-1.0, 1.0, size=(samples, 3)) * box
        rho = tuple(jnp.asarray(v) for v in pts[idx])
        v = np.einsum("ni,ij,nj->n", x_n, p, x_n)
        base = np.einsum("ni,ij,nj->n", x_n, acl.T @ p + p @ acl, x_n)
        for name, fn in filters.items():
            if name == "lpv":
                d = np.zeros_like(du)
            elif fn is None:
                d = du
            else:
                d = np.asarray(jitted[name](f, rho, jnp.asarray(x_n * xs), jnp.asarray(du)))
            vdot = base + 2.0 * np.einsum("ni,ij,jk,nk->n", x_n, p, b_t, d / us)
            out[name]["decay"][idx] = np.mean(vdot <= -2.0 * f.decay * v + 1e-9)
            out[name]["neg"][idx] = np.mean(vdot < 0.0)
    return out
