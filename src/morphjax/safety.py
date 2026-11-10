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


# ----------------------------------------------------------------------------- round 3
# Sampled-data (discrete-time) Lyapunov filter.
#
# The continuous-time half-space above is vacuous at x -> 0 (a = 2B'Px -> 0 and b -> 0,
# so any residual is admitted), and a residual held for one 20 ms sample can then
# increase V: on the 50 Hz ZOH model the certified residual fails V(x+) <= rho_d V(x)
# for about half of the sampled states (scripts/r3_certificate.py). The filter below
# enforces the discrete condition directly. With the frozen ZOH model of the lateral
# loop at the interpolated rho, Phi_cl = Phi + Gamma K (normalized coordinates, LPV
# gain K) and x+ = Phi_cl x + Gamma du:
#
#     V(x+) - rho_d V(x) + g_d [x' D x + du' R du] <= 0,
#     D = rho_d P - Phi_cl' P Phi_cl,   R = Gamma' P Gamma,
#
# i.e. du' H du + 2 c' du + e <= 0 with H = (1 + g_d) R, c = Gamma' P Phi_cl x,
# e = -(1 - g_d) x' D x: an ellipsoid in du. rho_d = exp(-2 lambda T). D > 0 is the
# discrete decrease margin of the LPV loop itself (its V ratio is at most 0.991 < rho_d
# on the interpolated model at all check points). g_d is a relative bound on the model
# mismatch, exactly like gamma-bar: for z = (x, du), the true-minus-interpolated
# quadratic form of V(x+) satisfies |z' dM z| <= g z' N z with N = blkdiag(D, R)
# (generalized eigenvalue), and g_d = factor * max g over random conditions. Then
# V_true(x+) <= rho_d V(x) whenever the filtered du satisfies the inequality and g <= g_d.
# du = 0 is feasible wherever D > 0 and g_d < 1, and at x = 0 only du = 0 is admitted
# (small-control property). The projection
# min |du - du0|^2 s.t. the ellipsoid is du(nu) = (I + nu H)^-1 (du0 - nu c), with the
# multiplier nu >= 0 found by bisection on the (monotone) constraint value.


class DTSafetyFilter(NamedTuple):
    phi: Table  # rho -> Phi (8 x 8) of the normalized open-loop lateral model, ZOH at DT
    gam: Table  # rho -> Gamma (8 x 3)
    k_n: Table  # rho -> normalized LPV gain (3 x 8)
    p: jnp.ndarray
    xs: jnp.ndarray
    us: jnp.ndarray
    rho_d: float  # exp(-2 lambda T)
    margin: float = 0.0  # g_d


def _zoh(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = b.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = a, b
    e = scipy.linalg.expm(blk * dt)
    return e[:n, :n], e[:n, n:]


def make_dt_safety_filter(setup: Setup, dt: float = 0.02) -> DTSafetyFilter:
    f = make_safety_filter(setup)
    grid = setup.grid
    shp = grid.residual.shape
    xs, us = np.asarray(f.xs), np.asarray(f.us)
    ph = np.zeros((*shp, 8, 8))
    gm = np.zeros((*shp, 8, 3))
    for idx in np.ndindex(*shp):
        a, b = design._normalize(*design.lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        ph[idx], gm[idx] = _zoh(a, b, dt)
    axes = f.a_n.grids
    return DTSafetyFilter(
        Table(axes, jnp.asarray(ph.reshape(*shp, -1))),
        Table(axes, jnp.asarray(gm.reshape(*shp, -1))),
        f.k_n, f.p, f.xs, f.us, float(np.exp(-2.0 * f.decay * dt)),
    )  # fmt: skip


def dt_ellipsoid(f: DTSafetyFilter, rho, dx: jnp.ndarray):
    """``(H, c, e)`` of ``du_n' H du_n + 2 c' du_n + e <= 0`` at ``rho`` for deviation ``dx``."""
    ph = interp(f.phi, *rho).reshape(8, 8)
    gm = interp(f.gam, *rho).reshape(8, 3)
    k = interp(f.k_n, *rho).reshape(3, 8)
    x = dx / f.xs
    phc = ph + gm @ k
    r = gm.T @ f.p @ gm
    h = (1.0 + f.margin) * r
    c = gm.T @ f.p @ (phc @ x)
    e = (1.0 - f.margin) * (x @ (phc.T @ f.p @ phc) @ x - f.rho_d * (x @ f.p @ x))
    return h, c, e


def dt_project(f: DTSafetyFilter, rho, dx: jnp.ndarray, du: jnp.ndarray, iters: int = 60):
    """Residual ``du`` (physical units) projected onto the discrete-time ellipsoid."""
    h, c, e = dt_ellipsoid(f, rho, dx)
    du0 = du / f.us

    def at(nu):
        return jnp.linalg.solve(jnp.eye(3) + nu * h, du0 - nu * c)

    def g(nu):
        d = at(nu)
        return d @ h @ d + 2.0 * c @ d + e

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        ok = g(jnp.tan(mid)) <= 0.0  # nu = tan(t), t in [0, pi/2)
        return jnp.where(ok, lo, mid), jnp.where(ok, mid, hi)

    _, hi = jax.lax.fori_loop(0, iters, body, (0.0, 0.5 * jnp.pi - 1e-9))
    out = jnp.where(g(0.0) <= 0.0, du0, at(jnp.tan(hi)))
    return out * f.us


def dt_relative_error(f: DTSafetyFilter, grid: design.DesignGrid, dt: float = 0.02) -> np.ndarray:
    """Per point of ``grid``: ``g`` with ``|z' (M_true - M_int) z| <= g z' N z``."""
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    pts = np.stack(np.meshgrid(grid.v_syn, grid.alt, grid.eta_s, indexing="ij"), axis=-1)
    out = np.zeros(grid.residual.shape)
    for idx in np.ndindex(*grid.residual.shape):
        a, b = design._normalize(*design.lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        out[idx] = _dt_gamma(f, p, a, b, tuple(jnp.asarray(v) for v in pts[idx]), dt)
    return out


def _dt_gamma(f: DTSafetyFilter, p, a, b, rho, dt) -> float:
    ph_t, gm_t = _zoh(a, b, dt)
    ph = np.asarray(interp(f.phi, *rho)).reshape(8, 8)
    gm = np.asarray(interp(f.gam, *rho)).reshape(8, 3)
    k = np.asarray(interp(f.k_n, *rho)).reshape(3, 8)

    def form(phi, gam):
        m = np.hstack([phi + gam @ k, gam])  # x+ = m z, z = (x, du)
        return m.T @ p @ m

    dm = form(ph_t, gm_t) - form(ph, gm)
    phc = ph + gm @ k
    n = scipy.linalg.block_diag(f.rho_d * p - phc.T @ p @ phc, gm.T @ p @ gm)
    ev = scipy.linalg.eigh(dm, n, eigvals_only=True)
    return float(np.abs(ev).max())


def estimate_dt_margin(
    setup: Setup, f: DTSafetyFilter, factor: float = 2.0, per_axis: int = 10, seed: int = 0
) -> tuple[DTSafetyFilter, dict]:
    """``g_d = factor * max g`` on the random conditions of :func:`estimate_margin`."""
    rng = np.random.default_rng(seed)
    axes = (
        np.sort(rng.uniform(60.0, 120.0, per_axis)),
        np.sort(rng.uniform(0.0, 10000.0, per_axis)),
        np.sort(rng.uniform(-25.0, 0.0, per_axis)),
    )
    g = dt_relative_error(f, design.build_grid(setup.plant, *axes))
    margin = factor * float(g.max())
    return f._replace(margin=margin), {"max_g": float(g.max()), "margin": margin, "factor": factor}
