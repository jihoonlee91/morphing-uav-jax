"""Model-based controller synthesis on the paper's 63-point design grid.

Every design point is trimmed and linearized with JAX (vectorized with
``vmap``). Three lateral-directional designs are built on top:

* ``fixed``  — one LQR designed at the nominal point and used everywhere.
* ``gs``     — pointwise LQR at every grid point, gains interpolated online
  (classical gain scheduling: no guarantee while the parameters move).
* ``lpv``    — a guaranteed-cost state feedback with one Lyapunov matrix shared
  by all grid points, found by LMI optimization. A common quadratic Lyapunov
  function certifies stability for arbitrarily fast parameter variation on the
  gridded LPV model, which is the property the paper relies on.

The longitudinal axis uses a gain-scheduled LQR airspeed/altitude hold shared
by all lateral controllers, so the comparison isolates the lateral design.
"""

from __future__ import annotations

from typing import NamedTuple

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from morphjax.dynamics import Cmd, Plant, W
from morphjax.trim import linearize, synthetic_to_true, trim

# Scheduling grid of Sec. 2.2.1: synthetic airspeed (kt), altitude (ft), eta_s (%).
V_SYN = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)
ALT = (0.0, 5000.0, 10000.0)
ETA_S = (-25.0, -12.5, 0.0)
NOMINAL = (90.0, 5000.0, -12.5)

# Lateral design model: state [beta p r phi da dr eta_a  int(phi - phi_cmd)].
LAT_STATES = (W.BETA, W.P, W.R, W.PHI, W.DA, W.DR, W.ETA_A)
LAT_INPUTS = (Cmd.DA, Cmd.DR, Cmd.ETA_A)
# Bryson scaling: "how large is acceptable" for each state and input.
LAT_X_SCALE = np.array([np.deg2rad(2), np.deg2rad(40), np.deg2rad(20), np.deg2rad(8),
                        30.0, 30.0, 12.5, np.deg2rad(4)])  # fmt: skip
LAT_U_SCALE = np.array([20.0, 20.0, 10.0])


class LatWeights(NamedTuple):
    """Multipliers on the default Bryson scales (smaller = tighter control).

    ``scripts/tune_lqr.py`` searches these on the PPO training distribution
    with the PPO reward, so the model-based baselines are tuned for the same
    objective as the learned policy.
    """

    phi: float = 1.0  # roll-angle error
    integ: float = 1.0  # roll-error integral
    beta: float = 1.0  # sideslip
    eta_a: float = 1.0  # asymmetric-morphing command (larger = cheaper to use)
    effort: float = 1.0  # all lateral commands (smaller = costlier = lower gains)

    def scales(self) -> tuple[np.ndarray, np.ndarray]:
        xs = LAT_X_SCALE.copy()
        xs[3] *= self.phi
        xs[7] *= self.integ
        xs[0] *= self.beta
        us = LAT_U_SCALE.copy() * self.effort
        us[2] *= self.eta_a
        return xs, us


# Longitudinal design model: [V alpha q theta h de thr  int(V-Vc) int(h-hc)].
LON_STATES = (W.V, W.ALPHA, W.Q, W.THETA, W.H, W.DE, W.THR)
LON_INPUTS = (Cmd.DE, Cmd.THR)
LON_X_SCALE = np.array([3.0, np.deg2rad(3), np.deg2rad(15), np.deg2rad(5), 10.0,
                        20.0, 50.0, 3.0, 15.0])  # fmt: skip
LON_U_SCALE = np.array([8.0, 30.0])


class DesignGrid(NamedTuple):
    """Trim points and linear models on the scheduling grid, shape (7, 3, 3, ...)."""

    v_syn: np.ndarray
    alt: np.ndarray
    eta_s: np.ndarray
    xw: np.ndarray  # (..., 18) trim state
    cmd: np.ndarray  # (..., 6) trim command
    A: np.ndarray  # (..., 18, 18)
    B: np.ndarray  # (..., 18, 6)
    residual: np.ndarray


def build_grid(plant: Plant, v_syn=V_SYN, alt=ALT, eta_s=ETA_S) -> DesignGrid:
    """Trim and linearize on a scheduling grid (default: the paper's 63 points)."""
    vs, hs, es = np.meshgrid(v_syn, alt, eta_s, indexing="ij")
    vt = synthetic_to_true(vs, hs)

    def one(v, h, e):
        tp = trim(plant, v, h, e)
        A, B = linearize(plant, tp.xw, tp.cmd)
        return tp.xw, tp.cmd, A, B, tp.residual

    flat = [jnp.asarray(a.ravel()) for a in (vt, hs, es)]
    xw, cmd, A, B, res = jax.vmap(one)(*flat)
    shape = vs.shape
    return DesignGrid(
        np.asarray(v_syn, dtype=float),
        np.asarray(alt, dtype=float),
        np.asarray(eta_s, dtype=float),
        np.asarray(xw).reshape(*shape, -1),
        np.asarray(cmd).reshape(*shape, -1),
        np.asarray(A).reshape(*shape, *A.shape[1:]),
        np.asarray(B).reshape(*shape, *B.shape[1:]),
        np.asarray(res).reshape(shape),
    )


def lateral_model(A: np.ndarray, B: np.ndarray):
    """Augmented lateral model with a roll-angle error integrator."""
    s, i = list(LAT_STATES), list(LAT_INPUTS)
    a = np.zeros((8, 8))
    a[:7, :7] = A[np.ix_(s, s)]
    a[7, 3] = 1.0  # d/dt int(phi - phi_cmd) = phi - phi_cmd
    b = np.zeros((8, 3))
    b[:7] = B[np.ix_(s, i)]
    return a, b


def longitudinal_model(A: np.ndarray, B: np.ndarray):
    """Augmented longitudinal model with airspeed and altitude integrators."""
    s, i = list(LON_STATES), list(LON_INPUTS)
    a = np.zeros((9, 9))
    a[:7, :7] = A[np.ix_(s, s)]
    a[7, 0] = 1.0
    a[8, 4] = 1.0
    b = np.zeros((9, 2))
    b[:7] = B[np.ix_(s, i)]
    return a, b


def _normalize(a, b, xs, us):
    dx, du = np.diag(xs), np.diag(us)
    return np.linalg.solve(dx, a @ dx), np.linalg.solve(dx, b @ du)


def lqr_gain(a: np.ndarray, b: np.ndarray, xs: np.ndarray, us: np.ndarray) -> np.ndarray:
    """LQR gain ``K`` (``u = K x``), Bryson weights ``Q = diag(1/xs^2)``, ``R = diag(1/us^2)``."""
    an, bn = _normalize(a, b, xs, us)
    p = scipy.linalg.solve_continuous_are(an, bn, np.eye(len(xs)), np.eye(len(us)))
    kn = -bn.T @ p
    return np.diag(us) @ kn @ np.diag(1.0 / xs)


class Gains(NamedTuple):
    """Gain tables on the grid: ``lat`` (7,3,3,3,8) and ``lon`` (7,3,3,2,9)."""

    lat: np.ndarray
    lon: np.ndarray


def _grid_iter(shape):
    return np.ndindex(*shape)


def delayed_spectral_radius(a, b, k, dt: float, delay_steps: int) -> float:
    """Spectral radius of the sampled loop ``u_k = K x_{k-d}`` (ZOH, ``d`` samples late).

    The augmented state is ``[x_k, u_{k-1}, ..., u_{k-d}]``; ``< 1`` means the
    frozen-point loop tolerates that much transport delay. This is the
    sampled-data analogue of a classical phase-margin requirement.
    """
    n, m = b.shape
    ad = scipy.linalg.expm(a * dt)
    # ZOH input matrix via the block-matrix exponential.
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = a, b
    bd = scipy.linalg.expm(blk * dt)[:n, n:]
    d = delay_steps
    if d == 0:
        return float(np.abs(np.linalg.eigvals(ad + bd @ k)).max())
    size = n + d * m
    big = np.zeros((size, size))
    big[:n, :n] = ad
    big[:n, n + (d - 1) * m :] = bd  # oldest command acts now
    big[n : n + m, :n] = k  # newest command from the current state
    for j in range(1, d):
        big[n + j * m : n + (j + 1) * m, n + (j - 1) * m : n + j * m] = np.eye(m)
    return float(np.abs(np.linalg.eigvals(big)).max())


def lateral_delay_margin_ok(grid: DesignGrid, gains: np.ndarray, dt: float, steps: int) -> bool:
    """True if every frozen grid point stays stable with ``steps`` samples of delay."""
    for idx in _grid_iter(grid.residual.shape):
        a, b = lateral_model(grid.A[idx], grid.B[idx])
        if delayed_spectral_radius(a, b, gains[idx], dt, steps) >= 1.0:
            return False
    return True


def max_stable_delay(grid: DesignGrid, gains: np.ndarray, dt: float, max_steps: int = 10) -> int:
    """Largest delay (samples) every frozen point tolerates; -1 if unstable even without."""
    best = -1
    for d in range(max_steps + 1):
        if not lateral_delay_margin_ok(grid, gains, dt, d):
            break
        best = d
    return best


def longitudinal_gains(grid: DesignGrid) -> np.ndarray:
    out = np.zeros((*grid.residual.shape, 2, 9))
    for idx in _grid_iter(grid.residual.shape):
        a, b = longitudinal_model(grid.A[idx], grid.B[idx])
        out[idx] = lqr_gain(a, b, LON_X_SCALE, LON_U_SCALE)
    return out


def lateral_gains_gs(grid: DesignGrid, weights: LatWeights | None = None) -> np.ndarray:
    xs, us = (weights or LatWeights()).scales()
    out = np.zeros((*grid.residual.shape, 3, 8))
    for idx in _grid_iter(grid.residual.shape):
        a, b = lateral_model(grid.A[idx], grid.B[idx])
        out[idx] = lqr_gain(a, b, xs, us)
    return out


def nominal_index(grid: DesignGrid) -> tuple[int, int, int]:
    return (
        int(np.argmin(np.abs(grid.v_syn - NOMINAL[0]))),
        int(np.argmin(np.abs(grid.alt - NOMINAL[1]))),
        int(np.argmin(np.abs(grid.eta_s - NOMINAL[2]))),
    )


def lateral_gains_fixed(grid: DesignGrid, weights: LatWeights | None = None) -> np.ndarray:
    """The nominal-point LQR gain broadcast to every grid point."""
    gs = lateral_gains_gs(grid, weights)
    return np.broadcast_to(gs[nominal_index(grid)], gs.shape).copy()


class LPVResult(NamedTuple):
    gains: np.ndarray  # (7, 3, 3, 3, 8), physical units
    X: np.ndarray  # common Lyapunov matrix (normalized coordinates)
    status: str
    cost_bound: float
    lyapunov_margin: float  # max eigenvalue of the Lyapunov derivative (< 0 = certified)
    decay: float = 0.1  # guaranteed decay rate lambda of the synthesis


def lateral_gains_lpv(
    grid: DesignGrid,
    weights: LatWeights | None = None,
    decay: float = 0.1,
    radius: float = 30.0,
    solver: str = "CLARABEL",
    disk: bool = True,
) -> LPVResult:
    """Guaranteed-cost LPV state feedback with a common Lyapunov matrix.

    For every grid point ``i`` (normalized coordinates, ``Q = R = I``):

        [[A_i X + X A_i' + B_i Y_i + Y_i' B_i' + 2 decay X,  X,  Y_i'],
         [X, -I, 0], [Y_i, 0, -I]]  <= 0,        X > 0,

    minimizing ``trace(Z)`` with ``[[Z, I], [I, X]] >= 0`` (a bound on the
    quadratic cost from unit initial conditions). Then ``K_i = Y_i X^-1``.

    A disk constraint ``[[-r X, A X + B Y], [*, -r X]] <= 0`` keeps every
    frozen-point closed-loop pole inside ``|s| < radius`` so the design does
    not outrun the 5 Hz surface actuators or the 50 Hz control loop.
    ``disk=False`` drops it (an ablation of the revision).
    """
    xs, us = (weights or LatWeights()).scales()
    n, m = 8, 3
    X = cp.Variable((n, n), symmetric=True)
    Z = cp.Variable((n, n), symmetric=True)
    cons: list[cp.Constraint] = [
        X >> 1e-6 * np.eye(n),
        cp.bmat([[Z, np.eye(n)], [np.eye(n), X]]) >> 0,
    ]
    Ys = {}
    for idx in _grid_iter(grid.residual.shape):
        a, b = _normalize(*lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        Y = cp.Variable((m, n))
        Ys[idx] = Y
        lyap = a @ X + X @ a.T + b @ Y + Y.T @ b.T + 2 * decay * X
        blk = cp.bmat(
            [
                [lyap, X, Y.T],
                [X, -np.eye(n), np.zeros((n, m))],
                [Y, np.zeros((m, n)), -np.eye(m)],
            ]
        )
        cons.append(0.5 * (blk + blk.T) << 0)
        if disk:
            acl = a @ X + b @ Y
            dsk = cp.bmat([[-radius * X, acl], [acl.T, -radius * X]])
            cons.append(0.5 * (dsk + dsk.T) << 0)
    prob = cp.Problem(cp.Minimize(cp.trace(Z)), cons)
    prob.solve(solver=solver)
    if X.value is None:
        raise RuntimeError(f"LPV synthesis failed: {prob.status}")
    Xv = 0.5 * (X.value + X.value.T)
    Xinv = np.linalg.inv(Xv)
    out = np.zeros((*grid.residual.shape, m, n))
    worst = -np.inf
    for idx, Y in Ys.items():
        kn = Y.value @ Xinv
        out[idx] = np.diag(us) @ kn @ np.diag(1.0 / xs)
        # Interior-point solvers may report "optimal_inaccurate"; check the
        # certificate itself: (A+BK)'P + P(A+BK) < 0 with the common P = X^-1.
        a, b = _normalize(*lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        acl = a + b @ kn
        worst = max(worst, float(np.linalg.eigvalsh(acl.T @ Xinv + Xinv @ acl).max()))
    if worst >= 0:
        raise RuntimeError(f"common Lyapunov certificate does not hold (max eig {worst:.3g})")
    return LPVResult(out, Xv, str(prob.status), float(np.asarray(prob.value)), worst, decay)


def lateral_gains_lpv_pointwise(
    grid: DesignGrid,
    weights: LatWeights | None = None,
    decay: float = 0.1,
    radius: float = 30.0,
    solver: str = "CLARABEL",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ablation: the LMIs of :func:`lateral_gains_lpv` with a separate ``X_i`` per point.

    Same decay rate, pole disk and ``trace(Z_i)`` objective, but no common
    Lyapunov matrix, so the problem decouples into one SDP per design point
    (a pointwise guaranteed-cost design, i.e. what the common ``P`` adds).
    Returns ``(gains, X per point, solver status per point)``.
    """
    xs, us = (weights or LatWeights()).scales()
    n, m = 8, 3
    out = np.zeros((*grid.residual.shape, m, n))
    xs_out = np.zeros((*grid.residual.shape, n, n))
    statuses = []
    for idx in _grid_iter(grid.residual.shape):
        a, b = _normalize(*lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        X = cp.Variable((n, n), symmetric=True)
        Z = cp.Variable((n, n), symmetric=True)
        Y = cp.Variable((m, n))
        lyap = a @ X + X @ a.T + b @ Y + Y.T @ b.T + 2 * decay * X
        blk = cp.bmat([[lyap, X, Y.T], [X, -np.eye(n), np.zeros((n, m))],
                       [Y, np.zeros((m, n)), -np.eye(m)]])  # fmt: skip
        acl = a @ X + b @ Y
        dsk = cp.bmat([[-radius * X, acl], [acl.T, -radius * X]])
        cons: list[cp.Constraint] = [
            X >> 1e-6 * np.eye(n),
            cp.bmat([[Z, np.eye(n)], [np.eye(n), X]]) >> 0,
            0.5 * (blk + blk.T) << 0,
            0.5 * (dsk + dsk.T) << 0,
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(Z)), cons)
        prob.solve(solver=solver)
        if X.value is None or Y.value is None:
            raise RuntimeError(f"pointwise LPV synthesis failed at {idx}: {prob.status}")
        Xv = 0.5 * (X.value + X.value.T)
        kn = Y.value @ np.linalg.inv(Xv)
        out[idx] = np.diag(us) @ kn @ np.diag(1.0 / xs)
        xs_out[idx] = Xv
        statuses.append(str(prob.status))
    return out, xs_out, statuses
