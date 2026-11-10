"""Runtime controllers: scheduled feedforward + feedback, all pure JAX.

Every controller sees the same measurement (wind-axis state) and the same
references. Gains and trim values are interpolated on the design grid at the
current flight condition ``rho = (V_syn, h, eta_s)``, clipped to the grid.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from morphjax.airframe import KT_TO_FPS, G
from morphjax.data import Table
from morphjax.design import DesignGrid
from morphjax.dynamics import Cmd, W
from morphjax.interp import interp
from morphjax.trim import true_to_synthetic


class Schedule(NamedTuple):
    trim_x: Table  # rho -> trim wind-axis state (18)
    trim_u: Table  # rho -> trim command (6)
    k_lon: Table  # rho -> (2, 9) flattened
    k_lat: Table  # rho -> (3, 8) flattened


def make_schedule(grid: DesignGrid, k_lon: np.ndarray, k_lat: np.ndarray) -> Schedule:
    grids = (jnp.asarray(grid.v_syn), jnp.asarray(grid.alt), jnp.asarray(grid.eta_s))
    shp = grid.residual.shape

    def tab(a):
        return Table(grids, jnp.asarray(np.asarray(a).reshape(*shp, -1)))

    return Schedule(tab(grid.xw), tab(grid.cmd), tab(k_lon), tab(k_lat))


def scheduling_point(xw: jnp.ndarray, sched: Schedule) -> tuple[jnp.ndarray, ...]:
    """``(V_syn, h, eta_s)`` of the current state, clipped to the design grid."""
    v_kt = xw[W.V] / KT_TO_FPS
    h = xw[W.H]
    rho = (true_to_synthetic(v_kt, jnp.clip(h, 0.0, 25000.0)), h, xw[W.ETA_S])
    return tuple(jnp.clip(r, g[0], g[-1]) for r, g in zip(rho, sched.trim_x.grids, strict=True))


def turn_rates(xw: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Steady coordinated-turn ``(q, r)`` at the current bank angle."""
    phi, theta, v = xw[W.PHI], xw[W.THETA], jnp.maximum(xw[W.V], 30.0)
    psi_dot = G * jnp.tan(phi) / v
    return psi_dot * jnp.sin(phi) * jnp.cos(theta), psi_dot * jnp.cos(phi) * jnp.cos(theta)


def longitudinal_law(sched: Schedule, xw, v_cmd, h_cmd, integ_v, integ_h):
    """Shared airspeed/altitude hold: returns ``(elevator, throttle)`` commands."""
    rho = scheduling_point(xw, sched)
    xt, ut = interp(sched.trim_x, *rho), interp(sched.trim_u, *rho)
    k = interp(sched.k_lon, *rho).reshape(2, 9)
    q_turn, _ = turn_rates(xw)
    dx = jnp.array(
        [
            xw[W.V] - v_cmd,
            xw[W.ALPHA] - xt[W.ALPHA],
            xw[W.Q] - q_turn,
            xw[W.THETA] - xt[W.THETA],
            xw[W.H] - h_cmd,
            xw[W.DE] - ut[Cmd.DE],
            xw[W.THR] - ut[Cmd.THR],
            integ_v,
            integ_h,
        ]
    )
    return jnp.array([ut[Cmd.DE], ut[Cmd.THR]]) + k @ dx


def lateral_error_state(sched: Schedule, xw, phi_cmd, integ_phi):
    """Deviation state of the lateral design model and the trim command."""
    rho = scheduling_point(xw, sched)
    xt, ut = interp(sched.trim_x, *rho), interp(sched.trim_u, *rho)
    _, r_turn = turn_rates(xw)
    dx = jnp.array(
        [
            xw[W.BETA] - xt[W.BETA],
            xw[W.P],
            xw[W.R] - r_turn,
            xw[W.PHI] - phi_cmd,
            xw[W.DA] - ut[Cmd.DA],
            xw[W.DR] - ut[Cmd.DR],
            xw[W.ETA_A],
            integ_phi,
        ]
    )
    return dx, jnp.array([ut[Cmd.DA], ut[Cmd.DR], 0.0]), rho


def lateral_law(sched: Schedule, xw, phi_cmd, integ_phi):
    """Scheduled state feedback: returns ``(aileron, rudder, eta_a)`` commands."""
    dx, u0, rho = lateral_error_state(sched, xw, phi_cmd, integ_phi)
    k = interp(sched.k_lat, *rho).reshape(3, 8)
    return u0 + k @ dx


def make_delayed_lateral(sched: Schedule, k_table, delay: int):
    """Round 5: discrete delay-aware LQR (``design.lateral_gains_dlqr``) as a lateral law.

    ``k_table`` is a :class:`Table` of the (3, 8 + 3 d) gains; the law uses the ``delay``
    pending commands from the command buffer (row 0 = computed one sample ago).
    """
    lat_idx = jnp.array([Cmd.DA, Cmd.DR, Cmd.ETA_A])

    def lateral(xw, phi_cmd, integ_phi, buf=None, delay_now=None):
        dx, u0, rho = lateral_error_state(sched, xw, phi_cmd, integ_phi)
        k = interp(k_table, *rho).reshape(3, 8 + 3 * delay)
        u = u0 + k[:, :8] @ dx
        for j in range(delay):
            assert buf is not None
            u = u + k[:, 8 + 3 * j : 11 + 3 * j] @ (buf[j][lat_idx] - u0)
        return u

    lateral.uses_history = delay > 0  # type: ignore[attr-defined]
    return lateral
