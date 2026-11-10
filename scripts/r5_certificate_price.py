# ruff: noqa: E501
"""Round 5: what the common Lyapunov certificate costs, and dense checks of the dt filter.

    uv run python scripts/r5_certificate_price.py

(a) Common-P test for fixed gains: for the GS LQR, the no-margin LQR (original and wide
    grid) and the fixed-point LQR, minimize t over P with trace(P) = n and
    A_cl,i' P + P A_cl,i + 2 lambda P <= t I at all 63 design points (normalized
    coordinates of the LPV synthesis), for lambda = 0 and 0.1. t* < 0: a common quadratic
    Lyapunov function (with that decay) exists; t* >= 0: none does.
(b) LPV synthesis with pole-disk radius in {30, 60, none} x decay rate in {0.05, 0.1,
    0.2}: held-out nominal RMSE (192 scenarios of ``r5_sim_frontier``), the simulated
    delay tolerance (both rules) and the frozen cumulative delay margin at the 63 design
    points and the 135 envelope points.
(c) residual_safe_dt on 2,025 between-grid conditions (25 x 9 x 9, every coordinate off
    the design grid): D > 0 on the interpolated model and g <= g_d.
Writes ``r5/certificate_price.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from functools import partial
from pathlib import Path

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from r3_frontier import cumulative_delay, zoh
from r5_sim_frontier import DELAYS, nominal_split, per_scenario, tolerance, with_delay

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.interp import interp
from morphjax.safety import _dt_gamma, estimate_dt_margin, make_dt_safety_filter
from morphjax.sim import DT, rollout
from morphjax.verify import dense_grid, interpolate_gains

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"


def common_p(setup, gains, lam):
    xs, us = setup.weights.scales()
    n = 8
    P = cp.Variable((n, n), symmetric=True)
    t = cp.Variable()
    cons = [P >> 1e-6 * np.eye(n), cp.trace(P) == n]
    for idx in np.ndindex(*setup.grid.residual.shape):
        an, bn = design._normalize(
            *design.lateral_model(setup.grid.A[idx], setup.grid.B[idx]), xs, us
        )
        acl = an + bn @ (np.diag(1 / us) @ gains[idx] @ np.diag(xs))
        m = acl.T @ P + P @ acl + 2 * lam * P
        cons.append(0.5 * (m + m.T) << t * np.eye(n))
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver="CLARABEL")
    return float(t.value) if t.value is not None else None, str(prob.status)


def main() -> None:
    t0 = time.time()
    setup = pipeline.build_revision(ROOT / "results" / "lqr_weights.json", TM / "lqr_wide.json")
    res: dict = {"common_P": {}}
    for name in ("gs", "gs_hg", "gs_hg_wide", "fixed", "lpv"):
        for lam in (0.0, 0.1):
            tstar, status = common_p(setup, setup.lat_gains[name], lam)
            res["common_P"][f"{name}/lambda={lam}"] = {"t_star": tstar, "status": status,
                                                       "common_P_exists": tstar is not None and tstar < 0}  # fmt: skip
            print(name, lam, tstar, status, flush=True)

    # (b) LPV synthesis variants
    k_lon = design.longitudinal_gains(setup.grid)
    _, held = nominal_split()
    dense, _on_grid = dense_grid(setup)
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    env = (pts[..., 0] >= 70) & (pts[..., 0] <= 110) & (pts[..., 1] >= 500) & (pts[..., 1] <= 9500)
    env_idx = [i for i in np.ndindex(*env.shape) if env[i]]
    env_models = [zoh(*design.lateral_model(dense.A[i], dense.B[i])) for i in env_idx]
    grid_idx = list(np.ndindex(*setup.grid.residual.shape))
    grid_models = [zoh(*design.lateral_model(setup.grid.A[i], setup.grid.B[i])) for i in grid_idx]
    res["lpv_variants"] = {}
    for radius in (30.0, 60.0, None):
        for lam in (0.05, 0.1, 0.2):
            key = f"radius={radius}/lambda={lam}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    lpv = design.lateral_gains_lpv(setup.grid, setup.weights, decay=lam,
                                                   radius=radius or 30.0, disk=radius is not None)  # fmt: skip
            except Exception as e:
                res["lpv_variants"][key] = {"failed": str(e)[:200]}
                print(key, "failed", flush=True)
                continue
            sched = make_schedule(setup.grid, k_lon, lpv.gains)
            run = jax.jit(
                jax.vmap(
                    lambda s, sched=sched: rollout(
                        setup.plant, sched, partial(lateral_law, sched), s
                    )
                )
            )
            st = []
            for d in DELAYS:
                r, lost = per_scenario(run(with_delay(held, d)), held)
                st.append((float(np.nanmean(r)) if (~lost).any() else np.inf, bool(lost.any())))
            dg = interpolate_gains(setup.grid, lpv.gains, dense)
            res["lpv_variants"][key] = {
                "cost_bound": lpv.cost_bound,
                "heldout_rmse_by_delay": [s[0] if np.isfinite(s[0]) else None for s in st],
                "heldout_lost_by_delay": [s[1] for s in st],
                "sim_tolerance": tolerance(st),
                "frozen_delay_design_points": cumulative_delay(
                    grid_models, [lpv.gains[i] for i in grid_idx]
                ),
                "frozen_delay_envelope": cumulative_delay(env_models, [dg[i] for i in env_idx]),
            }
            print(
                key,
                res["lpv_variants"][key]["heldout_rmse_by_delay"][0],
                res["lpv_variants"][key]["sim_tolerance"],
                flush=True,
            )

    # (c) dt filter on between-grid conditions
    f, info = estimate_dt_margin(setup, make_dt_safety_filter(setup), 2.0)
    axes = (60.0 + (np.arange(25) + 0.5) * 60.0 / 25, (np.arange(9) + 0.5) * 10000.0 / 9,
            -25.0 + (np.arange(9) + 0.5) * 25.0 / 9)  # fmt: skip
    grid = design.build_grid(setup.plant, *axes)
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    gp = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    dmin, gs = [], []
    for idx in np.ndindex(*grid.residual.shape):
        rho = tuple(jnp.asarray(v) for v in gp[idx])
        ph = np.asarray(interp(f.phi, *rho)).reshape(8, 8)
        gm = np.asarray(interp(f.gam, *rho)).reshape(8, 3)
        k = np.asarray(interp(f.k_n, *rho)).reshape(3, 8)
        phc = ph + gm @ k
        dmat = f.rho_d * p - phc.T @ p @ phc
        dmin.append(float(np.linalg.eigvalsh(0.5 * (dmat + dmat.T)).min()))
        a, b = design._normalize(*design.lateral_model(grid.A[idx], grid.B[idx]), xs, us)
        gs.append(_dt_gamma(f, p, a, b, rho, DT))
    dmin, gs = np.array(dmin), np.array(gs)
    res["dt_filter_between_grid"] = {"conditions": int(dmin.size), "D_positive_definite": int((dmin > 0).sum()),
                                     "min_eig_D": float(dmin.min()), "g_d": info["margin"],
                                     "max_g": float(gs.max()), "covered_g_le_g_d": int((gs <= info["margin"]).sum())}  # fmt: skip
    print(res["dt_filter_between_grid"], flush=True)
    res["wall_seconds"] = time.time() - t0
    (TM / "r5" / "certificate_price.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
