# ruff: noqa: E501
"""Round 3: the discrete-time (sampled-data) Lyapunov filter for the LPV residual.

    uv run python scripts/r3_dt_filter.py

Derivation in ``morphjax.safety`` (section "round 3"). This script

1. estimates the relative model-mismatch bound g on the 1,000 random conditions used for
   gamma-bar (``safety.estimate_margin``: seed 0, 10 x 10 x 10) and sets
   g_d = 2 x max g (the factor of the certified residual);
2. checks feasibility of du = 0, i.e. V(Phi_cl x) < rho_d V(x) (D > 0) on the
   interpolated model, at the 325 check points and 63 design points;
3. validates g_d on fresh conditions: the 2,000 random points and the offset 13x5x5 grid
   of ``validate_margin.py`` and the 1,331-point corner grid of ``r3_certificate.py``;
4. repeats the sampled-data check of ``r3_certificate.py`` (same states and residuals,
   seed 0) for the filtered residual, on the interpolated model the filter uses and on
   the true linearization at each check point.

Writes ``results/tmlr/r3/dt_filter.json``. The filter is evaluated in simulation as
``residual_safe_dt`` (the residual_dr policies, deploy-only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import design, pipeline
from morphjax.interp import interp
from morphjax.rl import RESIDUAL_SCALE
from morphjax.safety import (
    _dt_gamma,
    _zoh,
    dt_project,
    dt_relative_error,
    estimate_dt_margin,
    make_dt_safety_filter,
)
from morphjax.sim import DT
from morphjax.trim import linearize, synthetic_to_true, trim
from morphjax.verify import dense_grid, interpolate_gains

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
FACTOR = 2.0


def gamma_scattered(setup, f, pts):
    def one(v, h, e):
        tp = trim(setup.plant, synthetic_to_true(v, h), h, e)
        return linearize(setup.plant, tp.xw, tp.cmd)

    A, B = jax.vmap(one)(*(jnp.asarray(pts[:, i]) for i in range(3)))
    A, B = np.asarray(A), np.asarray(B)
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    return np.array([
        _dt_gamma(f, p, *design._normalize(*design.lateral_model(A[i], B[i]), xs, us),
                  tuple(jnp.asarray(v) for v in pts[i]), DT)
        for i in range(len(pts))
    ])  # fmt: skip


def main() -> None:
    t0 = time.time()
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    f = make_dt_safety_filter(setup, DT)
    f, info = estimate_dt_margin(setup, f, FACTOR)
    g_est, g_d = np.array(info["max_g"]), info["margin"]
    print(f"g max {g_est.max():.4g} -> g_d {g_d:.4g}; rho_d {f.rho_d:.6f}", flush=True)

    # feasibility of du = 0 on the interpolated model
    p = np.asarray(f.p)
    pinv_h = np.linalg.inv(np.linalg.cholesky(p))

    def ratio(rho):
        ph = np.asarray(interp(f.phi, *rho)).reshape(8, 8)
        gm = np.asarray(interp(f.gam, *rho)).reshape(8, 3)
        k = np.asarray(interp(f.k_n, *rho)).reshape(3, 8)
        phc = ph + gm @ k
        m = pinv_h @ phc.T @ p @ phc @ pinv_h.T
        return float(np.linalg.eigvalsh(0.5 * (m + m.T)).max())

    dense, _ = dense_grid(setup)
    dpts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    r_dense = np.array(
        [ratio(tuple(jnp.asarray(v) for v in dpts[i])) for i in np.ndindex(*dense.residual.shape)]
    )
    feas = {"max_V_ratio_LPV_interpolated_dense": float(r_dense.max()),
            "required_at_most": f.rho_d,
            "feasible_points_dense": int((r_dense < f.rho_d).sum()), "points": int(r_dense.size)}  # fmt: skip
    print(feas, flush=True)

    # validation on fresh conditions
    mv = np.load(TM / "margin_validation.npz")
    g_rand = gamma_scattered(setup, f, mv["random_points"])
    g_off = gamma_scattered(setup, f, mv["offset_points"])
    corner_axes = (np.arange(60.0, 70.0 + 1e-9, 1.0), np.arange(0.0, 2500.0 + 1e-9, 250.0),
                   np.arange(-25.0, 0.0 + 1e-9, 2.5))  # fmt: skip
    g_corner = dt_relative_error(f, design.build_grid(setup.plant, *corner_axes), DT)
    g_dense = dt_relative_error(f, dense, DT)
    valid = {name: {"points": int(g.size), "max_g": float(g.max()), "covered": int((g <= g_d).sum())}
             for name, g in (("random_2000", g_rand), ("offset_13x5x5", g_off),
                             ("corner_1331", g_corner), ("check_grid_325", g_dense))}  # fmt: skip
    print(json.dumps(valid), flush=True)

    # sampled-data check, same draws as r3_certificate.py
    lpv_dense = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    xs, us = np.asarray(f.xs), np.asarray(f.us)
    rng = np.random.default_rng(0)
    proj = jax.jit(jax.vmap(dt_project, in_axes=(None, None, 0, 0)))
    box = np.asarray(RESIDUAL_SCALE)
    shp = dense.residual.shape
    share = {m: np.zeros(shp) for m in ("interp", "true")}
    worst = {m: 0.0 for m in share}
    du_norm = np.zeros(shp)
    for idx in np.ndindex(*shp):
        a_t, b_t = design._normalize(*design.lateral_model(dense.A[idx], dense.B[idx]), xs, us)
        ad, bd = _zoh(a_t, b_t, DT)
        k_true = np.diag(1.0 / us) @ lpv_dense[idx] @ np.diag(xs)
        direction = rng.normal(size=(4096, 8))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        x_n = direction * 10.0 ** rng.uniform(-3.0, 1.0, size=(4096, 1))
        du = rng.uniform(-1.0, 1.0, size=(4096, 3)) * box
        rho = tuple(jnp.asarray(v) for v in dpts[idx])
        d = np.asarray(proj(f, rho, jnp.asarray(x_n * xs), jnp.asarray(du))) / us
        du_norm[idx] = float(
            np.linalg.norm(d, axis=1).mean() / np.linalg.norm(du / us, axis=1).mean()
        )
        v = np.einsum("ni,ij,nj->n", x_n, p, x_n)
        ph = np.asarray(interp(f.phi, *rho)).reshape(8, 8)
        gm = np.asarray(interp(f.gam, *rho)).reshape(8, 3)
        k_i = np.asarray(interp(f.k_n, *rho)).reshape(3, 8)
        for m, (A_, B_, K_) in (("interp", (ph, gm, k_i)), ("true", (ad, bd, k_true))):
            xp = x_n @ (A_ + B_ @ K_).T + d @ B_.T
            vp = np.einsum("ni,ij,nj->n", xp, p, xp)
            share[m][idx] = np.mean(vp <= f.rho_d * v * (1 + 1e-9))
            worst[m] = max(worst[m], float((vp / v).max()))
    check = {m: {"points_all_samples": int((share[m] >= 1).sum()), "min_share": float(share[m].min()),
                 "max_V_ratio": worst[m]} for m in share}  # fmt: skip
    check["mean_filtered_over_raw_residual_norm"] = float(du_norm.mean())
    res = {"rho_d": f.rho_d, "factor": FACTOR, "g_estimation_max": float(g_est.max()), "g_d": g_d,
           "feasibility": feas, "validation": valid, "sampled_data_check": check,
           "wall_seconds": time.time() - t0}  # fmt: skip
    out = TM / "r3" / "dt_filter.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
