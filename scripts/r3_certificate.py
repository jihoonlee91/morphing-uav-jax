# ruff: noqa: E501
"""Round 3: discrete-time check of the certified residual and the margin's supremum.

    uv run python scripts/r3_certificate.py

(a) Sampled-data decrease. The certificate is a continuous-time condition
``V-dot <= -2 lambda V`` on the frozen model. With the command held over a 50 Hz
sample (ZOH), the natural discrete analogue is ``V(x+) <= exp(-2 lambda T) V(x)``,
i.e. ``V(x+) - V(x) <= -(1 - exp(-2 lambda T)) V(x)``, with
``x+ = Ad x + Bd (K x + du)`` from the *true* linearization at each of the 325 check
points, the interpolated LPV gain, and the residual ``du`` filtered (with the 2x margin)
at ``x``. States and residuals are sampled exactly as in ``safety.sampled_decrease``
(seed 0, 4,096 per point). Also reported: plain ``V(x+) < V(x)``, and the same for the
LPV law alone and the unfiltered residual.

(b) Supremum of gamma in the corner where it peaks: a 1 kt x 250 ft x 2.5 % grid over
60-70 kt, 0-2,500 ft and the full span range (1,331 conditions), with the 2 kt x 500 ft
x 5 % subgrid for comparison.

Writes ``results/tmlr/r3/certificate.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from morphjax import design, pipeline
from morphjax.rl import RESIDUAL_SCALE
from morphjax.safety import _relative_error, estimate_margin, make_safety_filter, project
from morphjax.sim import DT
from morphjax.verify import dense_grid, interpolate_gains

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t0 = time.time()
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    filt = make_safety_filter(setup)
    f2, info2 = estimate_margin(setup, filt, factor=2.0)
    dense, _ = dense_grid(setup)
    lpv_dense = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    xs, us, p = np.asarray(f2.xs), np.asarray(f2.us), np.asarray(f2.p)
    lam = f2.decay
    rho_d = np.exp(-2 * lam * DT)
    shp = dense.residual.shape
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    rng = np.random.default_rng(0)  # the draws of safety.sampled_decrease
    proj = jax.jit(jax.vmap(project, in_axes=(None, None, 0, 0)))
    box = np.asarray(RESIDUAL_SCALE)
    names = ("lpv", "residual_dr", "residual_safe2")
    share = {
        n: {"discrete_decay": np.zeros(shp), "discrete_decrease": np.zeros(shp)} for n in names
    }
    worst_ratio = {n: np.zeros(shp) for n in names}
    # Largest normalized state norm among samples that violate the discrete condition:
    # above it, every sample at that point satisfies it (0 = none violate).
    fail_norm = {n: np.zeros(shp) for n in names}
    for idx in np.ndindex(*shp):
        a_t, b_t = design._normalize(*design.lateral_model(dense.A[idx], dense.B[idx]), xs, us)
        n, m = b_t.shape
        blk = np.zeros((n + m, n + m))
        blk[:n, :n], blk[:n, n:] = a_t, b_t
        e = scipy.linalg.expm(blk * DT)
        ad, bd = e[:n, :n], e[:n, n:]
        k_n = np.diag(1.0 / us) @ lpv_dense[idx] @ np.diag(xs)
        direction = rng.normal(size=(4096, 8))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        x_n = direction * 10.0 ** rng.uniform(-3.0, 1.0, size=(4096, 1))
        du = rng.uniform(-1.0, 1.0, size=(4096, 3)) * box
        rho = tuple(jnp.asarray(v) for v in pts[idx])
        du_safe = np.asarray(proj(f2, rho, jnp.asarray(x_n * xs), jnp.asarray(du)))
        v = np.einsum("ni,ij,nj->n", x_n, p, x_n)
        for name, d in (
            ("lpv", np.zeros_like(du)),
            ("residual_dr", du),
            ("residual_safe2", du_safe),
        ):
            xp = x_n @ (ad + bd @ k_n).T + (d / us) @ bd.T
            vp = np.einsum("ni,ij,nj->n", xp, p, xp)
            share[name]["discrete_decay"][idx] = np.mean(vp <= rho_d * v + 1e-12 * v)
            share[name]["discrete_decrease"][idx] = np.mean(vp < v)
            worst_ratio[name][idx] = float((vp / v).max())
            bad = vp > rho_d * v + 1e-12 * v
            fail_norm[name][idx] = (
                float(np.linalg.norm(x_n[bad], axis=1).max()) if bad.any() else 0.0
            )
    out_a = {
        "condition": "V(x+) <= exp(-2*lambda*T) V(x), T = 0.02 s, lambda = 0.1 (ZOH, true linearization)",
        "exp_minus_2_lambda_T": rho_d,
        "margin_gamma_bar": info2["margin"],
    }
    for name in names:
        s = share[name]
        out_a[name] = {
            "points_all_samples_decay": int((s["discrete_decay"] >= 1).sum()),
            "points_all_samples_decrease": int((s["discrete_decrease"] >= 1).sum()),
            "min_share_decay": float(s["discrete_decay"].min()),
            "median_share_decay": float(np.median(s["discrete_decay"])),
            "max_V_ratio": float(worst_ratio[name].max()),
            "max_violating_state_norm_normalized": float(fail_norm[name].max()),
            "median_violating_state_norm_normalized": float(np.median(fail_norm[name])),
        }
    print(json.dumps(out_a, indent=1), flush=True)

    # (b) refined corner
    fine_axes = (np.arange(60.0, 70.0 + 1e-9, 1.0), np.arange(0.0, 2500.0 + 1e-9, 250.0),
                 np.arange(-25.0, 0.0 + 1e-9, 2.5))  # fmt: skip
    fine = design.build_grid(setup.plant, *fine_axes)
    g, _ = _relative_error(filt, fine)
    sub = g[::2, ::2, ::2]
    i = np.unravel_index(np.argmax(g), g.shape)
    out_b = {
        "grid": "60-70 kt step 1, 0-2500 ft step 250, eta_s -25..0 step 2.5 (1,331 points)",
        "sup_gamma": float(g.max()),
        "argmax": {"v_syn": float(fine_axes[0][i[0]]), "h": float(fine_axes[1][i[1]]),
                   "eta_s": float(fine_axes[2][i[2]])},
        "sup_gamma_half_resolution_subgrid": float(sub.max()),
        "points_above_margin_0.281": int((g > info2["margin"]).sum()),
        "margin": info2["margin"],
        "max_gamma_per_airspeed": [float(x) for x in g.max(axis=(1, 2))],
    }  # fmt: skip
    res = {"discrete_decrease": out_a, "corner_refinement": out_b, "wall_seconds": time.time() - t0}
    out = ROOT / "results" / "tmlr" / "r3" / "certificate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(
        out.with_suffix(".npz"),
        corner_gamma=g,
        **{f"{n}/{k}": v for n in names for k, v in share[n].items()},
    )
    print(json.dumps(out_b, indent=1))


if __name__ == "__main__":
    main()
