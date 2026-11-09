"""Revision: validate the interpolation-error margin on fresh held-out flight conditions.

    uv run python scripts/validate_margin.py [--out results/tmlr]

The certified residual tightens the Lyapunov half-space by a relative margin
``gamma_bar = factor * max gamma`` over 1,000 random conditions (factor 1.5 -> 0.21,
factor 2 -> 0.28; ``safety.estimate_margin``); the factor was raised from 1.5 to 2
after the 325-point check grid showed 322/325 coverage. This script measures gamma on
conditions independent of both sets:

* 2,000 random points in the envelope box: 1,500 uniform inside, 400 on the faces,
  92 on the edges, and the 8 corners (seed 2026, independent of the estimation seed 0);
* an offset 13 x 5 x 5 grid (cell centres of the check grid's spacing, so no point
  coincides with the check grid or the design grid).

It reports coverage of both margins and where gamma peaks. Writes
``margin_validation.json`` and ``margin_validation.npz`` (points and gamma).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from morphjax import design, pipeline
from morphjax.interp import interp
from morphjax.safety import _relative_error, estimate_margin, make_safety_filter
from morphjax.trim import linearize, synthetic_to_true, trim
from morphjax.verify import dense_grid

ROOT = Path(__file__).resolve().parents[1]
LO = np.array([60.0, 0.0, -25.0])
HI = np.array([120.0, 10000.0, 0.0])


def random_points(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(1500, 3))
    face = rng.uniform(size=(400, 3))
    ax = rng.integers(0, 3, 400)
    face[np.arange(400), ax] = rng.integers(0, 2, 400)
    edge = rng.integers(0, 2, size=(92, 3)).astype(float)
    free = rng.integers(0, 3, 92)
    edge[np.arange(92), free] = rng.uniform(size=92)
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T
    unit = np.concatenate([u, face, edge, corners.astype(float)])
    kind = np.array(["interior"] * 1500 + ["face"] * 400 + ["edge"] * 92 + ["corner"] * 8)
    return LO + unit * (HI - LO), kind


def gamma_at(setup, f, pts: np.ndarray) -> np.ndarray:
    """Relative error gamma (see safety._relative_error) at scattered points."""

    def one(v, h, e):
        tp = trim(setup.plant, synthetic_to_true(v, h), h, e)
        return linearize(setup.plant, tp.xw, tp.cmd)

    A, B = jax.vmap(one)(*(jnp.asarray(pts[:, i]) for i in range(3)))
    A, B = np.asarray(A), np.asarray(B)
    xs, us, p = np.asarray(f.xs), np.asarray(f.us), np.asarray(f.p)
    out = np.zeros(len(pts))
    for i, rho in enumerate(pts):
        a_t, b_t = design._normalize(*design.lateral_model(A[i], B[i]), xs, us)
        r = tuple(jnp.asarray(v) for v in rho)
        a_i = np.asarray(interp(f.a_n, *r)).reshape(8, 8)
        b_i = np.asarray(interp(f.b_n, *r)).reshape(8, 3)
        k_i = np.asarray(interp(f.k_n, *r)).reshape(3, 8)
        acl_t, acl_i = a_t + b_t @ k_i, a_i + b_i @ k_i
        m_i = acl_i.T @ p + p @ acl_i
        dm = (acl_t.T @ p + p @ acl_t) - m_i
        out[i] = float(scipy.linalg.eigh(dm, -(m_i + 2.0 * f.decay * p), eigvals_only=True).max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmlr")
    args = ap.parse_args()
    t0 = time.time()
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    filt = make_safety_filter(setup)
    margins = {}
    for factor in (1.5, 2.0):
        _, info = estimate_margin(setup, filt, factor=factor)
        margins[str(factor)] = info
    est_max = margins["2.0"]["max_gamma"]

    pts, kind = random_points()
    g_rand = gamma_at(setup, filt, pts)
    off_axes = (
        60.0 + (np.arange(13) + 0.5) * 60.0 / 13,
        (np.arange(5) + 0.5) * 10000.0 / 5,
        -25.0 + (np.arange(5) + 0.5) * 25.0 / 5,
    )
    off = design.build_grid(setup.plant, *off_axes)
    g_off, _ = _relative_error(filt, off)
    off_pts = np.stack(np.meshgrid(*off_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    check, _ = dense_grid(setup)
    g_check, _ = _relative_error(filt, check)

    def summary(g: np.ndarray, where: np.ndarray) -> dict:
        order = np.argsort(-g)[:5]
        return {
            "points": int(g.size),
            "max_gamma": float(g.max()),
            "quantiles_50_90_99": [float(q) for q in np.quantile(g, [0.5, 0.9, 0.99])],
            "covered_by_1.5x_margin": int((g <= margins["1.5"]["margin"]).sum()),
            "covered_by_2x_margin": int((g <= margins["2.0"]["margin"]).sum()),
            "exceeds_estimation_max": int((g > est_max).sum()),
            "top5": [{"v_syn": float(where[i][0]), "h": float(where[i][1]),
                      "eta_s": float(where[i][2]), "gamma": float(g[i])} for i in order],
        }  # fmt: skip

    out = {
        "margins": margins,
        "random_2000": summary(g_rand, pts)
        | {
            k: summary(g_rand[kind == k], pts[kind == k])
            for k in ("interior", "face", "edge", "corner")
        },
        "offset_grid_13x5x5": summary(g_off.ravel(), off_pts),
        "check_grid_325": summary(
            g_check.ravel(),
            np.stack(np.meshgrid(check.v_syn, check.alt, check.eta_s, indexing="ij"), -1).reshape(
                -1, 3
            ),
        ),
        "seed": 2026,
        "wall_seconds": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "margin_validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    arrays = dict(random_points=pts, random_kind=kind, random_gamma=g_rand,
                  offset_points=off_pts, offset_gamma=g_off.ravel())  # fmt: skip
    np.savez_compressed(args.out / "margin_validation.npz", **arrays)
    print(json.dumps({k: (v if not isinstance(v, dict) or "max_gamma" not in v else
                          {kk: v[kk] for kk in ("points", "max_gamma", "covered_by_1.5x_margin",
                                                "covered_by_2x_margin")})
                      for k, v in out.items() if k != "margins"}, indent=1))  # fmt: skip
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
