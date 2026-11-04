"""Linear checks of every controller on a 325-point grid (262 points off the design grid).

    uv run python scripts/verify_designs.py

Model-based designs use their interpolated gains; learned policies are
linearized with jax.jacfwd at each trim point (see morphjax.verify). Writes
``results/verification.json`` (summaries) and ``results/verification_points.npz``
(per-point eigenvalue, tolerated delay, and LPV-certificate arrays).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from morphjax import pipeline
from morphjax.rl import RESIDUAL_SCALE, PPOConfig, load_params
from morphjax.safety import (
    _relative_error,
    estimate_margin,
    make_safety_filter,
    project,
    sampled_decrease,
)
from morphjax.verify import check_gains, dense_grid, interpolate_gains, policy_gains, summarize

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t0 = time.time()
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    dense, on_grid = dense_grid(setup)
    out: dict = {"grid": {"points": int(on_grid.size), "on_design_grid": int(on_grid.sum())}}
    points: dict[str, np.ndarray] = {"on_grid": on_grid}

    def record(name, checks):
        for field, arr in checks._asdict().items():
            points[f"{name}/{field}"] = arr

    designs = {}
    for name, gains in setup.lat_gains.items():
        checks = check_gains(setup, dense, interpolate_gains(setup.grid, gains, dense))
        designs[name] = summarize(checks, on_grid)._asdict()
        record(name, checks)
    out["designs"] = designs

    learned: dict[str, list] = {}
    for name, (method, variant) in pipeline.LEARNED.items():
        if name in pipeline.PROJECTED:
            continue  # not linearizable at trim; checked by sampling below
        for f in sorted((ROOT / "results" / "policies").glob(f"{method}_{variant}_seed*.npz")):
            params = load_params(f, PPOConfig())
            if name in pipeline.RESIDUAL_BASE:
                # Residual policies act on top of the base law: K = K_base + K_residual.
                base = setup.lat_gains[pipeline.RESIDUAL_BASE[name]]
                gains, saturated = policy_gains(params, dense, RESIDUAL_SCALE)
                gains = gains + interpolate_gains(setup.grid, base, dense)
            else:
                gains, saturated = policy_gains(params, dense)
            checks = check_gains(setup, dense, gains)
            row = summarize(checks, on_grid)._asdict()
            row["seed_file"] = f.name
            row["points_with_saturated_trim_action"] = int(saturated.sum())
            learned.setdefault(name, []).append(row)
            record(f"{name}/{f.stem}", checks)
    out["learned_policies_linearized"] = learned

    # One entry per controller, pooled over seeds, in the same fields as "designs".
    pooled = {}
    for name, rows in learned.items():
        pooled[name] = {
            "seeds": len(rows),
            "points": sum(r["points"] for r in rows),
            "stable": sum(r["stable"] for r in rows),
            "worst_real_eig": max(r["worst_real_eig"] for r in rows),
            "min_delay_ms": min(r["min_delay_ms"] for r in rows),
            "certified": sum(r["certified"] for r in rows),
        }

    # Residual variants: the projected laws are not differentiable at trim when the
    # residual is nonzero there, so the certificate condition is checked directly on
    # the true linearization at every point, for random states and box residuals.
    # The check exercises the filter, not a particular policy, so residual_proj and
    # residual_proj_trained (same filter) share one result.
    filt = make_safety_filter(setup)
    filt_safe, margin_info = estimate_margin(setup, filt)
    lpv_dense = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    dec = sampled_decrease(filt, dense, lpv_dense, {"lpv": None, "residual_dr": None,
                                                    "residual_proj": project})  # fmt: skip
    dec |= sampled_decrease(filt_safe, dense, lpv_dense, {"residual_safe": project})
    dec["residual_proj_trained"] = dec["residual_proj"]
    gamma, _ = _relative_error(filt, dense)
    proven = gamma <= filt_safe.margin
    sampled = {}
    for name, d in dec.items():
        for k, v in d.items():
            points[f"sampled/{name}/{k}"] = v
        sampled[name] = {
            "points": int(d["decay"].size),
            "decay_holds_for_all_samples": int((d["decay"] >= 1.0).sum()),
            "negative_for_all_samples": int((d["neg"] >= 1.0).sum()),
            "worst_share_decay": float(d["decay"].min()),
            "worst_share_negative": float(d["neg"].min()),
        }
    points["sampled/residual_safe/gamma"] = gamma
    out["sampled_decrease"] = {
        "samples_per_point": 4096,
        "state_norm": "log-uniform 1e-3..10 (normalized), directions uniform",
        "residuals": "uniform in the +/-20 % box",
        "condition": "decay: V-dot <= -2*lambda*V (lambda = 0.1); negative: V-dot < 0",
        "rows": sampled,
        "margin": margin_info | {"points_where_margin_covers_gamma": int(proven.sum())},
    }
    for name in ("residual_proj", "residual_proj_trained", "residual_safe"):
        s = sampled[name]
        pooled[name] = {
            "seeds": None,  # a property of the filter for any residual in the box
            "points": s["points"],
            "stable": s["negative_for_all_samples"],
            "worst_real_eig": None,  # not linearizable at trim
            "min_delay_ms": None,  # the guarantee assumes no delay; see the delay sweep
            "certified": s["decay_holds_for_all_samples"],
        }
    pooled["residual_safe"]["certified_for_all_states"] = int(proven.sum())
    out["learned_policies_pooled"] = pooled

    (ROOT / "results" / "verification.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(ROOT / "results" / "verification_points.npz", **points)

    print(f"{'controller':12s} {'stable':>9s} {'min delay':>10s} {'LPV-certified':>14s}")
    rows = list(designs.items()) + [
        (f"{n}[{i}]", r) for n, rs in learned.items() for i, r in enumerate(rs)
    ]
    for name, c in rows:
        print(f"{name:12s} {c['stable']:>4d}/{c['points']:<4d} {c['min_delay_ms']:8.0f}ms "
              f"{c['certified']:>9d}/{c['points']}")  # fmt: skip
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
