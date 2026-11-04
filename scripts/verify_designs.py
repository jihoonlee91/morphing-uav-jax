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
from morphjax.safety import make_safety_filter, sampled_decrease
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

    # The projected residual: the filtered law is not differentiable at trim when
    # the residual is nonzero there, so check the certificate condition directly on
    # the true linearization at every point, for random states and residuals.
    filt = make_safety_filter(setup)
    dec = sampled_decrease(
        filt, dense, interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    )
    for k, v in dec.items():
        points[f"residual_proj/{k}"] = v
    holds = dec["filtered_decay"] >= 1.0
    pooled["residual_proj"] = {
        "seeds": None,  # the check covers any residual in the box, not a particular policy
        "points": int(holds.size),
        "stable": int((dec["filtered_neg"] >= 1.0).sum()),
        "worst_real_eig": None,
        "min_delay_ms": None,
        "certified": int(holds.sum()),
        "samples_per_point": 4096,
        "min_fraction_decay_holds": float(dec["filtered_decay"].min()),
        "unfiltered_points_decay_always_holds": int((dec["raw_decay"] >= 1.0).sum()),
        "unfiltered_min_fraction_decay_holds": float(dec["raw_decay"].min()),
        "note": (
            "certified = V-dot <= -2*lambda*V held for every sampled state and residual on the "
            "true frozen linearization; stable = V-dot < 0 for every sample. Not a delay check."
        ),
    }
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
