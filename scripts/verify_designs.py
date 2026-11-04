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
from morphjax.rl import PPOConfig, load_params
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
        for f in sorted((ROOT / "results" / "policies").glob(f"{method}_{variant}_seed*.npz")):
            gains, saturated = policy_gains(load_params(f, PPOConfig()), dense)
            checks = check_gains(setup, dense, gains)
            row = summarize(checks, on_grid)._asdict()
            row["seed_file"] = f.name
            row["points_with_saturated_trim_action"] = int(saturated.sum())
            learned.setdefault(name, []).append(row)
            record(f"{name}/{f.stem}", checks)
    out["learned_policies_linearized"] = learned

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
