"""Redraw the figures from results/ without re-running the evaluation.

    uv run python scripts/make_figures.py

Stress tests and learning curves come straight from ``metrics.json`` and the
policy JSON files; the example rollout re-simulates one scenario (the same
seeded batch ``evaluate.py`` uses), so it needs the trained policies.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluate import CONDITIONS, controllers, scenarios
from figures import PLOTTED, example_rollout, learning_curves, stress_tests

from morphjax import pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results = ROOT / "results"
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    fig_dir = results / "figures"
    names = [k for k in PLOTTED if k in metrics["table"]["nominal"]]
    stress_tests(metrics["sweeps"], names, fig_dir / "stress_tests.png")
    residual = [k for k in ("lpv", "residual_dr", "residual_proj_trained", "residual_safe")
                if k in metrics["table"]["nominal"]]  # fmt: skip
    if len(residual) > 2:
        stress_tests(metrics["sweeps"], residual, fig_dir / "residual_filter.png")
    learning_curves(
        results / "policies",
        fig_dir / "learning_curve.png",
        metrics["meta"]["training_distribution_return"].get("gs"),
    )
    setup = pipeline.build_for_evaluation(results / "lqr_weights.json")
    ctrls = controllers(setup, results / "policies", max_seeds=1)
    i = list(CONDITIONS).index("aileron_fail")
    _, kw = CONDITIONS["aileron_fail"]
    scen = scenarios(metrics["meta"]["n_per_condition"], seed=1000 + i, **dict(kw))
    shown = ("gs", "lpv", "ppo", "ppo_dr", "diffsim", "diffsim_dr", "residual_dr")
    example = {k: [(ctrls[k][0], lat) for lat in ctrls[k][1]] for k in shown if k in ctrls}
    example_rollout(setup, example, scen, fig_dir / "example_rollout.png")
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
