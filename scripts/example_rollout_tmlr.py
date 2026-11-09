"""Redraw the appendix example rollout from the revision policies in results/tmlr/.

    uv run python scripts/example_rollout_tmlr.py

Same scenario as make_figures.py (aileron at about 10 %, fast symmetric morphing; the seeded
batch evaluate.py uses), one seed per learned controller.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluate import CONDITIONS, controllers, scenarios
from figures import example_rollout

from morphjax import pipeline

ROOT = Path(__file__).resolve().parents[1]
SHOWN = ("gs", "lpv", "ppo", "ppo_dr", "diffsim", "diffsim_dr", "residual_dr")


def main() -> None:
    res = ROOT / "results" / "tmlr"
    metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8"))
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    ctrls = controllers(setup, res / "policies", max_seeds=1)
    i = list(CONDITIONS).index("aileron_fail")
    _, kw = CONDITIONS["aileron_fail"]
    scen = scenarios(metrics["meta"]["n_per_condition"], seed=1000 + i, **dict(kw))
    example = {k: [(ctrls[k][0], lat) for lat in ctrls[k][1]] for k in SHOWN if k in ctrls}
    (res / "figures").mkdir(exist_ok=True)
    example_rollout(setup, example, scen, res / "figures" / "example_rollout.png")
    print(f"figures written to {res / 'figures'}")


if __name__ == "__main__":
    main()
