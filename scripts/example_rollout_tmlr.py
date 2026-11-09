"""Example rollout (appendix figure) from the revision policies in results/tmlr/policies."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import CONDITIONS, controllers, scenarios
from figures import example_rollout
from morphjax import pipeline

res = ROOT / "results" / "tmlr"
metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8"))
setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
ctrls = controllers(setup, res / "policies", max_seeds=1)
i = list(CONDITIONS).index("aileron_fail")
_, kw = CONDITIONS["aileron_fail"]
scen = scenarios(metrics["meta"]["n_per_condition"], seed=1000 + i, **dict(kw))
shown = ("gs", "lpv", "ppo", "ppo_dr", "diffsim", "diffsim_dr", "residual_dr")
(res / "figures").mkdir(exist_ok=True)
example_rollout(setup, {k: [(ctrls[k][0], lat) for lat in ctrls[k][1]] for k in shown if k in ctrls}, scen, res / "figures" / "example_rollout.png")
print("ok", sorted(p.name for p in (res / "figures").iterdir()))

