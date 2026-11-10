# ruff: noqa: E501
"""Round 5: pick the PPO and APG search winners on the LQR tuning scenarios; write queue B.

    uv run python scripts/r5_select.py

Every search policy (``r5/search``: 16 PPO and 8 APG configs x seeds 0-2) flies the LQR
tuning scenarios (``tune_lqr``: seed 7, 96 scenarios of the nominal training
distribution, 10 s) deterministically; a config's score is the mean return over its seeds
(PPO reward, actuator-limit action scale, -10 at a departure). The winners are retrained
with 10 seeds (``ppo_tuned``, ``diffsim_tuned``) and, with domain randomization, 5 seeds
(``ppo_dr_tuned``, ``diffsim_dr_tuned``). Writes ``r5/search_selection.json`` and
``r5/jobs/train_r5b.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
from tune_lqr import SEED, N, episode_return

from morphjax import pipeline
from morphjax.rl import ACT_LIMITS, PPOConfig, load_policy, make_policy_lateral, training_scenario
from morphjax.sim import rollout

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "tmlr" / "r5"


def main() -> None:
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    lon = setup.schedules["gs"]
    cfg = PPOConfig()
    sc = jax.vmap(lambda k: training_scenario(k, cfg))(
        jax.random.split(jax.random.PRNGKey(SEED), N)
    )
    configs = json.loads((R5 / "search_configs.json").read_text(encoding="utf-8"))
    res = {}
    for method, key in (("ppo", "ppo"), ("diffsim", "apg")):
        rows = []
        for i, c in enumerate(configs[key]):
            files = sorted((R5 / "search").glob(f"{method}_s{i:02d}_seed*.npz"))
            rets = []
            for f in files:
                params, pcfg = load_policy(f)
                lat = make_policy_lateral(lon, params, cfg=pcfg)
                tr = jax.jit(jax.vmap(lambda s, lat=lat: rollout(setup.plant, lon, lat, s)))(sc)
                rets.append(float(np.mean(episode_return(tr, sc, ACT_LIMITS))))
            rows.append({"config": i, "override": c, "seeds": len(rets), "returns": rets,
                         "mean": float(np.mean(rets)) if rets else None})  # fmt: skip
            print(method, i, rows[-1]["mean"], flush=True)
        ok = [r for r in rows if r["mean"] is not None and r["seeds"] == 3]
        best = max(ok, key=lambda r: r["mean"])
        res[method] = {"rows": rows, "winner": best, "default_row": rows[0]}
    (R5 / "search_selection.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pol = str(R5 / "policies").replace("\\", "/")
    lines = ["# Round 5 queue B: winners of the search, 10 seeds nominal, 5 seeds DR"]
    for seed in range(10):
        for method in ("ppo", "diffsim"):
            o = json.dumps(res[method]["winner"]["override"], separators=(",", ":"))
            lines.append(
                f"scripts/train_policy.py --method {method} --variant nominal --name tuned --seed {seed} --override {o} --out {pol}"
            )
            if seed < 5:
                lines.append(
                    f"scripts/train_policy.py --method {method} --variant dr --name drtuned --seed {seed} --override {o} --out {pol}"
                )
    (R5 / "jobs" / "train_r5b.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(
        {
            m: (r["winner"]["config"], r["winner"]["mean"], r["default_row"]["mean"])
            for m, r in res.items()
        }
    )


if __name__ == "__main__":
    main()
