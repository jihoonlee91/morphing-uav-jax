"""Revision: sample and wall-clock efficiency from deterministic learning curves.

    uv run python scripts/learning_curves_revision.py [--policies results/tmlr/policies]

Every revision run evaluates its mean-action policy on a fixed held-out batch of its
training distribution (``rl.EVAL_SEED``, 64 scenarios of 10 s) during training, and
times the updates separately from JIT compilation (``timing.compile_s``). This script
scores the gain-scheduled LQRs on the same batch and reports, per method and seed, the
environment steps and training wall-clock (compile excluded) to reach several return
thresholds:

* ``lqr_gs``: the 60 ms-margin GS LQR's return on the batch;
* ``lqr_no_margin``: the no-margin LQR's return;
* ``ppo_final``: the mean final return of the PPO seeds;
* ``final99``: 99 % of the run's own final return (i.e. "converged").

Thresholds come from the nominal batch, so only nominal-variant runs are compared
across methods. Also reports the final PPO exploration std (revision and committed
policies). Writes ``learning_curves.json``.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import numpy as np
from tune_lqr import episode_return

from morphjax import pipeline
from morphjax.control import lateral_law
from morphjax.rl import ACT_LIMITS, PPOConfig, eval_scenarios
from morphjax.sim import rollout

ROOT = Path(__file__).resolve().parents[1]


def first_reach(hist: list[dict], thr: float):
    for h in hist:
        if "eval_return" in h and h["eval_return"] >= thr:
            return h["env_steps"], h["wall_s"], h.get("cpu_s")
    return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=Path, default=ROOT / "results" / "tmlr" / "policies")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmlr" / "learning_curves.json")
    args = ap.parse_args()

    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    cfg = PPOConfig()
    scen = eval_scenarios(cfg, cfg.steps)
    lqr = {}
    for name in ("gs", "gs_hg"):
        s = setup.schedules[name]
        traj = jax.jit(
            jax.vmap(lambda sc, s=s: rollout(setup.plant, s, partial(lateral_law, s), sc))
        )(scen)
        lqr[name] = float(np.mean(episode_return(traj, scen, ACT_LIMITS)))

    runs: dict[str, list[dict]] = {}
    for f in sorted(args.policies.glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        stem = f.stem
        group = stem.rsplit("_seed", 1)[0]
        hist = meta["history"]
        evals = [h for h in hist if "eval_return" in h]
        runs.setdefault(group, []).append({
            "seed": int(stem.rsplit("_seed", 1)[1]),
            "timing": meta.get("timing"),
            "final_eval_return": evals[-1]["eval_return"] if evals else None,
            "final_train_return": hist[-1].get("return"),
            "final_std": hist[-1].get("std"),
            "env_steps_total": hist[-1]["env_steps"],
            "hist": hist,
        })  # fmt: skip

    ppo_final = float(np.mean([r["final_eval_return"] for r in runs.get("ppo_nominal", [])]))
    thresholds = {"lqr_gs": lqr["gs"], "lqr_no_margin": lqr["gs_hg"], "ppo_final": ppo_final}
    batch = {"seed": "rl.EVAL_SEED", "n": cfg.eval_envs, "steps": cfg.steps}
    out: dict = {"eval_batch": batch, "lqr_return_on_eval_batch": lqr,
                 "thresholds": thresholds, "groups": {}}  # fmt: skip
    for group, rs in sorted(runs.items()):
        rows = []
        for r in rs:
            row = {k: v for k, v in r.items() if k != "hist"}
            reach = {}
            for tname, thr in [
                *thresholds.items(),
                ("final99", 0.99 * (r["final_eval_return"] or 0)),
            ]:
                steps, wall, cpu = first_reach(r["hist"], thr)
                reach[tname] = {"threshold": thr, "env_steps": steps, "wall_s": wall, "cpu_s": cpu}
            row["reach"] = reach
            row["curve"] = [
                (h["env_steps"], h["eval_return"]) for h in r["hist"] if "eval_return" in h
            ]
            rows.append(row)
        summary = {}
        for tname in [*thresholds, "final99"]:
            st = [x["reach"][tname]["env_steps"] for x in rows]
            wl = [x["reach"][tname]["wall_s"] for x in rows]
            got = [s for s in st if s is not None]
            summary[tname] = {
                "reached": len(got),
                "of": len(rows),
                "median_env_steps": float(np.median(got)) if got else None,
                "range_env_steps": [min(got), max(got)] if got else None,
                "median_wall_s": float(np.median([w for w in wl if w is not None]))
                if got
                else None,
            }
        fin = [x["final_eval_return"] for x in rows]
        out["groups"][group] = {
            "seeds": len(rows),
            "final_eval_return_mean": float(np.mean(fin)),
            "final_eval_return_range": [float(min(fin)), float(max(fin))],
            "compile_s_median": float(np.median([x["timing"]["compile_s"] for x in rows])),
            "train_wall_s_median": float(np.median([x["timing"]["train_wall_s"] for x in rows])),
            "train_cpu_s_total": float(sum(x["timing"]["train_cpu_s"] for x in rows)),
            "final_std": [x["final_std"] for x in rows],
            "to_threshold": summary,
            "runs": rows,
        }
    # The committed (bbafe40) PPO policies: final exploration std at the end of training.
    legacy = {}
    for f in sorted((ROOT / "results" / "policies").glob("ppo_*_seed*.json")):
        h = json.loads(f.read_text(encoding="utf-8"))["history"]
        legacy[f.stem] = {"final_std": h[-1].get("std"), "final_train_return": h[-1]["return"]}
    out["committed_ppo_final_std"] = legacy

    for a, b in (("diffsim_nominal", "ppo_nominal"), ("diffsim_lat", "ppo_lat")):
        if a in out["groups"] and b in out["groups"]:
            ga, gb = out["groups"][a]["to_threshold"], out["groups"][b]["to_threshold"]
            out[f"ratio_{b}_over_{a}"] = {
                t: {
                    "env_steps": (gb[t]["median_env_steps"] / ga[t]["median_env_steps"])
                    if gb[t]["median_env_steps"] and ga[t]["median_env_steps"]
                    else None,
                    "wall_s": (gb[t]["median_wall_s"] / ga[t]["median_wall_s"])
                    if gb[t]["median_wall_s"] and ga[t]["median_wall_s"]
                    else None,
                }
                for t in ga
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1))
    for g, v in out["groups"].items():
        print(g, v["seeds"], round(v["final_eval_return_mean"], 2), v["to_threshold"])


if __name__ == "__main__":
    main()
