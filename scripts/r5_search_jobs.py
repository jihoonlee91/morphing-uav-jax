# ruff: noqa: E501
"""Round 5: random hyperparameter search configs for PPO and APG, and the training queue.

    uv run python scripts/r5_search_jobs.py

PPO: config 0 is the paper's setting, configs 1-15 are random draws (seed 5) over the
learning rate (log-uniform 1e-4..1e-3), epochs, minibatches, clip range, entropy
coefficient, initial log-std, observation / return normalization, GAE lambda, gamma and
value-loss clipping. APG: config 0 is the default, 1-7 random over the learning rate
(log-uniform 3e-4..3e-3), window length and gradient-clip norm. Each config is trained
with seeds 0-2 on the nominal distribution; the winner is chosen by mean return on the
LQR tuning scenarios (``r5_select.py``). Also queued: PPO, APG, PPO+DR, APG+DR trained
with actuator rate limits (100 deg/s, 25 %/s; seeds 0-4) and the smoothness-priced PPO
with initial log-std -2 (seeds 0-2). Writes ``results/tmlr/r5/search_configs.json`` and
``results/tmlr/r5/jobs/train_r5a.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "tmlr" / "r5"


def main() -> None:
    rng = np.random.default_rng(5)
    ppo = [{}]
    for _ in range(15):
        ppo.append({
            "lr": float(10 ** rng.uniform(-4, -3)),
            "epochs": int(rng.choice([4, 6, 10])),
            "minibatches": int(rng.choice([8, 16, 32])),
            "clip": float(rng.choice([0.1, 0.2, 0.3])),
            "ent_coef": float(rng.choice([0.0, 1e-3, 1e-2])),
            "init_log_std": float(rng.choice([-2.0, -1.0, 0.0])),
            "obs_norm": bool(rng.integers(2)),
            "reward_norm": bool(rng.integers(2)),
            "lam": float(rng.choice([0.9, 0.95, 0.98])),
            "gamma": float(rng.choice([0.98, 0.99, 0.995])),
            "vf_clip": bool(rng.integers(2)),
        })  # fmt: skip
    apg = [{}]
    for _ in range(7):
        apg.append({
            "lr": float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
            "window": int(rng.choice([50, 100, 125, 250])),
            "max_grad_norm": float(10 ** rng.uniform(-0.5, 0.5)),
        })  # fmt: skip
    R5.mkdir(parents=True, exist_ok=True)
    (R5 / "jobs").mkdir(exist_ok=True)
    (R5 / "search_configs.json").write_text(
        json.dumps({"ppo": ppo, "apg": apg}, indent=1), encoding="utf-8"
    )
    out = str(R5 / "search").replace("\\", "/")
    pol = str(R5 / "policies").replace("\\", "/")
    lines = [
        "# Round 5 queue A: search (seeds 0-2), rate-limited training, smoothness PPO log-std -2"
    ]
    for seed in range(3):
        for i, c in enumerate(ppo):
            o = json.dumps(c, separators=(",", ":"))
            lines.append(
                f"scripts/train_policy.py --method ppo --variant nominal --name s{i:02d} --seed {seed} --override {o} --out {out}"
            )
        for i, c in enumerate(apg):
            o = json.dumps(c, separators=(",", ":"))
            lines.append(
                f"scripts/train_policy.py --method diffsim --variant nominal --name s{i:02d} --seed {seed} --override {o} --out {out}"
            )
    for seed in range(5):
        for m, v in (("ppo", "nominal"), ("diffsim", "nominal"), ("ppo", "dr"), ("diffsim", "dr")):
            name = "rl" if v == "nominal" else "drrl"
            lines.append(
                f"scripts/train_policy.py --method {m} --variant {v} --name {name} --seed {seed} --rate-limits 100,25 --out {pol}"
            )
    for seed in range(3):
        lines.append(
            f'scripts/train_policy.py --method ppo --variant smooth --name smoothls2 --seed {seed} --override {{"init_log_std":-2.0}} --out {pol}'
        )
    (R5 / "jobs" / "train_r5a.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(len(lines) - 1, "jobs")


if __name__ == "__main__":
    main()
