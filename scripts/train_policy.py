"""Train a learned lateral controller, by PPO or by gradients through the simulator.

    uv run python scripts/train_policy.py --method ppo --variant nominal --seed 0
    uv run python scripts/train_policy.py --method diffsim --variant dr --seed 0

``--method residual`` trains PPO as a residual (at most 20 % of each actuator
range) on top of the certified LPV law. ``--variant dr`` adds domain
randomization (model error, delay, sensor noise). Writes
``results/policies/<method>_<variant>_seed<k>.npz`` and a JSON learning curve
next to it. All methods produce the same network, so ``evaluate.py`` loads
them the same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morphjax import diffsim, pipeline, rl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", choices=["ppo", "diffsim", "residual"], default="ppo")
    ap.add_argument("--variant", choices=["nominal", "dr"], default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="PPO updates / diff-sim iterations")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "policies")
    args = ap.parse_args()

    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json", verbose=True)
    # The policy shares the longitudinal hold; the lateral gains of this
    # schedule are unused (the policy replaces them).
    lon = setup.schedules["gs"]
    dr = args.variant == "dr"
    if args.method in ("ppo", "residual"):
        # "residual": PPO learns a bounded correction on top of the certified LPV law.
        base = setup.schedules["lpv"] if args.method == "residual" else None
        cfg = rl.PPOConfig(seed=args.seed, domain_randomization=dr)
        if args.iters:
            cfg = cfg._replace(updates=args.iters)
        params, history = rl.train(setup.plant, lon, cfg, base=base)
    else:
        cfg = diffsim.DiffSimConfig(seed=args.seed, domain_randomization=dr)
        if args.iters:
            cfg = cfg._replace(iters=args.iters)
        params, history = diffsim.train(setup.plant, lon, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.variant}_seed{args.seed}"
    rl.save_params(args.out / f"{stem}.npz", params)
    meta = {"method": args.method, "config": cfg._asdict(), "history": history}
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {args.out / stem}.npz")


if __name__ == "__main__":
    main()
