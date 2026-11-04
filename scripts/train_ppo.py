"""Train the PPO lateral controller.

    uv run python scripts/train_ppo.py --variant nominal --seed 0
    uv run python scripts/train_ppo.py --variant dr --seed 0

Writes ``results/policies/ppo_<variant>_seed<k>.npz`` and a JSON learning curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morphjax import pipeline
from morphjax.rl import PPOConfig, save_params, train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variant", choices=["nominal", "dr"], default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    defaults = PPOConfig._field_defaults
    ap.add_argument("--updates", type=int, default=defaults["updates"])
    ap.add_argument("--envs", type=int, default=defaults["num_envs"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "policies")
    args = ap.parse_args()

    setup = pipeline.build(verbose=True)
    cfg = PPOConfig(
        num_envs=args.envs,
        updates=args.updates,
        seed=args.seed,
        domain_randomization=args.variant == "dr",
    )
    # The learned policy shares the longitudinal hold; the lateral gains of
    # this schedule are unused (the policy replaces them).
    params, history = train(setup.plant, setup.schedules["gs"], cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"ppo_{args.variant}_seed{args.seed}"
    save_params(args.out / f"{stem}.npz", params)
    meta = {"config": cfg._asdict(), "history": history}
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {args.out / stem}.npz")


if __name__ == "__main__":
    main()
