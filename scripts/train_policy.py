"""Train a learned lateral controller, by PPO or by gradients through the simulator.

    uv run python scripts/train_policy.py --method ppo --variant nominal --seed 0
    uv run python scripts/train_policy.py --method diffsim --variant dr --seed 0

``--method ppo_clean`` is PPO with CleanRL-style defaults (observation and reward
normalization, separate policy/value gradient clipping, initial log-std 0; see
``rl.CLEANRL``). ``--method residual`` trains PPO as a residual (at most 20 % of each actuator
range) on top of the certified LPV law. ``--variant dr`` adds domain
randomization (model error, delay, sensor noise). ``--variant lat`` trains on
the nominal model with a fixed one-sample (20 ms) command delay and the pending
command appended to the observation, so the delayed problem stays Markov. Writes
``results/policies/<method>_<variant>_seed<k>.npz`` and a JSON learning curve
next to it. All methods produce the same network, so ``evaluate.py`` loads
them the same way.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from morphjax import diffsim, pipeline, rl, safety

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--method",
        choices=["ppo", "ppo_clean", "diffsim", "residual", "residual_proj"],
        default="ppo",
    )
    ap.add_argument("--variant", choices=["nominal", "dr", "lat"], default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="PPO updates / diff-sim iterations")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "policies")
    args = ap.parse_args()

    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json", verbose=True)
    # The policy shares the longitudinal hold; the lateral gains of this
    # schedule are unused (the policy replaces them).
    lon = setup.schedules["gs"]
    dr = args.variant == "dr"
    # "lat": a fixed 20 ms delay in every training scenario, pending command observed.
    lat = {"delay": 1, "history": 1} if args.variant == "lat" else {}
    if args.method in ("ppo", "ppo_clean", "residual", "residual_proj"):
        # "residual": PPO learns a bounded correction on top of the certified LPV law;
        # "residual_proj" also projects it onto the LPV certificate inside the loop.
        base = setup.schedules["lpv"] if args.method.startswith("residual") else None
        filt = None
        if args.method == "residual_proj":
            filt = partial(safety.project, safety.make_safety_filter(setup))
        # "ppo_clean": the same PPO with CleanRL-style defaults (rl.CLEANRL).
        clean = rl.CLEANRL if args.method == "ppo_clean" else {}
        cfg = rl.PPOConfig(seed=args.seed, domain_randomization=dr, **lat, **clean)
        if args.iters:
            cfg = cfg._replace(updates=args.iters)
        params, history, timing = rl.train(setup.plant, lon, cfg, base=base, residual_filter=filt)
    else:
        cfg = diffsim.DiffSimConfig(seed=args.seed, domain_randomization=dr, **lat)
        if args.iters:
            cfg = cfg._replace(iters=args.iters)
        params, history, timing = diffsim.train(setup.plant, lon, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.variant}_seed{args.seed}"
    rl.save_params(args.out / f"{stem}.npz", params)
    meta = {"method": args.method, "config": cfg._asdict(), "timing": timing, "history": history}
    (args.out / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {args.out / stem}.npz")


if __name__ == "__main__":
    main()
