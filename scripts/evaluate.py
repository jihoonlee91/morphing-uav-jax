"""Evaluate every controller on the same scenarios; write the results table and figures.

    uv run python scripts/evaluate.py            # full run (~6 min on a laptop CPU)
    uv run python scripts/evaluate.py --quick    # smoke test used in CI

Outputs (``results/``): ``metrics.json``, ``results.md``, ``figures/*.png``.
Every controller sees identical scenarios (common random numbers), so
differences between rows are paired comparisons, not sampling noise.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from morphjax import design, pipeline
from morphjax.control import lateral_law
from morphjax.evaluation import Metrics, batch_rollout, scenario_metrics, summarize
from morphjax.rl import PPOConfig, load_params, make_policy_lateral, training_scenario
from morphjax.scenarios import sample_scenario
from morphjax.sim import DT

ROOT = Path(__file__).resolve().parents[1]
STEPS = 750  # 15 s

# Table conditions. "Fast morphing" = symmetric-span steps between far-apart
# targets every 2.5 s, limited only by the 0.5 Hz morphing actuator.
CONDITIONS = {
    "nominal": ("Frozen shape", dict(morph_rate=0.0)),
    "fast_morph": ("Fast morphing", dict(morph_rate=math.inf)),
    "noise_delay": ("+ noise, 40 ms delay", dict(morph_rate=math.inf, delay=2, noise=1.0)),
    "model_error": (
        "+ model error",
        dict(morph_rate=math.inf, morph_aero=(0.6, 1.4), inertia=(0.8, 1.2)),
    ),
    "aileron_fail": ("+ aileron at 0-30 %", dict(morph_rate=math.inf, aileron_effect=(0.0, 0.3))),
}
SWEEPS = {
    "morph_rate": ("Symmetric-morphing command rate (%/s)",
                   [0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, math.inf],
                   lambda v: dict(morph_rate=v)),
    "aileron": ("Aileron effectiveness (fast morphing)",
                [1.0, 0.75, 0.5, 0.3, 0.15, 0.0],
                lambda v: dict(morph_rate=math.inf, aileron_effect=(v, v))),
    "delay": ("Command transport delay, ms (fast morphing)",
              [0, 1, 2, 3, 4],
              lambda v: dict(morph_rate=math.inf, delay=int(v))),
}  # fmt: skip


def scenarios(n: int, seed: int, **kw):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    rate = kw.pop("morph_rate")
    return jax.vmap(lambda k: sample_scenario(k, STEPS, rate, **kw))(keys)


def controllers(setup, policy_dir: Path, max_seeds: int | None = None):
    """name -> list of (longitudinal schedule, lateral fn); several entries = seeds."""
    out = {}
    for name in pipeline.MODEL_BASED:
        s = setup.schedules[name]
        out[name] = [(s, lambda xw, pc, ig, s=s: lateral_law(s, xw, pc, ig))]
    lon = setup.schedules["gs"]
    for variant, name in (("nominal", "ppo"), ("dr", "ppo_dr")):
        files = sorted(policy_dir.glob(f"ppo_{variant}_seed*.npz"))[:max_seeds]
        if files:
            out[name] = [
                (lon, make_policy_lateral(lon, load_params(f, PPOConfig()))) for f in files
            ]
    return out


def pooled(metrics: list[Metrics]) -> Metrics:
    return Metrics(*[np.concatenate(x) for x in zip(*metrics, strict=True)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny batches (CI smoke test)")
    ap.add_argument("--n", type=int, default=256, help="scenarios per table condition")
    ap.add_argument("--n-sweep", type=int, default=128, help="scenarios per sweep point")
    ap.add_argument("--policies", type=Path, default=ROOT / "results" / "policies")
    ap.add_argument("--weights", type=Path, default=ROOT / "results" / "lqr_weights.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if args.quick:
        args.n, args.n_sweep = 4, 2

    t0 = time.time()
    setup = pipeline.build(pipeline.load_weights(args.weights), verbose=True)
    ctrls = controllers(setup, args.policies, max_seeds=1 if args.quick else None)
    runners = {
        name: [batch_rollout(setup.plant, lon, lat) for lon, lat in entries]
        for name, entries in ctrls.items()
    }
    print(f"controllers: {', '.join(f'{k} x{len(v)}' for k, v in runners.items())}")

    def run(scen) -> dict[str, Metrics]:
        return {
            name: pooled([scenario_metrics(f(scen), scen) for f in fns])
            for name, fns in runners.items()
        }

    table: dict[str, dict] = {}
    raw_example = None
    for i, (cond, (label, kw)) in enumerate(CONDITIONS.items()):
        scen = scenarios(args.n, seed=1000 + i, **dict(kw))
        res = run(scen)
        table[cond] = {"label": label, **{k: summarize(m) for k, m in res.items()}}
        print(f"[{time.time() - t0:6.0f}s] {label:24s} " + "  ".join(
            f"{k}: {v['roll_rmse_deg']:.2f} deg / {100 * v['success_rate']:.0f}%"
            for k, v in table[cond].items() if k != "label"
        ))  # fmt: skip
        if cond == "aileron_fail":
            raw_example = scen

    sweeps: dict[str, dict] = {}
    for j, (sweep, (label, values, make_kw)) in enumerate(SWEEPS.items()):
        rows = []
        for v in values:
            scen = scenarios(args.n_sweep, seed=2000 + 10 * j, **make_kw(v))
            rows.append({"value": v, **{k: summarize(m) for k, m in run(scen).items()}})
        sweeps[sweep] = {"label": label, "rows": rows}
        print(f"[{time.time() - t0:6.0f}s] sweep {sweep} done")

    # Mean PPO-reward return on the LQR tuning scenarios (PPO training distribution),
    # scored exactly as scripts/tune_lqr.py scores LQR candidates.
    train_return = {}
    if not args.quick:
        from tune_lqr import SEED as TUNE_SEED
        from tune_lqr import N as TUNE_N
        from tune_lqr import episode_return

        cfg = PPOConfig()
        keys = jax.random.split(jax.random.PRNGKey(TUNE_SEED), TUNE_N)
        tscen = jax.vmap(lambda k: training_scenario(k, cfg))(keys)
        for name, fns in runners.items():
            train_return[name] = float(np.mean([episode_return(f(tscen), tscen) for f in fns]))
        print("training-distribution return:", {k: round(v, 2) for k, v in train_return.items()})

    meta = {
        "training_distribution_return": train_return,
        "steps": STEPS,
        "dt": DT,
        "n_per_condition": args.n,
        "n_per_sweep_point": args.n_sweep,
        "seeds_per_controller": {k: len(v) for k, v in runners.items()},
        "lqr_weights": setup.weights._asdict(),
        "linear_delay_margin_ms": {
            name: 1000 * DT * design.max_stable_delay(setup.grid, setup.lat_gains[name], DT)
            for name in pipeline.MODEL_BASED
        },
        "lpv": {"status": setup.lpv.status, "lyapunov_margin": setup.lpv.lyapunov_margin,
                "cost_bound": setup.lpv.cost_bound},
        "max_trim_residual": float(setup.grid.residual.max()),
        "wall_seconds": time.time() - t0,
    }  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "table": table, "sweeps": sweeps}
    (args.out / "metrics.json").write_text(
        json.dumps(_finite(payload), indent=1, default=_json, allow_nan=False), encoding="utf-8"
    )
    (args.out / "results.md").write_text(markdown_table(table), encoding="utf-8")
    print(markdown_table(table))

    if not args.quick:
        from figures import example_rollout, learning_curves, stress_tests

        fig_dir = args.out / "figures"
        fig_dir.mkdir(exist_ok=True)
        stress_tests(sweeps, list(runners), fig_dir / "stress_tests.png")
        example_rollout(setup, ctrls, raw_example, fig_dir / "example_rollout.png")
        learning_curves(args.policies, fig_dir / "learning_curve.png")
    print(f"done in {time.time() - t0:.0f}s")


def _json(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def _finite(x):
    """Replace NaN/inf by None so metrics.json is strict JSON."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_finite(v) for v in x]
    if isinstance(x, float | np.floating) and not math.isfinite(float(x)):
        return None
    return x


def markdown_table(table: dict) -> str:
    names = [k for k in next(iter(table.values())) if k != "label"]
    head = "| Controller | " + " | ".join(v["label"] for v in table.values()) + " |"
    sep = "|---|" + "---:|" * len(table)
    lines = [head, sep]
    for name in names:
        cells = []
        for v in table.values():
            s = v[name]
            rmse = "—" if not math.isfinite(s["roll_rmse_deg"]) else f"{s['roll_rmse_deg']:.2f}°"
            cells.append(f"{rmse} · {100 * s['success_rate']:.1f}%")
        lines.append(f"| {pipeline.LABELS[name]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Cells: roll-tracking RMS error over successful runs · share of runs without loss of "
        "control. 95 % bootstrap intervals are in `metrics.json`."
    )
    if "fast_morph" in table:
        fm = table["fast_morph"]
        lines += [
            "",
            "| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |",
            "|---|---:|---:|---:|",
        ]
        for name in names:
            s = fm[name]
            lines.append(
                f"| {pipeline.LABELS[name]} | {s['max_beta_deg_p95']:.2f}° | "
                f"{s['cmd_activity']:.2f} /s | {s['rms_eta_a']:.2f} % |"
            )
        lines += [
            "",
            "Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, "
            "each divided by its full range (higher = busier actuators).",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
