"""Round 3: per-step traces of the appendix example rollout, for the paper to draw.

    uv run python scripts/r3_rollout_traces.py [--policies DIR ...]

Same scenario as ``example_rollout_tmlr.py`` / ``figures.example_rollout``: from the
256-scenario "aileron at 0-30 %" batch of ``evaluate.py`` (seed 1004), the scenario
whose aileron effectiveness is closest to 10 %; one seed (the first file) per learned
controller. Writes ``results/tmlr/r3/example_rollout_traces.npz`` with ``t`` (s),
``phi_cmd`` (deg), ``eta_s_cmd`` (%), the scenario scalars, and per controller
``<name>/phi`` (deg), ``<name>/eta_a`` (%), ``<name>/beta`` (deg),
``<name>/failure_index`` (first failed step, -1 if none).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from evaluate import CONDITIONS, controllers, scenarios

from morphjax import pipeline
from morphjax.dynamics import W
from morphjax.sim import DT, rollout

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=Path, nargs="+", default=[TM / "policies"])
    ap.add_argument("--out", type=Path, default=TM / "r3" / "example_rollout_traces.npz")
    args = ap.parse_args()
    setup = pipeline.build_revision(
        ROOT / "results" / "lqr_weights.json",
        TM / "lqr_wide.json",
        lpv_tuned_path=TM / "r3" / "lpv_tuned.json",
    )
    ctrls = {}
    for d in args.policies:
        ctrls.update(controllers(setup, d, max_seeds=1))
    i = list(CONDITIONS).index("aileron_fail")
    _, kw = CONDITIONS["aileron_fail"]
    batch = scenarios(256, seed=1000 + i, **dict(kw))
    eff = np.asarray(batch.aileron_effect)
    k = int(np.argmin(np.abs(eff - 0.1)))
    scen = jax.tree_util.tree_map(lambda x: x[k], batch)
    t = np.arange(len(scen.phi_cmd)) * DT
    out = {
        "t": t,
        "phi_cmd": np.rad2deg(np.asarray(scen.phi_cmd)),
        "eta_s_cmd": np.asarray(scen.eta_s_cmd),
        "scenario_index": np.array(k),
        "aileron_effect": np.asarray(scen.aileron_effect),
        "v_kt": np.asarray(scen.v_kt),
        "h_ft": np.asarray(scen.h),
    }
    for name, (lon, lats, _, _) in ctrls.items():
        tr = jax.jit(lambda s, lon=lon, lat=lats[0]: rollout(setup.plant, lon, lat, s))(scen)
        xw, failed = np.asarray(tr.xw), np.asarray(tr.failed)
        out[f"{name}/phi"] = np.rad2deg(xw[:, W.PHI])
        out[f"{name}/eta_a"] = xw[:, W.ETA_A]
        out[f"{name}/beta"] = np.rad2deg(xw[:, W.BETA])
        out[f"{name}/failure_index"] = np.array(int(np.argmax(failed)) if failed.any() else -1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"scenario {k}: aileron {float(scen.aileron_effect):.3f}; {len(ctrls)} controllers")


if __name__ == "__main__":
    main()
