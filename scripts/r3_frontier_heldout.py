# ruff: noqa: E501
"""Round 3: score the frontier's Pareto designs on scenarios they were not selected on.

    uv run python scripts/r3_frontier_heldout.py

The Pareto designs of ``frontier.json`` were picked on the first 64 of the 256 nominal
(frozen-shape) evaluation scenarios (seed 1000) and scored on the same 64. Here they, and
gs / gs_hg / gs_hg_wide for reference, fly the remaining 192 (held out from the
selection) and all 256, reporting RMSE over runs that keep control, loss of control and
actuator activity (``evaluation.command_activity``, as in the tables), plus the
evaluation delay sweep (fast morphing, seed 2020, 128 scenarios) at 20/40/60 ms.
Writes ``results/tmlr/r3/frontier_heldout.json``.
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import jax
import numpy as np
from evaluate import SWEEPS, scenarios
from tune_lqr import GRID

from morphjax import design, pipeline
from morphjax.control import lateral_law, make_schedule
from morphjax.evaluation import scenario_metrics
from morphjax.sim import rollout

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"


def main() -> None:
    fr = json.loads((TM / "r3" / "frontier.json").read_text(encoding="utf-8"))
    wide = json.loads((TM / "lqr_wide.json").read_text(encoding="utf-8"))
    legacy = json.loads((ROOT / "results" / "lqr_weights.json").read_text(encoding="utf-8"))
    designs = {}
    for key in ("pareto_envelope", "pareto_design_points"):
        for p in fr[key]:
            designs.setdefault(tuple(p["weights"][k] for k in GRID), []).append(
                f"{key}@{p['min_delay_steps']}"
            )
    for name, w in (("gs", legacy["weights"]), ("gs_hg", legacy["unconstrained_best"]["weights"]),
                    ("gs_hg_wide", wide["gs_hg_wide"]["weights"])):  # fmt: skip
        designs.setdefault(tuple(w[k] for k in GRID), []).append(name)
    setup = pipeline.build()
    k_lon = design.longitudinal_gains(setup.grid)
    nominal = scenarios(256, seed=1000, morph_rate=0.0)
    j = list(SWEEPS).index("delay")
    sweep = {d: scenarios(128, seed=2000 + 10 * j, morph_rate=math.inf, delay=d) for d in (1, 2, 3)}
    out = []
    for w, labels in designs.items():
        sched = make_schedule(
            setup.grid, k_lon, design.lateral_gains_gs(setup.grid, design.LatWeights(*w))
        )
        run = jax.jit(
            jax.vmap(
                lambda s, sched=sched: rollout(setup.plant, sched, partial(lateral_law, sched), s)
            )
        )

        def summ(m, sl=slice(None)):
            f = m.failed[sl]
            ok = ~f
            return {"rmse": float(np.nanmean(m.roll_rmse_deg[sl])) if ok.any() else None,
                    "loc": float(f.mean()),
                    "activity": float(np.mean(m.cmd_activity[sl][ok])) if ok.any() else None}  # fmt: skip

        m = scenario_metrics(run(nominal), nominal)
        row = {"weights": dict(zip(GRID, w, strict=True)), "labels": labels,
               "selection_64": summ(m, slice(0, 64)), "heldout_192": summ(m, slice(64, None)),
               "all_256": summ(m)}  # fmt: skip
        for d, sc in sweep.items():
            row[f"delay_{20 * d}ms"] = summ(scenario_metrics(run(sc), sc))
        out.append(row)
        print(
            labels,
            row["heldout_192"],
            {k: row[k] for k in row if k.startswith("delay")},
            flush=True,
        )
    (TM / "r3" / "frontier_heldout.json").write_text(
        json.dumps({"description": __doc__, "designs": out}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
