"""Round 3: LPV design with a parameter-dependent Lyapunov matrix X(rho).

    uv run python scripts/r3_lpv_pd.py

``design.lateral_gains_lpv_pd`` with the paper's weights, decay rate and pole disk,
``X`` affine in synthetic airspeed and symmetric span, and parameter-rate bounds
|dV/dt| <= 5 kt/s (the airspeed is held; this covers the hold's transients) and
|d eta_s/dt| <= 80 %/s (the peak rate of the 0.5 Hz morphing actuator on a full 25 %
step, 2 pi 0.5 25 = 79 %/s). Writes ``results/tmlr/r3/lpv_pd_gains.npy`` (evaluated as
``lpv_pd``) and ``lpv_pd.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from morphjax import design, pipeline
from morphjax.sim import DT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tmlr" / "r3"


def main() -> None:
    t0 = time.time()
    setup = pipeline.build(pipeline.load_weights(ROOT / "results" / "lqr_weights.json"))
    res = {}
    for rb in ((5.0, 80.0), (5.0, 40.0)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                g, info = design.lateral_gains_lpv_pd(setup.grid, setup.weights, rb)
            info["linear_delay_margin_ms"] = 1000 * DT * design.max_stable_delay(setup.grid, g, DT)
            if rb == (5.0, 80.0):
                np.save(OUT / "lpv_pd_gains.npy", g)
        except Exception as e:  # solver failure is reported, not hidden
            info = {"failed": str(e)[:200]}
        res[f"rates_{rb[0]:g}kt_s_{rb[1]:g}pct_s"] = info
    res["common_X_cost_bound"] = setup.lpv.cost_bound
    res["wall_seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lpv_pd.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
