"""One-call setup shared by the scripts: plant, design grid, and controller schedules."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import NamedTuple

from morphjax import design
from morphjax.aero import build_aero_model
from morphjax.control import Schedule, make_schedule
from morphjax.data import load_aero_tables
from morphjax.dynamics import Plant

MODEL_BASED = ("fixed", "gs", "lpv")
LABELS = {
    "fixed": "Fixed-point LQR",
    "gs": "Gain-scheduled LQR",
    "lpv": "LPV (common Lyapunov)",
    "ppo": "PPO",
    "ppo_dr": "PPO + domain rand.",
    "diffsim": "Diff-sim policy",
    "diffsim_dr": "Diff-sim + domain rand.",
}
# Learned controllers: name -> (training method, variant); files are
# results/policies/<method>_<variant>_seed<k>.npz
LEARNED = {
    "ppo": ("ppo", "nominal"),
    "ppo_dr": ("ppo", "dr"),
    "diffsim": ("diffsim", "nominal"),
    "diffsim_dr": ("diffsim", "dr"),
}


class Setup(NamedTuple):
    plant: Plant
    grid: design.DesignGrid
    schedules: dict[str, Schedule]  # lateral design name -> schedule (shared longitudinal)
    lpv: design.LPVResult
    weights: design.LatWeights
    lat_gains: dict  # lateral design name -> (7, 3, 3, 3, 8) gain table


def load_weights(path: Path | None) -> design.LatWeights:
    """Tuned lateral weights written by ``scripts/tune_lqr.py`` (defaults if absent)."""
    if path is None or not Path(path).exists():
        return design.LatWeights()
    return design.LatWeights(**json.loads(Path(path).read_text(encoding="utf-8"))["weights"])


def build(weights: design.LatWeights | None = None, verbose: bool = False) -> Setup:
    weights = weights or design.LatWeights()
    plant = Plant(build_aero_model(load_aero_tables()))
    grid = design.build_grid(plant)
    k_lon = design.longitudinal_gains(grid)
    with warnings.catch_warnings():
        # Clarabel often ends "optimal_inaccurate" here; the certificate is
        # re-checked explicitly inside lateral_gains_lpv.
        warnings.simplefilter("ignore", UserWarning)
        lpv = design.lateral_gains_lpv(grid, weights)
    k_lat = {
        "fixed": design.lateral_gains_fixed(grid, weights),
        "gs": design.lateral_gains_gs(grid, weights),
        "lpv": lpv.gains,
    }
    schedules = {name: make_schedule(grid, k_lon, k) for name, k in k_lat.items()}
    if verbose:
        print(
            f"design grid: {grid.residual.size} points, "
            f"max trim residual {grid.residual.max():.1e}; "
            f"LPV {lpv.status}, Lyapunov margin {lpv.lyapunov_margin:.3f}; weights {weights}"
        )
    return Setup(plant, grid, schedules, lpv, weights, k_lat)
