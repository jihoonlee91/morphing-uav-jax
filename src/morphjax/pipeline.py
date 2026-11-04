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
    "gs_hg": "Gain-sched. LQR, no delay margin",
    "lpv": "LPV (common Lyapunov)",
    "ppo": "PPO",
    "ppo_dr": "PPO + domain rand.",
    "diffsim": "Diff-sim policy",
    "diffsim_dr": "Diff-sim + domain rand.",
    "residual_dr": "LPV + residual PPO (DR)",
    "residual_proj": "LPV + residual, Lyapunov-projected",
    "residual_proj_trained": "LPV + residual, projected in training",
}
# Learned controllers: name -> (training method, variant); files are
# results/policies/<method>_<variant>_seed<k>.npz
LEARNED = {
    "ppo": ("ppo", "nominal"),
    "ppo_dr": ("ppo", "dr"),
    "diffsim": ("diffsim", "nominal"),
    "diffsim_dr": ("diffsim", "dr"),
    "residual_dr": ("residual", "dr"),
    # Same trained residuals, projected at run time onto the LPV certificate.
    "residual_proj": ("residual", "dr"),
    # Residuals trained with the projection inside the PPO loop.
    "residual_proj_trained": ("residual_proj", "dr"),
}
# Learned controllers that are residuals on top of a model-based law.
RESIDUAL_BASE = {"residual_dr": "lpv", "residual_proj": "lpv", "residual_proj_trained": "lpv"}
# Residuals passed through the Lyapunov projection of morphjax.safety.
PROJECTED = ("residual_proj", "residual_proj_trained")


class Setup(NamedTuple):
    plant: Plant
    grid: design.DesignGrid
    schedules: dict[str, Schedule]  # lateral design name -> schedule (shared longitudinal)
    lpv: design.LPVResult
    weights: design.LatWeights
    lat_gains: dict  # lateral design name -> (7, 3, 3, 3, 8) gain table


def load_weights(path: Path | None, key: str = "weights") -> design.LatWeights:
    """Lateral weights written by ``scripts/tune_lqr.py`` (defaults if absent).

    ``key="weights"`` is the tuned design that meets the 60 ms delay-margin
    requirement; ``key="unconstrained_best"`` is the best one without it.
    """
    if path is None or not Path(path).exists():
        return design.LatWeights()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = data[key]
    return design.LatWeights(**entry.get("weights", entry))


def build(
    weights: design.LatWeights | None = None,
    extra_gs: dict[str, design.LatWeights] | None = None,
    verbose: bool = False,
) -> Setup:
    """Trim, linearize, and synthesize every model-based design.

    ``extra_gs`` adds further gain-scheduled LQR designs under the given names
    (e.g. ``{"gs_hg": <weights without the delay requirement>}``).
    """
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
    for name, w in (extra_gs or {}).items():
        k_lat[name] = design.lateral_gains_gs(grid, w)
    schedules = {name: make_schedule(grid, k_lon, k) for name, k in k_lat.items()}
    if verbose:
        print(
            f"design grid: {grid.residual.size} points, "
            f"max trim residual {grid.residual.max():.1e}; "
            f"LPV {lpv.status}, Lyapunov margin {lpv.lyapunov_margin:.3f}; weights {weights}"
        )
    return Setup(plant, grid, schedules, lpv, weights, k_lat)


def build_for_evaluation(weights_path: Path, verbose: bool = False) -> Setup:
    """The evaluated set: tuned designs plus the no-delay-margin LQR as ``gs_hg``."""
    extra = {}
    if Path(weights_path).exists():
        extra["gs_hg"] = load_weights(weights_path, "unconstrained_best")
    return build(load_weights(weights_path), extra_gs=extra, verbose=verbose)
