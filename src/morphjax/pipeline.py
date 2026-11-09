"""One-call setup shared by the scripts: plant, design grid, and controller schedules."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

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
    "residual_safe": "LPV + residual, projected with error margin",
    "residual_safe2": "LPV + residual, projected with 2x error margin",
    "ppo_lat": "PPO, trained with 20 ms delay",
    "diffsim_lat": "Diff-sim, trained with 20 ms delay",
    "ppo_clean": "PPO, CleanRL-style settings",
    "gs_wide": "Gain-scheduled LQR, wide grid",
    "gs_hg_wide": "Gain-sched. LQR, no margin, wide grid",
    "gs_dr": "Gain-scheduled LQR tuned on DR",
    "lpv_pw": "LPV, pointwise Lyapunov (ablation)",
    "lpv_nodisk": "LPV, no pole disk (ablation)",
    "lpv_pw_nodisk": "LPV, pointwise, no pole disk (ablation)",
    "lpv_tuned": "LPV, tuned multipliers (60 ms rule)",
    "lpv_pd": "LPV, parameter-dependent Lyapunov matrix",
    "ppo_lat40": "PPO, trained with 40 ms delay",
    "diffsim_lat40": "Diff-sim, trained with 40 ms delay",
    "ppo_lat60": "PPO, trained with 60 ms delay",
    "diffsim_lat60": "Diff-sim, trained with 60 ms delay",
    "ppo_ddr": "PPO, delay 0-60 ms randomized",
    "diffsim_ddr": "Diff-sim, delay 0-60 ms randomized",
    "ppo_ddrh": "PPO, delay randomized + command buffer",
    "diffsim_ddrh": "Diff-sim, delay randomized + command buffer",
    "diffsim_lin": "Diff-sim, linear gain-scheduled policy",
    "ppo_smooth": "PPO, action-rate penalty x1000",
    "diffsim_smooth": "Diff-sim, action-rate penalty x1000",
    "ppo_rate100": "PPO, action-rate penalty x100",
    "ppo_ls0": "PPO, initial log-std 0",
    "ppo_ent": "PPO, entropy coefficient 0.01",
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
    # The same trained policies, projected with an interpolation-error margin.
    "residual_safe": ("residual_proj", "dr"),
    "residual_safe2": ("residual_proj", "dr"),
    # Revision: trained with a fixed 20 ms delay, pending command observed.
    "ppo_lat": ("ppo", "lat"),
    "diffsim_lat": ("diffsim", "lat"),
    # Revision: PPO with CleanRL-style settings (rl.CLEANRL), nominal model.
    "ppo_clean": ("ppo_clean", "nominal"),
    # Round 3 (results/tmlr/r3): fixed 40 / 60 ms with pending commands observed,
    # delay-only randomization without / with the (masked) command buffer, a learned
    # linear gain-scheduled policy, smoothness-priced reward, PPO sensitivity.
    "ppo_lat40": ("ppo", "lat40"),
    "diffsim_lat40": ("diffsim", "lat40"),
    "ppo_lat60": ("ppo", "lat60"),
    "diffsim_lat60": ("diffsim", "lat60"),
    "ppo_ddr": ("ppo", "ddr"),
    "diffsim_ddr": ("diffsim", "ddr"),
    "ppo_ddrh": ("ppo", "ddrh"),
    "diffsim_ddrh": ("diffsim", "ddrh"),
    "diffsim_lin": ("diffsim", "lin"),
    "ppo_smooth": ("ppo", "smooth"),
    "diffsim_smooth": ("diffsim", "smooth"),
    "ppo_rate100": ("ppo", "rate100"),
    "ppo_ls0": ("ppo", "ls0"),
    "ppo_ent": ("ppo", "ent"),
}
# Learned controllers that are residuals on top of a model-based law.
RESIDUAL_BASE = {
    "residual_dr": "lpv",
    "residual_proj": "lpv",
    "residual_proj_trained": "lpv",
    "residual_safe": "lpv",
    "residual_safe2": "lpv",
}
# Residuals passed through the Lyapunov projection of morphjax.safety.
PROJECTED = ("residual_proj", "residual_proj_trained", "residual_safe", "residual_safe2")
# Of those, the ones projected with an interpolation-error margin, and its safety
# factor on the worst relative error seen on 1,000 random flight conditions.
MARGINED = {"residual_safe": 1.5, "residual_safe2": 2.0}


class Setup(NamedTuple):
    plant: Plant
    grid: design.DesignGrid
    schedules: dict[str, Schedule]  # lateral design name -> schedule (shared longitudinal)
    lpv: design.LPVResult
    weights: design.LatWeights
    lat_gains: dict  # lateral design name -> (7, 3, 3, 3, 8) gain table
    # Revision: Lyapunov matrices of the LPV ablations ("lpv_nodisk": common X,
    # "lpv_pw": one X per design point), in the LPV design's normalized coordinates.
    lyapunov: dict | None = None


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


# Revision rows (results/tmlr): LQR designs from the bracketed search of
# scripts/tune_lqr_wide.py and the two LPV ablations.
WIDE_LQR = ("gs_wide", "gs_hg_wide", "gs_dr")
LPV_ABLATIONS = ("lpv_pw", "lpv_nodisk", "lpv_pw_nodisk", "lpv_tuned", "lpv_pd")


def build_revision(
    weights_path: Path,
    wide_path: Path | None,
    verbose: bool = False,
    lpv_tuned_path: Path | None = None,
) -> Setup:
    """The evaluated set of the revision.

    The original designs keep the committed weights (``results/lqr_weights.json``),
    so the LPV law, the residual base and the safety filter are unchanged. Added:
    ``gs_wide`` / ``gs_hg_wide`` / ``gs_dr`` (if ``wide_path`` exists) and the LPV
    ablations ``lpv_pw`` (pointwise ``X_i``) and ``lpv_nodisk`` (common ``X``, no
    pole-disk constraint), both with the same weights, decay rate and objective.
    """
    extra = {}
    if Path(weights_path).exists():
        extra["gs_hg"] = load_weights(weights_path, "unconstrained_best")
    if wide_path is not None and Path(wide_path).exists():
        for name in WIDE_LQR:
            extra[name] = load_weights(wide_path, name)
    setup = build(load_weights(weights_path), extra_gs=extra, verbose=verbose)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nodisk = design.lateral_gains_lpv(setup.grid, setup.weights, disk=False)
        pw_gains, pw_x, pw_status = design.lateral_gains_lpv_pointwise(setup.grid, setup.weights)
        # Round 3: the missing ablation cell (pointwise X and no pole disk), and the
        # LPV design with multipliers tuned like the LQR (scripts/r3_tune_lpv.py).
        pwnd_gains, _, _ = design.lateral_gains_lpv_pointwise(setup.grid, setup.weights, disk=False)
        tuned = None
        if (
            lpv_tuned_path is not None
            and Path(lpv_tuned_path).exists()
            and json.loads(Path(lpv_tuned_path).read_text(encoding="utf-8")).get("lpv_tuned")
        ):
            tuned = design.lateral_gains_lpv(setup.grid, load_weights(lpv_tuned_path, "lpv_tuned"))
    k_lon = design.longitudinal_gains(setup.grid)
    extra_lpv = [("lpv_nodisk", nodisk.gains), ("lpv_pw", pw_gains), ("lpv_pw_nodisk", pwnd_gains)]
    if tuned is not None:
        extra_lpv.append(("lpv_tuned", tuned.gains))
    # Round 3: parameter-dependent X(rho) design (design.lateral_gains_lpv_pd), solved
    # once by scripts/r3_lpv_pd.py and stored next to lpv_tuned.json.
    pd_path = Path(lpv_tuned_path).parent / "lpv_pd_gains.npy" if lpv_tuned_path else None
    if pd_path is not None and pd_path.exists():
        extra_lpv.append(("lpv_pd", np.load(pd_path)))
    for name, g in extra_lpv:
        setup.lat_gains[name] = g
        setup.schedules[name] = make_schedule(setup.grid, k_lon, g)
    if verbose:
        print(f"LPV no-disk {nodisk.status} (margin {nodisk.lyapunov_margin:.3f}); "
              f"pointwise statuses {sorted(set(pw_status))}")  # fmt: skip
    return setup._replace(
        lyapunov={
            "lpv": setup.lpv.X,
            "lpv_nodisk": nodisk.X,
            "lpv_pw": pw_x,
            "lpv_nodisk_info": nodisk._replace(gains=None, X=None)._asdict(),
            "lpv_pw_status": pw_status,
        }
    )
