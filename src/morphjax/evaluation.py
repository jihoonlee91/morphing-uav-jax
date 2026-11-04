"""Batch evaluation: per-scenario metrics and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from morphjax.control import Schedule
from morphjax.dynamics import Cmd, Plant, W
from morphjax.sim import DT, LateralFn, Scenario, Trajectory, rollout


class Metrics(NamedTuple):
    """One value per scenario."""

    failed: np.ndarray  # departed controlled flight
    roll_rmse_deg: np.ndarray  # RMS roll-tracking error (failed runs: nan)
    max_beta_deg: np.ndarray
    max_alt_dev_ft: np.ndarray
    rms_aileron_deg: np.ndarray
    rms_eta_a: np.ndarray
    cmd_activity: np.ndarray  # lateral command travel per second, in units of each limit


def scenario_metrics(traj: Trajectory, scen: Scenario) -> Metrics:
    xw = np.asarray(traj.xw)
    failed = np.asarray(traj.failed).any(axis=-1)
    e_phi = np.rad2deg(xw[..., W.PHI] - np.asarray(scen.phi_cmd))
    rmse = np.sqrt(np.mean(e_phi**2, axis=-1))
    cmd = np.asarray(traj.cmd)
    return Metrics(
        failed=failed,
        roll_rmse_deg=np.where(failed, np.nan, rmse),
        max_beta_deg=np.rad2deg(np.abs(xw[..., W.BETA])).max(axis=-1),
        max_alt_dev_ft=np.abs(xw[..., W.H] - np.asarray(scen.h)[..., None]).max(axis=-1),
        rms_aileron_deg=np.sqrt(np.mean(cmd[..., 1] ** 2, axis=-1)),
        rms_eta_a=np.sqrt(np.mean(xw[..., W.ETA_A] ** 2, axis=-1)),
        cmd_activity=command_activity(cmd),
    )


def command_activity(cmd: np.ndarray) -> np.ndarray:
    """Mean |du/dt| of the lateral commands, each scaled by its position limit.

    1.0 means the commands travel one limit (e.g. 30 deg of aileron) per second;
    it separates smooth control from chattering that RMS tracking error hides.
    """
    limits = np.array([30.0, 30.0, 12.5])  # aileron, rudder (deg), eta_a (%)
    lat = np.clip(cmd[..., [Cmd.DA, Cmd.DR, Cmd.ETA_A]], -limits, limits) / limits
    return np.abs(np.diff(lat, axis=-2)).sum(axis=-1).mean(axis=-1) / DT


def batch_rollout(plant: Plant, lon: Schedule, lateral: LateralFn) -> Callable:
    """Jitted, vmapped rollout over a batch of scenarios."""
    return jax.jit(jax.vmap(lambda s: rollout(plant, lon, lateral, s)))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of ``stat`` over the finite entries of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = stat(x[rng.integers(0, x.size, size=(n, x.size))], axis=1)
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return float(stat(x)), float(lo), float(hi)


def summarize(m: Metrics) -> dict[str, float]:
    rmse, lo, hi = bootstrap_ci(m.roll_rmse_deg)
    succ, slo, shi = bootstrap_ci((~m.failed).astype(float))
    ok = ~m.failed
    return {
        "n": int(m.failed.size),
        "success_rate": succ,
        "success_ci": (slo, shi),
        "roll_rmse_deg": rmse,
        "roll_rmse_ci": (lo, hi),
        "max_beta_deg_p95": float(np.percentile(m.max_beta_deg[ok], 95)) if ok.any() else np.nan,
        "max_alt_dev_ft_p95": float(np.percentile(m.max_alt_dev_ft[ok], 95))
        if ok.any()
        else np.nan,
        "rms_aileron_deg": float(np.mean(m.rms_aileron_deg[ok])) if ok.any() else np.nan,
        "rms_eta_a": float(np.mean(m.rms_eta_a[ok])) if ok.any() else np.nan,
        "cmd_activity": float(np.mean(m.cmd_activity[ok])) if ok.any() else np.nan,
    }


def stack_scenarios(scenarios: list[Scenario]) -> Scenario:
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scenarios)
