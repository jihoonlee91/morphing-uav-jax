"""Figures for the README (matplotlib, static PNG).

Colors follow a fixed categorical order (one hue per controller, never
reassigned), with marker shape as a second channel so identity never depends
on color alone. The numbers behind every panel are in ``results/metrics.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morphjax import pipeline
from morphjax.dynamics import W
from morphjax.sim import DT, rollout

SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
COLORS = {
    "fixed": "#2a78d6",
    "gs": "#eb6834",
    "lpv": "#1baf7a",
    "ppo": "#eda100",
    "ppo_dr": "#e87ba4",
    "diffsim": "#008300",
    "diffsim_dr": "#6250d6",
    "gs_hg": "#e34948",
    # Eight categorical slots for nine controllers: the fixed-point LQR is left
    # out of the line charts (it tracks the gain-scheduled LQR everywhere) and
    # its blue goes to the residual policy there.
    "residual_dr": "#2a78d6",
}
PLOTTED = ("gs", "gs_hg", "lpv", "ppo", "ppo_dr", "diffsim", "diffsim_dr", "residual_dr")
MARKERS = {
    "fixed": "o",
    "gs": "s",
    "lpv": "D",
    "ppo": "^",
    "ppo_dr": "v",
    "diffsim": "P",
    "diffsim_dr": "X",
    "gs_hg": "h",
    "residual_dr": "*",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5,
})  # fmt: skip


def _legend(fig, names):
    handles = [
        plt.Line2D([], [], color=COLORS[n], marker=MARKERS[n], label=pipeline.LABELS[n])
        for n in names
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=min(len(names), 4), bbox_to_anchor=(0.5, 1.0)
    )


def plain_log_ticks(axis) -> None:
    """Log-scale axis labelled with plain numbers (0.3, 1, 10) instead of 3x10^-1."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    # 1-2-5 steps so that axes spanning less than a decade still get labels.
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _save(fig, path: Path) -> None:
    """PNG for the README, vector PDF for papers."""
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


COLUMN_TITLES = {
    "morph_rate": "Morphing rate",
    "aileron": "Aileron failure",
    "delay": "Command delay",
}


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def stress_tests(sweeps: dict, names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex="col")
    for col, (key, sweep) in enumerate(sweeps.items()):
        axes[0, col].set_title(COLUMN_TITLES.get(key, key), loc="left")
        axes[0, col].set_yscale("log")
        plain_log_ticks(axes[0, col].yaxis)
        # metrics.json stores the "step" morphing rate (infinity) as null.
        values = [math.inf if r["value"] is None else r["value"] for r in sweep["rows"]]
        if key == "morph_rate":
            x = np.arange(len(values))
            ticks = [
                "0" if v == 0 else ("step" if not math.isfinite(v) else f"{v:g}") for v in values
            ]
        elif key == "delay":
            x = np.asarray(values) * DT * 1000
            ticks = [f"{v:g}" for v in x]
        else:
            x = np.asarray(values) * 100
            ticks = [f"{v:g}" for v in x]
        for name in names:
            # metrics.json stores NaN (no successful run) as null.
            rmse = [_num(r[name]["roll_rmse_deg"]) for r in sweep["rows"]]
            lo = [_num(r[name]["roll_rmse_ci"][0]) for r in sweep["rows"]]
            hi = [_num(r[name]["roll_rmse_ci"][1]) for r in sweep["rows"]]
            succ = [100 * r[name]["success_rate"] for r in sweep["rows"]]
            kw = dict(color=COLORS[name], marker=MARKERS[name])
            axes[0, col].plot(x, rmse, **kw)
            axes[0, col].fill_between(x, lo, hi, color=COLORS[name], alpha=0.12, linewidth=0)
            axes[1, col].plot(x, succ, **kw)
        axes[1, col].set_xticks(x, ticks)
        axes[1, col].set_xlabel(sweep["label"])
        if key == "aileron":
            axes[1, col].invert_xaxis()
        axes[1, col].set_ylim(-5, 105)
    axes[0, 0].set_ylabel("Roll RMS error (deg, log)")
    axes[1, 0].set_ylabel("Runs in control (%)")
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, path)


def example_rollout(setup, ctrls: dict, scen_batch, path: Path) -> None:
    """Time history of one degraded-aileron scenario for every controller."""
    eff = np.asarray(scen_batch.aileron_effect)
    i = int(np.argmin(np.abs(eff - 0.1)))
    scen = jax.tree_util.tree_map(lambda x: x[i], scen_batch)
    t = np.arange(len(scen.phi_cmd)) * DT
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.4), sharex=True)
    axes[0].plot(t, np.rad2deg(scen.phi_cmd), color=INK2, linestyle="--", linewidth=1.2,
                 label="reference")  # fmt: skip
    for name, entries in ctrls.items():
        lon, lat = entries[0]
        tr = jax.jit(lambda s, lon=lon, lat=lat: rollout(setup.plant, lon, lat, s))(scen)
        xw = np.asarray(tr.xw)
        failed = np.asarray(tr.failed)
        end = int(np.argmax(failed)) if failed.any() else len(t)
        kw = dict(color=COLORS[name], linewidth=1.4)
        axes[0].plot(t[:end], np.rad2deg(xw[:end, W.PHI]), **kw)
        axes[1].plot(t[:end], xw[:end, W.ETA_A], **kw)
        axes[2].plot(t[:end], np.rad2deg(xw[:end, W.BETA]), **kw)
        if end < len(t):
            axes[0].plot(t[end - 1], np.rad2deg(xw[end - 1, W.PHI]), "x", color=COLORS[name],
                         markersize=8, mew=2)  # fmt: skip
    eta_s = np.asarray(jnp.asarray(scen.eta_s_cmd))
    axes[1].plot(t, eta_s, color=MUTED, linewidth=1.0, linestyle=":")
    axes[1].text(0.01, 0.04, "dotted: symmetric-span command η_s", transform=axes[1].transAxes,
                 color=MUTED, fontsize=8)  # fmt: skip
    axes[0].set_ylabel("Roll angle (deg)")
    axes[1].set_ylabel("Asymmetric span η_a (%)")
    axes[2].set_ylabel("Sideslip (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(
        f"Aileron at {100 * float(scen.aileron_effect):.0f} % effectiveness, "
        f"fast symmetric morphing, {float(scen.v_kt):.0f} kt, {float(scen.h):,.0f} ft",
        loc="left",
    )
    _legend(fig, list(ctrls))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, path)


def learning_curves(policy_dir: Path, path: Path, lqr_return: float | None = None) -> None:
    """Training return against simulated environment steps (log scale)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = []
    for name, (method, variant) in pipeline.LEARNED.items():
        runs = sorted(policy_dir.glob(f"{method}_{variant}_seed*.json"))
        if not runs:
            continue
        names.append(name)
        curves, steps = [], np.array([])
        for f in runs:
            hist = json.loads(f.read_text(encoding="utf-8"))["history"]
            steps = np.array([h["env_steps"] for h in hist]) / 1e6
            ret = np.array([h["return"] for h in hist])
            curves.append(ret)
            ax.plot(steps, ret, color=COLORS[name], alpha=0.25, linewidth=0.8)
        n = min(len(c) for c in curves)
        ax.plot(steps[:n], np.mean([c[:n] for c in curves], axis=0), color=COLORS[name])
    if lqr_return is not None:
        ax.axhline(lqr_return, color=COLORS["gs"], linestyle="--", linewidth=1.2)
        ax.annotate(
            f"tuned gain-scheduled LQR ({lqr_return:.0f})", (0.035, lqr_return),
            xytext=(2, -11), textcoords="offset points", color=INK2, fontsize=8,
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(440, 500)
    ax.set_xlabel("Simulated environment steps (millions, log scale)")
    ax.set_ylabel("Episode return (max 500)")
    ax.set_title(
        "Training return: thin lines are seeds, thick lines the mean (PPO includes exploration)",
        loc="left",
    )
    if names:
        _legend(fig, names)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, path)
