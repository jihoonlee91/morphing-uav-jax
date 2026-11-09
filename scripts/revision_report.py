# ruff: noqa: E501
"""Revision: the numbers behind each paper claim, with seed + scenario bootstrap intervals.

    uv run python scripts/revision_report.py [--dir results/tmlr]

Reads ``metrics.json`` and ``per_scenario.npz`` of an ``evaluate.py`` run and writes
``claims.json``. Intervals are 95 % percentile bootstraps that resample seeds (per
controller) and scenarios (shared by all controllers, so comparisons stay paired).
Ratios are ratios of means. Failed runs (loss of control) have no RMSE and are left
out of RMSE means, as in ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 2000


class Data:
    def __init__(self, d: Path):
        self.m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        self.ps = np.load(d / "per_scenario.npz")
        self.rng = np.random.default_rng(0)

    def arr(self, tag: str, name: str, field: str = "roll_rmse_deg") -> np.ndarray:
        return np.asarray(self.ps[f"{tag}/{name}/{field}"], dtype=float)

    def has(self, tag: str, name: str) -> bool:
        return f"{tag}/{name}/roll_rmse_deg" in self.ps

    def boot(self, fn, arrays: list[np.ndarray]):
        """fn(list of (seeds x scenarios) arrays) -> float; seeds and scenarios resampled."""
        n_sc = arrays[0].shape[1]
        est = fn(arrays)
        out = []
        for _ in range(B):
            sc = self.rng.integers(0, n_sc, n_sc)
            res = [a[self.rng.integers(0, a.shape[0], a.shape[0])][:, sc] for a in arrays]
            out.append(fn(res))
        lo, hi = np.nanquantile(out, [0.025, 0.975])
        return [float(est), float(lo), float(hi)]


def mean(a):
    return float(np.nanmean(a))


def seed_means(a):
    return [float(x) for x in np.nanmean(a, axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "tmlr")
    args = ap.parse_args()
    D = Data(args.dir)
    m = D.m
    t = m["table"]
    names = [k for k in t["nominal"] if k != "label"]
    out: dict = {"controllers": names, "seeds": m["meta"]["seeds_per_controller"]}

    # Per controller: table cells, seed means, activity, LOC, delay sweep.
    ctrl = {}
    delay_rows = m["sweeps"]["delay"]["rows"]
    ail_rows = m["sweeps"]["aileron"]["rows"]
    for c in names:
        row = {}
        for cond in t:
            if cond == "label":
                continue
            s = t[cond][c]
            a = D.arr(f"table/{cond}", c)
            row[cond] = {"rmse": s["roll_rmse_deg"], "loc": 1 - s["success_rate"],
                         "activity": s["cmd_activity"], "seed_means": seed_means(a),
                         "ci": D.boot(lambda x: mean(x[0]), [a])}  # fmt: skip
        row["delay"] = {
            str(20 * r["value"]): {"rmse": r[c]["roll_rmse_deg"], "loc": 1 - r[c]["success_rate"],
                                   "activity": r[c]["cmd_activity"]}
            for r in delay_rows
        }  # fmt: skip
        row["aileron"] = {str(r["value"]): {"rmse": r[c]["roll_rmse_deg"],
                                            "loc": 1 - r[c]["success_rate"]} for r in ail_rows}  # fmt: skip
        row["train_return"] = m["meta"]["training_distribution_return"].get(c)
        row["train_return_per_seed"] = (
            m["meta"].get("training_distribution_return_per_seed", {}).get(c)
        )
        ho = m["meta"].get("heldout_return") or {}
        row["heldout_return"] = {k: v.get(c) for k, v in ho.items()}
        ctrl[c] = row
    out["per_controller"] = ctrl

    def ratio(a_name, b_name, tag="table/nominal"):
        """mean(a) / mean(b), paired by scenario."""
        return D.boot(lambda x: mean(x[0]) / mean(x[1]), [D.arr(tag, a_name), D.arr(tag, b_name)])

    def diff(a_name, b_name, tag="table/nominal"):
        return D.boot(lambda x: mean(x[0]) - mean(x[1]), [D.arr(tag, a_name), D.arr(tag, b_name)])

    def seed_t_interval(a_name, b_name, tag="table/nominal"):
        """t-interval over seeds of (a - b): b's seed means against a's pooled mean."""
        a, b = D.arr(tag, a_name), D.arr(tag, b_name)
        d = (
            [np.nanmean(a) - x for x in np.nanmean(b, axis=1)]
            if a.shape[0] == 1
            else [x - y for x, y in zip(np.nanmean(a, axis=1), np.nanmean(b, axis=1), strict=False)]
        )
        d = np.asarray(d)
        from scipy import stats

        half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
        return [float(d.mean()), float(d.mean() - half), float(d.mean() + half), int(d.size)]

    claims = {}
    have = set(names)
    # F1: morphing speed.
    claims["F1"] = {c: {"frozen": ctrl[c]["nominal"]["rmse"], "fast": ctrl[c]["fast_morph"]["rmse"]}
                    for c in names}  # fmt: skip
    # F2: learned accuracy vs GS LQR without delay.
    claims["F2"] = {f"gs/{c}": ratio("gs", c) for c in ("ppo", "diffsim", "ppo_lat", "diffsim_lat",
                                                         "ppo_clean", "ppo_dr", "diffsim_dr") if c in have}  # fmt: skip
    # F3: one-sample cliff.
    claims["F3"] = {c: {"growth_20ms": ctrl[c]["delay"]["20"]["rmse"] / ctrl[c]["delay"]["0"]["rmse"]
                        if ctrl[c]["delay"]["20"]["rmse"] else None,
                        "loc_80ms": ctrl[c]["delay"]["80"]["loc"], "loc_60ms": ctrl[c]["delay"]["60"]["loc"]}
                    for c in names}  # fmt: skip
    # F4: no-margin LQR vs PPO.
    f4 = {}
    for lqr in ("gs_hg", "gs_hg_wide"):
        if lqr not in have:
            continue
        for c in ("ppo", "ppo_clean", "diffsim"):
            if c in have:
                f4[f"{lqr}-{c}"] = {"diff_seed_scenario_boot": diff(lqr, c),
                                    "seed_t_interval": seed_t_interval(lqr, c),
                                    "activity_ratio": ctrl[c]["fast_morph"]["activity"]
                                    / ctrl[lqr]["fast_morph"]["activity"]}  # fmt: skip
    claims["F4"] = f4
    # F5: DR cost and robustness.
    claims["F5"] = {
        "nominal_cost": {f"{d}/{n}": ratio(d, n) for d, n in (("ppo_dr", "ppo"), ("diffsim_dr", "diffsim"))
                         if d in have and n in have},
        "noise_delay_rank": sorted((ctrl[c]["noise_delay"]["rmse"] or 1e9, c) for c in names),
        "delay80": {c: (ctrl[c]["delay"]["80"]["rmse"], ctrl[c]["delay"]["80"]["loc"]) for c in names},
    }  # fmt: skip
    # F6: LPV cost and payoff.
    f6 = {"lpv/gs": ratio("lpv", "gs")}
    for c in ("lpv_pw", "lpv_nodisk"):
        if c in have:
            f6[f"{c}/gs"] = ratio(c, "gs")
            f6[f"lpv/{c}"] = ratio("lpv", c)
    tag0 = f"sweep/aileron/{[r['value'] for r in ail_rows].index(0.0)}"
    for c in ("gs", "gs_wide", "lpv_pw", "lpv_nodisk", "ppo", "diffsim", "ppo_dr", "diffsim_dr"):
        if c in have and D.has(tag0, c):
            f6[f"aileron0: lpv-{c}"] = diff("lpv", c, tag0)
    claims["F6"] = f6
    # F7: activity.
    claims["F7"] = {c: ctrl[c]["fast_morph"]["activity"] for c in names}
    # F9: residuals vs the LPV -> PPO gap.
    lpv_e, ppo_e = ctrl["lpv"]["nominal"]["rmse"], ctrl["ppo"]["nominal"]["rmse"]
    claims["F9"] = {c: {"rmse": ctrl[c]["nominal"]["rmse"],
                        "gap_closed": (lpv_e - ctrl[c]["nominal"]["rmse"]) / (lpv_e - ppo_e),
                        "delay": {k: v["rmse"] for k, v in ctrl[c]["delay"].items()},
                        "aileron0": ctrl[c]["aileron"]["0.0"]}
                    for c in names if c.startswith("residual")}  # fmt: skip
    out["claims"] = claims
    (args.dir / "claims.json").write_text(
        json.dumps(out, indent=1, default=float), encoding="utf-8"
    )

    # Compact printout.
    print(
        "controller        seeds  nominal  fast   n+40ms  model  ail0-30  act    d20    d40    d60    d80(LOC)"
    )
    for c in names:
        r = ctrl[c]
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
        print(f"{c:18s}{len(r['nominal']['seed_means']):3d}  {f(r['nominal']['rmse'])}  {f(r['fast_morph']['rmse'])}"
              f"  {f(r['noise_delay']['rmse'])}  {f(r['model_error']['rmse'])}  {f(r['aileron_fail']['rmse'])}"
              f"({100 * r['aileron_fail']['loc']:.1f}%)  {f(r['fast_morph']['activity'])}  "
              + "  ".join(f"{f(r['delay'][d]['rmse'])}" for d in ("20", "40", "60"))
              + f"  {f(r['delay']['80']['rmse'])}({100 * r['delay']['80']['loc']:.0f}%)")  # fmt: skip
    print(json.dumps({k: v for k, v in claims.items() if k in ("F2", "F4", "F5", "F6")}, indent=1,
                     default=float)[:6000])  # fmt: skip


if __name__ == "__main__":
    main()
