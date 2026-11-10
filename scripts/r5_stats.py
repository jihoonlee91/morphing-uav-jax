# ruff: noqa: E501
"""Round 5: statistics the reviewers asked for, from the merged evaluation.

    uv run python scripts/r5_stats.py [--dir results/tmlr/r5]

1. Seed-level 95 % t intervals of the nominal RMSE (and of the noise + 40 ms RMSE) for every
   learned controller: one mean per seed over the shared scenarios, t on n_seeds - 1 dof.
2. Loss-of-control rates per table condition and delay with 95 % cluster-bootstrap
   intervals (resampling seeds; scenarios for single-design controllers), 2,000 draws.
3. TOST equivalence of the no-margin LQR and PPO nominal RMSE: per-seed paired
   differences (same scenarios), equivalent at margin delta if the 90 % t interval lies
   inside +-delta; delta = 0.05 deg (one sixth of PPO's RMSE) primary, 0.03 deg also given.
4. Linear audit vs simulation, per delay (primary) with 95 % bootstrap intervals that
   resample controllers (designs), and a delay-only comparator (Spearman of the delay
   itself with the degradation rank, pooled), from ``r3/audit.json`` without duplicates.
5. Sample efficiency without the "PPO final return" threshold: Kaplan-Meier median env
   steps, training wall-clock and process CPU seconds (compile excluded) to the GS-LQR
   and no-margin-LQR returns and to absolute returns 490 and 494, from the fine curves
   (evaluation after every update = every 32,000 steps, the batch size; denser is not
   possible without changing the batch).
Writes ``stats.json`` in ``--dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats
from r3_audit_facts import DUPLICATES
from r3_learning_curves import first, km_median

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
B = 2000


def t_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return [float(x.mean()) if x.size else None, None, None, int(x.size)]
    h = scipy.stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h), int(x.size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=TM / "r5")
    args = ap.parse_args()
    m = json.loads((args.dir / "metrics.json").read_text(encoding="utf-8"))
    ps = np.load(args.dir / "per_scenario.npz")
    rng = np.random.default_rng(0)
    names = [k for k in m["table"]["nominal"] if k != "label"]
    seeds = m["meta"]["seeds_per_controller"]
    out: dict = {"seed_t_intervals": {}, "loc_cluster_bootstrap": {}}
    for n in names:
        if seeds.get(n, 1) < 2:
            continue
        row = {}
        for cond in ("nominal", "noise_delay", "aileron_fail"):
            a = np.asarray(ps[f"table/{cond}/{n}/roll_rmse_deg"], float)
            row[cond] = t_ci(np.nanmean(a, axis=1))
        out["seed_t_intervals"][n] = row
    tags = [(c, f"table/{c}") for c in m["table"]] + [
        (f"delay_{20 * r['value']}ms", f"sweep/delay/{i}")
        for i, r in enumerate(m["sweeps"]["delay"]["rows"])
    ]
    for n in names:
        row = {}
        for label, tag in tags:
            key = f"{tag}/{n}/failed"
            if key not in ps:
                continue
            f = np.asarray(ps[key], float)  # seeds x scenarios
            est = float(f.mean())
            boots = []
            for _ in range(B):
                if f.shape[0] > 1:
                    boots.append(f[rng.integers(0, f.shape[0], f.shape[0])].mean())
                else:
                    boots.append(f[:, rng.integers(0, f.shape[1], f.shape[1])].mean())
            row[label] = [est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        out["loc_cluster_bootstrap"][n] = row
    # TOST
    tost = {}
    a = np.asarray(ps["table/nominal/gs_hg/roll_rmse_deg"], float)[0]
    for other in ("ppo", "ppo_clean", "ppo_tuned"):
        if f"table/nominal/{other}/roll_rmse_deg" not in ps:
            continue
        b = np.asarray(ps[f"table/nominal/{other}/roll_rmse_deg"], float)
        d = np.array([np.nanmean(a) - np.nanmean(x) for x in b])
        lo, hi = t_ci(d, 0.90)[1:3]
        tost[f"gs_hg-{other}"] = {"mean_diff_deg": float(d.mean()), "ci90": [lo, hi], "n_seeds": int(d.size),
                                  **{f"equivalent_at_{dl}": bool(lo > -dl and hi < dl) for dl in (0.03, 0.05)}}  # fmt: skip
    out["tost"] = tost
    # audit per delay with design bootstrap
    audit = json.loads((TM / "r3" / "audit.json").read_text(encoding="utf-8"))
    pairs = [p for p in audit["pairs"]["pairs"] if p["controller"] not in DUPLICATES]

    def ranks(sub):
        key = [
            (
                p["loc_share"] > 0,
                p["loc_share"],
                p["rmse_growth"] if p["rmse_growth"] is not None else np.inf,
            )
            for p in sub
        ]
        order = sorted(set(key))
        return scipy.stats.rankdata([order.index(k) for k in key])

    def rho(sub, field="share_x0"):
        if len(sub) < 3:
            return np.nan
        x = np.array([p[field] for p in sub])
        r = ranks(sub)
        if np.all(x == x[0]) or np.all(r == r[0]):
            return np.nan
        return scipy.stats.spearmanr(x, -r).statistic

    per = {}
    ctrls = sorted({p["controller"] for p in pairs})
    for d in (20, 40, 60, 80):
        sub = [p for p in pairs if p["delay_ms"] == d]
        boots = []
        for _ in range(B):
            pick = rng.choice(ctrls, len(ctrls))
            bs = [p for c in pick for p in sub if p["controller"] == c]
            boots.append(rho(bs))
        per[str(d)] = {
            "rho": float(rho(sub)),
            "ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
            "n": len(sub),
        }
    pooled_boot = []
    for _ in range(B):
        pick = rng.choice(ctrls, len(ctrls))
        pooled_boot.append(rho([p for c in pick for p in pairs if p["controller"] == c]))
    delay_only = scipy.stats.spearmanr([-p["delay_ms"] for p in pairs], -ranks(pairs)).statistic
    out["audit"] = {"per_delay": per, "pooled": {"rho": float(rho(pairs)), "ci95": [float(np.nanquantile(pooled_boot, 0.025)), float(np.nanquantile(pooled_boot, 0.975))]},
                    "delay_only_comparator_rho": float(delay_only), "n_pairs": len(pairs), "controllers": len(ctrls)}  # fmt: skip
    # sample efficiency
    thr = json.loads((TM / "learning_curves_fine.json").read_text(encoding="utf-8"))["thresholds"]
    thresholds = {
        "lqr_gs": thr["lqr_gs"],
        "lqr_no_margin": thr["lqr_no_margin"],
        "abs_490": 490.0,
        "abs_494": 494.0,
    }
    runs = {g: [json.loads(f.read_text(encoding="utf-8")) for f in sorted((TM / "curves_fine").glob(f"{g}_seed*.json"))]
            for g in ("ppo_nominal", "diffsim_nominal")}  # fmt: skip
    eff = {}
    for t, v in thresholds.items():
        row = {}
        for key in ("env_steps", "wall_s", "cpu_s"):
            meds = {}
            for g, rs in runs.items():
                vals = [first(r["history"], v, key) for r in rs]
                times = np.array([x[0] for x in vals], float)
                got = np.array([x[1] for x in vals])
                med = km_median(times, got)
                boots = []
                for _ in range(B):
                    i = rng.integers(0, len(times), len(times))
                    boots.append(km_median(times[i], got[i]))
                bb = [b for b in boots if b is not None]
                meds[g] = {"km_median": med, "reached": int(got.sum()), "of": len(got),
                           "ci95": [float(np.quantile(bb, 0.025)), float(np.quantile(bb, 0.975))] if len(bb) > 0.95 * B else None}  # fmt: skip
            p, a_ = meds["ppo_nominal"]["km_median"], meds["diffsim_nominal"]["km_median"]
            row[key] = {**meds, "ratio_ppo_over_apg": (p / a_) if p and a_ else None}
        eff[t] = {"threshold": v, **row}
    out["sample_efficiency"] = eff
    out["sample_efficiency_note"] = (
        "evaluation every update = 32,000 env steps (the batch); compile time excluded; cpu_s is process CPU time"
    )
    (args.dir / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"tost": tost, "audit": out["audit"]}, indent=1))
    for t, r in eff.items():
        print(
            t,
            {
                k: (
                    r[k]["ppo_nominal"]["km_median"],
                    r[k]["diffsim_nominal"]["km_median"],
                    r[k]["ratio_ppo_over_apg"],
                )
                for k in ("env_steps", "wall_s", "cpu_s")
            },
        )


if __name__ == "__main__":
    main()
