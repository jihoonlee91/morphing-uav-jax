"""Round 3: steps and wall-clock to return thresholds, 10 seeds, censoring-aware medians.

    uv run python scripts/r3_learning_curves.py

Reads the fine learning curves (``results/tmlr/curves_fine``: deterministic held-out
return after every update, seeds 0-9) and the thresholds of
``results/tmlr/learning_curves_fine.json`` (GS-LQR and no-margin-LQR returns on the
same held-out batch, PPO's mean final return), plus 99 % of each run's own final
return. A seed that never reaches a threshold is right-censored at its last update;
medians are Kaplan-Meier medians (the first time the estimated share not yet reached
drops to 0.5 or below; ``None`` if it never does), with 95 % percentile bootstrap
intervals over seeds (2,000 resamples) for each median and for the PPO / APG ratio.
Wall-clock excludes JIT compile (reported separately). Writes
``results/tmlr/r3/learning_curves_km.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
B = 2000


def km_median(times, reached):
    """Kaplan-Meier median of event times with right censoring."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(reached)[order]
    s, n = 1.0, len(t)
    for i in range(n):
        if e[i]:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return float(t[i])
    return None


def first(hist, thr, key):
    for h in hist:
        if "eval_return" in h and h["eval_return"] >= thr:
            return h[key], True
    return hist[-1][key], False


def main() -> None:
    thr = json.loads((TM / "learning_curves_fine.json").read_text(encoding="utf-8"))["thresholds"]
    runs = {}
    for g in ("ppo_nominal", "diffsim_nominal"):
        runs[g] = [
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted((TM / "curves_fine").glob(f"{g}_seed*.json"))
        ]
    rng = np.random.default_rng(0)
    out = {"thresholds": thr, "seeds": {g: len(r) for g, r in runs.items()}, "results": {}}
    for tname in [*thr, "final99"]:
        row = {}
        per = {}
        for g, rs in runs.items():
            for key in ("env_steps", "wall_s"):
                vals = []
                for r in rs:
                    hist = r["history"]
                    t = (
                        thr[tname]
                        if tname != "final99"
                        else 0.99 * [h for h in hist if "eval_return" in h][-1]["eval_return"]
                    )
                    vals.append(first(hist, t, key))
                times = np.array([v[0] for v in vals], dtype=float)
                got = np.array([v[1] for v in vals])
                per[(g, key)] = (times, got)
                med = km_median(times, got)
                boots = []
                for _ in range(B):
                    i = rng.integers(0, len(times), len(times))
                    boots.append(km_median(times[i], got[i]))
                bb = [b for b in boots if b is not None]
                row[f"{g}/{key}"] = {
                    "km_median": med,
                    "ci95": [float(np.quantile(bb, 0.025)), float(np.quantile(bb, 0.975))]
                    if len(bb) > 0.95 * B
                    else None,
                    "reached": int(got.sum()),
                    "of": len(got),
                }
        for key in ("env_steps", "wall_s"):
            (tp, gp), (ta, ga) = per[("ppo_nominal", key)], per[("diffsim_nominal", key)]
            ratios = []
            for _ in range(B):
                i = rng.integers(0, len(tp), len(tp))
                j = rng.integers(0, len(ta), len(ta))
                mp, ma = km_median(tp[i], gp[i]), km_median(ta[j], ga[j])
                if mp is not None and ma:
                    ratios.append(mp / ma)
            mp, ma = km_median(tp, gp), km_median(ta, ga)
            row[f"ratio_ppo_over_apg/{key}"] = {
                "estimate": (mp / ma) if mp and ma else None,
                "ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]
                if len(ratios) > 0.95 * B
                else None,
                "bootstrap_defined": len(ratios),
            }
        out["results"][tname] = row
    out["compile_s"] = {g: [r["timing"]["compile_s"] for r in rs] for g, rs in runs.items()}
    dst = TM / "r3" / "learning_curves_km.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for t, r in out["results"].items():
        print(t, {k: (v.get("km_median", v.get("estimate")), v.get("ci95")) for k, v in r.items()})


if __name__ == "__main__":
    main()
