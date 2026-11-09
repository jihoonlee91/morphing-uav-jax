"""Round 3: facts from the revision audit data (no new simulation).

    uv run python scripts/r3_audit_facts.py

1. Closed-loop equilibrium of the learned policies (``audit_points.npz``, ``eq/state``):
   raw offset of every lateral state from trim, the roll-error integrator against the
   simulator's limit (0.3 rad s, ``sim.INTEG_LIMITS``), and the offsets normalized by
   the tuned Bryson scales (which is what the first revision report quoted).
2. Per-delay audit statistics without duplicate designs (``gs_wide`` and ``gs_dr`` are
   the same gains as ``gs``): Spearman at each delay, pooled, and the 2 x 2 table
   (stable share 0 vs > 0) x (error growth >= 5x or any loss of control vs not).
3. Unprojected residual: per-point share of sampled states/residuals meeting the decay
   condition (``verification_points.npz``).

Writes ``results/tmlr/r3/audit_facts.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats

from morphjax import pipeline
from morphjax.sim import INTEG_LIMITS

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
STATES = ("beta", "p", "r", "phi_err", "da", "dr", "eta_a", "integ_phi")
UNITS = ("deg", "deg/s", "deg/s", "deg", "deg", "deg", "%", "rad s")
TO_UNIT = np.array([np.rad2deg(1), np.rad2deg(1), np.rad2deg(1), np.rad2deg(1), 1, 1, 1, 1])
DUPLICATES = ("gs_wide", "gs_dr")  # identical gains to gs (results/tmlr/lqr_wide.json)


def equilibrium_facts(pts, xs) -> dict:
    env = pts["envelope"]
    out = {}
    for name in ("ppo", "ppo_dr", "ppo_clean", "ppo_lat", "diffsim", "diffsim_dr", "diffsim_lat",
                 "residual_dr"):  # fmt: skip
        keys = sorted(k for k in pts.files if k.startswith(f"{name}/") and k.endswith("/eq/state"))
        if not keys:
            continue
        z = np.stack([pts[k] for k in keys])  # seeds x 13 x 5 x 5 x 11
        dx = z[..., :8]
        e = np.broadcast_to(env, dx.shape[:-1])
        raw = np.abs(dx) * TO_UNIT
        norm = np.abs(dx) / xs
        integ = np.abs(dx[..., 7])
        row = {
            "seeds": len(keys),
            "median_abs_raw_envelope": {
                f"{s} ({u})": float(np.median(raw[..., i][e]))
                for i, (s, u) in enumerate(zip(STATES, UNITS, strict=True))
            },
            "median_abs_raw_all": {
                f"{s} ({u})": float(np.median(raw[..., i]))
                for i, (s, u) in enumerate(zip(STATES, UNITS, strict=True))
            },
            "integrator_median_rad_s_envelope": float(np.median(integ[e])),
            "integrator_median_rad_s_all": float(np.median(integ)),
            "share_integrator_beyond_limit_envelope": float((integ[e] > INTEG_LIMITS[2]).mean()),
            "share_integrator_beyond_limit_all": float((integ > INTEG_LIMITS[2]).mean()),
            "median_normalized_norm_all": float(np.median(np.linalg.norm(norm, axis=-1))),
            "dominant_normalized_state_counts": {
                s: int((np.argmax(norm, axis=-1) == i).sum()) for i, s in enumerate(STATES)
            },
            "integ_normalization_scale_rad_s": float(xs[7]),
        }
        out[name] = row
    return out


def per_delay(audit: dict) -> dict:
    pairs = [p for p in audit["pairs"]["pairs"] if p["controller"] not in DUPLICATES]

    def bad(p):  # >= 5x error growth or any loss of control
        return p["loc_share"] > 0 or (p["rmse_growth"] is not None and p["rmse_growth"] >= 5)

    def rank_key(p):
        # Pairs with loss of control rank above any finite error growth, ordered by the
        # share lost; pairs where every run is lost tie at the top (average rank).
        g = p["rmse_growth"] if p["rmse_growth"] is not None else np.inf
        return (p["loc_share"] > 0, p["loc_share"], g)

    def rho(sub, mode):
        keys = [rank_key(p) for p in sub]
        order = sorted(set(keys))
        r = np.array([order.index(k) for k in keys], dtype=float)
        r = scipy.stats.rankdata(r)  # ties -> average rank
        share = np.array([p[f"share_{mode}"] for p in sub])
        res = scipy.stats.spearmanr(share, -r)
        return {"rho": float(res.statistic), "p": float(res.pvalue), "n": len(sub)}

    out: dict = {"n_pairs_without_duplicates": len(pairs), "removed": list(DUPLICATES)}
    for mode in ("x0", "eq"):
        out[f"pooled_{mode}"] = rho(pairs, mode)
        out[f"by_delay_{mode}"] = {
            d: rho([p for p in pairs if p["delay_ms"] == d], mode) for d in (20, 40, 60, 80)
        }
        z = np.array([p[f"share_{mode}"] == 0 for p in pairs])
        b = np.array([bad(p) for p in pairs])
        out[f"confusion_{mode}"] = {
            "share0_bad": int((z & b).sum()),
            "share0_notbad": int((z & ~b).sum()),
            "sharepos_bad": int((~z & b).sum()),
            "sharepos_notbad": int((~z & ~b).sum()),
            "share0_notbad_pairs": [
                (p["controller"], p["delay_ms"])
                for p, zz, bb in zip(pairs, z, b, strict=True)
                if zz and not bb
            ],
            "sharepos_bad_pairs": [
                (p["controller"], p["delay_ms"], round(p[f"share_{mode}"], 3))
                for p, zz, bb in zip(pairs, z, b, strict=True)
                if bb and not zz
            ],
        }
    out["ranking_rule"] = (
        "degradation rank: pairs with any loss of control above all others, ordered by the "
        "share lost, then by RMSE growth; pairs where every run is lost share the top rank "
        "(average rank for ties)"
    )
    return out


def main() -> None:
    setup = pipeline.build_for_evaluation(ROOT / "results" / "lqr_weights.json")
    xs, _ = setup.weights.scales()
    pts = np.load(TM / "audit_points.npz")
    audit = json.loads((TM / "audit.json").read_text(encoding="utf-8"))
    vp = np.load(TM / "verification_points.npz")
    res = {
        "equilibrium": equilibrium_facts(pts, xs),
        "audit_per_delay": per_delay(audit),
        "unprojected_residual_decay_share": {
            k: {
                "min": float(vp[f"sampled/{k}/decay"].min()),
                "median": float(np.median(vp[f"sampled/{k}/decay"])),
                "max": float(vp[f"sampled/{k}/decay"].max()),
                "points_all_samples": int((vp[f"sampled/{k}/decay"] >= 1).sum()),
            }
            for k in ("residual_dr", "residual_proj", "residual_safe2")
            if f"sampled/{k}/decay" in vp
        },
    }
    out = TM / "r3" / "audit_facts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:9000])


if __name__ == "__main__":
    main()
