"""Revision: linear audit of every controller, at trim and at the closed-loop equilibrium.

    uv run python scripts/audit_revision.py [--policies results/tmlr/policies]
                                            [--metrics results/tmlr/metrics.json]

On the 325-point check grid (``verify.DENSE``) every controller's frozen loop is
checked for stability and for the largest sampled-data delay it tolerates
(0-80 ms), and against the LPV Lyapunov matrices (the design's and, for the
no-disk ablation, its own). Learned policies are linearized two ways:

* ``x0``: the Jacobian of the unclipped command at the trim point (dx = 0), as in
  ``scripts/verify_designs.py``;
* ``eq``: at the closed-loop equilibrium of the linearized plant under the
  (clipped) policy, i.e. the fixed point ``a dx + b u = 0, u = pi(dx, u)``. A
  policy whose output at dx = 0 is not the trim command settles there instead.

Policies that observe their pending command (``history = 1``) are checked with
``u_k = Kx x_k + Ku u_{k-1}`` (``verify.spectral_radius_hist``). Projected residuals
are not linearizable at trim and are left to the sampled check of verify_designs.py.

With ``--metrics`` (an ``evaluate.py`` output) it also exports, for every controller x
delay pair (20-80 ms), the in-envelope share of points tolerating the delay next to
the simulated RMSE growth, activity growth and loss-of-control share, with rank
correlations over all pairs. Writes ``audit.json`` and ``audit_points.npz``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats

from morphjax import design, pipeline
from morphjax.rl import RESIDUAL_SCALE, load_policy
from morphjax.sim import DT
from morphjax.verify import dense_grid, interpolate_gains, policy_command_full

ROOT = Path(__file__).resolve().parents[1]
MAX_D = 4
NEWTON_ITERS = 40
STEPS_T = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0])
SKIP = set(pipeline.PROJECTED)  # filtered laws: checked by sampling, not by a Jacobian


def zoh(a, b):
    n, m = b.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = a, b
    e = scipy.linalg.expm(blk * DT)
    return e[:n, :n], e[:n, n:]


def radius(ad, bd, kx, ku, d):
    """verify.spectral_radius_hist with the ZOH matrices precomputed."""
    n, m = bd.shape
    big_d = max(d, 1)
    big = np.zeros((n + big_d * m, n + big_d * m))
    big[:n, :n] = ad
    if d == 0:
        big[:n, :n] += bd @ kx
        big[:n, n : n + m] += bd @ ku
    else:
        big[:n, n + (d - 1) * m : n + d * m] = bd
    big[n : n + m, :n] = kx
    big[n : n + m, n : n + m] = ku
    for j in range(1, big_d):
        big[n + j * m : n + (j + 1) * m, n + (j - 1) * m : n + j * m] = np.eye(m)
    return float(np.abs(np.linalg.eigvals(big)).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=Path, default=ROOT / "results" / "tmlr" / "policies")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "tmlr" / "metrics.json")
    ap.add_argument("--wide", type=Path, default=ROOT / "results" / "tmlr" / "lqr_wide.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmlr")
    args = ap.parse_args()
    t0 = time.time()
    setup = pipeline.build_revision(ROOT / "results" / "lqr_weights.json", args.wide, verbose=True)
    dense, on_grid = dense_grid(setup)
    shp = dense.residual.shape
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    envelope = ((pts[..., 0] >= 70) & (pts[..., 0] <= 110) & (pts[..., 1] >= 500)
                & (pts[..., 1] <= 9500))  # fmt: skip
    xs, us = setup.weights.scales()
    lat = {idx: design.lateral_model(dense.A[idx], dense.B[idx]) for idx in np.ndindex(*shp)}
    zohs = {idx: zoh(*lat[idx]) for idx in lat}
    p_lpv = np.linalg.inv(setup.lyapunov["lpv"])
    p_nodisk = np.linalg.inv(setup.lyapunov["lpv_nodisk"])

    def certified(a, b, kx, p):
        an, bn = design._normalize(a, b, xs, us)
        acl = an + bn @ (np.diag(1 / us) @ kx @ np.diag(xs))
        return bool(np.linalg.eigvalsh(acl.T @ p + p @ acl).max() < 0)

    def check(kx_all, ku_all=None):
        """Per-point arrays: stability, tolerated delay, certificates."""
        out = {k: np.zeros(shp) for k in ("max_real_eig", "max_delay_steps", "cert_lpv",
                                          "cert_nodisk", "radius0")}  # fmt: skip
        for idx in np.ndindex(*shp):
            a, b = lat[idx]
            kx = kx_all[idx]
            ku = np.zeros((3, 3)) if ku_all is None else ku_all[idx]
            ad, bd = zohs[idx]
            out["max_real_eig"][idx] = (float(np.real(np.linalg.eigvals(a + b @ kx)).max())
                                        if ku_all is None else np.nan)  # fmt: skip
            d_ok = -1
            for d in range(MAX_D + 1):
                r = radius(ad, bd, kx, ku, d)
                if d == 0:
                    out["radius0"][idx] = r
                if r >= 1.0:
                    break
                d_ok = d
            out["max_delay_steps"][idx] = d_ok
            out["cert_lpv"][idx] = certified(a, b, kx, p_lpv) if ku_all is None else False
            out["cert_nodisk"][idx] = certified(a, b, kx, p_nodisk) if ku_all is None else False
        return out

    def summary(c):
        stable = c["max_delay_steps"] >= 0  # sampled loop stable without delay
        env = envelope
        return {
            "points": int(stable.size),
            "stable_sampled": int(stable.sum()),
            "stable_continuous": int((c["max_real_eig"] < 0).sum())
            if np.isfinite(c["max_real_eig"]).all()
            else None,
            "envelope_points": int(env.sum()),
            "envelope_stable": int(stable[env].sum()),
            "envelope_tolerates_ms": {
                str(20 * d): float((c["max_delay_steps"][env] >= d).mean())
                for d in range(MAX_D + 1)
            },
            "all_points_tolerates_ms": {
                str(20 * d): float((c["max_delay_steps"] >= d).mean()) for d in range(MAX_D + 1)
            },
            "min_delay_ms": float(20 * c["max_delay_steps"].min()),
            "certified_lpv_P": int(c["cert_lpv"].sum()),
            "certified_nodisk_P": int(c["cert_nodisk"].sum()),
        }

    points: dict[str, np.ndarray] = {"on_grid": on_grid, "envelope": envelope}
    res: dict = {"model_based": {}, "learned": {}}
    for name, g in setup.lat_gains.items():
        c = check(interpolate_gains(setup.grid, g, dense))
        res["model_based"][name] = summary(c)
        for k, v in c.items():
            points[f"{name}/{k}"] = v
        tol = res["model_based"][name]["envelope_tolerates_ms"]
        print(f"[{time.time() - t0:5.0f}s] {name}: {tol}")

    lpv_dense = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    flat_pts = jnp.asarray(pts.reshape(-1, 3))
    a_all = jnp.asarray(np.stack([lat[i][0] for i in np.ndindex(*shp)]))
    b_all = jnp.asarray(np.stack([lat[i][1] for i in np.ndindex(*shp)]))
    kb_all = jnp.asarray(lpv_dense.reshape(-1, 3, 8))

    for name, (method, variant) in pipeline.LEARNED.items():
        if name in SKIP:
            continue
        files = sorted(args.policies.glob(f"{method}_{variant}_seed*.npz"))
        if not files:
            continue
        residual = name in pipeline.RESIDUAL_BASE
        rows = []
        for f in files:
            params, pcfg = load_policy(f)
            scale = jnp.asarray(RESIDUAL_SCALE if residual else pcfg.act_scale)
            hist = pcfg.history

            def cmd(dx, u, rho, kb, clip=True, params=params, scale=scale, hist=hist,
                    residual=residual):  # fmt: skip
                pol = policy_command_full(params, dx, u, rho, scale, hist, clip)
                return kb @ dx + pol if residual else pol

            def newton(a, b, rho, kb, cmd=cmd):
                def resid(z):
                    return jnp.concatenate(
                        [a @ z[:8] + b @ z[8:], z[8:] - cmd(z[:8], z[8:], rho, kb)]
                    )

                def it(z, _):
                    r = resid(z)
                    step = jnp.linalg.lstsq(jax.jacfwd(resid)(z), -r)[0]
                    cand = z[None] + STEPS_T[:, None] * step[None]
                    norms = jax.vmap(lambda zc: jnp.linalg.norm(resid(zc)))(cand)
                    return cand[jnp.argmin(norms)], None

                z, _ = jax.lax.scan(it, jnp.zeros(11), None, length=NEWTON_ITERS)
                kx = jax.jacfwd(lambda xx: cmd(xx, z[8:], rho, kb))(z[:8])
                ku = jax.jacfwd(lambda uu: cmd(z[:8], uu, rho, kb))(z[8:])
                kx0 = jax.jacfwd(lambda xx: cmd(xx, jnp.zeros(3), rho, kb, False))(jnp.zeros(8))
                ku0 = jax.jacfwd(lambda uu: cmd(jnp.zeros(8), uu, rho, kb, False))(jnp.zeros(3))
                return z, jnp.linalg.norm(resid(z)), kx, ku, kx0, ku0

            z, fres, kx, ku, kx0, ku0 = jax.jit(jax.vmap(newton))(a_all, b_all, flat_pts, kb_all)
            z, fres = np.asarray(z), np.asarray(fres)
            r3 = lambda x, s: np.asarray(x).reshape(*shp, *s)  # noqa: E731
            use_ku = hist > 0
            c_x0 = check(r3(kx0, (3, 8)), r3(ku0, (3, 3)) if use_ku else None)
            c_eq = check(r3(kx, (3, 8)), r3(ku, (3, 3)) if use_ku else None)
            dx_n = z[:, :8] / xs
            row = {
                "seed_file": f.name,
                "history": hist,
                "x0": summary(c_x0),
                "eq": summary(c_eq),
                "equilibrium": {
                    "converged": int((fres < 1e-8).sum()),
                    "points": int(fres.size),
                    "max_residual": float(fres.max()),
                    "max_normalized_offset": float(np.abs(dx_n).max()),
                    "median_normalized_offset": float(np.median(np.linalg.norm(dx_n, axis=1))),
                    "max_abs_command_deg_or_pct": float(np.abs(z[:, 8:]).max()),
                    "envelope_max_normalized_offset": float(np.abs(dx_n[envelope.ravel()]).max()),
                },
            }
            rows.append(row)
            for mode, c in (("x0", c_x0), ("eq", c_eq)):
                for k, v in c.items():
                    points[f"{name}/{f.stem}/{mode}/{k}"] = v
            points[f"{name}/{f.stem}/eq/state"] = z.reshape(*shp, 11)
        pooled = {}
        for mode in ("x0", "eq"):
            pooled[mode] = {
                "envelope_tolerates_ms": {
                    d: float(np.mean([r[mode]["envelope_tolerates_ms"][d] for r in rows]))
                    for d in rows[0][mode]["envelope_tolerates_ms"]
                },
                "envelope_stable_per_seed": [r[mode]["envelope_stable"] for r in rows],
                "certified_lpv_P_per_seed": [r[mode]["certified_lpv_P"] for r in rows],
                "seeds_tolerating_20ms_somewhere": int(
                    sum(r[mode]["envelope_tolerates_ms"]["20"] > 0 for r in rows)
                ),
            }
        res["learned"][name] = {"seeds": rows, "pooled": pooled}
        tx0, teq = pooled["x0"]["envelope_tolerates_ms"], pooled["eq"]["envelope_tolerates_ms"]
        print(f"[{time.time() - t0:5.0f}s] {name} x{len(rows)}: x0 {tx0} eq {teq}")

    if args.metrics.exists():
        res["pairs"] = all_pairs(json.loads(args.metrics.read_text(encoding="utf-8")), res)

    res["meta"] = {"grid_points": int(on_grid.size), "envelope_points": int(envelope.sum()),
                   "envelope": "V_syn 70-110 kt, h 500-9500 ft (the training/test envelope)",
                   "policies": str(args.policies), "wall_seconds": time.time() - t0}  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(args.out / "audit_points.npz", **points)
    print(f"done in {time.time() - t0:.0f}s")


def all_pairs(m: dict, res: dict) -> dict:
    """Linear share vs. simulated degradation for every controller x delay pair."""
    rows = {r["value"]: r for r in m["sweeps"]["delay"]["rows"]}
    base = rows[0]
    pairs = []
    for name in base:
        if name == "value":
            continue
        if name in res["model_based"]:
            shares = {"x0": res["model_based"][name]["envelope_tolerates_ms"]}
            shares["eq"] = shares["x0"]
        elif name in res["learned"]:
            shares = {mode: res["learned"][name]["pooled"][mode]["envelope_tolerates_ms"]
                      for mode in ("x0", "eq")}  # fmt: skip
        else:
            continue
        r0, a0 = base[name]["roll_rmse_deg"], base[name]["cmd_activity"]
        for d in range(1, MAX_D + 1):
            s = rows[d][name]
            rmse, act = s["roll_rmse_deg"], s["cmd_activity"]
            pairs.append({
                "controller": name,
                "delay_ms": 20 * d,
                "share_x0": shares["x0"][str(20 * d)],
                "share_eq": shares["eq"][str(20 * d)],
                "rmse_growth": None if rmse is None or r0 is None else rmse / r0,
                "activity_growth": None if act is None or a0 is None else act / a0,
                "loc_share": 1.0 - s["success_rate"],
            })  # fmt: skip

    def degradation(p):
        # Loss of control ranks above any finite error growth.
        g = p["rmse_growth"] if p["rmse_growth"] is not None else np.inf
        return (p["loc_share"] > 0, p["loc_share"], g)

    order = sorted(range(len(pairs)), key=lambda i: degradation(pairs[i]))
    rank = np.empty(len(pairs))
    rank[order] = np.arange(len(pairs))
    out = {"pairs": pairs, "n_pairs": len(pairs)}
    for mode in ("x0", "eq"):
        share = np.array([p[f"share_{mode}"] for p in pairs])
        rho, pval = scipy.stats.spearmanr(share, -rank)
        tau, tp = scipy.stats.kendalltau(share, -rank)
        act = np.array([p["activity_growth"] if p["activity_growth"] is not None else np.nan
                        for p in pairs])  # fmt: skip
        ok = np.isfinite(act)
        rho_a, pa = scipy.stats.spearmanr(share[ok], -act[ok])
        out[f"spearman_{mode}"] = {"rho": float(rho), "p": float(pval), "kendall_tau": float(tau),
                                   "kendall_p": float(tp),
                                   "vs_activity_growth_rho": float(rho_a),
                                   "vs_activity_growth_p": float(pa)}  # fmt: skip
        inter = (share > 0) & (share < 1)
        if inter.sum() >= 3:
            r_i, p_i = scipy.stats.spearmanr(share[inter], -rank[inter])
            out[f"spearman_{mode}_intermediate_shares"] = {"rho": float(r_i), "p": float(p_i),
                                                           "n": int(inter.sum())}  # fmt: skip
    return out


if __name__ == "__main__":
    main()
