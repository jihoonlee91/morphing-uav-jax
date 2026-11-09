# ruff: noqa: E501
"""Round 3: loop-at-a-time classical and disk margins at the plant input.

    uv run python scripts/r3_margins.py [--policies DIR ...]

For every controller and each of the 135 envelope points of the check grid, the frozen
continuous-time lateral loop (true linearization, gains as in the linear audit; learned
policies linearized at trim) is broken at one command channel at a time (aileron,
rudder, asymmetric span) with the other two closed:

    L_i(s) = -K_i (sI - A - B_{-i} K_{-i})^{-1} B_i      (closed loop: 1 + L_i = 0)

and from L_i(jw) on 600 log-spaced frequencies (0.01-1000 rad/s) we take the classical
gain margin (upper; at -180 deg crossings), the phase margin, the delay margin
PM / w_c, and the balanced (sigma = 0) disk margin alpha = 1 / max_w |S - 1/2|,
S = 1 / (1 + L) (Seiler, Packard, Gahinet 2020), with its gain range
[(2 - alpha)/(2 + alpha), (2 + alpha)/(2 - alpha)] and phase margin 2 atan(alpha/2).
Margins are computed only where the frozen closed loop is stable. Policies that observe
their pending command are omitted (their law has memory; see audit.json). Writes
``results/tmlr/r3/margins.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from morphjax import design, pipeline
from morphjax.rl import RESIDUAL_SCALE, load_policy
from morphjax.verify import dense_grid, interpolate_gains, policy_command_full

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "results" / "tmlr"
W = np.logspace(-2, 3, 600)
CHANNELS = ("aileron", "rudder", "eta_a")


def channel_margins(a, b, k):
    """Per channel: (disk alpha, upper GM, PM deg, delay margin s, crossover rad/s)."""
    out = []
    n = a.shape[0]
    for i in range(3):
        others = [j for j in range(3) if j != i]
        ac = a + b[:, others] @ k[others]
        res = np.linalg.solve(
            1j * W[:, None, None] * np.eye(n)[None] - ac[None],
            np.broadcast_to(b[:, i, None], (len(W), n, 1)),
        )[..., 0]
        L = -(res @ k[i])
        S = 1.0 / (1.0 + L)
        alpha = 1.0 / np.abs(S - 0.5).max()
        mag, ph = np.abs(L), np.unwrap(np.angle(L))
        gm = np.inf
        for j in range(len(W) - 1):  # phase crossings of -180 deg (mod 360)
            k0, k1 = (
                np.floor((ph[j] + np.pi) / (2 * np.pi)),
                np.floor((ph[j + 1] + np.pi) / (2 * np.pi)),
            )
            if k0 != k1 and mag[j] > 0:
                gm = min(gm, 1.0 / max(mag[j], 1e-12)) if mag[j] < 1 else gm
        pm, dm, wc = np.inf, np.inf, np.nan
        for j in range(len(W) - 1):
            if (mag[j] - 1) * (mag[j + 1] - 1) <= 0:
                pmj = np.rad2deg(np.angle(L[j])) + 180.0
                pmj = (pmj + 180) % 360 - 180
                if abs(pmj) < abs(pm):
                    pm, wc = pmj, W[j]
                    dm = np.deg2rad(pmj) / W[j] if pmj > 0 else 0.0
        out.append((alpha, gm, pm, dm, wc))
    return out


def summarize(per_point):
    """per_point: list over points of [3 channels x 5]; NaN rows = unstable point."""
    arr = np.array(per_point, dtype=float)  # points x 3 x 5
    stable = np.isfinite(arr[:, 0, 0])
    res = {"points": int(arr.shape[0]), "stable_points": int(stable.sum())}
    if not stable.any():
        return res
    s = arr[stable]
    for c, ch in enumerate(CHANNELS):
        alpha = s[:, c, 0]
        res[ch] = {
            "disk_alpha_median": float(np.median(alpha)),
            "disk_alpha_min": float(alpha.min()),
            "disk_gain_range_median": [
                float((2 - np.median(alpha)) / (2 + np.median(alpha)))
                if np.median(alpha) < 2
                else 0.0,
                float((2 + np.median(alpha)) / (2 - np.median(alpha)))
                if np.median(alpha) < 2
                else float("inf"),
            ],
            "disk_phase_margin_deg_median": float(np.rad2deg(2 * np.arctan(np.median(alpha) / 2))),
            "upper_gm_median": float(np.median(s[:, c, 1])),
            "phase_margin_deg_median": float(np.median(s[:, c, 2])),
            "delay_margin_ms_median": float(1000 * np.median(s[:, c, 3])),
            "crossover_rad_s_median": float(np.nanmedian(s[:, c, 4])),
        }
    weakest = np.argmin(s[:, :, 0], axis=1)
    res["weakest_channel_counts"] = {ch: int((weakest == c).sum()) for c, ch in enumerate(CHANNELS)}
    res["weakest_channel"] = CHANNELS[int(np.argmin([np.median(s[:, c, 0]) for c in range(3)]))]
    return res


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=Path, nargs="+", default=[TM / "policies"])
    ap.add_argument("--out", type=Path, default=TM / "r3" / "margins.json")
    args = ap.parse_args()
    t0 = time.time()
    setup = pipeline.build_revision(ROOT / "results" / "lqr_weights.json", TM / "lqr_wide.json")
    dense, _ = dense_grid(setup)
    pts = np.stack(np.meshgrid(dense.v_syn, dense.alt, dense.eta_s, indexing="ij"), -1)
    env = (pts[..., 0] >= 70) & (pts[..., 0] <= 110) & (pts[..., 1] >= 500) & (pts[..., 1] <= 9500)
    idxs = [i for i in np.ndindex(*env.shape) if env[i]]
    lat = {i: design.lateral_model(dense.A[i], dense.B[i]) for i in idxs}

    def evaluate(gains):
        rows = []
        for i in idxs:
            a, b = lat[i]
            k = gains[i]
            if np.real(np.linalg.eigvals(a + b @ k)).max() >= 0:
                rows.append([[np.nan] * 5] * 3)
            else:
                rows.append(channel_margins(a, b, k))
        return summarize(rows)

    res: dict = {"model_based": {}, "learned": {}}
    for name in ("gs", "gs_hg", "lpv", "fixed", "gs_hg_wide", "lpv_pw", "lpv_nodisk"):
        if name in setup.lat_gains:
            res["model_based"][name] = evaluate(
                interpolate_gains(setup.grid, setup.lat_gains[name], dense)
            )
            print(
                f"[{time.time() - t0:5.0f}s] {name}: weakest {res['model_based'][name].get('weakest_channel')}",
                flush=True,
            )

    lpv_dense = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    flat = jnp.asarray(pts.reshape(-1, 3))
    for d in args.policies:
        for name, (method, variant) in pipeline.LEARNED.items():
            if name in pipeline.PROJECTED:
                continue
            files = sorted(d.glob(f"{method}_{variant}_seed*.npz"))
            if not files:
                continue
            residual = name in pipeline.RESIDUAL_BASE
            seeds = []
            for f in files:
                params, pcfg = load_policy(f)
                if pcfg.history:
                    break
                scale = jnp.asarray(RESIDUAL_SCALE if residual else pcfg.act_scale)
                jac = jax.jit(jax.vmap(lambda r, params=params, scale=scale: jax.jacfwd(
                    lambda x: policy_command_full(params, x, jnp.zeros(3), r, scale, 0, False))(jnp.zeros(8))))(flat)  # fmt: skip
                g = np.asarray(jac).reshape(*env.shape, 3, 8)
                if residual:
                    g = g + lpv_dense
                seeds.append(evaluate(g))
            if not seeds:
                continue
            pooled = {
                "seeds": len(seeds),
                "stable_points_per_seed": [s["stable_points"] for s in seeds],
            }
            for ch in CHANNELS:
                vals = [s[ch]["disk_alpha_median"] for s in seeds if ch in s]
                dms = [s[ch]["delay_margin_ms_median"] for s in seeds if ch in s]
                if vals:
                    pooled[ch] = {"disk_alpha_median_of_seed_medians": float(np.median(vals)),
                                  "disk_alpha_seed_range": [float(min(vals)), float(max(vals))],
                                  "delay_margin_ms_median_of_seeds": float(np.median(dms))}  # fmt: skip
            pooled["weakest_channel_per_seed"] = [s.get("weakest_channel") for s in seeds]
            res["learned"][name] = {"pooled": pooled, "per_seed": seeds}
            print(
                f"[{time.time() - t0:5.0f}s] {name}: weakest {pooled['weakest_channel_per_seed']}",
                flush=True,
            )
    res["meta"] = {"frequencies_rad_s": [float(W[0]), float(W[-1]), len(W)], "points": len(idxs),
                   "wall_seconds": time.time() - t0}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
