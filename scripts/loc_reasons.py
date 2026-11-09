"""Revision: which loss-of-control criterion fires, in the 60 and 80 ms delay runs.

    uv run python scripts/loc_reasons.py [--policies results/tmlr/policies]

Flies the delay-sweep scenarios of ``evaluate.py`` (same seed and count, fast
morphing, 60 and 80 ms) and, for every run that departs, records which criteria of
``sim.departed`` hold at the first departed state: |bank| > 85 deg, |sideslip| > 25 deg,
angle of attack < -10 deg or > 25 deg, airspeed < 40 kt, or a non-finite state. Several
can hold at once. Writes ``loc_reasons.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from evaluate import STEPS, SWEEPS, controllers

from morphjax import pipeline
from morphjax.airframe import DEG, KT_TO_FPS
from morphjax.control import longitudinal_law
from morphjax.dynamics import Cmd, Uncertainty, W, body_to_wind, rk4_step
from morphjax.scenarios import sample_scenario
from morphjax.sim import DT, INTEG_LIMITS, MAX_DELAY, initial_carry, measure

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ("bank>85", "sideslip>25", "alpha<-10", "alpha>25", "speed<40kt", "non-finite")


def reasons(xw):
    return jnp.array([
        jnp.abs(xw[W.PHI]) > 85 * DEG,
        jnp.abs(xw[W.BETA]) > 25 * DEG,
        xw[W.ALPHA] < -10 * DEG,
        xw[W.ALPHA] > 25 * DEG,
        xw[W.V] < 40 * KT_TO_FPS,
        ~jnp.all(jnp.isfinite(xw)),
    ])  # fmt: skip


def rollout_reasons(plant, lon, lateral, scen):
    """sim.rollout, returning the criteria that hold at the first departure (or all False)."""
    with_buf = getattr(lateral, "uses_history", False)

    def body(state, inputs):
        carry, first = state
        phi_cmd, eta_s_cmd = inputs
        xw_meas, key = measure(carry, scen)
        carry = carry._replace(key=key)
        lat_cmd = (lateral(xw_meas, phi_cmd, carry.integ[2], carry.buf) if with_buf
                   else lateral(xw_meas, phi_cmd, carry.integ[2]))  # fmt: skip
        lon_cmd = longitudinal_law(lon, xw_meas, scen.v_kt * KT_TO_FPS, scen.h, *carry.integ[:2])
        cmd = jnp.zeros(6).at[Cmd.DE].set(lon_cmd[0]).at[Cmd.THR].set(lon_cmd[1])
        cmd = cmd.at[Cmd.DA].set(lat_cmd[0]).at[Cmd.DR].set(lat_cmd[1])
        cmd = cmd.at[Cmd.ETA_A].set(lat_cmd[2]).at[Cmd.ETA_S].set(eta_s_cmd)
        buf = jnp.roll(carry.buf, 1, axis=0).at[0].set(cmd)
        applied = buf[jnp.clip(scen.delay, 0, MAX_DELAY)]
        unc = Uncertainty(scen.morph_aero, scen.inertia, scen.aileron_effect)
        x_next = rk4_step(plant, carry.x, applied, DT, unc)
        why = reasons(body_to_wind(x_next))
        new_fail = ~carry.failed & jnp.any(why)
        first = jnp.where(new_fail, why, first)
        err = jnp.array([xw_meas[W.V] - scen.v_kt * KT_TO_FPS, xw_meas[W.H] - scen.h,
                         xw_meas[W.PHI] - phi_cmd])  # fmt: skip
        integ = jnp.clip(carry.integ + DT * err, -INTEG_LIMITS, INTEG_LIMITS)
        failed = carry.failed | jnp.any(why)
        x_next = jnp.where(failed, carry.x, x_next)
        return (carry._replace(x=x_next, integ=integ, buf=buf, failed=failed), first), None

    init = (initial_carry(plant, scen), jnp.zeros(len(CRITERIA), bool))
    (carry, first), _ = jax.lax.scan(body, init, (scen.phi_cmd, scen.eta_s_cmd))
    return carry.failed, first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=Path, default=ROOT / "results" / "tmlr" / "policies")
    ap.add_argument("--wide", type=Path, default=ROOT / "results" / "tmlr" / "lqr_wide.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmlr" / "loc_reasons.json")
    args = ap.parse_args()
    t0 = time.time()
    setup = pipeline.build_revision(ROOT / "results" / "lqr_weights.json", args.wide)
    ctrls = controllers(setup, args.policies)
    j = list(SWEEPS).index("delay")
    n_sweep = 128
    out: dict = {"criteria": CRITERIA, "n_per_controller_seed": n_sweep, "delays": {}}
    runs = {}
    for name, (lon, lats, factory, params) in ctrls.items():
        if factory is None:
            runs[name] = [jax.jit(jax.vmap(lambda s, lat=lat, lon=lon: rollout_reasons(
                setup.plant, lon, lat, s))) for lat in lats]  # fmt: skip
        else:
            g = jax.jit(jax.vmap(lambda p, s, lon=lon, fac=factory: rollout_reasons(
                setup.plant, lon, fac(p), s), in_axes=(None, 0)))  # fmt: skip
            runs[name] = [lambda s, p=p, g=g: g(p, s) for p in params]
    for delay in (3, 4):
        keys = jax.random.split(jax.random.PRNGKey(2000 + 10 * j), n_sweep)
        scen = jax.vmap(lambda k, d=delay: sample_scenario(k, STEPS, math.inf, delay=d))(keys)
        res = {}
        for name, fns in runs.items():
            failed_all, first_all = [], []
            for f in fns:
                failed, first = f(scen)
                failed_all.append(np.asarray(failed))
                first_all.append(np.asarray(first))
            failed, first = np.concatenate(failed_all), np.concatenate(first_all)
            n_fail = int(failed.sum())
            res[name] = {
                "runs": int(failed.size),
                "lost": n_fail,
                "criterion_counts": {
                    c: int(first[failed, i].sum()) for i, c in enumerate(CRITERIA)
                },
                "only_criterion_counts": {
                    c: int((first[failed, i] & (first[failed].sum(axis=1) == 1)).sum())
                    for i, c in enumerate(CRITERIA)
                },
            }
            print(f"[{time.time() - t0:5.0f}s] {20 * delay} ms {name}: {res[name]['lost']}/"
                  f"{res[name]['runs']} {res[name]['criterion_counts']}", flush=True)  # fmt: skip
        out["delays"][f"{20 * delay}ms"] = res
    out["wall_seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
