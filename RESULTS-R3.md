# Round-3 experiments (TMLR), branch `tmlr-revision`

All outputs are in **`results/tmlr/r3/`**. Nothing earlier was changed: the original
results and the first revision batch (`results/tmlr/`) are still there. The plant, reward
and evaluation protocol are as before, with two exceptions that the items asked for:
item 10 adds optional actuator rate limits (`Plant.rate_limits`, default none), and item 12
raises the action-rate weight (`rl.PPOConfig.rate_weight`, default 0.05). Nothing was
tuned on evaluation scenarios.

**For the paper:** `r3/metrics.json` and `r3/per_scenario.npz` hold all 39 controllers on the
standard protocol. They were merged by `scripts/merge_evaluations.py` from
`results/tmlr/` and `r3/eval_{dt,models,learn,sens}/`, and the merge checks that every part
flew the same scenarios. `r3/audit.json` (with `audit_points.npz`) and `r3/claims.json` cover
the same set.

**Compute** (Windows, i5-13600KF, CPU only; 4 training jobs × 5 pinned CPUs):
- Training: 74 jobs, 41.5 CPU-h for the updates alone, about 4.5 h wall.
- Evaluations: 3.1 h wall in total (`eval_dt` 41 min, `eval_models` 15, `eval_learn` 60,
  `eval_sens` 22, rate limits 49).
- Analyses: frontier 21 min, LPV tuning 74 min plus about 60 min of delay checks, everything
  else under 10 min each.

Most training ran from commit `dc42a4e` (`r3/jobs/train_r3.txt`, logs in `r3/logs/`).

## P1 — analyses of existing runs

### 1. Accuracy–delay frontier of the LQR (`r3/frontier.json`, `r3/frontier_heldout.json`)

**Setup.** 1,395 GS-LQR weight candidates: the 768 originals, 640 random points of the wide
grid, and the wide-search optima.
- **Accuracy:** nominal RMSE on the **first 64 of the 256 nominal (frozen-shape) evaluation
  scenarios** (seed 1000).
- **Delay margin (cumulative):** the largest d (samples) such that the frozen ZOH loop is
  stable for every d′ ≤ d. It is computed at all 135 envelope points of the check grid, and
  separately at the 63 design points.

**Pareto front, envelope rule.** Best RMSE for each delay margin:

| Delay margin | Best RMSE |
|---|---|
| unstable | 0.051° |
| 0 ms | 0.052° |
| 20 ms | **0.227°** |
| 40 ms | 0.479° |
| 60 ms | 0.710° |
| 80 ms | 1.437° |

On the design-point rule: 0.051 / 0.072 / 0.333 / 0.484 / 1.011 / 1.501°.

**Held out from selection.** On the other 192 nominal scenarios the Pareto designs score
within 0.001–0.009° of their in-sample values, so the frontier is not a selection
artefact. The 20 ms design flies 0.228° held-out, with activity 0.17 and no loss of
control. On the evaluation delay sweep (128 scenarios) that design scores:

| Delay | RMSE | Loss of control |
|---|---|---|
| 20 ms | 0.217° | 0 % |
| 40 ms | — | 100 % |

The 40 ms front design gives 0.465 / 0.480 / 3.64° at 20 / 40 / 60 ms.

**Learned controllers on the frontier.** Pooled RMSE on the same 64 scenarios; cumulative
delay margin at the 135 envelope points:
- PPO 0.352° tolerates 0 ms (9 seeds; 1 seed is unstable at some point). It is **dominated**:
  the 20 ms LQR is both more accurate (0.227°) and more tolerant.
- APG 0.208° tolerates 0 ms (all seeds). It lies between the LQR's 0 ms (0.052°) and
  20 ms (0.227°) front points.
- APG+DR 0.391° tolerates 40 ms (10/10 seeds). It sits **beyond** the LQR's 40 ms front
  point (0.479°).
- PPO+DR is unstable somewhere in the envelope for 8/10 seeds at the trim linearization.
- Round-3 additions are discussed under their items.

**Cumulative rule on the 768 originals.** 336 meet "every d ≤ 3 at the 63 design points",
the same 336 as the non-cumulative rule. The chosen GS design is still the optimum (return
485.06).

**Verdict:** the reviewer is right. At equal or better accuracy, an LQR keeps 20 ms that PPO
does not have, so PPO's fragility is not explained by loop gain alone. APG+DR is the one
learned controller beyond the LQR frontier.

### 2. Equilibrium vs. trim — correction (`r3/audit_facts.json`)

**The reviewer is right, and my first-round sentence was wrong.**

Raw state offsets at the closed-loop equilibrium (medians over envelope points, all seeds;
the integrator limit is 0.3 rad s):

| Controller | Integrator (rad s) | Share beyond limit, envelope / all points | Aileron | η_a | β |
|---|---|---|---|---|---|
| PPO | 0.0127 | 1.2 % / 1.1 % | 0.19° | 0.25 % | 0.03° |
| PPO+DR | 0.0259 | 0.4 % / 0.6 % | 0.49° | 0.65 % | 0.04° |
| CleanRL PPO | 0.0159 | 10.7 % / 11.0 % | 2.2° | 3.5 % | 0.06° |
| PPO, 20 ms + pending | 0.0245 | 2.6 % / 3.3 % | — | — | — |
| APG | 0.0005 | 0 | — | — | — |
| APG+DR | 0.0137 | 0 | — | — | — |
| residual | 0.0087 | 0 | — | — | — |

- Roll rate, yaw rate and the roll error are essentially zero for every row.
- The "normalized offsets 28–190" came from dividing by the tuned Bryson scale of the
  integrator, 4° s × 1/32 = 0.0022 rad s. Measured that way the integrator dominates the
  normalized offset at almost every point: 3,202 of 3,250 for PPO.

**Correct statement for the paper:** "the equilibria lie close to trim (median integrator
0.013–0.026 rad s, 4–9 % of its 0.3 rad s limit; beyond the limit at 0.4–1.2 % of points
for PPO/PPO+DR, 11 % for CleanRL-style PPO)."

### 3. Per-delay audit without duplicates (`r3/audit_facts.json`; with round-3 controllers `r3/audit_per_delay_all.json`)

`gs_wide` and `gs_dr` were dropped, leaving 60 pairs: 15 controllers × 20/40/60/80 ms.

**Ranking rule.** Any loss of control ranks above all others, ordered by the share lost,
then by RMSE growth. Pairs where every run is lost tie at the top (average rank).

**Spearman correlation** between the share of envelope points stable at the delay and the
degradation rank:

| Delay | Trim linearization | Equilibrium linearization |
|---|---|---|
| Pooled | 0.83 (p = 1e-16) | 0.84 |
| 20 ms | 0.74 (p = 0.002) | 0.78 |
| 40 ms | 0.86 | 0.89 |
| 60 ms | 0.89 | 0.88 |
| 80 ms | 0.54 (p = 0.036) | 0.54 |

**2 × 2 table** (trim linearization; identical at equilibrium):

| | ≥ 5× growth or any loss | Not |
|---|---|---|
| Stable share = 0 | 25 | 5 |
| Stable share > 0 | 3 | 27 |

- The 5 "share 0, not bad" pairs: LPV at 80 ms and no-disk LPV at every delay. All of them
  limit-cycle instead (activity 61–620×).
- The 3 "share > 0, bad" pairs: PPO at 20 ms (share 0.008), PPO+DR at 80 ms (0.21), APG+DR
  at 80 ms (0.30).

**With round-3 controllers** (128 pairs): ρ = 0.74 pooled, 0.58 / 0.70 / 0.83 / 0.49 at
20 / 40 / 60 / 80 ms; the table is 54 / 13 / 7 / 54. The new misses are the fixed-delay
learners at delays *shorter* than their training delay. They limit-cycle there with bounded
error.

### 4. Rollout traces (`r3/example_rollout_traces.npz`)

Same scenario as `example_rollout_tmlr.py`: from the "aileron 0–30 %" batch (seed 1004),
index chosen nearest 10 % effectiveness; one seed (the first file) per controller, for all
37 controllers that have one.
- Shared arrays: `t`, `phi_cmd` (deg), `eta_s_cmd` (%), `scenario_index`,
  `aileron_effect`, `v_kt`, `h_ft`.
- Per controller: `<name>/phi`, `<name>/eta_a`, `<name>/beta` (deg / %), and
  `<name>/failure_index` (−1 = no failure).

### 5. Certified residual (`r3/certificate.json`; the fix is in item 5d)

**(a) Sampled-data check.** Condition: V(x⁺) ≤ e^{−2λT} V(x) with T = 0.02 s, the true
linearization, ZOH, and the same 4,096 states and residuals per point as the
continuous-time check.

| Controller | Points where all samples hold | Share of samples holding | Worst V(x⁺)/V(x) |
|---|---|---|---|
| LPV alone | 325 / 325 | all | 0.949 |
| Unprojected residual | 0 / 325 | 0.37–0.42 | — |
| **residual_safe2** | **0 / 325** | median 0.45, min 0.39 | 2.9e6 |

For residual_safe2 every violation is at normalized |x| ≤ 4.5 (median per point 1.2).
Cause: near x = 0 the half-space a′du ≤ (1 − γ̄)b admits any du orthogonal to a, and
holding it for one sample raises V by O(T²|du|²).
**The continuous-time certificate does not hold for the 50 Hz loop.**

**(b) Corner refinement.** On a 1 kt × 250 ft × 2.5 % grid (60–70 kt, 0–2,500 ft, all span;
1,331 points), sup γ = **0.2399** at 65 kt, 0 ft, η_s = −2.5 %; 0.2323 on the half-resolution
subgrid. No point exceeds 0.281, and the per-airspeed maxima peak at 65–66 kt. It converges
below the margin.

**(c) Unprojected residual, continuous-time decay.** Per-point share of samples satisfying
it: min 0.68, median 0.71, max 0.73; 0/325 points with all samples. For comparison, the
plain projection gives 63/325 and the 2× margin 325/325.

### 5d. Discrete-time Lyapunov filter `residual_safe_dt` (lead's request; `r3/dt_filter.json`, `r3/eval_dt/`)

**Derivation** (also in the `morphjax.safety` "round 3" docstring). The frozen ZOH model
uses Φ and Γ tabulated at the 63 design points and interpolated, with the actuator states
included. With Φ_cl = Φ + ΓK_LPV, the filter enforces

V(x⁺) − ρ_d V(x) + g_d [x′Dx + du′Rdu] ≤ 0,  ρ_d = e^{−2λT} = 0.99601,
D = ρ_d P − Φ_cl′PΦ_cl,  R = Γ′PΓ.

- This is an ellipsoid in du. The policy's (clipped) du is projected onto it in closed
  form, du(ν) = (I + νH)⁻¹(du₀ − νc), with ν found by a 60-step bisection inside the jit.
- du = 0 is feasible wherever D > 0: the interpolated LPV ratio is at most 0.991 < ρ_d at all
  325 check points.
- At x = 0 only du = 0 is admitted, so the small-control property is restored.

**Robustness margin.** Built like γ̄: g = max |eig(ΔM, blkdiag(D, R))|, where ΔM is the
true-minus-interpolated quadratic form of V(x⁺) in z = (x, du). g_d = 2 × max g over the
same 1,000 estimation conditions = 2 × 0.368 = **0.735**.

Fresh conditions, all covered by g_d:

| Condition set | Max g |
|---|---|
| 2,000 random | 0.375 |
| Offset 13×5×5 grid | 0.331 |
| 1,331-point corner grid | 0.404 |
| 325-point check grid | 0.357 |

**Check.** Every sampled state and residual satisfies V(x⁺) ≤ ρ_d V(x) at all 325 points,
both on the interpolated model (worst ratio 0.943) and on the **true** linearization (worst
ratio 0.942). The filtered residual has on average 46 % of the raw residual's norm.

**Simulation** (residual_dr policies, 10 seeds, deploy-only; no retrain-with-filter run):
- Table: nominal 1.24°, fast morphing 1.20°, noise + 40 ms 1.23°, model error 1.25°,
  aileron 0–30 % 3.15°. No loss of control in any cell.
- Activity 0.32, against 0.82 for residual_safe2.
- Delay sweep (RMSE / activity): 1.20 / 0.33, 1.21 / 0.41, 1.20 / 3.41, 1.24 / 9.95 and
  5.67 / 9.42 at 0 / 20 / 40 / 60 / 80 ms. No loss of control. At 80 ms it beats
  residual_safe2 (10.95°).
- Held-out return: 480.4 nominal and 473.9 DR, vs 483.3 / 477.4 for residual_safe2.
- With no aileron it scores 5.09°, vs 5.93° for residual_safe2 and 3.64° for LPV.

**Verdict:** the paper's "certified residual" should become residual_safe_dt. It is
certified on the sampled-data frozen model, including the interpolation error on 3,656
fresh conditions. It costs 0.12° of nominal accuracy vs residual_safe2 and closes 22 % of
the LPV→PPO gap (vs 33 %).

### 6. Loop-at-a-time and disk margins at the plant input (`r3/margins.json`, `r3/margins_r3.json`)

Frozen continuous-time loops broken one channel at a time with the other two closed;
balanced disk margin α. Medians over 135 envelope points:

| Controller | Aileron α (delay margin) | Rudder α (delay margin) | Weakest channel |
|---|---|---|---|
| GS | 1.57 (129 ms) | 1.55 (106 ms) | — |
| No-margin LQR | 1.17 (50 ms) | 1.18 (33 ms) | — |
| No-margin LQR, wide grid | 0.93 (13 ms) | 1.07 (13 ms) | — |
| LPV | 1.21 (upper GM 4.3) | 1.33 (126 ms) | aileron at all points |
| **PPO** | **0.55** (18.5 ms) | 0.94 (27 ms) | **aileron, 10/10 seeds** |
| **APG** | 0.62 (18.4 ms) | **0.51** (24 ms) | **rudder by disk margin, 10/10 seeds** |
| PPO+DR | 0.85 | 2.0 | η_a in 9/10 seeds (α 0.35) |
| APG+DR | 0.78 | — | aileron or η_a |
| Residual | 0.35 | — | aileron |
| CleanRL PPO | 0.43 (12.7 ms) | — | — (stable at only 0–133 points per seed) |
| PPO entropy 0.01 / log-std 0 / rate ×100 | 0.54 / 0.56 / 0.48 | — | aileron |
| APG smooth | — | 0.80 | rudder |
| Delay-randomized APG | 0.85 (71 ms) | — | aileron |

The η_a loop has α ≈ 2 (very large margin) for GS and the no-margin LQR.

**Answer:** the PPO family breaks first in the **aileron** loop, with an 18 ms per-loop delay
margin, which matches the 20 ms cliff. For APG the **rudder** loop has the smallest disk
margin, but the aileron loop has the smaller delay margin (18 ms).

## P2 — new training and evaluation

Seeds: 5 each unless noted; same budget and hyperparameters as the main learners. All
numbers come from `r3/metrics.json`.

### 7. Fixed 40 / 60 ms training with all pending commands observed

| Controller | 0 ms | 20 ms | 40 ms | 60 ms | 80 ms |
|---|---|---|---|---|---|
| PPO, 40 ms | 0.56 (act 162) | 1.08 (act 127) | **0.48** (act 1.3) | 3.28 | 11.1 (3 % lost) |
| APG, 40 ms | 0.56 (act 92) | 0.94 | **0.25** (act 1.0) | 12.9 | 20.3 (16 % lost) |
| PPO, 60 ms | 1.02 (act 239) | 0.96 | 1.07 | **0.60** (act 1.8) | 3.62 |
| APG, 60 ms | 0.79 (act 67) | 0.78 | 0.55 | **0.31** (act 1.8) | 19.0 (30 % lost) |

- In the linear audit (per-lag gains), each is stable **only at its training delay**: PPO
  91 % / 76 % and APG 99 % / 98 % of envelope points for the 40 / 60 ms variants, and 0 %
  at every other delay.

**Verdict:** a learner trained at d samples is excellent at d (APG 0.25° at 40 ms and 0.31°
at 60 ms, vs GS 1.04 / 1.08°). It limit-cycles below d and falls off the cliff at d + 1.
The cliff follows the training distribution.

### 8. Delay-only randomization (uniform 0–60 ms), without and with the command buffer

The buffer variant (`ddrh`) observes the last 3 commands. Rows the current delay has
already applied are zeroed, so the controller is assumed to know its own latency; that is
what makes the problem Markov.

| Controller | 0 ms | 20 ms | 40 ms | 60 ms | 80 ms | Aileron 0–30 % | Noise + 40 ms |
|---|---|---|---|---|---|---|---|
| PPO ddr | 0.43 | 0.44 | 0.46 | 0.55 | 1.97 | **3.46 (19 % lost)** | — |
| APG ddr | 0.37 | 0.39 | 0.40 | 0.52 | 3.76 (39 % lost) | 3.34 | — |
| PPO ddrh | 0.37 | 0.40 | 0.47 | 0.60 | 6.14 (act 90) | 1.70 | — |
| **APG ddrh** | **0.23** | 0.22 | 0.26 | 0.34 | 1.93 (99 % lost) | 0.64 (32 % lost) | 0.51 (10 % lost) |

Other figures:
- Activity at 0 ms is 0.16–0.28.
- Held-out nominal return 493.8–495.6.
- Audit: APG ddrh is stable at 96–100 % of envelope points for 0–60 ms and 0 % at 80 ms;
  APG ddr is stable at 100 / 100 / 92 / 67 / 36 % at 0–80 ms.

**Verdict:** with delay randomized and pending commands known, APG keeps nominal-level
accuracy (0.23°) through 60 ms. It beats GS at every delay up to 60 ms by 3–5×. It is
fragile only outside what it was trained on: weak aileron, sensor noise, and 80 ms.

### 9. Learned linear gain-scheduled policy (APG, `diffsim_lin`)

The policy is u = u0 + K(ρ)x, with K multilinear on the 63-point grid, starting from zero.
- Nominal 0.58°, fast morphing 0.56°, held-out return 492.4.
- Delay: 9.7° at 20 ms, 0 % lost; 76 % lost at 60 ms and 97 % at 80 ms.
- Aileron 0–30 % 2.70°.
- Frozen-unstable at every envelope point: the slowest mode has a median real part of
  +0.19 s⁻¹, a 5 s time constant, which the 15 s episodes do not expose.
- On the frontier: 0.58° with cumulative margin −1. It is **dominated** by every LQR
  front point down to 40 ms.

**Verdict:** the learner's objective with the LQR's function class reproduces the same
no-margin trade, and is less accurate. Fragility comes from the objective, not from the
network.

### 10. Actuator rate limits (evaluation only; `r3/ratelimit/`)

**Values.** Surfaces 100 °/s. Each wingtip 25 %/s: full span travel in 1 s, below the
0.5 Hz actuator's 39 %/s peak on a 12.5 % step. GTM DesignSim's actuator rate limits are not
in the downloaded aero tables, so I used the suggested typical value. Commercial small-UAV
servos are faster unloaded (300–600 °/s), so this is conservative.

Delay sweep, RMSE and loss of control (unlimited → rate-limited). Activity is in the JSON.

| Controller | 0 ms | 20 ms | 40 ms | 60 ms | 80 ms |
|---|---|---|---|---|---|
| GS | 1.01 → 1.01 | — | 1.04 → 1.07 | 1.08 → 3.21 (30 % lost) | 1.90 → 5.02 (94 %) |
| LPV | — | — | — | 1.60 → 3.17 (1 %) | 2.35 → 5.66 (31 %) |
| No-margin LQR | — | 18 % lost | 100 % lost | — | — |
| PPO | 10 % lost | 97 % lost | — | — | — |
| APG | **41 % lost** | 100 % lost | — | — | — |
| PPO+DR | — | — | 0.50 → 0.66 | 0.66 → 18.5 (88 %) | — |
| APG+DR | — | — | 0.43 → 0.47 | 0.68 → 6.73 (98 %) | — |
| Certified residual (safe2) | — | — | 1.03 → 1.73 | 1.12 → 12.4 (98 %) | — |
| PPO, 20 ms + pending | 75 % lost | 0.42 → 0.78 (10 %) | — | — | — |

**Verdict:** rate limits move every cliff earlier. The ordering is unchanged:
model-based ≫ DR learners ≫ nominal learners. The nominal learners now also lose control
without delay.

### 11. LPV

**(a) Pointwise X and no disk.** Nominal 1.04°, aileron 0–30 % 2.60°, delay sweep up to
60 ms 1.00–1.06° (2.04° at 80 ms), cumulative margin 60 ms at the envelope points. It
behaves like pointwise X with the disk (1.04°), so removing the disk matters only together
with a common P.

**(b) Tuned multipliers** (`r3/lpv_tuned.json`), lead's question answered:
- **No** LPV design on the 3⁵ grid meets the cumulative 0–60 ms rule, including the original
  weights.
- 203 of 243 are feasible. Best cumulative delay: **40 ms** (76 designs), 20 ms (93), 0 ms (34).
- The original weights also reach 40 ms.
- Best under a 40 ms rule (φ 0.5, integ 1/32, β 1/8, η_a 1, effort 1/8):
  - Tuning return 477.61 (original 476.15).
  - **Held-out return 475.93 nominal / 465.99 DR**, vs 474.38 / 466.54 for the original.
  - In the evaluation: nominal 1.39° (original 1.50°), aileron 0–30 % 3.27° (original 2.74°).

So the LPV price is not an untuned-weights artefact: tuning recovers 0.1° and loses some of
the weak-aileron advantage.

**(c) Parameter-dependent X(ρ)** (`r3/lpv_pd.json`, `lpv_pd`). Affine in V_syn and η_s, with
|V̇| ≤ 5 kt/s and |η̇_s| ≤ 80 %/s (the 0.5 Hz actuator's peak rate on a full step).
- Certificate holds: max eigenvalue of the derivative −1.03. Cost bound 537, vs 1,924 with a
  common X.
- With 40 %/s the solver failed numerically.
- Nominal **1.01°** (GS 1.06°), aileron 0–30 % **2.28°** (best of all designs), held-out
  return 484.2.
- But the linear delay margin is only 20 ms: 1.83° at 60 ms, and 9 % of runs lost at 80 ms.

The price of the common P is mostly the *constant* P: a rate-bounded parameter-dependent P
recovers GS accuracy and keeps the weak-aileron benefit, but not the delay margin.

### 12. Reward that prices smoothness (action-rate weight × 1000, 5 seeds)

**Why × 1000.** At × 1000, PPO's observed chattering (0.51 limits/s) would cost about 1
return unit per episode. That is the size of the PPO vs no-margin-LQR differences. × 100
costs only about 0.1.

| Controller | Nominal | Activity | 20 ms | 40 ms | 80 ms | Held-out return |
|---|---|---|---|---|---|---|
| **PPO, smooth** | 0.79° | 0.12 | — | — | — | 162 |
| APG, smooth | 0.29° | **0.09** | 1.33° (act 20.9) | 8.3° | 22° (13 % lost) | 495.2 |

PPO with the smoothness price **fails**: it loses control in 80 % of nominal runs. The
penalty dominates the advantage scale and the policy collapses.

**Verdict:** pricing smoothness removes the nominal chattering (APG has the lowest activity
of all controllers) but not the delay cliff.

### 13. PPO sensitivity (3 seeds each)

| Setting | Nominal | Activity | 20 ms error growth |
|---|---|---|---|
| Initial log-std 0 | 0.32° | 0.66 | 25.7× |
| Entropy coefficient 0.01 | 0.30° | 0.57 | 23.8× |
| Action-rate weight × 100 | 1.38° | 0.21 | 6.0×, 28 % lost at 20 ms |
| Paper PPO (10 seeds) | 0.36° | 0.51 | 18× |

No setting removes the chattering or the 20 ms cliff (F3 and F7 hold). The no-margin LQR
still matches these PPOs in accuracy within 0.06° (F4 holds).

### 14. Learning curves, 10 seeds (`r3/learning_curves_km.json`)

Kaplan–Meier medians: seeds that never reach a threshold are censored. 95 % intervals are
seed bootstraps. Wall-clock excludes JIT compile, which is 29 s for PPO and 66–76 s for APG.

| Threshold | PPO env steps | APG env steps | Ratio PPO/APG, steps | Ratio PPO/APG, wall-clock |
|---|---|---|---|---|
| GS-LQR return | 0.352 M | 0.128 M | **2.75×** [2.75, 2.75] | 1.71× [1.39, 2.17] |
| No-margin LQR return | 1.57 M | 0.288 M | **5.44×** [5.22, 5.56] | 3.36× [3.03, 3.84] |
| PPO's final return (KM median) | 2.82 M | 0.416 M | **6.8×** [6.0, 8.5] | 4.4× [3.8, 5.2] |
| 99 % of own final | 0.74 M | 0.19 M | **3.8×** [3.1, 4.3] | 2.3× [1.9, 2.9] |

At PPO's final return, not every PPO seed reaches it; the KM median accounts for that.

**F8 should read "2.75–6.8× fewer samples depending on the threshold".** With compile time
included, PPO is still faster in wall-clock to the GS-LQR threshold.

## Verdicts per item

| Item | Verdict |
|---|---|
| 1 Frontier | PPO is dominated by a 20 ms-margin LQR; APG+DR lies beyond the LQR frontier at 40 ms. Held out from selection: unchanged. GS is still the cumulative-rule optimum (336/768 qualify). |
| 2 Equilibrium | My first-round statement was wrong; corrected numbers above. |
| 3 Per-delay audit | ρ = 0.74–0.89 at 20–60 ms, 0.54 at 80 ms; table 25/5/3/27; misses explained by limit cycles. |
| 4 Traces | Saved. |
| 5 Certified residual | Continuous-time certificate **fails** on the 50 Hz ZOH model; γ sup 0.240 < 0.281 confirmed; unprojected residual 0.68–0.73. **Fixed by residual_safe_dt** (passes on interpolated and true models, margin validated on 3,656 fresh points). |
| 6 Margins | Aileron loop breaks first for PPO (α 0.55, 18 ms); rudder has the smallest disk margin for APG. |
| 7 40/60 ms training | Cliff moves with the training delay. |
| 8 Delay-only DR | APG with the masked buffer flies 0.23–0.34° from 0 to 60 ms; fails outside its randomization. |
| 9 Linear APG | Same no-margin trade, frozen-unstable, dominated by the LQR. |
| 10 Rate limits | All cliffs move earlier; ordering unchanged; nominal learners lose control even at 0 ms. |
| 11 LPV | No LPV meets 60 ms (best 40 ms); tuning gains 0.1°; P(ρ) recovers GS accuracy but only 20 ms of margin. |
| 12 Smoothness | APG: activity down, cliff stays; PPO collapses. |
| 13 PPO sensitivity | Chattering and cliff persist under every setting. |
| 14 F8 | 2.75–6.8× by threshold, with intervals. |

## Skipped or partial

- **Item 5d:** the optional retrain-with-filter run was not done; the filter is
  deploy-only on the existing residual_dr policies.
- **Item 11c:** the 40 %/s rate bound failed in Clarabel (numerical); only the 80 %/s
  design is reported.
- **Item 6:** margins are not computed for policies that observe their pending commands
  (the law has memory). Their per-lag discrete audit is in `r3/audit.json`.
- **PPO smoothness (item 12):** the collapse at × 1000 is reported as is. I did not tune the
  weight further (there was no non-evaluation criterion to tune it on).
- **Correction to the first-round `lqr_wide.json` note:** nothing in `results/tmlr/` was
  changed. The bug in the `lpv_tuned.json` count label was fixed in place: the 76 is the
  40 ms count; 0 meet 60 ms.
