# Revision experiments (TMLR), branch `tmlr-revision`

Everything here is new; nothing committed at `bbafe40` was changed or removed. The paper's
current numbers (from `results/`) stay reproducible: policies saved before the revision load
with their original ±25° action scale, and the original evaluation path is unchanged.
All new results are in **`results/tmlr/`**. The plant and aerodynamic model are unchanged.

Machine: Windows 11, i5-13600KF (20 logical CPUs), CPU only, JAX 0.12. Training ran from a
frozen worktree (`scripts/run_queue.py`, 4 slots × 5 pinned CPUs). CPU times below are the
process CPU time of the updates only (compile and held-out evaluation excluded).

## What changed in the code

| Commit | Change |
|---|---|
| `2a31261` | **Identical actuation**: learners command ±30° aileron/rudder and ±12.5 % asymmetric span around trim (were ±25°), stored in each policy's config. **APG schedule fix**: cosine decay spans `iters × windows` optimizer steps (it reached its floor at iteration 40). **Latency-aware learners**: fixed training delay + pending commands in the observation (`--variant lat`). Deterministic held-out learning curves, compile time timed separately. |
| `0b6bca8` | CleanRL-style PPO option (`--method ppo_clean`), LPV ablations, wide LQR search, revision evaluation (`--revision`), audit, margin validation, LOC causes, learning-curve analysis. |
| `68fbfb6`, `e48b648`, `bc1d6f3` | `--eval-every`; finalize the LQR search from its log; `evaluate.py --only` + `merge_evaluations.py`. |

The training reward, network, scenarios and evaluation protocol are unchanged. Evaluation uses
`evaluate.py` exactly as before (256 paired scenarios per table condition, 128 per sweep point,
same seeds); only the returns are scored with the ±30° action scale (it enters only the tiny
action penalties). Nothing was tuned on evaluation scenarios.

## Files for the paper (`results/tmlr/`)

`scripts/snapshot.py` needs no code change if it is pointed at these paths instead of
`results/…` (same formats as before):

| File | Content |
|---|---|
| `metrics.json`, `per_scenario.npz` | Full evaluation, 21 controllers (merged from `eval_main/` and `eval_clean/`). Same keys as before plus `meta.training_distribution_return_per_seed`, `meta.heldout_return.{nominal,dr}` (seed 1007, 256 scenarios each). |
| `verification.json`, `verification_points.npz` | `verify_designs.py` on the revision policies (history policies skipped there). |
| `audit.json`, `audit_points.npz` | New audit: per controller, stability / delay tolerance (cumulative `envelope_tolerates_ms` and per-delay `envelope_stable_at_ms`) at trim (`x0`) and at the closed-loop equilibrium (`eq`), certificates, and `pairs` (all controller × delay pairs with Spearman/Kendall). |
| `lqr_wide.json` | Bracketed LQR search: rounds, optima, edge positions, held-out returns; `gs_wide`, `gs_hg_wide`, `gs_dr` weights. |
| `margin_validation.json/.npz` | γ on 2,000 random + offset 13×5×5 held-out conditions. |
| `learning_curves.json`, `learning_curves_fine.json` | Steps and wall-clock to thresholds (fine = evaluation after every update, seeds 0–4). |
| `loc_reasons.json` | First departure criterion in the 60 / 80 ms runs. |
| `claims.json` | Numbers per claim with seed + scenario bootstrap intervals (`scripts/revision_report.py`). |
| `policies/` | 10 seeds each: `ppo_nominal`, `ppo_dr`, `diffsim_nominal`, `diffsim_dr`, `residual_dr`, `residual_proj_dr`, `ppo_lat`, `diffsim_lat`, `ppo_clean_nominal`. JSON `history` now has `eval_return` rows and an `update: -1` row (untrained) without `return`; `timing` has `compile_s`. snapshot.py's learning-curve reader must skip rows without `return` (or use `eval_return`). |
| `logs/`, `jobs/` | Every job log and queue. |

New controller names: `ppo_lat`, `diffsim_lat` (20 ms + pending command), `ppo_clean`,
`gs_wide`, `gs_hg_wide`, `gs_dr` (wide-grid / DR-tuned GS LQR), `lpv_pw` (pointwise X),
`lpv_nodisk` (no pole disk). The original `fixed`, `gs`, `gs_hg`, `lpv` keep the committed
weights, so the LPV law, the residual base and the safety filter are identical to `bbafe40`.

## What was run

| Experiment | Seeds / size | Compute |
|---|---|---|
| P1 retraining, identical actuation + APG fix: PPO, PPO+DR, APG, APG+DR, residual (DR), residual projected in training | 0–9 each | PPO-type 7.2–7.7 CPU-h, APG 3.3 CPU-h per 10 seeds; ≈ 4.4 h wall for all 80 jobs incl. latency |
| P1 latency-aware PPO and APG (fixed 20 ms, pending command observed) | 0–9 each | 7.6 / 3.5 CPU-h |
| P2 PPO sanity variant (`rl.CLEANRL`) | 0–9 | 11.9 CPU-h |
| P2 LQR bracketing (3 rounds, 21,870 candidates) + held-out scoring | — | 4.1 h wall |
| P2 LPV ablations, margin validation | — | minutes |
| P3 fine learning curves (PPO, APG nominal) | 0–4 (bitwise the same policies) | 5.4 CPU-h |
| Evaluation of all 21 controllers | 10 seeds per learner | 2.3 h wall |

## Key numbers (10 seeds; brackets = 95 % seed + scenario bootstrap)

Roll RMSE (°), share of runs lost in parentheses where > 0; activity = limit-normalized command
travel per second (fast morphing, 0 ms).

| Controller | Frozen | Fast | +noise, 40 ms | Model error | Aileron 0–30 % | Activity | 20 ms | 40 ms | 60 ms | 80 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| GS LQR (60 ms) | 1.06 | 1.02 | 1.07 | 1.06 | 2.64 | 0.13 | 1.03 | 1.04 | 1.08 | 1.90 |
| GS LQR, wide grid (= DR-tuned) | 1.06 | 1.02 | 1.07 | 1.06 | 2.64 | 0.13 | 1.03 | 1.04 | 1.08 | 1.89 |
| No-margin LQR | 0.36 | 0.35 | 1.91 | 0.36 | 2.13 (5.1 %) | 0.15 | 0.35 | 1.80 | lost | lost |
| No-margin LQR, wide grid | 0.06 | 0.06 | lost (100 %) | 0.06 | 20.1 (62 %) | 0.27 | lost | lost | lost | lost |
| LPV | 1.50 | 1.45 | 1.55 | 1.51 | 2.74 | 0.17 | 1.49 | 1.53 | 1.60 | 2.35 |
| LPV, pointwise X (ablation) | 1.04 | 1.00 | 1.05 | 1.03 | 2.93 | 0.12 | 1.00 | 1.02 | 1.06 | 3.43 |
| LPV, no pole disk (ablation) | 1.21 | 1.17 | 1.96 | 1.22 | 2.34 | 0.13 | 1.39 | 1.98 | 2.83 | 4.74 |
| PPO | 0.36 | 0.35 | 11.7 | 0.36 | 2.54 (3.5 %) | 0.51 | 6.39 | 11.4 | 19.6 (16 %) | 30.2 (80 %) |
| PPO, CleanRL-style | 0.33 | 0.33 | 12.4 | 0.33 | 3.45 (29 %) | 0.83 | 6.93 | 12.6 | 16.2 | 20.9 (4 %) |
| PPO + DR | 0.48 | 0.47 | 0.55 | 0.48 | 2.50 (0.1 %) | 0.16 | 0.48 | 0.50 | 0.66 | 6.19 (2 %) |
| APG | 0.21 | 0.20 | 17.8 | 0.21 | 2.69 (1.0 %) | 0.25 | 9.44 | 17.8 | 26.5 (62 %) | 24.3 (92 %) |
| APG + DR | 0.40 | 0.39 | 0.50 | 0.40 | 2.34 | 0.15 | 0.40 | 0.43 | 0.68 | 9.38 (1 %) |
| PPO, 20 ms + pending cmd | 0.64 | 0.62 | 8.48 | 0.65 | 2.45 (0.9 %) | **182** | 0.42 | 8.20 | 14.8 (3 %) | 28.5 (53 %) |
| APG, 20 ms + pending cmd | 0.63 | 0.64 | 12.2 | 0.65 | 3.24 (1.0 %) | **195** | 0.22 | 12.1 | 21.9 (52 %) | 23.0 (89 %) |
| LPV + residual (DR) | 0.96 | 0.92 | 1.02 | 0.98 | 3.54 | 0.47 | 0.93 | 0.96 | 1.26 | 11.4 |
| …projected at run time | 1.17 | 1.13 | 1.15 | 1.19 | 3.50 | 0.61 | 1.16 | 1.14 | 1.37 | 11.2 |
| …projected in training | 1.14 | 1.11 | 1.12 | 1.17 | 3.51 | 0.53 | 1.14 | 1.11 | 1.18 | 11.4 |
| …with margin (1.5×) | 1.11 | 1.07 | 1.06 | 1.14 | 3.41 | 0.72 | 1.06 | 1.02 | 1.12 | 11.2 |
| …with 2× margin (certified) | 1.12 | 1.08 | 1.06 | 1.14 | 3.36 | 0.82 | 1.07 | 1.03 | 1.12 | 11.0 |

Delay columns are the fast-morphing delay sweep (128 scenarios × seeds). "lost" = every run
lost control. The latency-aware policies' activity at 0 ms (182/195) is a limit cycle: they
are designed for one sample of delay; at 20 ms their activity is 0.57 / 0.37.

Activity under delay (sweep, 0 → 20 → 40 → 60 → 80 ms): GS 0.13 → 0.14 → 0.28 → 1.04 → 4.0;
LPV 0.16 → 0.20 → 1.74 → 8.6 → 9.95; certified residual 0.83 → 1.25 → 3.15 → 9.0 → 9.65;
PPO+DR 0.16 → 0.18 → 0.38 → 1.56 → 5.7; APG+DR 0.15 → 0.18 → 0.36 → 1.40 → 7.2.

**Returns** (PPO reward, ±30° scale; tuning scenarios / held-out nominal / held-out DR):
GS 485.07 / 483.95 / 480.03; no-margin LQR 494.81 / 494.51 / 351.2; no-margin wide 496.33 /
496.13 / 134.3; PPO 494.96 [seeds 494.68–495.16] / 494.69 / 247.4; APG 496.01 / 495.79 / 198.5;
PPO CleanRL 493.21 [481.6–495.6] / 492.99 / 234.5; PPO+DR 493.69 / 493.27 / 490.65; APG+DR
494.57 / 494.21 / 491.61; LPV 476.15 / 474.38 / 466.54.

## Per experiment

**P1 identical actuation.** With ±30° the nominal learners are about as accurate as before
(PPO 0.36° vs 0.35°; APG 0.21° vs 0.23°) but worse with a weak aileron: at 0–30 %
aileron PPO is 2.54° with 3.5 % loss of control (was 1.81°, 0.3 %), APG 2.69° with 1.0 % (was
1.73°, 0.3 %). The paper's "learned policies handle a degraded aileron better" (README #5)
**no longer holds** for the nominal learners: GS LQR is 2.64° with no loss of control. With the
wider range they also leave more of the envelope frozen-unstable at trim (PPO+DR 60 % of
envelope points stable at 20 ms vs 78 % before).

**P1 10 seeds.** Seed spread: PPO seed means 0.30–0.41°, APG 0.208–0.210°, PPO+DR 0.46–0.50°,
APG+DR 0.39–0.41°, residual 0.92–0.99°, certified residual 1.11–1.15°. No-margin LQR − PPO =
+0.000° [−0.020, +0.021] (seed+scenario bootstrap); seed t-interval [−0.023, +0.023] (n = 10),
so "within 0.1°" is now supported.

**P1 latency-aware learners** (fixed 20 ms, pending command observed). At their design delay
they are the most accurate controllers in the study: 20 ms sweep PPO 0.42°, APG 0.22° (GS LQR
1.03°), activity 0.57 / 0.37. But the cliff is only moved by one sample: at 40 ms 8.2° /
12.1°, at 80 ms 53 % / 89 % lost; at 0 ms they limit-cycle (activity ≈ 190) and lose 0.9–1.0 %
with a weak aileron. Linear audit (per-delay stability, envelope points, 10 seeds): stable at
20 ms at 96 % (PPO) / 100 % (APG) of points, at 0 ms and ≥ 40 ms at 0 %.

**P2 PPO sanity variant (CleanRL-style, 10 seeds).** Final held-out training return 492.8
(481.4–495.1; our PPO 495.05), nominal RMSE 0.33° [0.20, 0.51] with a wide seed spread
(0.15–1.06°), activity **0.83** (our PPO 0.51, LQR 0.13), 20 ms ×21 error growth, 40 ms+noise
12.4°, and **29 % loss of control at 0–30 % aileron**. It loses only 4 % at 80 ms (vs 80 % for
our PPO) but at 6.9–20.9° error. Linear audit: frozen-stable at 0–99 % of points depending on
seed (30 % pooled), 0 % tolerate 20 ms. Final PPO exploration std (action units): revision
PPO 0.042–0.044, committed PPO (`bbafe40`) 0.044–0.046, CleanRL-style 0.032–0.036.

**P2 LQR bracketing** (`lqr_wide.json`). Started from the original grid extended one step
beyond every edge with an optimum; extended an edge while an optimum sat on it and gained
≥ 0.05. 60 ms-constrained optimum: 485.088 → 485.093 → 485.093 (stopped improving after round
1; still on the low edge of φ, β, effort — a flat ridge along which these weights scale
together); **the DR-tuned constrained optimum is the same design** (`gs_dr = gs_wide`).
Held-out: 483.98 vs 483.95 for the original GS LQR — identical within the SE (0.60). The
no-margin optimum kept drifting to higher loop gain (495.98 → 496.28 → 496.33, still on edges
in round 2; search stopped there for time and finalized from the log, so `lqr_wide.json` has
no full candidate list). That design (`gs_hg_wide`: φ 8, integ 1/256, β 1/16, η_a 1/16,
effort 1) flies 0.06° nominal and returns 496.13 held-out — above PPO (494.69) and APG
(495.79) — but is frozen-unstable at 26 of 325 check points, loses every run at 20 ms and
183/256 held-out DR runs.

**P2 LPV ablations.** Nominal: LPV/GS 1.41× [1.40, 1.42]; pointwise-X LPV/GS 0.98× [0.98, 0.98];
no-disk LPV/GS 1.14×. So the 41 % cost is the common P (LPV/pointwise 1.45×); the pole disk
accounts for part of it (LPV/no-disk 1.24×). With no aileron, LPV − GS = −0.41° [−0.55, −0.26],
LPV − pointwise = −1.49° [−1.75, −1.24], but LPV − no-disk = +0.36° [+0.22, +0.53]: the
common P buys the no-aileron robustness and the disk costs some of it. Dense-grid audit: LPV
certified 325/325 (own P); pointwise 8/325 with the LPV P, tolerates 60 ms at 100 % of envelope
points (LPV 27 %); no-disk certified 325/325 with its own P (214/325 with the LPV P) but
tolerates 0 % at 20 ms; in simulation it degrades modestly (1.39° at 20 ms) while its
activity jumps from 0.13 to 78 (actuator limit cycle).

**P2 margin validation** (`margin_validation.json`). γ on 2,000 new random conditions (1,500
interior, 400 faces, 92 edges, 8 corners): max 0.240 (median 0.013); offset 13×5×5 grid: max
0.213. γ̄ = 0.28 (2×) covers 2,000/2,000 and 325/325; γ̄ = 0.21 (1.5×) misses 1 and 1. 41 of
2,000 points exceed the 1,000-point estimation maximum (0.141). γ peaks at 63–67 kt near sea
level, between the 60 and 70 kt design airspeeds (below the 70 kt training envelope).

**P3 learning curves** (`learning_curves_fine.json`, deterministic mean-action policy on a fixed
held-out batch after every update, seeds 0–4, medians; wall-clock excludes JIT compile, which is
29 s for PPO and 76 s for APG; all runs shared the CPU four ways, so wall-clock is comparative
only). Steps to threshold, APG vs PPO: GS-LQR return (481.96 on that batch): 0.128 M vs 0.352 M
(**2.75×**, wall 12 s vs 23 s); no-margin LQR return (493.84): 0.29 M vs 1.57 M (**5.4×**, 27 s vs
95 s); PPO's mean final return (495.02): 0.42 M vs 2.50 M (**6.0×**, 39 s vs 155 s; 4/5 PPO seeds
reach it); 99 % of own final: 0.19 M vs 0.70 M (3.7×). With compile time added, PPO is faster to
the GS-LQR threshold (52 s vs 88 s) and APG is faster to the higher ones (103 s vs 124 s;
115 s vs 184 s).

**P3 audit** (`audit.json`). All 68 controller × delay pairs (20–80 ms; 17 controllers with a
linearization): Spearman between the envelope share stable at that delay and the simulated
degradation rank ρ = 0.85 (x0) / 0.86 (equilibrium), p < 1e-19; Kendall τ = 0.69; restricted to
intermediate shares (0 < share < 1) ρ = 0.77 (n = 19) / 0.75 (n = 21). Against activity growth
ρ = 0.49. Extremes: all 19 pairs with share 1 grow ≤ 12 % in RMSE (max APG+DR at 40 ms); of 30
pairs with share 0, 25 grow ≥ 5× or lose control, the other 5 (LPV at 80 ms, no-disk LPV at
every delay) grow only 1.2–4.0× in RMSE but 61–620× in activity (limit cycles). Equilibrium
linearization: Newton on the linearized plant under the clipped policy converges at 71–100 % of
points; shares change by ≤ 4 points vs x0 for every controller. For PPO, CleanRL-PPO and the
latency-aware PPO the equilibrium sits far from trim (median normalized offset 28–190, mostly
in the roll-error integrator, far beyond the simulator's integrator limit), so it is not a
physically reachable operating point; for APG it is close to trim (≤ 1.3) and gives identical
results. Two PPO seeds tolerate 20 ms at a few envelope points (0.8 % of seed-points); no APG
seed does.

**P3 loss-of-control criterion** (`loc_reasons.json`, first departure, several can fire): at
80 ms PPO 1,019/1,280 lost — bank > 85° 578, α > 25° 179, sideslip > 25° 170, α < −10° 97; APG
1,182 — α < −10° 486, sideslip 363, bank 227, α > 25° 111; no-margin LQR 128/128 — bank 95,
α < −10° 30; DR learners: bank only (27 and 8 runs). Airspeed and non-finite never fire.

## Verdict on the paper's claims

| # | Verdict | Revised number |
|---|---|---|
| F1 Morphing speed does not change lateral tracking | **Holds** | Frozen vs fast: GS 1.06 / 1.02°, PPO 0.36 / 0.35°, APG 0.21 / 0.20°, every controller within 0.04°. |
| F2 Learned policies 3–5× more accurate without delay | **Holds (APG ratio grows)** | GS/PPO 2.97× [2.82, 3.15], GS/APG 5.09× [5.04, 5.14] (was 4.7×; APG 0.21°). CleanRL PPO 3.2× [2.1, 5.3]. Latency-aware learners only 1.7× at 0 ms (they limit-cycle there). |
| F3 Cliff at one sample of delay; the audit predicts it | **Holds, qualified** | 20 ms: PPO ×18, APG ×46; 80 ms lost 80 % / 92 %. Two PPO seeds tolerate 20 ms at a few envelope points, so "no seed at any point" must become "≤ 0.8 % of points". Audit over all 68 pairs: Spearman 0.85 (0.77 on intermediate shares). "Zero share ⇒ ≥ 5× error" fails for 5/30 pairs (LPV 80 ms, no-disk LPV), which instead limit-cycle (61–620× activity). New: training with the delay moves the cliff by exactly one sample (20 ms: 0.42° / 0.22°; 40 ms: 8.2° / 12.1°) — the cliff belongs to the training distribution, not to learning. |
| F4 PPO's nominal edge is a gain-for-margin trade | **Holds, and is stronger** | No-margin LQR − PPO = +0.000° [−0.020, +0.021] (10 seeds), return 494.81 vs 494.96 (held-out 494.51 vs 494.69), activity 3.4× lower; 1.80° at 40 ms, all lost at 60 ms. On the widened grid the no-margin optimum keeps buying accuracy with gain (0.06°, return 496.33 > PPO and APG) while losing every run at 20 ms. Holds also for CleanRL-PPO (+0.03° [−0.16, +0.16]). |
| F5 DR buys robustness back, inside its range only | **Holds; APG cost changes** | Nominal cost PPO+DR/PPO 1.34× [1.27, 1.43], APG+DR/APG 1.91× [1.87, 1.94] (paper: ×1.4 [1.2, 1.8] for both). At 40 ms + noise APG+DR 0.50°, PPO+DR 0.55° are best. 80 ms: 9.38° (1 % lost) / 6.19° (2 % lost) vs GS 1.90° — DR learners now lose a few runs at 80 ms (were 0 %). |
| F6 LPV certificate costs 41 % and pays off only with no aileron | **Holds; attribution now isolated** | LPV/GS 1.41× [1.40, 1.42]; 0 % aileron −0.41° [−0.55, −0.26]. Ablations: pointwise X (no common P) is as accurate as GS (0.98×) and worse with no aileron (5.14° vs 3.64°) → both the cost and the payoff come from the common P; removing the pole disk cuts the cost to 1.14× and *improves* the no-aileron case (3.28°) but destroys delay tolerance (limit cycle from 20 ms). |
| F7 RMS error hides chattering | **Holds, stronger** | Activity PPO 0.51, CleanRL-PPO 0.83, APG 0.25 vs GS 0.13 and no-margin LQR 0.15 at PPO's accuracy. Action-rate penalty is in the reward for all; reword "although (2) penalizes". |
| F8 APG reaches PPO's return with 14× fewer samples | **Changes** | Deterministic curves, fixed APG schedule: 6.0× fewer steps to PPO's final return (0.42 M vs 2.50 M), 5.4× to the no-margin LQR return, 2.75× to the GS-LQR return; wall-clock (compile excluded) 3.9× / 3.5× / 1.9× faster; with compile (76 s vs 29 s) PPO is faster to the GS-LQR threshold. "14×" should become "3–6× depending on the threshold". "Makes the same trade" holds (20 ms ×46). |
| F9 Residual keeps the certificate only when projected with a margin | **Holds** | Residual 0.96° (48 % of the LPV→PPO gap; 0/325 certified); certified residual 1.12° (33 % of the gap), no loss of control, 1.03–1.12° up to 60 ms, 11.0° at 80 ms. Margin validated on 2,325 fresh points: 2× covers all (max γ 0.240 < 0.28); 1.5× would miss 2. Its activity grows 0.83 → 9.0 at 60 ms, so "flat to 60 ms" holds in RMSE only. |
| README #5 learned policies handle a degraded aileron better | **Fails** with identical actuation | 0–30 % aileron: PPO 2.54° (3.5 % lost), APG 2.69° (1.0 %) vs GS 2.64° (0 %). |

## Not done or partial

- **LQR bracketing**: stopped after round 2 (21,870 candidates); the no-margin optimum was still
  moving to higher gain (+0.05 in the last round, the stopping threshold). The final JSON was
  rebuilt from the per-round log, so the candidate list / top-5 are missing for that run.
- **Fine learning curves** only for PPO and APG nominal, seeds 0–4 (same policies as seeds 0–4
  of the main runs, verified bitwise); the 10-seed runs have curves every 10 updates (PPO) /
  5 iterations (APG).
- **Equilibrium linearization** is on the plant linearized at trim; where the equilibrium is far
  from trim (PPO-family, see above) it is not physically meaningful.
- Not in scope / not run: SAC/TD-MPC2/MPC/predictor-LQR baselines, reward-weight sweep, own
  common-P search per learned policy, discrete-time decrease check, per-policy figures.
