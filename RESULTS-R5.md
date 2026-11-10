# Round-5 experiments (TMLR), branch `tmlr-revision` — compact

Outputs are in `results/tmlr/r5/`. Earlier results are unchanged.
- `r5/metrics.json` and `r5/per_scenario.npz`: 52 controllers on the standard protocol.
- `r5/rate100/`: everyone with actuator rate limits.
- `r5/stats.json`, `r5/claims.json`: statistics.

**Compute.** Training: 121 jobs (search 72, rate-limited 20, smoothness 3, tuned 26) on 4 × 5 CPUs, about 8 h wall. Evaluations and analyses: about 4 h wall.

**Reduced to fit the time budget:**
- DR versions of the tuned configs: 3 seeds, not 5.
- Rate-limited learners: 5 seeds.

## P1

**1. Simulated frontier, one rule for everyone** (`sim_frontier_lqr.json`, `sim_frontier_learners.json`)

Setup:
- 1,395 LQR candidates. Frontier designs are chosen on 64 nominal scenarios and reported on the 192 held-out ones.
- Delay tolerance = largest d such that, for every d′ ≤ d, there is no loss of control and RMSE(d′) ≤ 2·RMSE(0) (rule "rel", primary).
- The variant "abs" uses RMSE ≤ 1° instead. I don't recommend it as primary: it fails every design coarser than 1° even at 0 ms.

Held-out rel frontier:

| Tolerance | RMSE |
|---|---|
| 0 ms | 0.051° |
| 20 ms | 0.227° |
| 40 ms | 0.478° |
| 60 ms | 0.687° |
| 80 ms | 1.006° |

Frozen-rule frontiers:
- Envelope (135 points): 0.052 / 0.228 / 0.480 / 0.712 / 1.442°.
- Design points (63) and all 325 points give the same frontier: 0.072 / 0.333 / 0.486 / 1.014 / 1.510°.

Held-out return of the frontier designs (95 % CI in `heldout_designs`): 494–496 at 0 ms, 493.6 at 20 ms, 478–480 at 40–80 ms. For comparison, GS LQR scores 484.0.

Learner gap to the rel frontier at the learner's own tolerance level (95 % bootstrap over seeds and scenarios):

| Controller | Tolerance | Gap |
|---|---|---|
| PPO | 0 ms | +0.31° [0.29, 0.33] |
| APG | 0 ms | +0.16° [0.15, 0.17] |
| PPO+DR | 60 ms | **−0.20° [−0.22, −0.19]** |
| APG+DR | 60 ms | **−0.29°** |
| Delay-randomized APG with masked buffer | 60 ms | **−0.45°** |
| No-margin LQR | 20 ms | +0.13° |
| LPV | 80 ms | +0.50° |
| Certified residual (dt) | 60 ms | +0.56° |

**Verdict:** under one rule, the nominal learners are dominated by the LQR frontier, and the DR learners beat it within their training range.

**2. Delay-aware discrete LQR** (`dlqr.json`, `eval_dlqr/`)

Design: exact ZOH model with the d pending commands; the same Bryson multipliers, tuned on the learners' reward at delay d (216 candidates per d).

| Design | At its design delay | Held-out return | Off-design |
|---|---|---|---|
| dlqr0 (0 ms) | 0.28° | 495.1 | — |
| dlqr1 (20 ms) | **0.34°** | — | at 0 ms: limit cycle (activity 190); at 60 ms: 78 % lost |
| dlqr2 (40 ms) | **0.42°** | — | at 0 ms: 8 % lost, activity 170 |
| dlqr3 (60 ms) | **0.58°** | — | 44–48 % lost at shorter delays |

Learners with a fixed delay and buffer, at their training delay: PPO 0.42 / 0.48 / 0.60° and APG 0.22 / 0.25 / 0.31° at 20 / 40 / 60 ms.

**Verdict:**
- The delay-aware model-based design shows the same pattern as the delay-trained learners. It is excellent at its design delay and fails off it.
- APG beats it at matched delay by 0.12–0.27°; PPO is about equal.
- dlqr0 (0.28°) is more accurate than the continuous LQR through the hold.

**3. Tuning budget** (`search_selection.json`; 16 PPO and 8 APG configurations × 3 seeds, chosen on the LQR tuning scenarios)

The paper's default ranks 2nd of 16 for PPO (495.00 vs the winner's 495.46) and 3rd of 8 for APG (496.01 vs 496.05).

| Controller | Seeds | Nominal (95 % seed-t CI) | Activity | Delay sweep 20/40/60/80 ms | Aileron 0–30 % |
|---|---|---|---|---|---|
| PPO, tuned (lr 2.7e-4, 4 epochs, 32 minibatches, clip 0.1, entropy 0.01, log-std 0, obs-norm) | 10 | **0.21° [0.17, 0.25]** | 0.69 | 7.35 / 13.1 / 16.5 (2 % lost) / 23.7 (30 % lost) | 20 % lost |
| APG, tuned | 10 | 0.20° | 0.21 | 15.2° at 20 ms (29 % lost); 99 % lost at 80 ms | — |
| PPO+DR, tuned config | 3 | 0.47° | 0.22 | — | 27 % lost |
| APG+DR, tuned config | 3 | 0.38° | 0.15 | 0.73° at 60 ms | — |

**Verdict:**
- Tuning makes PPO as accurate as APG.
- No-margin LQR − tuned PPO = +0.15° [0.12, 0.18], not equivalent. **F4 "PPO matches the no-margin LQR" holds only for the default PPO.**
- Tuned PPO is still dominated by the LQR frontier (+0.16° at 0 ms tolerance), and the 20 ms cliff and chattering (F3, F7) hold.

**4. Actuator rate limits**

No GTM/T2 rate figure exists in the repository or its downloaded sources, so I kept 100 °/s surfaces and 25 %/s wingtips. Sensitivity runs used 60 and 200 °/s.

With limits (`rate100/`), RMSE and share of runs lost:

| Controller | Nominal | 60 ms |
|---|---|---|
| GS | 1.06° | 3.21° (30 %) |
| LPV | 1.50° | 3.17° (1 %) |
| PPO | 0.72° (13 %) | — |
| APG | 0.87° (43 %) | — |
| PPO+DR | 0.48° | 88 % lost |
| APG+DR | 0.40° | 98 % lost |
| Certified residual (dt) | 1.24° | 92 % lost |

Trained with limits (5 seeds), flown with limits:

| Controller | Nominal | 40 ms | 60 ms | Aileron 0–30 % |
|---|---|---|---|---|
| PPO-rl | 0.46° | 28° (88 % lost) | — | — |
| APG-rl | 0.22° | 100 % lost at 20 ms | — | — |
| **PPO+DR-rl** | 0.71° | 0.70° | **1.08° (0 %)** | 15 % lost |
| APG+DR-rl | 0.54° | 0.54° | 4.5° (20 %) | — |

Sensitivity:
- At 60 °/s, PPO and APG lose 52 % and 85 % of nominal runs. GS still holds to 40 ms (48 % lost at 60 ms).
- At 200 °/s the effects are milder (GS 1.68° at 60 ms), but the nominal learners still fail at 20 ms (PPO 39 %, APG 92 % lost).

**Verdict:**
- Rate limits hurt the nominal learners most; training with them removes the nominal losses.
- PPO+DR-rl is the most delay-tolerant controller under limits, but loses 15 % with a weak aileron.
- The ordering of the paper's claims is unchanged.

## P2

**5. Statistics** (`stats.json`)
- Seed-level t intervals are given for every variant with 3–10 seeds.
- Loss of control has cluster-bootstrap intervals over seeds. For example, PPO at 80 ms loses 80 % [70, 87].
- TOST, no-margin LQR vs PPO: difference +0.000°, 90 % CI [−0.019, +0.019], equivalent at δ = 0.03° and 0.05°. Not equivalent for CleanRL-style or tuned PPO.
- Audit Spearman per delay, with bootstrap over designs: 20 ms 0.58 [0.24, 0.81], 40 ms 0.70 [0.45, 0.85], 60 ms 0.83 [0.67, 0.90], 80 ms 0.49 [0.21, 0.73]. Pooled 0.74 [0.63, 0.85]; delay alone gives 0.50.
- Sample efficiency, APG vs PPO, with Kaplan–Meier medians over 10 seeds, as steps / wall-clock / CPU-seconds:

| Threshold | Steps | Wall-clock | CPU-seconds |
|---|---|---|---|
| GS-LQR return | 2.75× | 1.71× | 3.08× |
| No-margin-LQR return | 5.44× | 3.36× | 5.78× |
| Return 490 | 3.83× | 2.35× | 4.14× |
| Return 494 | 5.1× | 3.4× | 5.9× |

  Evaluation is at most every update (32k steps, the batch size), so denser curves would need a smaller batch. Not done.

**6. Horizon and smoothness**
- 10 s episodes (`horizon10s/`) give the same picture as 15 s: PPO 0.34° nominal and 6.09° at 20 ms; APG 0.20° and 9.12°; DR learners flat to 60 ms.
- Smoothness-priced PPO with initial log-std −2 (3 seeds) no longer collapses: 0.89°, activity 0.11, return 487.0. But it degrades from 20 ms (1.12° at 20 ms, 3.98° at 40 ms). The earlier collapse was the exploration-noise penalty.

**7. Price of the certificate** (`certificate_price.json`)
- No common quadratic Lyapunov function exists for the GS, no-margin or fixed LQR gains (best t* = +0.0013 to +0.084 > 0); LPV is feasible (t* = −0.013).
- LPV synthesis grid:
  - Pole-disk radius 30: 1.49–1.51°, frozen margin 40 ms, simulated tolerance 80 ms.
  - Radius 60 or no disk: 1.19–1.27°, frozen 0 ms, simulated 40 ms.
  - Decay rate 0.05–0.2 changes RMSE by at most 0.03°.
- residual_safe_dt on 2,025 between-grid conditions: D ≻ 0 at all of them (min eigenvalue 0.019), g ≤ g_d at all of them (max 0.361 < 0.735).

## Verdicts on the review points
- **R1-3 / R2-M1–M2:** frontier with one rule: the nominal learners are dominated, the DR learners beat the frontier.
- **R2-M4 / AE-2:** the delay-aware LQR behaves like the delay-trained learners; APG is better at matched delay.
- **R2-M3:** tuning helps PPO's accuracy (0.21°) and breaks F4's equivalence. The cliff and chattering remain.
- **R1-1 / R2-M5 / AE-1:** rate limits change the numbers, not the ordering. Training with the limits fixes nominal failures.
- **R2-M6–M8:** the statistics above.
- **R1-6/7:** the certificate's price comes from the common P plus the disk; LQR designs admit no common P; the dt filter conditions hold between grid points.
