| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Fixed-point LQR | 1.05° · 100.0% | 1.01° · 100.0% | 1.07° · 100.0% | 1.05° · 100.0% | 2.75° · 100.0% |
| Gain-scheduled LQR | 1.06° · 100.0% | 1.02° · 100.0% | 1.07° · 100.0% | 1.06° · 100.0% | 2.64° · 100.0% |
| Gain-sched. LQR, no delay margin | 0.36° · 100.0% | 0.35° · 100.0% | 1.91° · 100.0% | 0.36° · 100.0% | 2.13° · 94.9% |
| LPV (common Lyapunov) | 1.50° · 100.0% | 1.45° · 100.0% | 1.55° · 100.0% | 1.51° · 100.0% | 2.74° · 100.0% |
| Gain-scheduled LQR, wide grid | 1.06° · 100.0% | 1.02° · 100.0% | 1.07° · 100.0% | 1.06° · 100.0% | 2.64° · 100.0% |
| Gain-sched. LQR, no margin, wide grid | 0.06° · 100.0% | 0.06° · 100.0% | — · 0.0% | 0.06° · 100.0% | 20.13° · 37.9% |
| Gain-scheduled LQR tuned on DR | 1.06° · 100.0% | 1.02° · 100.0% | 1.07° · 100.0% | 1.06° · 100.0% | 2.64° · 100.0% |
| LPV, pointwise Lyapunov (ablation) | 1.04° · 100.0% | 1.00° · 100.0% | 1.05° · 100.0% | 1.03° · 100.0% | 2.93° · 100.0% |
| LPV, no pole disk (ablation) | 1.21° · 100.0% | 1.17° · 100.0% | 1.96° · 100.0% | 1.22° · 100.0% | 2.34° · 100.0% |
| PPO | 0.36° · 100.0% | 0.35° · 100.0% | 11.70° · 91.8% | 0.36° · 100.0% | 2.54° · 96.5% |
| PPO + domain rand. | 0.48° · 100.0% | 0.47° · 100.0% | 0.55° · 100.0% | 0.48° · 100.0% | 2.50° · 99.9% |
| Diff-sim policy | 0.21° · 100.0% | 0.20° · 100.0% | 17.81° · 93.3% | 0.21° · 100.0% | 2.69° · 99.0% |
| Diff-sim + domain rand. | 0.40° · 100.0% | 0.39° · 100.0% | 0.50° · 100.0% | 0.40° · 100.0% | 2.34° · 100.0% |
| LPV + residual PPO (DR) | 0.96° · 100.0% | 0.92° · 100.0% | 1.02° · 100.0% | 0.98° · 100.0% | 3.54° · 100.0% |
| LPV + residual, Lyapunov-projected | 1.17° · 100.0% | 1.13° · 100.0% | 1.15° · 100.0% | 1.19° · 100.0% | 3.50° · 100.0% |
| LPV + residual, projected in training | 1.14° · 100.0% | 1.11° · 100.0% | 1.12° · 100.0% | 1.17° · 100.0% | 3.51° · 100.0% |
| LPV + residual, projected with error margin | 1.11° · 100.0% | 1.07° · 100.0% | 1.06° · 100.0% | 1.14° · 100.0% | 3.41° · 100.0% |
| LPV + residual, projected with 2x error margin | 1.12° · 100.0% | 1.08° · 100.0% | 1.06° · 100.0% | 1.14° · 100.0% | 3.36° · 100.0% |
| PPO, trained with 20 ms delay | 0.64° · 100.0% | 0.62° · 100.0% | 8.48° · 99.5% | 0.65° · 100.0% | 2.45° · 99.1% |
| Diff-sim, trained with 20 ms delay | 0.63° · 100.0% | 0.64° · 100.0% | 12.15° · 99.5% | 0.65° · 100.0% | 3.24° · 99.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Fixed-point LQR | 1.00° | 0.13 /s | 0.32 % |
| Gain-scheduled LQR | 0.71° | 0.13 /s | 0.33 % |
| Gain-sched. LQR, no delay margin | 0.16° | 0.15 /s | 0.25 % |
| LPV (common Lyapunov) | 1.24° | 0.17 /s | 0.40 % |
| Gain-scheduled LQR, wide grid | 0.71° | 0.13 /s | 0.33 % |
| Gain-sched. LQR, no margin, wide grid | 0.12° | 0.27 /s | 0.00 % |
| Gain-scheduled LQR tuned on DR | 0.71° | 0.13 /s | 0.33 % |
| LPV, pointwise Lyapunov (ablation) | 0.70° | 0.12 /s | 0.33 % |
| LPV, no pole disk (ablation) | 1.60° | 0.13 /s | 0.23 % |
| PPO | 0.26° | 0.51 /s | 1.27 % |
| PPO + domain rand. | 0.49° | 0.16 /s | 0.86 % |
| Diff-sim policy | 0.08° | 0.25 /s | 0.25 % |
| Diff-sim + domain rand. | 0.45° | 0.15 /s | 0.59 % |
| LPV + residual PPO (DR) | 0.68° | 0.47 /s | 3.29 % |
| LPV + residual, Lyapunov-projected | 0.77° | 0.61 /s | 2.21 % |
| LPV + residual, projected in training | 0.71° | 0.53 /s | 2.04 % |
| LPV + residual, projected with error margin | 0.72° | 0.72 /s | 1.22 % |
| LPV + residual, projected with 2x error margin | 0.73° | 0.82 /s | 1.07 % |
| PPO, trained with 20 ms delay | 0.31° | 181.45 /s | 1.50 % |
| Diff-sim, trained with 20 ms delay | 0.42° | 194.99 /s | 0.67 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
