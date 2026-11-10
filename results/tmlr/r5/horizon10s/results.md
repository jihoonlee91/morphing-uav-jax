| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Gain-scheduled LQR | 1.00° · 100.0% | 0.96° · 100.0% | 1.06° · 100.0% | 1.02° · 100.0% | 2.39° · 100.0% |
| Gain-sched. LQR, no delay margin | 0.35° · 100.0% | 0.34° · 100.0% | 1.89° · 100.0% | 0.36° · 100.0% | 1.95° · 99.6% |
| LPV (common Lyapunov) | 1.38° · 100.0% | 1.33° · 100.0% | 1.49° · 100.0% | 1.42° · 100.0% | 2.52° · 100.0% |
| PPO | 0.34° · 100.0% | 0.33° · 100.0% | 11.50° · 92.3% | 0.34° · 100.0% | 2.35° · 97.6% |
| PPO + domain rand. | 0.48° · 100.0% | 0.46° · 100.0% | 0.56° · 100.0% | 0.49° · 100.0% | 2.26° · 100.0% |
| Diff-sim policy | 0.20° · 100.0% | 0.20° · 100.0% | 17.28° · 97.1% | 0.21° · 100.0% | 1.86° · 99.7% |
| Diff-sim + domain rand. | 0.40° · 100.0% | 0.38° · 100.0% | 0.50° · 100.0% | 0.40° · 100.0% | 2.25° · 100.0% |
| LPV + residual, projected with 2x error margin | 1.05° · 100.0% | 1.01° · 100.0% | 1.05° · 100.0% | 1.10° · 100.0% | 3.01° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Gain-scheduled LQR | 0.65° | 0.13 /s | 0.33 % |
| Gain-sched. LQR, no delay margin | 0.16° | 0.15 /s | 0.23 % |
| LPV (common Lyapunov) | 1.17° | 0.17 /s | 0.39 % |
| PPO | 0.23° | 0.55 /s | 1.19 % |
| PPO + domain rand. | 0.44° | 0.17 /s | 0.81 % |
| Diff-sim policy | 0.08° | 0.28 /s | 0.21 % |
| Diff-sim + domain rand. | 0.41° | 0.16 /s | 0.59 % |
| LPV + residual, projected with 2x error margin | 0.67° | 0.80 /s | 0.87 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
