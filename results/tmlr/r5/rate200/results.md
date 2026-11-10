| Controller | Frozen shape | Fast morphing | + aileron at 0-30 % |
|---|---:|---:|---:|
| Gain-scheduled LQR | 1.06° · 100.0% | 1.02° · 100.0% | 2.81° · 100.0% |
| LPV (common Lyapunov) | 1.50° · 100.0% | 1.44° · 100.0% | 2.79° · 100.0% |
| PPO | 0.36° · 100.0% | 0.35° · 100.0% | 4.06° · 92.0% |
| PPO + domain rand. | 0.48° · 100.0% | 0.47° · 100.0% | 2.64° · 99.8% |
| Diff-sim policy | 0.21° · 100.0% | 0.20° · 100.0% | 10.74° · 74.7% |
| Diff-sim + domain rand. | 0.40° · 100.0% | 0.39° · 100.0% | 2.39° · 100.0% |
| LPV + residual, discrete-time Lyapunov filter | 1.24° · 100.0% | 1.20° · 100.0% | 3.19° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Gain-scheduled LQR | 0.71° | 0.14 /s | 0.35 % |
| LPV (common Lyapunov) | 1.26° | 0.20 /s | 0.41 % |
| PPO | 0.26° | 0.52 /s | 1.28 % |
| PPO + domain rand. | 0.49° | 0.16 /s | 0.87 % |
| Diff-sim policy | 0.08° | 0.26 /s | 0.25 % |
| Diff-sim + domain rand. | 0.45° | 0.15 /s | 0.60 % |
| LPV + residual, discrete-time Lyapunov filter | 0.88° | 0.38 /s | 0.53 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
