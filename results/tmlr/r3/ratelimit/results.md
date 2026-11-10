| Controller | Fast morphing |
|---|---:|
| Gain-scheduled LQR | 1.02° · 100.0% |
| Gain-sched. LQR, no delay margin | 0.35° · 100.0% |
| LPV (common Lyapunov) | 1.44° · 100.0% |
| PPO | 0.63° · 87.9% |
| PPO + domain rand. | 0.47° · 100.0% |
| Diff-sim policy | 0.61° · 57.0% |
| Diff-sim + domain rand. | 0.39° · 100.0% |
| LPV + residual, projected with 2x error margin | 1.08° · 100.0% |
| PPO, trained with 20 ms delay | 14.68° · 24.2% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Gain-scheduled LQR | 0.71° | 0.14 /s | 0.35 % |
| Gain-sched. LQR, no delay margin | 0.16° | 0.15 /s | 0.25 % |
| LPV (common Lyapunov) | 1.26° | 0.20 /s | 0.41 % |
| PPO | 0.29° | 0.57 /s | 1.25 % |
| PPO + domain rand. | 0.49° | 0.16 /s | 0.87 % |
| Diff-sim policy | 0.07° | 0.26 /s | 0.27 % |
| Diff-sim + domain rand. | 0.45° | 0.15 /s | 0.60 % |
| LPV + residual, projected with 2x error margin | 0.73° | 0.89 /s | 1.06 % |
| PPO, trained with 20 ms delay | 10.62° | 23.36 /s | 2.28 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
