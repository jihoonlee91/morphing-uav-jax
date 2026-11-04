| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Fixed-point LQR | 1.05° · 100.0% | 1.01° · 100.0% | 1.07° · 100.0% | 1.05° · 100.0% | 2.75° · 100.0% |
| Gain-scheduled LQR | 1.06° · 100.0% | 1.02° · 100.0% | 1.07° · 100.0% | 1.06° · 100.0% | 2.64° · 100.0% |
| Gain-sched. LQR, no delay margin | 0.36° · 100.0% | 0.35° · 100.0% | 1.91° · 100.0% | 0.36° · 100.0% | 2.13° · 94.9% |
| LPV (common Lyapunov) | 1.50° · 100.0% | 1.45° · 100.0% | 1.55° · 100.0% | 1.51° · 100.0% | 2.74° · 100.0% |
| PPO | 0.35° · 100.0% | 0.35° · 100.0% | 11.59° · 100.0% | 0.36° · 100.0% | 1.81° · 99.7% |
| PPO + domain rand. | 0.49° · 100.0% | 0.48° · 100.0% | 0.56° · 100.0% | 0.50° · 100.0% | 2.66° · 99.7% |
| Diff-sim policy | 0.23° · 100.0% | 0.22° · 100.0% | 9.86° · 100.0% | 0.23° · 100.0% | 1.73° · 99.7% |
| Diff-sim + domain rand. | 0.42° · 100.0% | 0.41° · 100.0% | 0.51° · 100.0% | 0.42° · 100.0% | 2.00° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Fixed-point LQR | 1.00° | 0.13 /s | 0.32 % |
| Gain-scheduled LQR | 0.71° | 0.13 /s | 0.33 % |
| Gain-sched. LQR, no delay margin | 0.16° | 0.15 /s | 0.25 % |
| LPV (common Lyapunov) | 1.24° | 0.17 /s | 0.40 % |
| PPO | 0.34° | 0.56 /s | 1.43 % |
| PPO + domain rand. | 0.56° | 0.16 /s | 0.90 % |
| Diff-sim policy | 0.23° | 0.33 /s | 0.38 % |
| Diff-sim + domain rand. | 0.44° | 0.14 /s | 0.62 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
