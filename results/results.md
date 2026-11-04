| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Fixed-point LQR | 1.05° · 100% | 1.01° · 100% | 1.07° · 100% | 1.05° · 100% | 2.75° · 100% |
| Gain-scheduled LQR | 1.06° · 100% | 1.02° · 100% | 1.07° · 100% | 1.06° · 100% | 2.64° · 100% |
| LPV (common Lyapunov) | 1.50° · 100% | 1.45° · 100% | 1.55° · 100% | 1.51° · 100% | 2.74° · 100% |
| PPO | 0.35° · 100% | 0.35° · 100% | 11.59° · 100% | 0.36° · 100% | 1.81° · 100% |
| PPO + domain rand. | 0.49° · 100% | 0.48° · 100% | 0.56° · 100% | 0.50° · 100% | 2.66° · 100% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Fixed-point LQR | 1.00° | 0.13 /s | 0.32 % |
| Gain-scheduled LQR | 0.71° | 0.13 /s | 0.33 % |
| LPV (common Lyapunov) | 1.24° | 0.17 /s | 0.40 % |
| PPO | 0.34° | 0.56 /s | 1.43 % |
| PPO + domain rand. | 0.56° | 0.16 /s | 0.90 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its full range (higher = busier actuators).
