| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Gain-scheduled LQR | 1.06° · 100.0% | 1.02° · 100.0% | 1.16° · 100.0% | 1.06° · 100.0% | 2.81° · 100.0% |
| Gain-sched. LQR, no delay margin | 0.36° · 100.0% | 0.35° · 100.0% | — · 0.0% | 0.36° · 100.0% | 2.56° · 75.8% |
| LPV (common Lyapunov) | 1.50° · 100.0% | 1.44° · 100.0% | 1.58° · 100.0% | 1.51° · 100.0% | 2.79° · 100.0% |
| PPO | 0.72° · 87.3% | 0.63° · 87.9% | 33.23° · 0.7% | 0.71° · 87.0% | 7.05° · 81.6% |
| PPO + domain rand. | 0.48° · 100.0% | 0.47° · 100.0% | 1.74° · 95.8% | 0.48° · 100.0% | 2.66° · 99.9% |
| Diff-sim policy | 0.87° · 57.2% | 0.61° · 57.0% | — · 0.0% | 0.57° · 54.1% | 19.23° · 36.4% |
| Diff-sim + domain rand. | 0.40° · 100.0% | 0.39° · 100.0% | 1.28° · 87.0% | 0.40° · 100.0% | 2.38° · 100.0% |
| LPV + residual, discrete-time Lyapunov filter | 1.24° · 100.0% | 1.20° · 100.0% | 1.64° · 100.0% | 1.25° · 100.0% | 3.19° · 100.0% |

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
| LPV + residual, discrete-time Lyapunov filter | 0.88° | 0.38 /s | 0.53 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
