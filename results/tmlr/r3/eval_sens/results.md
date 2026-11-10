| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| PPO, action-rate penalty x100 | 1.38° · 100.0% | 1.34° · 100.0% | 13.58° · 64.6% | 1.40° · 100.0% | 4.44° · 76.4% |
| PPO, initial log-std 0 | 0.32° · 100.0% | 0.31° · 100.0% | 13.14° · 100.0% | 0.33° · 100.0% | 4.73° · 85.3% |
| PPO, entropy coefficient 0.01 | 0.30° · 100.0% | 0.30° · 100.0% | 13.19° · 99.2% | 0.31° · 100.0% | 1.61° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| PPO, action-rate penalty x100 | 1.47° | 0.21 /s | 1.03 % |
| PPO, initial log-std 0 | 0.54° | 0.66 /s | 5.65 % |
| PPO, entropy coefficient 0.01 | 0.22° | 0.57 /s | 1.29 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
