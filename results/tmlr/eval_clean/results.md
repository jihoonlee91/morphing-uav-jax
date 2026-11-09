| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| PPO, CleanRL-style settings | 0.33° · 100.0% | 0.33° · 100.0% | 12.42° · 99.8% | 0.33° · 100.0% | 3.45° · 70.9% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| PPO, CleanRL-style settings | 0.56° | 0.83 /s | 3.44 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
