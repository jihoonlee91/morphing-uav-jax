| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| LPV + residual, discrete-time Lyapunov filter | 1.24° · 100.0% | 1.20° · 100.0% | 1.23° · 100.0% | 1.25° · 100.0% | 3.15° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| LPV + residual, discrete-time Lyapunov filter | 0.87° | 0.32 /s | 0.54 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
