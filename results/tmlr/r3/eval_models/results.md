| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| LPV, pointwise, no pole disk (ablation) | 1.04° · 100.0% | 1.01° · 100.0% | 1.05° · 100.0% | 1.04° · 100.0% | 2.60° · 100.0% |
| LPV, tuned multipliers (60 ms rule) | 1.39° · 100.0% | 1.35° · 100.0% | 1.47° · 100.0% | 1.40° · 100.0% | 3.27° · 100.0% |
| LPV, parameter-dependent Lyapunov matrix | 1.01° · 100.0% | 0.98° · 100.0% | 1.02° · 100.0% | 1.01° · 100.0% | 2.28° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| LPV, pointwise, no pole disk (ablation) | 0.71° | 0.13 /s | 0.32 % |
| LPV, tuned multipliers (60 ms rule) | 1.43° | 0.18 /s | 0.35 % |
| LPV, parameter-dependent Lyapunov matrix | 1.08° | 0.12 /s | 0.38 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
