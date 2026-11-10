| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| Discrete LQR (ZOH), 0 ms | 0.28° · 100.0% | 0.27° · 100.0% | 5.36° · 8.6% | 0.28° · 100.0% | 2.83° · 91.8% |
| Discrete delay-aware LQR, 20 ms | 0.37° · 100.0% | 0.37° · 100.0% | 0.62° · 100.0% | 0.37° · 100.0% | 1.88° · 100.0% |
| Discrete delay-aware LQR, 40 ms | 11.35° · 88.3% | 11.54° · 86.3% | 0.46° · 100.0% | 12.17° · 90.2% | 13.87° · 89.1% |
| Discrete delay-aware LQR, 60 ms | 25.46° · 46.5% | 26.07° · 46.9% | 27.00° · 53.5% | 25.02° · 44.5% | 26.04° · 41.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| Discrete LQR (ZOH), 0 ms | 0.12° | 0.16 /s | 0.36 % |
| Discrete delay-aware LQR, 20 ms | 1.19° | 190.02 /s | 0.39 % |
| Discrete delay-aware LQR, 40 ms | 11.45° | 167.23 /s | 0.84 % |
| Discrete delay-aware LQR, 60 ms | 12.61° | 141.77 /s | 0.69 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
