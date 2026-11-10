| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| PPO, trained with rate limits | 0.46° · 100.0% | 0.45° · 100.0% | 9.08° · 100.0% | 0.47° · 100.0% | 1.56° · 87.9% |
| Diff-sim, trained with rate limits | 0.22° · 100.0% | 0.21° · 100.0% | 17.31° · 99.5% | 0.22° · 100.0% | 2.83° · 97.0% |
| PPO + DR, trained with rate limits | 0.71° · 100.0% | 0.69° · 100.0% | 0.73° · 100.0% | 0.71° · 100.0% | 14.23° · 85.2% |
| Diff-sim + DR, trained with rate limits | 0.54° · 100.0% | 0.52° · 100.0% | 0.58° · 100.0% | 0.54° · 100.0% | 4.16° · 87.3% |
| PPO, rate penalty x1000, log-std -2 | 0.89° · 100.0% | 0.85° · 100.0% | 4.12° · 100.0% | 0.88° · 100.0% | 2.58° · 100.0% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| PPO, trained with rate limits | 0.30° | 0.28 /s | 1.99 % |
| Diff-sim, trained with rate limits | 0.14° | 0.17 /s | 0.26 % |
| PPO + DR, trained with rate limits | 0.55° | 0.14 /s | 0.81 % |
| Diff-sim + DR, trained with rate limits | 0.59° | 0.12 /s | 0.96 % |
| PPO, rate penalty x1000, log-std -2 | 0.66° | 0.11 /s | 0.25 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
