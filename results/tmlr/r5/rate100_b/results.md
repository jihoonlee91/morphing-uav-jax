| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| PPO, trained with rate limits | 0.46° · 99.9% | 0.45° · 100.0% | 28.50° · 11.4% | 0.47° · 99.7% | 5.23° · 86.6% |
| Diff-sim, trained with rate limits | 0.22° · 100.0% | 0.21° · 100.0% | — · 0.0% | 0.22° · 100.0% | 10.79° · 42.3% |
| PPO + DR, trained with rate limits | 0.71° · 100.0% | 0.69° · 100.0% | 0.74° · 100.0% | 0.71° · 100.0% | 14.15° · 85.1% |
| Diff-sim + DR, trained with rate limits | 0.54° · 100.0% | 0.52° · 100.0% | 0.62° · 100.0% | 0.54° · 100.0% | 4.32° · 86.2% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| PPO, trained with rate limits | 0.30° | 0.29 /s | 1.99 % |
| Diff-sim, trained with rate limits | 0.20° | 0.20 /s | 0.26 % |
| PPO + DR, trained with rate limits | 0.55° | 0.14 /s | 0.81 % |
| Diff-sim + DR, trained with rate limits | 0.60° | 0.12 /s | 0.96 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
