| Controller | Frozen shape | Fast morphing | + noise, 40 ms delay | + model error | + aileron at 0-30 % |
|---|---:|---:|---:|---:|---:|
| PPO, trained with 40 ms delay | 0.57° · 100.0% | 0.56° · 100.0% | 0.54° · 100.0% | 0.58° · 100.0% | 2.44° · 98.7% |
| Diff-sim, trained with 40 ms delay | 0.56° · 100.0% | 0.56° · 100.0% | 0.64° · 100.0% | 0.57° · 100.0% | 3.28° · 99.4% |
| PPO, trained with 60 ms delay | 1.01° · 100.0% | 1.00° · 100.0% | 1.10° · 100.0% | 1.05° · 100.0% | 2.29° · 98.5% |
| Diff-sim, trained with 60 ms delay | 0.81° · 100.0% | 0.79° · 100.0% | 0.64° · 100.0% | 0.81° · 100.0% | 3.88° · 100.0% |
| PPO, delay 0-60 ms randomized | 0.44° · 100.0% | 0.43° · 100.0% | 0.53° · 100.0% | 0.45° · 100.0% | 3.46° · 81.5% |
| Diff-sim, delay 0-60 ms randomized | 0.39° · 100.0% | 0.38° · 100.0% | 0.50° · 100.0% | 0.39° · 100.0% | 3.34° · 100.0% |
| PPO, delay randomized + command buffer | 0.36° · 100.0% | 0.36° · 100.0% | 0.53° · 100.0% | 0.37° · 100.0% | 1.70° · 99.7% |
| Diff-sim, delay randomized + command buffer | 0.23° · 100.0% | 0.23° · 100.0% | 0.51° · 90.4% | 0.23° · 100.0% | 0.64° · 67.8% |
| Diff-sim, linear gain-scheduled policy | 0.58° · 100.0% | 0.56° · 100.0% | 20.66° · 89.7% | 0.58° · 100.0% | 2.70° · 100.0% |
| PPO, action-rate penalty x1000 | 0.79° · 20.0% | 0.77° · 20.0% | 1.45° · 0.1% | 0.79° · 20.0% | 13.02° · 35.2% |
| Diff-sim, action-rate penalty x1000 | 0.29° · 100.0% | 0.28° · 100.0% | 8.21° · 100.0% | 0.29° · 100.0% | 4.18° · 98.6% |

Cells: roll-tracking RMS error over successful runs · share of runs without loss of control. 95 % bootstrap intervals are in `metrics.json`.

| Controller (fast morphing) | Sideslip, p95 of max | Command activity | RMS η_a |
|---|---:|---:|---:|
| PPO, trained with 40 ms delay | 0.64° | 161.36 /s | 1.97 % |
| Diff-sim, trained with 40 ms delay | 0.40° | 92.41 /s | 0.62 % |
| PPO, trained with 60 ms delay | 1.03° | 239.04 /s | 1.22 % |
| Diff-sim, trained with 60 ms delay | 0.52° | 67.41 /s | 0.94 % |
| PPO, delay 0-60 ms randomized | 0.42° | 0.18 /s | 0.98 % |
| Diff-sim, delay 0-60 ms randomized | 0.33° | 0.16 /s | 0.87 % |
| PPO, delay randomized + command buffer | 0.41° | 0.28 /s | 1.04 % |
| Diff-sim, delay randomized + command buffer | 0.27° | 0.16 /s | 0.55 % |
| Diff-sim, linear gain-scheduled policy | 0.47° | 0.29 /s | 0.28 % |
| PPO, action-rate penalty x1000 | 0.88° | 0.12 /s | 0.29 % |
| Diff-sim, action-rate penalty x1000 | 0.29° | 0.09 /s | 0.26 % |

Command activity: mean |du/dt| of aileron, rudder and asymmetric-span commands, each divided by its position limit (higher = busier actuators).
