"""
Validate a generated maze dataset: wall clearance, control bounds,
heading coverage.

Usage:  python validate_dataset.py maze_sim.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze_config import sketch_maze

cfg = sketch_maze()
df = pd.read_csv(Path(__file__).resolve().parent / "results" / sys.argv[1])

xy = df[["x", "y"]].to_numpy()
d = np.full(len(xy), np.inf)
for w in cfg.walls:
    dx = np.maximum(np.maximum(w.xmin - xy[:, 0], 0.0), xy[:, 0] - w.xmax)
    dy = np.maximum(np.maximum(w.ymin - xy[:, 1], 0.0), xy[:, 1] - w.ymax)
    d = np.minimum(d, np.hypot(dx, dy))

print(f"rows: {len(df)}   trajectories: {df['traj_id'].nunique()}")
print(f"wall distance:  min {d.min():.3f}  p1 {np.percentile(d, 1):.3f}  "
      f"median {np.median(d):.3f}  max {d.max():.3f}")
print(f"states with distance < robot radius ({cfg.robot_radius}): {(d < cfg.robot_radius).sum()}")
print(f"states within 0.3 m of a wall: {100 * (d < 0.3).mean():.1f}%")
print(f"v in [{df['v'].min():.3f}, {df['v'].max():.3f}]   "
      f"omega in [{df['omega'].min():.3f}, {df['omega'].max():.3f}]")
print(f"theta coverage: std {df['theta'].std():.2f} rad")
