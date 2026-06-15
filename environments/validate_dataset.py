"""
Validate a generated dataset against an environment's geometry:
clearance to obstacles, control bounds, heading coverage. Works for any
environment (maze, single_obstacle, …).

Usage
-----
    python environments/validate_dataset.py --env maze --csv maze_sim.csv
    python environments/validate_dataset.py --env maze --csv runs/datasets/<tag>/unsafe.csv

`--csv` may be an absolute/relative path, or a bare filename resolved
inside environments/<env>/results/ (where build_dataset.py caches the
trajectory CSVs <env>_sim.csv / <env>_unsafe_sim.csv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.environment import build_env, nearest_obstacle_distance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="environment name (maze, single_obstacle, …)")
    p.add_argument("--csv", required=True, help="dataset CSV (path or bare filename in env results)")
    args = p.parse_args()

    env = build_env(args.env)

    given = Path(args.csv)
    fallback = Path(__file__).resolve().parent / args.env / "results" / args.csv
    csv_path = given if given.exists() else fallback
    if not csv_path.exists():
        sys.exit(f"[validate_dataset] CSV not found. Looked in:\n"
                 f"  {given}\n  {fallback}\n"
                 f"Generate it first:\n"
                 f"  python environments/build_dataset.py  (set ENV='{args.env}')")
    df = pd.read_csv(csv_path)

    xy = df[["x", "y"]].to_numpy()
    d = nearest_obstacle_distance(env, xy[:, 0], xy[:, 1])

    print(f"env: {env.name}   file: {csv_path.name}   rows: {len(df)}")
    if "traj_id" in df.columns:
        print(f"trajectories: {df['traj_id'].nunique()}")
    print(f"obstacle distance:  min {d.min():.3f}  p1 {np.percentile(d, 1):.3f}  "
          f"median {np.median(d):.3f}  max {d.max():.3f}")
    print(f"states with distance < robot radius ({env.robot_radius}): {int((d < env.robot_radius).sum())}")
    print(f"states within 0.3 m of an obstacle: {100 * (d < 0.3).mean():.1f}%")
    if {"v", "omega"}.issubset(df.columns):
        print(f"v in [{df['v'].min():.3f}, {df['v'].max():.3f}]   "
              f"omega in [{df['omega'].min():.3f}, {df['omega'].max():.3f}]")
    if "theta" in df.columns:
        print(f"theta coverage: std {df['theta'].std():.2f} rad")


if __name__ == "__main__":
    main()
