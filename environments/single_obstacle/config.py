"""
Single-obstacle world configuration.

A square workspace with one circular obstacle — the original planar
scenario from the project proposal. Returns a generic `Environment`, so
the shared trajectory generator and plotting work on it unchanged.

Run this file directly to render the layout to results/single_obstacle_layout.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.environment import Environment
from environments.obstacles import CircleObstacle, WallObstacle


def single_obstacle_env() -> Environment:
    """
    Square 4x4 m world with a single circular obstacle at the centre.

    The square is bounded by four border walls just outside the workspace
    (same convention as the maze): the robot is confined to the box, and
    "outside the square" is part of the unsafe set the CBF must learn.
    """
    obstacles = [
        CircleObstacle(2.0, 2.0, 0.5),
        # ── border walls (just outside the 4x4 workspace) ──
        WallObstacle(-0.2, 4.2, -0.2, 0.0),  # bottom
        WallObstacle(-0.2, 4.2, 4.0, 4.2),   # top
        WallObstacle(-0.2, 0.0, -0.2, 4.2),  # left
        WallObstacle(4.0, 4.2, -0.2, 4.2),   # right
    ]
    return Environment(
        name="single_obstacle",
        bounds=((0.0, 0.0), (4.0, 4.0)),
        obstacles=obstacles,
        start=np.array([0.4, 0.4, 0.0]),
        goal=np.array([3.6, 3.6, np.pi / 2]),
    )


# ── Demo: render the layout ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from environments.environment import plot_environment

    env = single_obstacle_env()
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_environment(ax, env, show_start_goal=False)
    ax.set_title("Single-obstacle world")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / "single_obstacle_layout.png", dpi=110)
    print(f"Saved {out / 'single_obstacle_layout.png'}")
