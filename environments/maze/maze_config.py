"""
Maze environment configuration.

A 6x6 m maze of axis-aligned walls transcribed from a hand-drawn sketch:
two verticals hanging from the top-left, an L on the left, a hook in the
top-right enclosing the goal pocket, a middle L, a bottom-left hook and a
bottom-right vertical. Four border walls sit just outside the workspace
so the trajectory generator's collision check keeps the robot inside.

Every border-adjacent interior wall leaves a robot-passable gap (0.5 m).

`sketch_maze()` returns a generic `Environment`, so the shared trajectory
generator and plotting work on it unchanged.

Run this file directly to render the maze layout to results/maze_layout.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.environment import Environment
from environments.obstacles import WallObstacle


def sketch_maze() -> Environment:
    """Maze from the hand-drawn sketch (see docstring at top of module)."""
    walls = [
        # ── interior walls ──
        WallObstacle(0.5, 0.8, 2.9, 6.0),   # W1  left vertical, hangs from top (gap to left border)
        WallObstacle(0.5, 2.3, 2.6, 2.9),   # W2  horizontal foot of the left L
        WallObstacle(1.7, 2.0, 3.5, 6.0),   # W3  inner vertical, hangs from top
        WallObstacle(2.9, 4.6, 4.4, 4.7),   # W4  top-right horizontal
        WallObstacle(4.3, 4.6, 3.4, 4.7),   # W5  corner dropping from W4 to W6
        WallObstacle(4.6, 5.5, 3.4, 3.7),   # W6  horizontal toward right border (gap at border)
        WallObstacle(3.4, 3.7, 1.3, 3.5),   # W7  middle vertical
        WallObstacle(2.9, 3.7, 3.2, 3.5),   # W8  arm on top of W7 (middle L)
        WallObstacle(0.7, 2.8, 1.6, 1.9),   # W9  bottom-left horizontal
        WallObstacle(2.5, 2.8, 0.7, 1.9),   # W10 hook down from W9's right end
        WallObstacle(4.7, 5.0, 0.5, 2.3),   # W11 bottom-right vertical
        # ── border walls (just outside the 6x6 workspace) ──
        WallObstacle(-0.2, 6.2, -0.2, 0.0),  # bottom
        WallObstacle(-0.2, 6.2, 6.0, 6.2),   # top
        WallObstacle(-0.2, 0.0, -0.2, 6.2),  # left
        WallObstacle(6.0, 6.2, -0.2, 6.2),   # right
    ]
    return Environment(
        name="maze",
        bounds=((0.0, 0.0), (6.0, 6.0)),
        obstacles=walls,
        start=np.array([0.4, 0.4, 0.0]),
        goal=np.array([5.4, 4.2, -np.pi / 2]),
    )


# ── Demo: render the layout ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from environments.environment import plot_environment

    env = sketch_maze()
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_environment(ax, env)
    ax.set_title("Maze layout (sketch_maze)")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / "maze_layout.png", dpi=110)
    print(f"Saved {out / 'maze_layout.png'}")
