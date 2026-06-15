"""
Maze environment configuration.

A 6x6 m maze of axis-aligned walls transcribed from a hand-drawn sketch:
two verticals hanging from the top-left, an L on the left, a hook in the
top-right enclosing the goal pocket, a middle L, a bottom-left hook and a
bottom-right vertical. Four border walls sit just outside the workspace
so the existing RRT collision check keeps the robot inside.

Every border-adjacent interior wall leaves a robot-passable gap (0.5 m).

Run this file directly to render the maze layout to results/maze_layout.png.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "expert_data_generation"))

from rrt_diff_drive import WallObstacle


@dataclass
class MazeConfig:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin, ymin), (xmax, ymax))
    walls: List[WallObstacle]
    start: np.ndarray            # (x, y, theta)
    goal: np.ndarray             # (x, y, theta)
    robot_radius: float = 0.1


def sketch_maze() -> MazeConfig:
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
    return MazeConfig(
        bounds=((0.0, 0.0), (6.0, 6.0)),
        walls=walls,
        start=np.array([0.4, 0.4, 0.0]),
        goal=np.array([5.4, 4.2, -np.pi / 2]),
    )


def plot_maze(ax, cfg: MazeConfig, show_start_goal: bool = False) -> None:
    """Draw walls, start and goal onto an existing matplotlib axis."""
    from matplotlib.patches import Rectangle

    for w in cfg.walls:
        ax.add_patch(Rectangle((w.xmin, w.ymin), w.xmax - w.xmin, w.ymax - w.ymin,
                               color="0.2", zorder=5))
    if show_start_goal:
        ax.plot(*cfg.start[:2], "go", markersize=12, label="start", zorder=10)
        ax.plot(*cfg.goal[:2], "r*", markersize=16, label="goal", zorder=10)
    (xmin, ymin), (xmax, ymax) = cfg.bounds
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


# ── Demo: render the layout ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = sketch_maze()
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_maze(ax, cfg)
    ax.set_title("Maze layout (sketch_maze)")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / "maze_layout.png", dpi=110)
    print(f"Saved {out / 'maze_layout.png'}")
