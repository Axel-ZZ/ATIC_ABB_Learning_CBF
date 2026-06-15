"""
The `Environment` abstraction.

A single, environment-agnostic description of a planar world the robot
moves in: workspace bounds, a list of obstacles, a start/goal pose, and
the robot footprint radius. Both the single-obstacle world and the maze
are just different `Environment` instances, so the same trajectory
generator, plotting, and (later) CBF code work on either one unchanged.

Obstacles only need to satisfy the duck-typed geometry interface defined
on `CircleObstacle` / `WallObstacle` in `obstacles.py`:
    collides(pt, r) · surface_distance(pt) · distance(X, Y)
    occupied(X, Y, r) · draw(ax, **kw) · bbox
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Environment:
    name: str
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin, ymin), (xmax, ymax))
    obstacles: List                                          # CircleObstacle | WallObstacle
    start: np.ndarray                                        # (x, y, theta)
    goal: np.ndarray                                         # (x, y, theta)
    robot_radius: float = 0.1


# Environment registry — add new worlds here (factories imported lazily to
# avoid a circular import with the config modules).
ENV_NAMES = ["maze", "single_obstacle"]


def build_env(name: str) -> Environment:
    if name == "maze":
        from environments.maze.maze_config import sketch_maze
        return sketch_maze()
    if name == "single_obstacle":
        from environments.single_obstacle.config import single_obstacle_env
        return single_obstacle_env()
    raise ValueError(f"Unknown env '{name}'. Choices: {ENV_NAMES}")


def nearest_obstacle_distance(env: Environment, X, Y):
    """Vectorized distance from points (X, Y) to the nearest obstacle surface."""
    d = None
    for o in env.obstacles:
        di = o.distance(X, Y)
        d = di if d is None else np.minimum(d, di)
    if d is None:                              # no obstacles
        return np.full(np.shape(X), np.inf)
    return d


def plot_environment(ax, env: Environment, show_start_goal: bool = False,
                     obstacle_color: str = "0.2") -> None:
    """Draw all obstacles (incl. the boundary walls) + optional start/goal."""
    for o in env.obstacles:
        o.draw(ax, color=obstacle_color, zorder=5)
    (xmin, ymin), (xmax, ymax) = env.bounds
    if show_start_goal:
        ax.plot(*env.start[:2], "go", markersize=12, label="start", zorder=10)
        ax.plot(*env.goal[:2], "r*", markersize=16, label="goal", zorder=10)
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
