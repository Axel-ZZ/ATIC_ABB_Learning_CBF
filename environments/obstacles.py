"""
Obstacle primitives for the planar worlds.

`CircleObstacle` and `WallObstacle` are duck-type compatible: both expose
the same geometry interface, so the trajectory generator, the dataset
sampler, and plotting handle circles and axis-aligned boxes through the
identical code path.

Interface:
    collides(pt, r) -> bool            point within r of the obstacle
    distance(X, Y)                     vectorized distance to surface (0 inside)
    surface_distance(pt) -> float      scalar version of `distance`
    occupied(X, Y, r) -> bool array    vectorized occupancy (inflated by r)
    draw(ax, **kw)                     render onto a matplotlib axis
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CircleObstacle:
    """Circular obstacle in the (x, y) plane."""
    x: float
    y: float
    radius: float

    def collides(self, pt: np.ndarray, robot_radius: float = 0.0) -> bool:
        dx = pt[0] - self.x
        dy = pt[1] - self.y
        return dx * dx + dy * dy <= (self.radius + robot_radius) ** 2

    def distance(self, X, Y):
        """Vectorized distance to the obstacle surface (0 if inside). X, Y array-like."""
        return np.maximum(0.0, np.hypot(np.asarray(X) - self.x, np.asarray(Y) - self.y) - self.radius)

    def surface_distance(self, pt: np.ndarray) -> float:
        """Scalar distance from pt to the obstacle surface (0 if inside)."""
        return float(self.distance(pt[0], pt[1]))

    def occupied(self, X: np.ndarray, Y: np.ndarray, inflation: float) -> np.ndarray:
        """Vectorized occupancy test on meshgrids X, Y (obstacle inflated by `inflation`)."""
        R = self.radius + inflation
        return (X - self.x) ** 2 + (Y - self.y) ** 2 <= R * R

    def draw(self, ax, **kw) -> None:
        import matplotlib.pyplot as plt
        ax.add_patch(plt.Circle((self.x, self.y), self.radius, **kw))


@dataclass
class WallObstacle:
    """
    Axis-aligned rectangular wall in the (x, y) plane.

    Duck-type compatible with CircleObstacle: callers only use the shared
    geometry interface, so walls and circles can be mixed in one list.
    """
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def collides(self, pt: np.ndarray, robot_radius: float = 0.0) -> bool:
        # Distance from point to the box (0 inside), compared to the
        # inflation radius — equivalent to inflating the box by a disc.
        dx = max(self.xmin - pt[0], 0.0, pt[0] - self.xmax)
        dy = max(self.ymin - pt[1], 0.0, pt[1] - self.ymax)
        return dx * dx + dy * dy <= robot_radius * robot_radius

    def distance(self, X, Y):
        """Vectorized distance to the box surface (0 if inside). X, Y array-like."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        dx = np.maximum(np.maximum(self.xmin - X, 0.0), X - self.xmax)
        dy = np.maximum(np.maximum(self.ymin - Y, 0.0), Y - self.ymax)
        return np.hypot(dx, dy)

    def surface_distance(self, pt: np.ndarray) -> float:
        """Scalar distance from pt to the box surface (0 if inside)."""
        return float(self.distance(pt[0], pt[1]))

    def occupied(self, X: np.ndarray, Y: np.ndarray, inflation: float) -> np.ndarray:
        """Vectorized occupancy test on meshgrids X, Y (box inflated by `inflation`)."""
        dx = np.maximum(np.maximum(self.xmin - X, 0.0), X - self.xmax)
        dy = np.maximum(np.maximum(self.ymin - Y, 0.0), Y - self.ymax)
        return dx * dx + dy * dy <= inflation * inflation

    def draw(self, ax, **kw) -> None:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((self.xmin, self.ymin),
                               self.xmax - self.xmin, self.ymax - self.ymin, **kw))
