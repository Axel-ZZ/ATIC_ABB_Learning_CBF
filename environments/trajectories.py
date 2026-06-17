"""
Expert-trajectory generation — shared by every `Environment`.

Defines the `Trajectory` record, the fast simulation generator, and CSV
export. The generator operates on an `Environment` (bounds + obstacles +
robot radius), so the single-obstacle world and the maze use the same
code path.

generate_sim_set (~100+ traj/s):
    A* grid path -> shortcut smoothing -> pure-pursuit controller ->
    forward simulation through the true RK4 dynamics (with injected
    control noise) -> collision check. Not optimal control — this samples
    a dense distribution over dynamically feasible, collision-free
    trajectories (feasible by construction, since the recorded actions are
    what was fed to the dynamics), biased toward the decision boundary a
    CBF must learn:
        A (70%): random A->B pairs
        B (20%): obstacle-hugging — both endpoints near an obstacle
        C (10%): corridor forcing — only pairs without line of sight
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.environment import Environment


# ── Trajectory bookkeeping ──────────────────────────────────────────────
@dataclass
class Trajectory:
    states: np.ndarray       # (N+1, 3)
    controls: np.ndarray     # (N, 2)
    dt: float


# ── Geometry helpers (obstacle-agnostic) ────────────────────────────────
def _pose_free(xy: np.ndarray, obstacles: List, r: float) -> bool:
    return not any(o.collides(xy, r) for o in obstacles)


def _dist_to_nearest(xy: np.ndarray, obstacles: List) -> float:
    """Distance to the nearest obstacle surface (inf if no obstacles)."""
    best = np.inf
    for o in obstacles:
        best = min(best, o.surface_distance(xy))
    return best


def _sample_free_xy(rng: np.random.Generator, env: Environment,
                    r: float, max_tries: int = 100) -> Optional[np.ndarray]:
    (xmin, ymin), (xmax, ymax) = env.bounds
    for _ in range(max_tries):
        xy = rng.uniform([xmin + r, ymin + r], [xmax - r, ymax - r])
        if _pose_free(xy, env.obstacles, r):
            return xy
    return None


# ── Occupancy grid + A* (fast geometric warm starts) ────────────────────
class _OccupancyGrid:
    """
    Coarse occupancy grid + 8-connected A* for fast geometric warm starts
    (milliseconds per query). Built generically from `obstacle.occupied`,
    inflated by the pose radius so paths keep the planned clearance.
    """

    def __init__(self, env: Environment, r: float, res: float = 0.05):
        (xmin, ymin), (xmax, ymax) = env.bounds
        self.xmin, self.ymin, self.res = xmin, ymin, res
        self.nx = int(round((xmax - xmin) / res)) + 1
        self.ny = int(round((ymax - ymin) / res)) + 1
        xs = xmin + np.arange(self.nx) * res
        ys = ymin + np.arange(self.ny) * res
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        for o in env.obstacles:
            occ |= o.occupied(Xg, Yg, r)
        self.occ = occ

    def _cell(self, xy: np.ndarray) -> Tuple[int, int]:
        return (int(round((xy[0] - self.xmin) / self.res)),
                int(round((xy[1] - self.ymin) / self.res)))

    def free(self, xy: np.ndarray) -> bool:
        i, j = self._cell(xy)
        return (0 <= i < self.nx and 0 <= j < self.ny) and not self.occ[i, j]

    def los(self, a_xy: np.ndarray, b_xy: np.ndarray) -> bool:
        """Vectorized line-of-sight check on the occupancy grid."""
        n = max(2, int(np.linalg.norm(b_xy - a_xy) / self.res) + 2)
        pts = a_xy + np.linspace(0.0, 1.0, n)[:, None] * (b_xy - a_xy)
        i = np.clip(np.round((pts[:, 0] - self.xmin) / self.res).astype(int), 0, self.nx - 1)
        j = np.clip(np.round((pts[:, 1] - self.ymin) / self.res).astype(int), 0, self.ny - 1)
        return not self.occ[i, j].any()

    def plan(self, a_xy: np.ndarray, b_xy: np.ndarray) -> Optional[np.ndarray]:
        """A* from a to b. Returns (M, 2) world-frame waypoints or None."""
        import heapq

        start, goal = self._cell(a_xy), self._cell(b_xy)
        if self.occ[start] or self.occ[goal]:
            return None
        sq2 = np.sqrt(2.0)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, sq2), (1, -1, sq2), (-1, 1, sq2), (-1, -1, sq2)]

        def h(c):
            dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
            return max(dx, dy) + (sq2 - 1.0) * min(dx, dy)

        g = {start: 0.0}
        came: dict = {}
        open_q = [(h(start), start)]
        closed = set()
        while open_q:
            _, cur = heapq.heappop(open_q)
            if cur == goal:
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                cells.reverse()
                pts = np.array([[self.xmin + i * self.res, self.ymin + j * self.res]
                                for i, j in cells])
                pts[0], pts[-1] = a_xy, b_xy
                return pts
            if cur in closed:
                continue
            closed.add(cur)
            for di, dj, c in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny):
                    continue
                if self.occ[nb]:
                    continue
                ng = g[cur] + c
                if ng < g.get(nb, np.inf):
                    g[nb] = ng
                    came[nb] = cur
                    heapq.heappush(open_q, (ng + h(nb), nb))
        return None


# ── Path post-processing ────────────────────────────────────────────────
def _shortcut(path_xy: np.ndarray, grid: _OccupancyGrid,
              rng: np.random.Generator, iters: int = 60) -> np.ndarray:
    """
    Random shortcutting: repeatedly replace path segments with straight
    lines when the grid says there is line of sight. Removes the A*
    staircase without ever leaving the clearance-inflated free space.
    """
    pts = list(path_xy)
    for _ in range(iters):
        if len(pts) <= 2:
            break
        i = rng.integers(0, len(pts) - 2)
        j = rng.integers(i + 2, len(pts))
        if grid.los(pts[i], pts[j]):
            del pts[i + 1:j]
    return np.array(pts)


def _densify(path_xy: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Resample a polyline at uniform arc-length spacing."""
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < step:
        return path_xy
    s = np.linspace(0.0, cum[-1], int(cum[-1] / step) + 2)
    return np.stack([np.interp(s, cum, path_xy[:, 0]),
                     np.interp(s, cum, path_xy[:, 1])], axis=1)


# ── Pure-pursuit tracking simulation ────────────────────────────────────
def _track_path(
    f_disc,                          # discrete dynamics step f(x, u, dt)
    path_xy: np.ndarray,             # (M, 2) densified reference path
    theta0: float,
    hard_grid: _OccupancyGrid,       # robot-radius-inflated grid for in-sim collision
    rng: np.random.Generator,
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    lookahead: float = 0.25,
    k_heading: float = 2.5,
    goal_tol: float = 0.15,
    sigma_v: float = 0.02,
    sigma_omega: float = 0.1,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Pure-pursuit tracking of a reference path, simulated through the true
    RK4 dynamics with clipped, noise-injected controls. The recorded
    (state, action) pairs are dynamically feasible by construction.
    Returns None on collision or timeout.
    """
    v_max = v_bounds[1]
    goal = path_xy[-1]
    path_len = np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
    max_steps = int(4.0 * path_len / (v_max * dt)) + 100

    state = np.array([path_xy[0, 0], path_xy[0, 1], theta0])
    states = [state]
    controls = []
    idx = 0
    for _ in range(max_steps):
        if np.linalg.norm(state[:2] - goal) < goal_tol:
            return np.array(states), np.array(controls)

        # Advance the lookahead target monotonically along the path.
        while (idx < len(path_xy) - 1
               and np.linalg.norm(path_xy[idx] - state[:2]) < lookahead):
            idx += 1
        target = path_xy[idx]

        alpha = np.arctan2(target[1] - state[1], target[0] - state[0]) - state[2]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        # Slow down for large heading errors (turn-then-go), pure-pursuit ω.
        v = v_max * max(0.1, np.cos(alpha)) if abs(alpha) < np.pi / 2 else 0.05
        om = k_heading * alpha
        v += rng.normal(0.0, sigma_v)
        om += rng.normal(0.0, sigma_omega)
        u = np.array([np.clip(v, max(0.0, v_bounds[0]), v_bounds[1]),
                      np.clip(om, omega_bounds[0], omega_bounds[1])])

        state = np.asarray(f_disc(state, u, dt)).flatten()
        if not hard_grid.free(state[:2]):
            return None     # collision (controller cut a corner / noise)
        states.append(state)
        controls.append(u)

    return None             # timeout — tracker got stuck


# ── Simulation-based generation (fast, NLP-free) ────────────────────────
def generate_sim_set(
    env: Environment,
    kinematics,
    n_trajs: int = 5000,
    mode_fracs: Tuple[float, float, float] = (0.7, 0.2, 0.1),  # A, B, C
    d_band_a: Tuple[float, float] = (1.5, 8.5),   # A: random pairs
    d_band_b: Tuple[float, float] = (1.0, 4.0),   # B: obstacle-hugging
    d_band_c: Tuple[float, float] = (2.0, 8.5),   # C: corridor (no-LOS) pairs
    wall_band: float = 0.25,            # B: endpoints within this of an obstacle
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    clearance: float = 0.02,            # slight wall standoff over the robot radius
                                        # (planning r_pose just above the collision margin)
    heading_noise: float = np.pi / 2,
    sigma_v: float = 0.02,
    sigma_omega: float = 0.1,
    lookahead: float = 0.25,
    seed: int = 0,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> List[Trajectory]:
    """
    Fast NLP-free generation: A* -> shortcut -> pure pursuit -> RK4 sim.

    Generates a sampling distribution over dynamically feasible
    trajectories, biased toward obstacles (mode B) and corridors (mode C),
    i.e. toward the decision boundary the CBF must learn.
    """
    r_pose = env.robot_radius + clearance + 0.02
    rng = np.random.default_rng(seed)
    astar = _OccupancyGrid(env, r_pose)
    # Hard grid for in-sim collision: robot radius + small margin, so noise
    # may eat into the planning clearance (good boundary data) but never
    # into the robot itself.
    hard = _OccupancyGrid(env, env.robot_radius + 0.03)
    f_disc = kinematics.discrete_dynamics(dt, method="rk4")

    def sample_endpoint(near_wall: bool) -> Optional[np.ndarray]:
        for _ in range(60):
            xy = _sample_free_xy(rng, env, r_pose)
            if xy is None:
                return None
            if not near_wall:
                return xy
            if _dist_to_nearest(xy, env.obstacles) < r_pose + wall_band:
                return xy
        return None

    trajs: List[Trajectory] = []
    kept_by_mode = {"A": 0, "B": 0, "C": 0}
    n_astar_fail = n_track_fail = 0
    attempts = 0
    max_attempts = 20 * n_trajs
    t_start = time.monotonic()

    while len(trajs) < n_trajs and attempts < max_attempts:
        if deadline is not None and time.monotonic() > deadline:
            if verbose:
                print(f"  [deadline] reached at {len(trajs)} trajectories.")
            break
        attempts += 1

        mode = rng.choice(["A", "B", "C"], p=mode_fracs)
        d_band = {"A": d_band_a, "B": d_band_b, "C": d_band_c}[mode]
        near_wall = mode == "B"

        a_xy = sample_endpoint(near_wall)
        if a_xy is None:
            continue
        b_xy = None
        for _ in range(20):
            ang, d = rng.uniform(-np.pi, np.pi), rng.uniform(*d_band)
            cand = a_xy + d * np.array([np.cos(ang), np.sin(ang)])
            (xmin, ymin), (xmax, ymax) = env.bounds
            if not (xmin + r_pose <= cand[0] <= xmax - r_pose
                    and ymin + r_pose <= cand[1] <= ymax - r_pose):
                continue
            if not _pose_free(cand, env.obstacles, r_pose):
                continue
            if near_wall and _dist_to_nearest(cand, env.obstacles) >= r_pose + wall_band:
                continue
            if mode == "C" and astar.los(a_xy, cand):
                continue    # corridor mode wants blocked pairs only
            b_xy = cand
            break
        if b_xy is None:
            continue

        path = astar.plan(a_xy, b_xy)
        if path is None:
            n_astar_fail += 1
            continue
        path = _densify(_shortcut(path, astar, rng), step=0.05)

        bearing0 = np.arctan2(path[min(4, len(path) - 1), 1] - path[0, 1],
                              path[min(4, len(path) - 1), 0] - path[0, 0])
        theta0 = bearing0 + rng.uniform(-heading_noise, heading_noise)

        res = _track_path(
            f_disc, path, theta0, hard, rng,
            dt=dt, v_bounds=v_bounds, omega_bounds=omega_bounds,
            lookahead=lookahead, sigma_v=sigma_v, sigma_omega=sigma_omega,
        )
        if res is None:
            n_track_fail += 1
            continue
        states, controls = res

        trajs.append(Trajectory(states=states, controls=controls, dt=dt))
        kept_by_mode[mode] += 1

        if verbose and len(trajs) % 500 == 0:
            rate = len(trajs) / (time.monotonic() - t_start)
            print(f"  {len(trajs):5d}/{n_trajs} trajectories ({rate:.1f}/s)")

    if verbose:
        elapsed = time.monotonic() - t_start
        print(f"  Attempts: {attempts}  A* failed: {n_astar_fail}  "
              f"tracking failed/collided: {n_track_fail}")
        print(f"  Kept by mode: A={kept_by_mode['A']}  B={kept_by_mode['B']}  "
              f"C={kept_by_mode['C']}")
        print(f"  Kept: {len(trajs)} in {elapsed:.0f}s "
              f"({len(trajs) / max(elapsed, 1e-9):.1f} traj/s)")
    return trajs


# ── CSV export ──────────────────────────────────────────────────────────
def export_csv(trajs: List[Trajectory], path: str) -> int:
    """Write (state, action) rows: traj_id, t, step, x, y, theta, v, omega."""
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_id", "t", "step", "x", "y", "theta", "v", "omega"])
        for tid, traj in enumerate(trajs):
            N = len(traj.controls)
            for k in range(N):
                w.writerow([
                    tid, f"{k * traj.dt:.4f}", k,
                    f"{traj.states[k, 0]:.6f}",
                    f"{traj.states[k, 1]:.6f}",
                    f"{traj.states[k, 2]:.6f}",
                    f"{traj.controls[k, 0]:.6f}",
                    f"{traj.controls[k, 1]:.6f}",
                ])
                rows += 1
    return rows
