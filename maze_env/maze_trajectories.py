"""
Expert trajectory generation in the maze environment.

Three modes, fastest first:

sim (default, ~10-50 ms/trajectory):
    A* grid path → shortcut smoothing → pure-pursuit controller →
    forward simulation through the true RK4 dynamics (with injected
    control noise) → collision check. Not optimal control — this
    generates a sampling distribution over dynamically feasible,
    collision-free trajectories, biased toward walls and corridors
    (the decision boundary a CBF must learn). Sampling modes:
        A (70%): random A→B pairs
        B (20%): wall-hugging — both endpoints near walls
        C (10%): corridor forcing — only pairs without line of sight

coverage (~3 s/trajectory):
    Same A→B sampling, but each warm start is refined by a CasADi NLP
    (effort/smoothness-optimal). Use for report figures.

mission (~1 min/trajectory):
    Full start→goal paths via kinodynamic RRT + NLP, one per seed.

All modes export the same CSV schema as dense_trajectories, so the
set-analysis notebook and train.py work without modification.
(via_angle/via_radius columns are 0 — polar-sweep metadata that does
not apply here.)

Run:  python maze_trajectories.py --trajs 5000 --budget 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "expert_data_generation"))

from model import RobotModel, DiffDriveKinematics
from rrt_diff_drive import RRTPlanner, WallObstacle
from dense_trajectories import Trajectory, _frechet, _is_smooth, export_csv
from maze_config import MazeConfig, sketch_maze, plot_maze


# ── Smooth wall clearance for the NLP ───────────────────────────────────
def _wall_clearance_constraints(opti: ca.Opti, X, wall: WallObstacle,
                                r: float, eps: float = 1e-4) -> None:
    """
    Keep every state at least `r` away from an axis-aligned box.

    Uses a smoothed box signed distance so IPOPT gets useful gradients:
        a = |p - c| - h        (per axis, |.| smoothed)
        s = max(a, 0)          (smoothed)
        ||s||^2 >= r^2
    Inside the box s ≈ 0, so the constraint is violated there and the
    solver pushes the trajectory out — but the RRT warm start is already
    collision-free, so in practice this only shapes the boundary.
    """
    cx, cy = wall.center
    hx, hy = wall.half_extents
    # Vectorized over all N+1 states (one matrix constraint instead of a
    # Python loop — much faster problem construction).
    ax_ = ca.sqrt((X[0, :] - cx) ** 2 + eps) - hx
    ay_ = ca.sqrt((X[1, :] - cy) ** 2 + eps) - hy
    sx = 0.5 * (ax_ + ca.sqrt(ax_ ** 2 + eps))
    sy = 0.5 * (ay_ + ca.sqrt(ay_ ** 2 + eps))
    opti.subject_to(sx * sx + sy * sy >= r * r)


# ── CasADi NLP smoothing (maze version of rrt_diff_drive.smooth_path) ──
def smooth_maze_path(
    kinematics: DiffDriveKinematics,
    states_init: np.ndarray,        # (N+1, 3) warm start (RRT path or straight guess)
    controls_init: np.ndarray,      # (N, 2)
    cfg: MazeConfig,
    goal: np.ndarray,               # (3,) — endpoint the NLP should hit exactly (xy)
    dt: float = 0.1,
    clearance: float = 0.05,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    w_effort: float = 1.0,
    w_smooth: float = 10.0,
    w_heading: float = 50.0,        # 0 → terminal heading free
    walls: Optional[List[WallObstacle]] = None,   # default: all cfg.walls
    max_iter: int = 600,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Polish a warm-started maze path. Same structure as the existing
    smooth_path / dense_trajectories._solve: fixed start, fixed terminal
    xy, soft terminal heading, effort + smoothness + jerk cost. Returns
    None if IPOPT fails.

    `walls` lets callers pass a pruned local subset (big speedup for
    short coverage trajectories); by default all maze walls are used.
    """
    if walls is None:
        walls = cfg.walls
    f_disc = kinematics.discrete_dynamics(dt, method="rk4")
    N = controls_init.shape[0]

    opti = ca.Opti()
    X = opti.variable(3, N + 1)
    U = opti.variable(2, N)

    # Dynamics — mapped over all N steps at once (faster construction)
    F = f_disc.map(N)
    opti.subject_to(X[:, 1:] == F(X[:, :-1], U, ca.repmat(dt, 1, N)))

    # Boundary: exact start, exact terminal position, soft terminal heading.
    # (RRT only reaches the goal within tolerance — the NLP closes the gap.)
    opti.subject_to(X[:, 0] == states_init[0])
    opti.subject_to(X[0, -1] == goal[0])
    opti.subject_to(X[1, -1] == goal[1])
    heading_err = ca.sin(0.5 * (X[2, -1] - goal[2]))

    # Control bounds
    opti.subject_to(opti.bounded(v_bounds[0], U[0, :], v_bounds[1]))
    opti.subject_to(opti.bounded(omega_bounds[0], U[1, :], omega_bounds[1]))

    # Wall clearance (border walls included in cfg.walls keep us in-bounds)
    r = cfg.robot_radius + clearance
    for wall in walls:
        _wall_clearance_constraints(opti, X, wall, r)

    # Cost — same shape as dense_trajectories._solve
    effort = ca.sumsqr(U)
    smooth = ca.sumsqr(U[:, 1:] - U[:, :-1])
    jerk = ca.sumsqr(U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2])
    w_jerk = 10.0 * w_smooth
    opti.minimize(w_effort * effort + w_smooth * smooth
                  + w_jerk * jerk + w_heading * heading_err ** 2)

    # Warm start from RRT
    opti.set_initial(X, states_init.T)
    opti.set_initial(U, controls_init.T)

    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "sb": "yes", "max_iter": max_iter, "tol": 1e-4},
    )

    try:
        sol = opti.solve()
    except RuntimeError:
        return None

    return np.array(sol.value(X)).T, np.array(sol.value(U)).T


# ── Helpers ─────────────────────────────────────────────────────────────
def _subsample(states: np.ndarray, max_pts: int = 100) -> np.ndarray:
    """Thin a long trajectory before the O(m·n) Fréchet comparison."""
    step = max(1, len(states) // max_pts)
    return states[::step]


def _jitter_goal(cfg: MazeConfig, rng: np.random.Generator,
                 jitter: float, clearance: float) -> np.ndarray:
    """Perturb the goal xy a little (collision-checked) to diversify routes."""
    r = cfg.robot_radius + clearance
    for _ in range(20):
        g = cfg.goal.copy()
        g[:2] += rng.uniform(-jitter, jitter, size=2)
        if not any(w.collides(g[:2], r) for w in cfg.walls):
            return g
    return cfg.goal.copy()


# ── Top-level generator ─────────────────────────────────────────────────
def generate_maze_set(
    kinematics: DiffDriveKinematics,
    cfg: MazeConfig,
    n_seeds: int = 40,
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    clearance: float = 0.05,
    w_effort: float = 1.0,
    w_smooth: float = 10.0,
    goal_jitter: float = 0.15,
    rrt_max_iters: int = 20000,
    rrt_goal_tol: Tuple[float, float] = (0.25, np.pi),
    dedup_thresh: float = 0.15,
    max_path_len: float = 30.0,     # absolute cap (m) — maze paths are long by design
    max_jerk: float = 0.8,          # relaxed vs. open-space: maze turns are sharper
    max_curv: float = 0.02,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> List[Trajectory]:
    """
    One RRT + NLP-smoothing attempt per seed; filter and dedup the results.
    """
    trajs: List[Trajectory] = []
    n_rrt_fail = n_nlp_fail = n_too_long = n_jagged = n_duped = 0
    timed_out = False

    for seed in range(n_seeds):
        if deadline is not None and time.monotonic() > deadline:
            timed_out = True
            break

        rng = np.random.default_rng(seed)
        goal = _jitter_goal(cfg, rng, goal_jitter, clearance)

        planner = RRTPlanner(
            kinematics,
            bounds=cfg.bounds,
            obstacles=cfg.walls,
            robot_radius=cfg.robot_radius,
            v_bounds=v_bounds,
            omega_bounds=omega_bounds,
            dt=dt,
            steps_per_extend=5,
            n_control_samples=15,
            goal_bias=0.15,
            goal_tol=rrt_goal_tol,
            max_iters=rrt_max_iters,
            seed=seed,
        )
        t0 = time.monotonic()
        try:
            states_rrt, controls_rrt = planner.plan(cfg.start, goal)
        except RuntimeError:
            n_rrt_fail += 1
            if verbose:
                print(f"  [seed {seed:3d}] RRT FAILED ({time.monotonic() - t0:.1f}s)")
            continue
        t_rrt = time.monotonic() - t0

        res = smooth_maze_path(
            kinematics, states_rrt, controls_rrt, cfg, goal,
            dt=dt, clearance=clearance,
            v_bounds=v_bounds, omega_bounds=omega_bounds,
            w_effort=w_effort, w_smooth=w_smooth,
        )
        if res is None:
            n_nlp_fail += 1
            if verbose:
                print(f"  [seed {seed:3d}] NLP INFEASIBLE (rrt {t_rrt:.1f}s)")
            continue
        states, controls = res

        path_len = np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
        if path_len > max_path_len:
            n_too_long += 1
            if verbose:
                print(f"  [seed {seed:3d}] TOO LONG ({path_len:.1f} m)")
            continue

        if not _is_smooth(states, controls, max_jerk=max_jerk, max_curv=max_curv):
            n_jagged += 1
            if verbose:
                print(f"  [seed {seed:3d}] JAGGED")
            continue

        cand = Trajectory(
            states=states, controls=controls, dt=dt,
            via_angle=0.0, via_radius=0.0,   # polar metadata — n/a in the maze
            clearance=clearance, w_effort=w_effort, w_smooth=w_smooth,
        )
        cand_sub = _subsample(states)
        if dedup_thresh > 0 and any(
            _frechet(cand_sub, _subsample(t.states)) < dedup_thresh for t in trajs
        ):
            n_duped += 1
            if verbose:
                print(f"  [seed {seed:3d}] DUPLICATE")
            continue

        trajs.append(cand)
        if verbose:
            print(f"  [seed {seed:3d}] KEPT — {len(controls)} steps, "
                  f"{path_len:.1f} m (rrt {t_rrt:.1f}s, total {time.monotonic() - t0:.1f}s)")

    if verbose:
        if timed_out:
            print(f"  ⏱  Deadline reached — returning {len(trajs)} trajectories collected so far.")
        print(f"  RRT failed: {n_rrt_fail}  NLP failed: {n_nlp_fail}  "
              f"Too long: {n_too_long}  Jagged: {n_jagged}  Duplicate: {n_duped}")
        print(f"  Kept: {len(trajs)}")
    return trajs


# ── Coverage sampling helpers ───────────────────────────────────────────
def _pose_free(xy: np.ndarray, walls: List[WallObstacle], r: float) -> bool:
    return not any(w.collides(xy, r) for w in walls)


def _sample_free_xy(rng: np.random.Generator, cfg: MazeConfig,
                    r: float, max_tries: int = 100) -> Optional[np.ndarray]:
    (xmin, ymin), (xmax, ymax) = cfg.bounds
    for _ in range(max_tries):
        xy = rng.uniform([xmin + r, ymin + r], [xmax - r, ymax - r])
        if _pose_free(xy, cfg.walls, r):
            return xy
    return None


def _segment_clear(a: np.ndarray, b: np.ndarray, walls: List[WallObstacle],
                   r: float, step: float = 0.05) -> bool:
    """Line-of-sight check: sample points along a→b against inflated walls."""
    d = np.linalg.norm(b - a)
    n = max(2, int(np.ceil(d / step)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        if not _pose_free(a + t * (b - a), walls, r):
            return False
    return True


def _local_walls(path_xy: np.ndarray, walls: List[WallObstacle],
                 margin: float = 0.6) -> List[WallObstacle]:
    """Walls whose box intersects the path's bounding box (inflated)."""
    lo = path_xy.min(axis=0) - margin
    hi = path_xy.max(axis=0) + margin
    return [w for w in walls
            if w.xmax >= lo[0] and w.xmin <= hi[0]
            and w.ymax >= lo[1] and w.ymin <= hi[1]]


def _straight_guess(a: np.ndarray, theta_a: float, b: np.ndarray,
                    N: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Straight-line initial guess A→B with constant-speed controls."""
    bearing = np.arctan2(b[1] - a[1], b[0] - a[0])
    s = np.linspace(0.0, 1.0, N + 1)
    xy = a[None, :] + s[:, None] * (b - a)[None, :]
    th = np.full(N + 1, bearing)
    th[0] = theta_a
    states = np.column_stack([xy, th])
    v_nom = np.linalg.norm(b - a) / (N * dt)
    controls = np.column_stack([np.full(N, v_nom), np.zeros(N)])
    return states, controls


def _polyline_guess(path_xy: np.ndarray, theta_a: float, v_max: float,
                    dt: float, max_N: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a geometric path (e.g. from A*) into an (states, controls) warm
    start: resample to N points at ~70% of max speed, headings along the
    path, controls from finite differences. Not exactly dynamically
    consistent — the NLP fixes that.
    """
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    N = int(np.clip(L / (0.7 * v_max * dt) + 20, 60, max_N))
    s = np.linspace(0.0, L, N + 1)
    xy = np.stack([np.interp(s, cum, path_xy[:, 0]),
                   np.interp(s, cum, path_xy[:, 1])], axis=1)
    dxy = np.gradient(xy, axis=0)
    th = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    th[0] = theta_a
    states = np.column_stack([xy, th])
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    om = np.diff(th) / dt
    controls = np.column_stack([v, om])
    return states, controls


class _AStarGrid:
    """
    Coarse occupancy grid + 8-connected A* for fast geometric warm
    starts through the maze (replaces RRT in coverage mode: milliseconds
    instead of ~10 s per query).

    The grid is inflated by the pose radius, so A* paths keep the same
    clearance the NLP will be asked to respect.
    """

    def __init__(self, cfg: MazeConfig, r: float, res: float = 0.05):
        (xmin, ymin), (xmax, ymax) = cfg.bounds
        self.xmin, self.ymin, self.res = xmin, ymin, res
        self.nx = int(round((xmax - xmin) / res)) + 1
        self.ny = int(round((ymax - ymin) / res)) + 1
        xs = xmin + np.arange(self.nx) * res
        ys = ymin + np.arange(self.ny) * res
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        for w in cfg.walls:
            dx = np.maximum(np.maximum(w.xmin - Xg, 0.0), Xg - w.xmax)
            dy = np.maximum(np.maximum(w.ymin - Yg, 0.0), Yg - w.ymax)
            occ |= dx * dx + dy * dy <= r * r
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


class _CoverageGrid:
    """Coarse visit-count grid used to bias A-sampling toward sparse cells."""

    def __init__(self, cfg: MazeConfig, n: int = 24):
        (self.xmin, self.ymin), (xmax, ymax) = cfg.bounds
        self.n = n
        self.dx = (xmax - self.xmin) / n
        self.dy = (ymax - self.ymin) / n
        self.counts = np.zeros((n, n))

    def _cell(self, xy: np.ndarray) -> Tuple[int, int]:
        i = int(np.clip((xy[0] - self.xmin) / self.dx, 0, self.n - 1))
        j = int(np.clip((xy[1] - self.ymin) / self.dy, 0, self.n - 1))
        return i, j

    def count(self, xy: np.ndarray) -> float:
        return self.counts[self._cell(xy)]

    def add_path(self, states: np.ndarray) -> None:
        for xy in states[:, :2]:
            self.counts[self._cell(xy)] += 1.0


# ── Coverage generator: one trajectory per random A→B pair ─────────────
def generate_coverage_set(
    kinematics: DiffDriveKinematics,
    cfg: MazeConfig,
    n_pairs: int = 300,
    d_band: Tuple[float, float] = (4.0, 8.5),   # A→B distance range (m)
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    clearance: float = 0.05,
    w_effort: float = 1.0,
    w_smooth: float = 10.0,
    heading_noise: float = np.pi / 2,   # start heading = bearing ± this
    coverage_bias: bool = True,
    n_bias_candidates: int = 8,
    max_length_factor: float = 3.0,     # vs. straight-line A→B distance
    max_jerk: float = 0.8,
    max_curv: float = 0.02,
    seed: int = 0,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> List[Trajectory]:
    """
    Dense coverage of the maze with fast, locally-optimal A→B trajectories:

        sample A, B in free space (distance in `d_band`)
          ├─ line-of-sight clear → straight-line warm start
          └─ blocked → A* grid path warm start (milliseconds)
        → NLP with only the walls near the path → filters → keep.

    One trajectory per pair, no dedup — diversity comes from sampling.
    """
    r_pose = cfg.robot_radius + clearance + 0.02
    rng = np.random.default_rng(seed)
    grid = _CoverageGrid(cfg) if coverage_bias else None
    astar = _AStarGrid(cfg, r_pose)

    trajs: List[Trajectory] = []
    n_los = n_astar_ok = n_astar_fail = n_nlp_fail = n_filtered = 0
    attempts = 0
    max_attempts = 20 * n_pairs
    t_start = time.monotonic()

    while len(trajs) < n_pairs and attempts < max_attempts:
        if deadline is not None and time.monotonic() > deadline:
            if verbose:
                print(f"  ⏱  Deadline reached at {len(trajs)} trajectories.")
            break
        attempts += 1

        # ── Sample A (optionally biased toward unvisited cells) and B ──
        if grid is not None:
            cands = [_sample_free_xy(rng, cfg, r_pose)
                     for _ in range(n_bias_candidates)]
            cands = [c for c in cands if c is not None]
            if not cands:
                continue
            a_xy = min(cands, key=grid.count)
        else:
            a_xy = _sample_free_xy(rng, cfg, r_pose)
            if a_xy is None:
                continue

        b_xy = None
        for _ in range(20):
            ang = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(*d_band)
            cand = a_xy + d * np.array([np.cos(ang), np.sin(ang)])
            (xmin, ymin), (xmax, ymax) = cfg.bounds
            if not (xmin + r_pose <= cand[0] <= xmax - r_pose
                    and ymin + r_pose <= cand[1] <= ymax - r_pose):
                continue
            if _pose_free(cand, cfg.walls, r_pose):
                b_xy = cand
                break
        if b_xy is None:
            continue

        dist = np.linalg.norm(b_xy - a_xy)
        bearing = np.arctan2(b_xy[1] - a_xy[1], b_xy[0] - a_xy[0])
        theta_a = bearing + rng.uniform(-heading_noise, heading_noise)
        goal = np.array([b_xy[0], b_xy[1], 0.0])    # terminal heading free (w_heading=0)

        # ── Warm start: straight line if visible, else A* grid path ──
        if _segment_clear(a_xy, b_xy, cfg.walls, r_pose):
            n_steps = int(np.clip(dist / (0.7 * v_bounds[1] * dt) + 20, 60, 300))
            states_ws, controls_ws = _straight_guess(a_xy, theta_a, b_xy, n_steps, dt)
            n_los += 1
        else:
            path_xy = astar.plan(a_xy, b_xy)
            if path_xy is None:
                n_astar_fail += 1
                continue
            states_ws, controls_ws = _polyline_guess(
                path_xy, theta_a, v_bounds[1], dt)
            n_astar_ok += 1

        # ── NLP with only nearby walls ──
        walls_loc = _local_walls(states_ws[:, :2], cfg.walls)
        res = smooth_maze_path(
            kinematics, states_ws, controls_ws, cfg, goal,
            dt=dt, clearance=clearance,
            v_bounds=v_bounds, omega_bounds=omega_bounds,
            w_effort=w_effort, w_smooth=w_smooth,
            w_heading=0.0, walls=walls_loc, max_iter=300,
        )
        if res is None:
            n_nlp_fail += 1
            continue
        states, controls = res

        # Safety net: the pruned NLP never saw far-away walls, so verify
        # the final path against the full wall list before keeping it.
        if not all(_pose_free(xy, cfg.walls, cfg.robot_radius)
                   for xy in states[:, :2]):
            n_nlp_fail += 1
            continue

        path_len = np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
        if path_len > max_length_factor * dist or \
           not _is_smooth(states, controls, max_jerk=max_jerk, max_curv=max_curv):
            n_filtered += 1
            continue

        trajs.append(Trajectory(
            states=states, controls=controls, dt=dt,
            via_angle=0.0, via_radius=0.0,
            clearance=clearance, w_effort=w_effort, w_smooth=w_smooth,
        ))
        if grid is not None:
            grid.add_path(states)

        if verbose and len(trajs) % 25 == 0:
            rate = len(trajs) / (time.monotonic() - t_start)
            print(f"  {len(trajs):4d}/{n_pairs} trajectories "
                  f"({rate:.1f}/s, attempts {attempts})")

    if verbose:
        elapsed = time.monotonic() - t_start
        print(f"  Attempts: {attempts}  LOS: {n_los}  A* ok/fail: {n_astar_ok}/{n_astar_fail}")
        print(f"  NLP failed: {n_nlp_fail}  Filtered: {n_filtered}")
        print(f"  Kept: {len(trajs)} in {elapsed:.0f}s "
              f"({len(trajs) / max(elapsed, 1e-9):.2f} traj/s)")
    return trajs


# ── Simulation-based generation (no NLP) ───────────────────────────────
def _dist_to_walls(xy: np.ndarray, walls: List[WallObstacle]) -> float:
    """Distance from a point to the nearest wall surface (0 if inside one)."""
    best = np.inf
    for w in walls:
        dx = max(w.xmin - xy[0], 0.0, xy[0] - w.xmax)
        dy = max(w.ymin - xy[1], 0.0, xy[1] - w.ymax)
        best = min(best, np.hypot(dx, dy))
    return best


def _shortcut(path_xy: np.ndarray, grid: _AStarGrid,
              rng: np.random.Generator, iters: int = 60) -> np.ndarray:
    """
    Random shortcutting: repeatedly replace path segments with straight
    lines when the grid says there is line of sight. Removes the A*
    staircase without ever leaving the clearance-inflated free space
    (unlike spline smoothing, which can bulge into walls).
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


def _track_path(
    f_disc,                          # CasADi discrete dynamics f(x, u, dt)
    path_xy: np.ndarray,             # (M, 2) densified reference path
    theta0: float,
    hard_grid: _AStarGrid,           # robot-radius-inflated grid for in-sim collision
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
    Pure-pursuit tracking of a reference path, simulated through the
    true RK4 dynamics with clipped, noise-injected controls. The
    recorded (state, action) pairs are dynamically feasible by
    construction. Returns None on collision or timeout.
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


def generate_sim_set(
    kinematics: DiffDriveKinematics,
    cfg: MazeConfig,
    n_trajs: int = 5000,
    mode_fracs: Tuple[float, float, float] = (0.7, 0.2, 0.1),  # A, B, C
    d_band_a: Tuple[float, float] = (1.5, 8.5),   # A: random pairs
    d_band_b: Tuple[float, float] = (1.0, 4.0),   # B: wall-hugging
    d_band_c: Tuple[float, float] = (2.0, 8.5),   # C: corridor (no-LOS) pairs
    wall_band: float = 0.25,            # B: endpoints within this of a wall
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    clearance: float = 0.05,
    heading_noise: float = np.pi / 2,
    sigma_v: float = 0.02,
    sigma_omega: float = 0.1,
    lookahead: float = 0.25,
    seed: int = 0,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> List[Trajectory]:
    """
    Fast NLP-free generation: A* → shortcut → pure pursuit → RK4 sim.

    We are not solving optimal control here — we are generating a
    sampling distribution over dynamically feasible trajectories,
    biased toward walls (mode B) and corridors (mode C), i.e. toward
    the decision boundary the CBF must learn.
    """
    r_pose = cfg.robot_radius + clearance + 0.02
    rng = np.random.default_rng(seed)
    astar = _AStarGrid(cfg, r_pose)
    # Hard grid for in-sim collision: robot radius + small margin, so
    # noise may eat into the planning clearance (good boundary data)
    # but never into the robot itself.
    hard = _AStarGrid(cfg, cfg.robot_radius + 0.03)
    f_disc = kinematics.discrete_dynamics(dt, method="rk4")

    def sample_endpoint(near_wall: bool) -> Optional[np.ndarray]:
        for _ in range(60):
            xy = _sample_free_xy(rng, cfg, r_pose)
            if xy is None:
                return None
            if not near_wall:
                return xy
            if _dist_to_walls(xy, cfg.walls) < r_pose + wall_band:
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
                print(f"  ⏱  Deadline reached at {len(trajs)} trajectories.")
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
            (xmin, ymin), (xmax, ymax) = cfg.bounds
            if not (xmin + r_pose <= cand[0] <= xmax - r_pose
                    and ymin + r_pose <= cand[1] <= ymax - r_pose):
                continue
            if not _pose_free(cand, cfg.walls, r_pose):
                continue
            if near_wall and _dist_to_walls(cand, cfg.walls) >= r_pose + wall_band:
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

        trajs.append(Trajectory(
            states=states, controls=controls, dt=dt,
            via_angle=0.0, via_radius=0.0,
            clearance=clearance, w_effort=0.0, w_smooth=0.0,
        ))
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


# ── Demo / entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert maze trajectories.")
    parser.add_argument("--mode", choices=["sim", "coverage", "mission"], default="sim",
                        help="sim: fast A*+pure-pursuit simulation (default); "
                             "coverage: NLP-refined random A→B pairs; "
                             "mission: full start→goal paths, one per RRT seed")
    parser.add_argument("--trajs", type=int, default=5000,
                        help="[sim] number of trajectories to generate")
    parser.add_argument("--pairs", type=int, default=300,
                        help="[coverage] number of A→B trajectories to generate")
    parser.add_argument("--dmin", type=float, default=4.0,
                        help="[coverage] min A→B distance (m)")
    parser.add_argument("--dmax", type=float, default=8.5,
                        help="[coverage] max A→B distance (m)")
    parser.add_argument("--seed", type=int, default=0,
                        help="[coverage] RNG seed (vary across runs to get new pairs)")
    parser.add_argument("--stem", type=str, default=None,
                        help="output file stem (default: maze_coverage / maze_trajectories)")
    parser.add_argument("--seeds", type=int, default=40,
                        help="[mission] number of RRT seeds to try")
    parser.add_argument("--budget", type=float, default=600.0, help="wall-clock budget (s)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: maze_env/results)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = RobotModel(wheel_radius=0.05, wheel_base=0.3)
    kin = DiffDriveKinematics(robot)
    cfg = sketch_maze()

    deadline = time.monotonic() + args.budget
    if args.mode == "sim":
        print(f"Sim generation: {args.trajs} trajectories, seed {args.seed}, "
              f"wall-clock budget {args.budget:.0f}s")
        trajs = generate_sim_set(kin, cfg, n_trajs=args.trajs,
                                 seed=args.seed, deadline=deadline)
        stem = args.stem or "maze_sim"
        show_start_goal = False
    elif args.mode == "coverage":
        print(f"Coverage generation: {args.pairs} pairs, d ∈ [{args.dmin}, {args.dmax}] m, "
              f"seed {args.seed}, wall-clock budget {args.budget:.0f}s")
        trajs = generate_coverage_set(kin, cfg, n_pairs=args.pairs,
                                      d_band=(args.dmin, args.dmax),
                                      seed=args.seed, deadline=deadline)
        stem = args.stem or "maze_coverage"
        show_start_goal = False
    else:
        print(f"Mission generation: {args.seeds} seeds, wall-clock budget {args.budget:.0f}s")
        trajs = generate_maze_set(kin, cfg, n_seeds=args.seeds, deadline=deadline)
        stem = args.stem or "maze_trajectories"
        show_start_goal = True
    print(f"\n{len(trajs)} trajectories generated.")

    out_csv = out_dir / f"{stem}.csv"
    n_rows = export_csv(trajs, str(out_csv))
    print(f"Wrote {n_rows} (state, action) rows to {out_csv}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 9))
        plot_maze(ax, cfg, show_start_goal=show_start_goal)
        n_plot = min(len(trajs), 1000)     # don't drown the figure / PNG size
        step = max(1, len(trajs) // n_plot)
        for t in trajs[::step]:
            ax.plot(t.states[:, 0], t.states[:, 1], "-",
                    color="tab:blue", alpha=0.15, linewidth=0.8)
        ax.set_title(f"{len(trajs)} expert maze trajectories ({args.mode}, "
                     f"{n_plot} shown)")
        if show_start_goal:
            ax.legend()
        plt.tight_layout()
        out_png = out_dir / f"{stem}.png"
        plt.savefig(out_png, dpi=110)
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")
