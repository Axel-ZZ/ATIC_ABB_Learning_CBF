"""
Dense expert trajectory generation around a single obstacle.

Strategy: polar grid of via-points around the obstacle center, each
defining one trajectory's initial guess. The CasADi NLP smooths each
into a locally-optimal trajectory respecting dynamics and clearance.

This gives dense angular coverage of paths "kissing" the obstacle from
every direction, plus radial coverage (close-hugging vs. wide berths).

Exports (state, action) pairs to CSV.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from model import RobotModel, DiffDriveKinematics
from rrt_diff_drive import CircleObstacle


# ── Trajectory bookkeeping ──────────────────────────────────────────────
@dataclass
class Trajectory:
    states: np.ndarray       # (N+1, 3)
    controls: np.ndarray     # (N, 2)
    dt: float
    via_angle: float         # phi in radians, where via-point sits
    via_radius: float        # R, distance from obstacle center
    clearance: float
    w_effort: float
    w_smooth: float


# ── Initial guess construction ──────────────────────────────────────────
def _build_via_initial_guess(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle: CircleObstacle,
    via_angle: float,
    via_radius: float,
    n_points: int,
) -> np.ndarray:
    """
    Arc-wrapping path: start -> tangent T1 on inflated circle -> arc to T2 -> goal.
    The arc is taken on the side that contains the via_angle, so a phi on the
    "far" side of start->goal wraps the obstacle smoothly. Resampled uniformly.
    """
    c = np.array([obstacle.x, obstacle.y])
    R = via_radius
    a_start = np.arctan2(start[1] - c[1], start[0] - c[0])
    a_goal  = np.arctan2(goal[1]  - c[1], goal[0]  - c[0])
    # Walk from a_start to a_goal in the direction that passes through via_angle.
    def _unwrap_to(a, ref):  # bring a within (ref - pi, ref + pi]
        while a - ref > np.pi:  a -= 2 * np.pi
        while a - ref <= -np.pi: a += 2 * np.pi
        return a
    a_via_pos = _unwrap_to(via_angle, a_start)        # via reached going CCW
    a_goal_ccw = _unwrap_to(a_goal, a_start)
    # Choose direction so the via lies on the arc between start-angle and goal-angle.
    if (a_via_pos - a_start) * (a_goal_ccw - a_start) >= 0 and \
       abs(a_via_pos - a_start) <= abs(a_goal_ccw - a_start):
        a_goal_used = a_goal_ccw
    else:
        # Go the other way around
        a_goal_used = a_goal_ccw - np.copysign(2 * np.pi, a_goal_ccw - a_start)
    # Sample arc points around the inflated circle.
    n_arc = max(8, n_points // 3)
    arc_angles = np.linspace(a_start, a_goal_used, n_arc)
    arc_xy = np.column_stack([c[0] + R * np.cos(arc_angles), c[1] + R * np.sin(arc_angles)])
    waypoints = np.vstack([start[:2], arc_xy, goal[:2]])

    seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        xy = np.tile(start[:2], (n_points, 1))
    else:
        s = np.linspace(0, total, n_points)
        xy = np.stack([
            np.interp(s, cum, waypoints[:, 0]),
            np.interp(s, cum, waypoints[:, 1]),
        ], axis=1)

    dxy = np.gradient(xy, axis=0)
    th = np.arctan2(dxy[:, 1], dxy[:, 0])
    th[0] = start[2]
    th[-1] = goal[2]
    return np.column_stack([xy, th])


# ── CasADi NLP (single obstacle) ────────────────────────────────────────
def _solve(
    kinematics: DiffDriveKinematics,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle: CircleObstacle,
    N: int,
    dt: float,
    clearance: float,
    w_effort: float,
    w_smooth: float,
    v_bounds: Tuple[float, float],
    omega_bounds: Tuple[float, float],
    initial_guess: np.ndarray,
    via_xy: Optional[np.ndarray] = None,    # single via target (legacy)
    w_via: float = 40.0,                    # weight pulling trajectory toward via point(s)
    via_anchor_fracs: Sequence[float] = (0.25, 0.5, 0.75),  # where along path to anchor
    via_arc: Optional[np.ndarray] = None,   # (M, 2) arc points — one per via_anchor_fracs entry
    w_hug: float = 0.0,                     # weight pulling mid-trajectory onto hug_radius
    hug_radius: Optional[float] = None,     # target distance from obstacle center
    hug_sigma_frac: float = 0.25,           # Gaussian width as fraction of N
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    f_disc = kinematics.discrete_dynamics(dt, method="rk4")

    opti = ca.Opti()
    X = opti.variable(3, N + 1)
    U = opti.variable(2, N)

    # Dynamics
    for k in range(N):
        opti.subject_to(X[:, k + 1] == f_disc(X[:, k], U[:, k], dt))

    # Boundary
    opti.subject_to(X[:, 0] == start)
    opti.subject_to(X[0, -1] == goal[0])
    opti.subject_to(X[1, -1] == goal[1])
    # Soft terminal heading (added to cost below)
    heading_err = ca.sin(0.5 * (X[2, -1] - goal[2]))

    # Control bounds
    opti.subject_to(opti.bounded(v_bounds[0], U[0, :], v_bounds[1]))
    opti.subject_to(opti.bounded(omega_bounds[0], U[1, :], omega_bounds[1]))

    # Obstacle clearance
    r = obstacle.radius + clearance
    for k in range(N + 1):
        dx = X[0, k] - obstacle.x
        dy = X[1, k] - obstacle.y
        opti.subject_to(dx * dx + dy * dy >= r * r)

    # Cost
    effort = ca.sumsqr(U)
    smooth = ca.sumsqr(U[:, 1:] - U[:, :-1])
    jerk = ca.sumsqr(U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2])  # 2nd diff of controls
    w_heading = 50.0
    w_jerk = 10.0 * w_smooth
    cost = w_effort * effort + w_smooth * smooth + w_jerk * jerk + w_heading * heading_err**2

    # Via-point anchor: pull the trajectory's midpoint toward the via target.
    # This is what creates dense coverage — without it, all paths collapse
    # to the same local optimum.
    if via_arc is not None:
        # Distinct target per anchor → forces path to wrap along the arc.
        n_anchors = len(via_anchor_fracs)
        assert via_arc.shape[0] == n_anchors, "via_arc must have one row per anchor frac"
        for i, frac in enumerate(via_anchor_fracs):
            k_a = int(frac * N)
            dx_via = X[0, k_a] - via_arc[i, 0]
            dy_via = X[1, k_a] - via_arc[i, 1]
            cost = cost + (w_via / n_anchors) * (dx_via**2 + dy_via**2)
    elif via_xy is not None:
        n_anchors = max(1, len(via_anchor_fracs))
        for frac in via_anchor_fracs:
            k_a = int(frac * N)
            dx_via = X[0, k_a] - via_xy[0]
            dy_via = X[1, k_a] - via_xy[1]
            cost = cost + (w_via / n_anchors) * (dx_via**2 + dy_via**2)

    # Hug cost: pull mid-trajectory onto an annulus at hug_radius.
    # Gaussian weighting centred at k_mid so start/goal aren't dragged.
    if w_hug > 0.0 and hug_radius is not None:
        k_mid = N // 2
        sigma = max(1.0, hug_sigma_frac * N)
        for k in range(N + 1):
            wk = float(np.exp(-((k - k_mid) ** 2) / (2.0 * sigma * sigma)))
            dxk = X[0, k] - obstacle.x
            dyk = X[1, k] - obstacle.y
            # Penalise (distance - hug_radius)² — gentle, well-scaled.
            d = ca.sqrt(dxk * dxk + dyk * dyk + 1e-8)
            cost = cost + w_hug * wk * (d - hug_radius) ** 2

    opti.minimize(cost)

    # Resample initial guess to (N+1, 3) if needed
    if initial_guess.shape[0] != N + 1:
        s = np.linspace(0, 1, initial_guess.shape[0])
        s_new = np.linspace(0, 1, N + 1)
        ig = np.stack([np.interp(s_new, s, initial_guess[:, i]) for i in range(3)], axis=1)
    else:
        ig = initial_guess
    opti.set_initial(X, ig.T)
    opti.set_initial(U, np.zeros((2, N)))

    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "sb": "yes", "max_iter": 300, "tol": 1e-4},
    )

    try:
        sol = opti.solve()
    except RuntimeError:
        return None

    return np.array(sol.value(X)).T, np.array(sol.value(U)).T


# ── Smoothness filter ──────────────────────────────────────────────────
def _max_control_jerk(controls: np.ndarray) -> float:
    """Max ‖U[k+1] - 2U[k] + U[k-1]‖ — large spikes indicate kinks."""
    if len(controls) < 3:
        return 0.0
    d2u = controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
    return float(np.linalg.norm(d2u, axis=1).max())


def _max_xy_curvature(states: np.ndarray) -> float:
    """Max ‖xy[k+1] - 2 xy[k] + xy[k-1]‖ — kink detector in path space."""
    xy = states[:, :2]
    if len(xy) < 3:
        return 0.0
    d2 = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]
    return float(np.linalg.norm(d2, axis=1).max())


def _is_smooth(states: np.ndarray, controls: np.ndarray,
               max_jerk: float = 0.6, max_curv: float = 0.01) -> bool:
    return (_max_control_jerk(controls) < max_jerk
            and _max_xy_curvature(states) < max_curv)


# ── Discrete Fréchet for dedup ──────────────────────────────────────────
def _frechet(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p[:, :2], q[:, :2]
    m, n = len(p), len(q)
    ca_mat = np.full((m, n), -1.0)
    ca_mat[0, 0] = np.linalg.norm(p[0] - q[0])
    for i in range(1, m):
        ca_mat[i, 0] = max(ca_mat[i - 1, 0], np.linalg.norm(p[i] - q[0]))
    for j in range(1, n):
        ca_mat[0, j] = max(ca_mat[0, j - 1], np.linalg.norm(p[0] - q[j]))
    for i in range(1, m):
        for j in range(1, n):
            d = np.linalg.norm(p[i] - q[j])
            ca_mat[i, j] = max(min(ca_mat[i - 1, j], ca_mat[i - 1, j - 1], ca_mat[i, j - 1]), d)
    return float(ca_mat[m - 1, n - 1])


# ── Top-level generator ─────────────────────────────────────────────────
def generate_dense_set(
    kinematics: DiffDriveKinematics,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle: CircleObstacle,
    n_angles: int = 36,
    radii: Sequence[float] = (None,),         # in absolute units (center-to-via). None = auto.
    radius_offsets: Sequence[float] = (0.1, 0.25, 0.5, 1.0),  # clearance levels added to obs radius
    clearances: Sequence[float] = (0.08, 0.2),
    cost_grid: Sequence[Tuple[float, float]] = ((1.0, 5.0),),
    hug_strengths: Sequence[float] = (0.0,),  # weights for boundary-hugging cost
    arc_spans_deg: Sequence[float] = (0.0,),  # ±angle (deg) the via-arc spans; >0 → wrap
    N_steps: int = 200,
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    dedup_thresh: float = 0.08,
    max_length_factor: float = 2.5,    # reject paths > factor * straight-line distance
    max_jerk: float = 0.6,             # reject if max 2nd-diff of U exceeds this
    max_curv: float = 0.01,            # reject if max 2nd-diff of xy exceeds this
    deadline: Optional[float] = None,  # absolute time.monotonic() — break early when exceeded
    verbose: bool = True,
) -> List[Trajectory]:
    """
    Polar sweep of via-points: for each (angle, radius-offset, clearance, cost),
    build an initial guess, solve, dedup.
    """
    trajs: List[Trajectory] = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    straight = np.linalg.norm(goal[:2] - start[:2])

    attempted = 0
    infeasible = 0
    too_long = 0
    duped = 0
    jagged = 0
    timed_out = False

    class _Timeout(Exception):
        pass

    try:
     for phi in angles:
        for r_off in radius_offsets:
            R = obstacle.radius + r_off
            for clear in clearances:
                for w_e, w_s in cost_grid:
                  for w_hug in hug_strengths:
                   for span_deg in arc_spans_deg:
                    if deadline is not None and time.monotonic() > deadline:
                        timed_out = True
                        raise _Timeout
                    attempted += 1
                    ig = _build_via_initial_guess(start, goal, obstacle, phi, R, N_steps + 1)
                    # Build an arc of 3 via points around phi at radius R.
                    span = np.deg2rad(span_deg)
                    arc_phis = np.linspace(phi - span, phi + span, 3)
                    via_arc = np.column_stack([
                        obstacle.x + R * np.cos(arc_phis),
                        obstacle.y + R * np.sin(arc_phis),
                    ])
                    hug_r = obstacle.radius + clear + 0.01
                    res = _solve(
                        kinematics, start, goal, obstacle,
                        N_steps, dt, clear, w_e, w_s,
                        v_bounds, omega_bounds, ig,
                        via_arc=via_arc,
                        w_via=80.0,                     # arc pull — let smoothness still shape ends
                        via_anchor_fracs=(0.25, 0.5, 0.75),
                        w_hug=w_hug, hug_radius=hug_r,
                    )
                    if res is None:
                        infeasible += 1
                        if verbose:
                            print(f"    [phi={np.degrees(phi):.0f}° r_off={r_off} clear={clear} hug={w_hug}] INFEASIBLE")
                        continue
                    states, controls = res

                    # Length filter: reject pathological wraparounds
                    path_len = np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
                    if path_len > max_length_factor * straight:
                        too_long += 1
                        if verbose:
                            print(f"    [phi={np.degrees(phi):.0f}° r_off={r_off} clear={clear} hug={w_hug}] TOO LONG ({path_len:.2f} > {max_length_factor*straight:.2f})")
                        continue

                    if not _is_smooth(states, controls, max_jerk=max_jerk, max_curv=max_curv):
                        jagged += 1
                        if verbose:
                            print(f"    [phi={np.degrees(phi):.0f}° r_off={r_off} clear={clear} hug={w_hug}] JAGGED "
                                  f"(jerk={_max_control_jerk(controls):.3f}, curv={_max_xy_curvature(states):.4f})")
                        continue

                    candidate = Trajectory(
                        states=states, controls=controls, dt=dt,
                        via_angle=phi, via_radius=R,
                        clearance=clear, w_effort=w_e, w_smooth=w_s,
                    )
                    if dedup_thresh > 0 and any(_frechet(candidate.states, t.states) < dedup_thresh for t in trajs):
                        duped += 1
                        continue
                    trajs.append(candidate)
    except _Timeout:
        pass

    if verbose:
        if timed_out:
            print(f"  ⏱  Deadline reached — returning {len(trajs)} trajectories collected so far.")
        print(f"  Attempted: {attempted}")
        print(f"  Infeasible: {infeasible}")
        print(f"  Too long (rejected): {too_long}")
        print(f"  Jagged (rejected): {jagged}")
        print(f"  Duplicate: {duped}")
        print(f"  Kept: {len(trajs)}")
    return trajs


# ── Grid-based streamline generator ────────────────────────────────────
def generate_grid_set(
    kinematics: DiffDriveKinematics,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle: CircleObstacle,
    grid_nx: int = 20,
    grid_ny: int = 20,
    bbox_margin: float = 0.8,
    grid_min_clearance: float = 0.05,    # skip via-points closer than this to obstacle
    clearance: float = 0.01,
    w_effort: float = 1.0,
    w_smooth: float = 20.0,
    N_steps: int = 160,
    dt: float = 0.1,
    v_bounds: Tuple[float, float] = (-0.2, 0.5),
    omega_bounds: Tuple[float, float] = (-2.0, 2.0),
    dedup_thresh: float = 0.04,
    max_length_factor: float = 3.0,
    max_jerk: float = 0.6,
    max_curv: float = 0.01,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> List[Trajectory]:
    """
    Place via-points on a Cartesian grid covering the bbox(start, goal).
    Each grid cell outside the inflated obstacle anchors one trajectory's
    midpoint -> fills the whole space with streamline-like paths.
    """
    xs = np.linspace(min(start[0], goal[0]) - bbox_margin,
                     max(start[0], goal[0]) + bbox_margin, grid_nx)
    ys = np.linspace(min(start[1], goal[1]) - bbox_margin,
                     max(start[1], goal[1]) + bbox_margin, grid_ny)
    straight = np.linalg.norm(goal[:2] - start[:2])
    r_skip = obstacle.radius + grid_min_clearance

    trajs: List[Trajectory] = []
    attempted = infeasible = too_long = duped = skipped = jagged = 0
    timed_out = False

    for gx in xs:
        if timed_out:
            break
        for gy in ys:
            if deadline is not None and time.monotonic() > deadline:
                timed_out = True
                break
            d_obs = np.hypot(gx - obstacle.x, gy - obstacle.y)
            if d_obs < r_skip:
                skipped += 1
                continue
            attempted += 1
            via_xy = np.array([gx, gy])
            # Initial guess: straight start -> via -> goal (polar arc not needed; grid
            # via already disambiguates side). _build_via_initial_guess works with
            # any radius, so reuse it via the angle/radius of this grid point.
            phi = np.arctan2(gy - obstacle.y, gx - obstacle.x)
            R   = max(d_obs, obstacle.radius + 0.05)
            ig = _build_via_initial_guess(start, goal, obstacle, phi, R, N_steps + 1)
            res = _solve(
                kinematics, start, goal, obstacle,
                N_steps, dt, clearance, w_effort, w_smooth,
                v_bounds, omega_bounds, ig,
                via_xy=via_xy,
                w_via=80.0,                          # strong: actually visit the grid cell
                via_anchor_fracs=(0.5,),             # single midpoint anchor only
            )
            if res is None:
                infeasible += 1
                continue
            states, controls = res
            path_len = np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
            if path_len > max_length_factor * straight:
                too_long += 1
                continue
            if not _is_smooth(states, controls, max_jerk=max_jerk, max_curv=max_curv):
                jagged += 1
                continue
            cand = Trajectory(
                states=states, controls=controls, dt=dt,
                via_angle=phi, via_radius=R,
                clearance=clearance, w_effort=w_effort, w_smooth=w_smooth,
            )
            if dedup_thresh > 0 and any(_frechet(cand.states, t.states) < dedup_thresh for t in trajs):
                duped += 1
                continue
            trajs.append(cand)

    if verbose:
        if timed_out:
            print(f"  ⏱  Deadline reached — returning {len(trajs)} trajectories collected so far.")
        print(f"  Grid cells: {grid_nx * grid_ny}, skipped (in obstacle): {skipped}")
        print(f"  Attempted: {attempted}  Infeasible: {infeasible}  Too long: {too_long}  Jagged: {jagged}  Duped: {duped}")
        print(f"  Kept: {len(trajs)}")
    return trajs


# ── CSV export ──────────────────────────────────────────────────────────
def export_csv(trajs: List[Trajectory], path: str) -> int:
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "traj_id", "t", "step",
            "x", "y", "theta",
            "v", "omega",
            "via_angle_deg", "via_radius", "clearance", "w_effort", "w_smooth",
        ])
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
                    f"{np.degrees(traj.via_angle):.1f}",
                    f"{traj.via_radius:.3f}",
                    f"{traj.clearance:.3f}",
                    f"{traj.w_effort:.3f}",
                    f"{traj.w_smooth:.3f}",
                ])
                rows += 1
    return rows


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    robot = RobotModel(wheel_radius=0.05, wheel_base=0.3)
    kin = DiffDriveKinematics(robot)

    obstacle = CircleObstacle(2.0, 2.0, 0.5)
    start = np.array([0.0, 0.0, 0.0])
    goal_center = np.array([4.0, 4.0, np.pi / 2])

    # ── Wall-clock budget. On timeout, partial results are saved/plotted. ──
    TIME_BUDGET_S = 600.0
    deadline = time.monotonic() + TIME_BUDGET_S
    print(f"Wall-clock budget: {TIME_BUDGET_S:.0f}s")

    # ── Goal sweep: offsets along a short line perpendicular to start→goal ──
    # Diversifies which side of the obstacle the optimum favors.
    se = goal_center[:2] - start[:2]
    n_perp = np.array([-se[1], se[0]]) / (np.linalg.norm(se) + 1e-9)
    GOAL_LINE_HALF_LEN = 0.30                   # ±0.30 m along the perpendicular
    GOAL_N            = 5                       # 5 goal positions
    lambdas           = np.linspace(-GOAL_LINE_HALF_LEN, GOAL_LINE_HALF_LEN, GOAL_N)
    goals = [np.array([goal_center[0] + lam * n_perp[0],
                       goal_center[1] + lam * n_perp[1],
                       goal_center[2]]) for lam in lambdas]
    print(f"Sweeping {GOAL_N} goals along ±{GOAL_LINE_HALF_LEN} m perpendicular to start→goal.")

    trajs: List[Trajectory] = []
    for gi, goal in enumerate(goals):
        if time.monotonic() > deadline:
            print(f"⏱  Deadline reached before goal {gi+1}/{GOAL_N} — stopping sweep.")
            break
        print(f"\n── Goal {gi+1}/{GOAL_N}: ({goal[0]:.2f}, {goal[1]:.2f}) ──")

        print("Generating streamline grid (fills space) ...")
        trajs_grid = generate_grid_set(
            kin, start, goal, obstacle,
            grid_nx=12, grid_ny=12,             # smaller per goal — budget shared across 5
            bbox_margin=0.7, grid_min_clearance=0.05,
            clearance=0.015, w_effort=1.0, w_smooth=25.0,
            N_steps=160, dt=0.1,
            v_bounds=(-0.2, 0.5), omega_bounds=(-2.0, 2.0),
            dedup_thresh=0.05, max_length_factor=2.5,
            max_jerk=0.6, max_curv=0.01,
            deadline=deadline,
        )
        print("Generating wall-hugging polar set (follows curvature) ...")
        trajs_hug = generate_dense_set(
            kin, start, goal, obstacle,
            n_angles=24,                        # 15° resolution per goal
            radius_offsets=(0.05, 0.14, 0.30),  # 3 bands: tight, mid, wide
            clearances=(0.015,),
            cost_grid=((1.0, 25.0),),
            hug_strengths=(2.0, 6.0),
            arc_spans_deg=(20.0, 40.0),
            N_steps=160, dt=0.1,
            v_bounds=(-0.2, 0.5), omega_bounds=(-2.0, 2.0),
            dedup_thresh=0.05, max_length_factor=2.5,
            max_jerk=0.6, max_curv=0.01,
            deadline=deadline,
        )
        trajs.extend(trajs_grid)
        trajs.extend(trajs_hug)
        print(f"  Goal {gi+1}: grid={len(trajs_grid)}  hug={len(trajs_hug)}  running total={len(trajs)}")

    # Use the central goal for plotting reference.
    goal = goal_center
    print(f"\n{len(trajs)} unique trajectories generated across {GOAL_N} goals.")

    out_csv = "/home/bb/Desktop/atic-cbfs/results/dense_trajectories.csv"
    n_rows = export_csv(trajs, out_csv)
    print(f"Wrote {n_rows} (state, action) rows to {out_csv}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.add_patch(plt.Circle((obstacle.x, obstacle.y), obstacle.radius,
                                color="tab:red", alpha=0.5, zorder=5))
        # Show all trajectories with alpha for density visualization
        for t in trajs:
            ax.plot(t.states[:, 0], t.states[:, 1], "-",
                    color="tab:blue", alpha=0.25, linewidth=0.9)
        ax.plot(*start[:2], "go", markersize=14, label="start", zorder=10)
        gxy = np.array([g[:2] for g in goals])
        ax.plot(gxy[:, 0], gxy[:, 1], "r*", markersize=14, label="goals", zorder=10)
        ax.set_aspect("equal")
        ax.set_title(f"{len(trajs)} dense expert trajectories — {GOAL_N}-goal sweep")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig("/home/bb/Desktop/atic-cbfs/results/dense_trajectories.png", dpi=110)
        print("Saved /results/dense_trajectories.png")
    except Exception as e:
        print(f"(plot skipped: {e})")