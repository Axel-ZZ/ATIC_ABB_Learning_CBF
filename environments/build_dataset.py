"""
Build a CBF training dataset — the single entry point.

Edit the CONFIG block below (at minimum `ENV`) and run this file (IDE "Run"
or `python environments/build_dataset.py`). It generates the expert
trajectories (cached so re-runs are fast) and writes the FOUR sets the CBF
learners consume to runs/datasets/<tag>/:

    expert_safe.csv    x,y,theta,v,omega   safe expert states (Z_dyn / X_safe),
                                           from the collision-free trajectories,
                                           with the NUTS boundary states below
                                           REMOVED (they go to expert_unsafe), so
                                           the two expert sets are disjoint.
    expert_unsafe.csv  x,y,theta,v,omega   SAMPLING unsafe set (X_N): the
                                           reverse-kNN boundary of the expert
                                           demonstrations (Lindemann et al.,
                                           "Learning Hybrid CBFs", Alg. 1 / NUTS)
                                           — expert states on the edge of the
                                           demonstrated region, at the clearance
                                           standoff from the walls.
    safe.csv           x,y,theta           sampled safe points (clearance ≥
                                           deep_margin).
    unsafe.csv         x,y,theta           GEOMETRIC unsafe set: fills S^c — the
                                           obstacles / boundary walls INFLATED by
                                           the robot radius (the C-space obstacle).

So two safe sets (expert + sampled) and two unsafe sets that encode the two
constructions the project compares: GEOMETRIC (unsafe.csv, fills the walls) vs
SAMPLING (expert_unsafe.csv, NUTS boundary of the demonstrations). Switch worlds
by changing `ENV`. CLI flags (see `--help`) override the CONFIG defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.environment import (ENV_NAMES, build_env,
                                       nearest_obstacle_distance)
from environments.vehicle_dynamics import RobotModel, DiffDriveKinematics
from environments.trajectories import generate_sim_set, export_csv
from environments.dataset_plot import preview_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================== CONFIG ===========================
ENV              = "single_obstacle"   # "single_obstacle" or "maze"
N_SAFE_TRAJS     = 5000                # safe (collision-free) expert trajectories
N_SAFE_SAMPLES   = 10000               # sampled safe points (clearance ≥ deep_margin)
N_UNSAFE_SAMPLES = 50000               # sampled unsafe points (fill S^c)
DEEP_MARGIN      = 0                 # None -> robot_radius + 0.05
# expert-unsafe = reverse-kNN boundary of the expert data (NUTS, Lindemann Alg. 1):
NUTS_CELL         = 0.05               # grid-thin to ~uniform density before kNN (m); 0 disables
NUTS_ETA          = 0.15               # neighbour radius (m) in (x, y)
NUTS_BOUNDARY_PCT = 20                 # flag the sparsest X% of (thinned) expert states as boundary
SEED             = 0
REGEN            = False               # True -> regenerate trajectories, ignore cache
TRAJ_BUDGET      = 600.0               # wall-clock budget for trajectory generation (s)
# ==============================================================


# ── Geometric sampling ────────────────────────────────────────────────────
def sample_unsafe(env, n: int, rng: np.random.Generator, inflate: float = 0.0,
                  batch: int = 200_000) -> np.ndarray:
    """
    GEOMETRIC unsafe set: fill S^c — the configuration-space obstacle, i.e. the
    obstacles / boundary walls INFLATED by the robot radius (`inflate`). A state
    is unsafe when the robot's CENTER is within `inflate` of an obstacle surface
    (clearance < inflate), because the robot disc then overlaps the obstacle.
    This closes the robot-radius "collision ring" that an interior-only fill
    (distance == 0) leaves empty around the walls. Rejection-sampled over the
    workspace expanded to include the border frame. Heading uniform.
    """
    (xmin, ymin), (xmax, ymax) = env.bounds
    m = 0.25 + inflate  # reach into the border-wall frame + its inflation
    lo, hi = np.array([xmin - m, ymin - m]), np.array([xmax + m, ymax + m])
    thresh = max(inflate, 1e-9)
    out, have = [], 0
    while have < n:
        xy = rng.uniform(lo, hi, size=(batch, 2))
        inside = xy[nearest_obstacle_distance(env, xy[:, 0], xy[:, 1]) < thresh]
        out.append(inside)
        have += len(inside)
    xy = np.vstack(out)[:n]
    theta = rng.uniform(-np.pi, np.pi, size=(len(xy), 1))
    return np.hstack([xy, theta])


def sample_safe(env, n: int, rng: np.random.Generator, deep_margin: float,
                batch: int = 200_000) -> np.ndarray:
    """Sampled safe points: clearance ≥ deep_margin, inside the workspace. Heading uniform."""
    (xmin, ymin), (xmax, ymax) = env.bounds
    lo, hi = np.array([xmin, ymin]), np.array([xmax, ymax])
    out, have = [], 0
    while have < n:
        xy = rng.uniform(lo, hi, size=(batch, 2))
        keep = xy[nearest_obstacle_distance(env, xy[:, 0], xy[:, 1]) >= deep_margin]
        out.append(keep)
        have += len(keep)
    xy = np.vstack(out)[:n]
    theta = rng.uniform(-np.pi, np.pi, size=(len(xy), 1))
    return np.hstack([xy, theta])


def nuts_boundary_mask(expert_all: np.ndarray, eta: float, boundary_pct: float,
                       cell: float = 0.05) -> np.ndarray:
    """
    SAMPLING unsafe set X_N — the reverse-kNN boundary of the expert data
    (Lindemann et al. "Learning Hybrid CBFs", Appendix C.1, Algorithm 1 / NUTS).

    Build a KD-tree on the expert (x, y), count neighbours within `eta`, and
    flag the sparsest `boundary_pct`% as boundary points — the edge of the
    demonstrated region, at the robot's clearance standoff from the walls.

    The expert states are first **grid-thinned** to ~uniform spatial density
    (one point per `cell`-sized cell). Without this, a raw neighbour count
    reflects how often the expert revisited a spot, so sparsely-visited
    interior corridors get flagged instead of the walls; after thinning the
    count reflects geometry (the ball is clipped only at the true data edge),
    giving a clean wall-hugging shell.

    Returns a boolean mask over the rows of `expert_all` (True = boundary /
    unsafe), so the caller can partition the expert set into disjoint safe and
    unsafe parts. Points dropped by grid-thinning are never flagged (mask
    False), i.e. they stay in the safe set.
    """
    from scipy.spatial import cKDTree

    if cell and cell > 0:
        q = np.floor((expert_all[:, :2] - expert_all[:, :2].min(axis=0)) / cell).astype(np.int64)
        _, sub_idx = np.unique(q, axis=0, return_index=True)  # indices into expert_all
    else:
        sub_idx = np.arange(len(expert_all))
    sub = expert_all[sub_idx]
    counts = cKDTree(sub[:, :2]).query_ball_point(sub[:, :2], r=eta, return_length=True)
    boundary_local = counts <= np.percentile(counts, boundary_pct)

    mask = np.zeros(len(expert_all), dtype=bool)
    mask[sub_idx[boundary_local]] = True
    return mask


# ── Trajectory caching ────────────────────────────────────────────────────
def _cached_or_generate(csv_path: Path, n_trajs: int, regen: bool, gen_fn) -> pd.DataFrame:
    """Reuse a cached trajectory CSV if it has enough trajectories, else regenerate."""
    if csv_path.exists() and not regen:
        df = pd.read_csv(csv_path)
        if df["traj_id"].nunique() >= n_trajs:
            print(f"  reuse cache {csv_path.name} ({df['traj_id'].nunique()} trajectories)")
            return df
        print(f"  cache {csv_path.name} too small — regenerating")
    trajs = gen_fn()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    export_csv(trajs, str(csv_path))
    return pd.read_csv(csv_path)


# ── CLI (defaults pulled from the CONFIG block) ───────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=ENV_NAMES, default=ENV)
    p.add_argument("--n-safe-trajs", type=int, default=N_SAFE_TRAJS)
    p.add_argument("--n-safe", type=int, default=N_SAFE_SAMPLES)
    p.add_argument("--n-unsafe", type=int, default=N_UNSAFE_SAMPLES)
    p.add_argument("--deep-margin", type=float, default=DEEP_MARGIN)
    p.add_argument("--cell", type=float, default=NUTS_CELL,
                   help="NUTS grid-thinning cell (m); 0 disables thinning")
    p.add_argument("--eta", type=float, default=NUTS_ETA, help="NUTS neighbour radius (m)")
    p.add_argument("--boundary-pct", type=float, default=NUTS_BOUNDARY_PCT,
                   help="NUTS: flag the sparsest X%% of (thinned) expert states as boundary")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--regen", action="store_true", default=REGEN,
                   help="regenerate trajectories even if a cache exists")
    p.add_argument("--budget", type=float, default=TRAJ_BUDGET)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    env = build_env(a.env)
    kin = DiffDriveKinematics(RobotModel(wheel_radius=0.05, wheel_base=0.3))
    rng = np.random.default_rng(a.seed)
    r = env.robot_radius
    deep_margin = a.deep_margin if a.deep_margin is not None else r + 0.05
    results = REPO_ROOT / "environments" / a.env / "results"
    print(f"[build_dataset] env={a.env}")

    # ── 1. SAFE expert trajectories -> expert_safe (Z_dyn / X_safe) ──
    print("Safe trajectories:")
    safe_df = _cached_or_generate(
        results / f"{a.env}_sim.csv", a.n_safe_trajs, a.regen,
        lambda: generate_sim_set(env, kin, n_trajs=a.n_safe_trajs, seed=a.seed,
                                 deadline=time.monotonic() + a.budget))
    expert_all = safe_df[["x", "y", "theta", "v", "omega"]].to_numpy()

    # ── 2. split expert into disjoint safe / unsafe sets ──
    # expert_unsafe (X_N, sampling) = reverse-kNN boundary of the expert data;
    # expert_safe = the remaining collision-free states (boundary REMOVED, so a
    # state is never labelled both safe and unsafe).
    boundary = nuts_boundary_mask(expert_all, eta=a.eta, boundary_pct=a.boundary_pct, cell=a.cell)
    expert_unsafe = expert_all[boundary]
    expert_safe = expert_all[~boundary]

    # ── 3. sampled safe (deep)  &  4. sampled unsafe (fill S^c) ──
    safe = sample_safe(env, a.n_safe, rng, deep_margin)
    unsafe = sample_unsafe(env, a.n_unsafe, rng, inflate=r)

    print(f"  expert_safe  {len(expert_safe):>7}   expert_unsafe {len(expert_unsafe):>7} (NUTS boundary)")
    print(f"  safe(sample) {len(safe):>7}   unsafe(sample) {len(unsafe):>7} (fills S^c)")

    # ── write dataset ──
    cfg = dict(env=a.env, n_safe_trajs=a.n_safe_trajs, n_safe=a.n_safe, n_unsafe=a.n_unsafe,
               deep_margin=deep_margin, nuts_cell=a.cell, nuts_eta=a.eta,
               nuts_boundary_pct=a.boundary_pct, robot_radius=r, seed=a.seed)
    tag = (f"{a.env}_es{len(expert_safe)}_eu{len(expert_unsafe)}"
           f"_sa{len(safe)}_un{len(unsafe)}__"
           + hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6])
    out = REPO_ROOT / "runs" / "datasets" / tag
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(expert_safe, columns=["x", "y", "theta", "v", "omega"]).to_csv(out / "expert_safe.csv", index=False)
    pd.DataFrame(expert_unsafe, columns=["x", "y", "theta", "v", "omega"]).to_csv(out / "expert_unsafe.csv", index=False)
    pd.DataFrame(safe, columns=["x", "y", "theta"]).to_csv(out / "safe.csv", index=False)
    pd.DataFrame(unsafe, columns=["x", "y", "theta"]).to_csv(out / "unsafe.csv", index=False)

    meta = dict(tag=tag, n_expert_safe=len(expert_safe), n_expert_unsafe=len(expert_unsafe),
                n_safe=len(safe), n_unsafe=len(unsafe),
                n_safe_trajectories=int(safe_df["traj_id"].nunique()))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    _preview(env, expert_safe, expert_unsafe, safe, unsafe, out, rng)
    print(f"[done] {out}")


def _preview(env, expert_safe, expert_unsafe, safe, unsafe, out, rng) -> None:
    """Four-set overlay — every set drawn as points (the data is points)."""
    try:
        # zorder above the obstacle patches so the S^c fill stays visible
        sets = [
            {"xy": unsafe,       "label": "unsafe (sampled, S^c)", "color": "tab:red",
             "s": 2, "alpha": 0.4, "zorder": 6, "subsample": 8000},
            {"xy": expert_safe[:, :2], "label": "expert safe", "color": "tab:blue",
             "s": 1, "alpha": 0.4, "zorder": 7, "subsample": 12000},
            {"xy": safe,         "label": "safe (sampled)", "color": "tab:green",
             "s": 2, "alpha": 0.5, "zorder": 7, "subsample": 4000},
            {"xy": expert_unsafe, "label": "expert unsafe (NUTS)", "color": "tab:orange",
             "s": 4, "alpha": 0.8, "zorder": 8, "subsample": 6000},
        ]
        path = preview_dataset(sets, out / "preview.png", env=env, rng=rng,
                               title=f"{env.name} dataset (4 sets)")
        print(f"  saved {path}")
    except Exception as e:
        print(f"  (preview skipped: {e})")


if __name__ == "__main__":
    main()
