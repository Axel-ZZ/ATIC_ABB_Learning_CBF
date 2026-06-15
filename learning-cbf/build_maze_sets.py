"""
Build generated-sets/set_02 (maze world) for CBF training.

Produces three files, ready for the NN training code:

    X_safe.csv       x,y,theta,v,omega   — expert states from maze_sim.csv
                     (same schema as set_01, loads without code changes)
    N.csv            x,y,theta,v,omega   — SAMPLED collision states:
                     uniform states with dist_to_wall < robot_radius,
                     plus a share concentrated in the thin band just
                     inside collision (pins the zero level set).
                     Unlike set_01, these are true negatives — not
                     relabeled near-wall expert states.
    transitions.csv  x,y,theta,v,omega,x_next,y_next,theta_next,dt
                     — exact RK4 transitions (consecutive sim steps),
                     for the discrete invariance loss
                         h(x') >= (1 - alpha*dt) * h(x)

Label semantics (documented in set_02/README.md):
    safe     d(x) >= 0.126 m   (all expert states satisfy this)
    unsafe   d(x) <  0.10  m   (= robot radius: true collision)
    buffer   0.10 - 0.126 m    unlabeled — the zero level set lives here

Run:  python build_maze_sets.py [--n-unsafe 50000] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "maze_env"))
sys.path.insert(0, str(REPO / "expert_data_generation"))

from maze_config import sketch_maze, plot_maze   # noqa: E402

BAND_WIDTH = 0.03            # unsafe-side boundary band: d in [r - 0.03, r)
HEADER = "x,y,theta,v,omega"


def dist_to_walls(xy: np.ndarray, walls) -> np.ndarray:
    """Vectorized distance from points (n, 2) to the nearest wall surface."""
    d = np.full(len(xy), np.inf)
    for w in walls:
        dx = np.maximum(np.maximum(w.xmin - xy[:, 0], 0.0), xy[:, 0] - w.xmax)
        dy = np.maximum(np.maximum(w.ymin - xy[:, 1], 0.0), xy[:, 1] - w.ymax)
        d = np.minimum(d, np.hypot(dx, dy))
    return d


def sample_unsafe(cfg, n: int, d_max: float, d_min: float,
                  rng: np.random.Generator,
                  control_pool: np.ndarray) -> np.ndarray:
    """
    Rejection-sample states with wall distance in [d_min, d_max).

    Controls are bootstrap-resampled *jointly* from the safe data
    (`control_pool`), so the (v, omega) distribution is identical in
    both classes and carries zero label information. (Range-matched
    uniform sampling is not enough: pure pursuit's controls are highly
    peaked, and a model with control inputs will happily classify on
    that density difference and ignore position entirely.)
    """
    (xmin, ymin), (xmax, ymax) = cfg.bounds
    out = []
    while sum(len(o) for o in out) < n:
        # Sample slightly beyond the workspace so the border walls get
        # negative samples too.
        xy = rng.uniform([xmin - 0.1, ymin - 0.1],
                         [xmax + 0.1, ymax + 0.1], size=(4096, 2))
        d = dist_to_walls(xy, cfg.walls)
        keep = xy[(d >= d_min) & (d < d_max)]
        out.append(keep)
    xy = np.concatenate(out)[:n]
    theta = rng.uniform(-np.pi, np.pi, n)
    vu = control_pool[rng.integers(0, len(control_pool), n)]
    return np.column_stack([xy, theta, vu])


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive rows within a trajectory are exact RK4 steps:
    row k holds (x_k, u_k) and row k+1 holds x_{k+1}.
    """
    nxt = df.groupby("traj_id")[["x", "y", "theta"]].shift(-1)
    tr = pd.DataFrame({
        "x": df["x"], "y": df["y"], "theta": df["theta"],
        "v": df["v"], "omega": df["omega"],
        "x_next": nxt["x"], "y_next": nxt["y"], "theta_next": nxt["theta"],
    }).dropna()
    tr["dt"] = 0.1
    return tr


def verify_transitions(tr: pd.DataFrame, n_check: int = 200,
                       tol: float = 1e-4) -> float:
    """Re-integrate a sample of transitions with the model's RK4."""
    from model import RobotModel, DiffDriveKinematics
    kin = DiffDriveKinematics(RobotModel(0.05, 0.3))
    f = kin.discrete_dynamics(dt=0.1, method="rk4")
    rows = tr.sample(n_check, random_state=0)
    err = 0.0
    for _, r in rows.iterrows():
        pred = np.asarray(f([r.x, r.y, r.theta], [r.v, r.omega], r["dt"])).flatten()
        err = max(err, np.abs(pred - [r.x_next, r.y_next, r.theta_next]).max())
    assert err < tol, f"transition mismatch vs RK4: {err:.2e} >= {tol}"
    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sim-csv", type=str,
                        default=str(REPO / "maze_env/results/maze_sim.csv"))
    parser.add_argument("--out", type=str,
                        default=str(REPO / "learning-cbf/generated-sets/set_02"))
    parser.add_argument("--n-unsafe", type=int, default=50000)
    parser.add_argument("--band-frac", type=float, default=0.4,
                        help="share of unsafe samples in the near-boundary band")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = sketch_maze()
    r = cfg.robot_radius
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Safe set: all expert sim states ──
    df = pd.read_csv(args.sim_csv)
    X_safe = df[["x", "y", "theta", "v", "omega"]].to_numpy()
    d_safe = dist_to_walls(X_safe[:, :2], cfg.walls)
    assert d_safe.min() >= r, "expert state inside collision radius?!"
    print(f"X_safe: {len(X_safe)} states (wall distance min {d_safe.min():.3f})")

    # ── Unsafe set: sampled true collision states ──
    pool = df[["v", "omega"]].to_numpy()
    n_band = int(args.band_frac * args.n_unsafe)
    n_uni = args.n_unsafe - n_band
    N_uni = sample_unsafe(cfg, n_uni, d_max=r, d_min=0.0, rng=rng,
                          control_pool=pool)
    N_band = sample_unsafe(cfg, n_band, d_max=r, d_min=r - BAND_WIDTH, rng=rng,
                           control_pool=pool)
    N = np.vstack([N_uni, N_band])
    rng.shuffle(N)
    print(f"N: {len(N)} states ({n_uni} uniform in collision, "
          f"{n_band} in band [{r - BAND_WIDTH:.2f}, {r:.2f}))")

    # ── Transitions for the invariance loss ──
    tr = build_transitions(df)
    err = verify_transitions(tr)
    print(f"transitions: {len(tr)} (RK4 consistency check: max err {err:.1e})")

    # ── Write ──
    np.savetxt(out / "X_safe.csv", X_safe, delimiter=",",
               header=HEADER, comments="", fmt="%.6f")
    np.savetxt(out / "N.csv", N, delimiter=",",
               header=HEADER, comments="", fmt="%.6f")
    tr.to_csv(out / "transitions.csv", index=False, float_format="%.6f")
    print(f"wrote {out / 'X_safe.csv'}")
    print(f"wrote {out / 'N.csv'}")
    print(f"wrote {out / 'transitions.csv'}")

    # ── Preview plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 9))
        sub_s = X_safe[rng.choice(len(X_safe), 20000, replace=False)]
        sub_n = N[rng.choice(len(N), 20000, replace=False)]
        ax.scatter(sub_s[:, 0], sub_s[:, 1], s=1, c="tab:green", alpha=0.2,
                   label=f"X_safe ({len(X_safe)})")
        ax.scatter(sub_n[:, 0], sub_n[:, 1], s=1, c="tab:red", alpha=0.2,
                   label=f"N ({len(N)})")
        plot_maze(ax, cfg)
        ax.legend(loc="upper right", markerscale=10)
        ax.set_title("set_02: safe (green) vs sampled unsafe (red), 20k each shown")
        plt.tight_layout()
        plt.savefig(out / "preview.png", dpi=110)
        print(f"wrote {out / 'preview.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")
