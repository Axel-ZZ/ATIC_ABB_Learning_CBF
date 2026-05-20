"""
Build the three-set CSV layout for CBF learning from a trajectories run.

Usage:
    python build_three_sets.py --traj <traj_tag>
        [--sigma 0.10] [--deep-target 3000] [--deep-margin 1.5]
        [--obstacle x,y,r]    # defaults to the obstacle in the traj config

Reads:
    runs/trajectories/<traj_tag>/trajectories.csv  +  config.json
Writes:
    runs/datasets/<ds_tag>/{expert_safe,unsafe,safe}.csv  +  config.json + meta + preview
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / "expert_data_generation"))

import pipeline_io as pio                                          # noqa: E402
from rrt_diff_drive import CircleObstacle                          # noqa: E402


def sample_deep_safe_interior(obstacle, bbox, n_target, margin_factor=1.5, seed=0):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox
    thresh = obstacle.radius * (1.0 + margin_factor)
    kept, n = [], 0
    while n < n_target:
        b = max(1024, n_target - n)
        x = rng.uniform(xmin, xmax, b)
        y = rng.uniform(ymin, ymax, b)
        ok = np.hypot(x - obstacle.x, y - obstacle.y) > thresh
        if ok.any():
            th = rng.uniform(0, 2 * np.pi, int(ok.sum()))
            kept.append(np.column_stack([x[ok], y[ok], th]))
            n += int(ok.sum())
    return np.vstack(kept)[:n_target]


def _parse_obstacle(s: str | None, traj_cfg: dict) -> CircleObstacle:
    if s is None:
        o = traj_cfg["obstacle"]
        return CircleObstacle(o["x"], o["y"], o["radius"])
    x, y, r = (float(v) for v in s.split(","))
    return CircleObstacle(x, y, r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="parent trajectories tag")
    ap.add_argument("--sigma", type=float, default=0.10,
                    help="N-ring half-width (unsafe band thickness)")
    ap.add_argument("--deep-target", type=int, default=3000)
    ap.add_argument("--deep-margin", type=float, default=1.5,
                    help="deep-safe threshold = radius * (1 + margin)")
    ap.add_argument("--obstacle", default=None,
                    help="override obstacle as 'x,y,r' (else read from traj config)")
    ap.add_argument("--synth-unsafe-n", type=int, default=0,
                    help="if >0, replace trajectory-derived unsafe with N uniform "
                         "samples on the annulus [R, R+sigma] × [0, 2pi)")
    ap.add_argument("--bbox", default=None,
                    help="override deep-safe bbox as 'xmin,xmax,ymin,ymax'")
    ap.add_argument("--subsample-expert", type=int, default=0,
                    help="if >0, voxel-decimate expert (x,u) rows to ~this many")
    ap.add_argument("--subsample-bins", default="25,25,6",
                    help="voxel grid as 'nx,ny,ntheta' for --subsample-expert")
    ap.add_argument("--neutral-shell-n", type=int, default=0,
                    help="if >0, sample N points in a frame around the bbox "
                         "for the |h|<=gamma_neutral cap constraint")
    ap.add_argument("--neutral-shell-width", type=float, default=0.4,
                    help="thickness of the neutral frame (m) added outside bbox")
    args = ap.parse_args()

    traj_dir = pio.resolve_existing(pio.STAGE_TRAJ, args.traj)
    traj_cfg = pio.load_config(traj_dir)
    src_csv = traj_dir / "trajectories.csv"
    obs = _parse_obstacle(args.obstacle, traj_cfg)

    R_IN  = obs.radius
    R_OUT = R_IN + args.sigma

    CFG = dict(
        traj_tag=args.traj,
        sigma=args.sigma,
        deep_target=args.deep_target,
        deep_margin=args.deep_margin,
        obstacle=dict(x=obs.x, y=obs.y, radius=obs.radius),
        synth_unsafe_n=args.synth_unsafe_n,
        bbox_override=args.bbox,
        subsample_expert=args.subsample_expert,
        subsample_bins=args.subsample_bins,
        neutral_shell_n=args.neutral_shell_n,
        neutral_shell_width=args.neutral_shell_width,
    )
    human_extra = ""
    if args.synth_unsafe_n:
        human_extra += f"_synU{args.synth_unsafe_n}"
    if args.bbox:
        human_extra += "_bboxOv"
    if args.subsample_expert:
        human_extra += f"_subE{args.subsample_expert}"
    if args.neutral_shell_n:
        human_extra += f"_neu{args.neutral_shell_n}"
    human = f"sig{args.sigma:.2f}_marg{args.deep_margin:.1f}_n{args.deep_target}{human_extra}"
    TAG = pio.make_tag(human, CFG)
    OUT = pio.run_dir(pio.STAGE_DATA, TAG)
    print(f"[stage 2] parent traj = {args.traj}\n          tag = {TAG}\n          out = {OUT}")

    df = pd.read_csv(src_csv)
    Z = df[["x", "y", "theta", "v", "omega"]].to_numpy()
    print(f"Loaded {Z.shape[0]} (x,u) pairs from {src_csv}")

    d = np.hypot(Z[:, 0] - obs.x, Z[:, 1] - obs.y)
    in_band = (d >= R_IN) & (d <= R_OUT)
    expert_safe = Z[~in_band]
    traj_unsafe = Z[in_band, :3]

    if args.subsample_expert > 0:
        nx, ny, nth = (int(v) for v in args.subsample_bins.split(","))
        ex = expert_safe[:, 0]; ey = expert_safe[:, 1]; et = expert_safe[:, 2]
        ix = np.clip(((ex - ex.min()) / ((ex.max() - ex.min()) + 1e-9) * nx).astype(int), 0, nx - 1)
        iy = np.clip(((ey - ey.min()) / ((ey.max() - ey.min()) + 1e-9) * ny).astype(int), 0, ny - 1)
        it = np.clip(((et % (2 * np.pi)) / (2 * np.pi) * nth).astype(int), 0, nth - 1)
        bin_id = ix * (ny * nth) + iy * nth + it
        # Keep the first row encountered per bin.
        _, first_idx = np.unique(bin_id, return_index=True)
        kept = expert_safe[np.sort(first_idx)]
        # If target is less than #unique-bins, uniformly sub-pick down to target.
        if kept.shape[0] > args.subsample_expert:
            rng = np.random.default_rng(0)
            sel = rng.choice(kept.shape[0], args.subsample_expert, replace=False)
            kept = kept[np.sort(sel)]
        print(f"  subsample expert: {expert_safe.shape[0]} → {kept.shape[0]} "
              f"(grid {nx}x{ny}x{nth})")
        expert_safe = kept

    if args.synth_unsafe_n > 0:
        rng = np.random.default_rng(0)
        # Uniform on annulus by sampling radius ~ sqrt for area-uniformity.
        u = rng.uniform(R_IN**2, R_OUT**2, size=args.synth_unsafe_n)
        rad = np.sqrt(u)
        ang = rng.uniform(0.0, 2 * np.pi, size=args.synth_unsafe_n)
        th  = rng.uniform(0.0, 2 * np.pi, size=args.synth_unsafe_n)
        unsafe = np.column_stack([obs.x + rad * np.cos(ang),
                                  obs.y + rad * np.sin(ang), th])
        print(f"  synth unsafe ring: {unsafe.shape} (replaces traj-derived {traj_unsafe.shape})")
    else:
        unsafe = traj_unsafe

    if args.bbox is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in args.bbox.split(","))
    else:
        xmin, xmax = float(Z[:, 0].min()) - 0.2, float(Z[:, 0].max()) + 0.2
        ymin, ymax = float(Z[:, 1].min()) - 0.2, float(Z[:, 1].max()) + 0.2
    t0 = time.monotonic()
    safe = sample_deep_safe_interior(
        obs, bbox=(xmin, xmax, ymin, ymax),
        n_target=args.deep_target,
        margin_factor=args.deep_margin, seed=0,
    )
    runtime_s = time.monotonic() - t0
    print(f"  expert_safe: {expert_safe.shape}")
    print(f"  unsafe     : {unsafe.shape}")
    print(f"  safe       : {safe.shape}")

    pd.DataFrame(expert_safe, columns=["x", "y", "theta", "v", "omega"]).to_csv(
        OUT / "expert_safe.csv", index=False)
    pd.DataFrame(unsafe, columns=["x", "y", "theta"]).to_csv(
        OUT / "unsafe.csv", index=False)
    pd.DataFrame(safe, columns=["x", "y", "theta"]).to_csv(
        OUT / "safe.csv", index=False)

    # Neutral frame: thin band around (and just outside) the bbox.
    if args.neutral_shell_n > 0:
        rng = np.random.default_rng(1)
        W = args.neutral_shell_width
        # Sample uniformly in the outer frame: outer bbox minus inner bbox.
        outer = (xmin - W, xmax + W, ymin - W, ymax + W)
        kept = []
        while sum(len(b) for b in kept) < args.neutral_shell_n:
            n = args.neutral_shell_n * 2
            x = rng.uniform(outer[0], outer[1], n)
            y = rng.uniform(outer[2], outer[3], n)
            outside_inner = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
            kept.append(np.column_stack([x[outside_inner], y[outside_inner]]))
        xy = np.vstack(kept)[:args.neutral_shell_n]
        th = rng.uniform(0.0, 2 * np.pi, args.neutral_shell_n)
        neutral = np.column_stack([xy, th])
        pd.DataFrame(neutral, columns=["x", "y", "theta"]).to_csv(
            OUT / "neutral.csv", index=False)
        print(f"  neutral frame: ({neutral.shape[0]}, 3)  width={W}")
    else:
        neutral = None

    pio.save_config(OUT, CFG)
    pio.save_meta(OUT, dict(
        n_expert_safe=int(expert_safe.shape[0]),
        n_unsafe=int(unsafe.shape[0]),
        n_safe=int(safe.shape[0]),
        bbox=[xmin, xmax, ymin, ymax],
        runtime_s=runtime_s,
    ))
    pio.append_index_row(pio.STAGE_DATA, dict(
        tag=TAG, traj_tag=args.traj,
        n_expert_safe=expert_safe.shape[0],
        n_unsafe=unsafe.shape[0],
        n_safe=safe.shape[0],
        sigma=args.sigma, deep_margin=args.deep_margin,
        path=str(OUT),
    ))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(safe[:, 0], safe[:, 1], s=4,
               c="tab:green", alpha=0.35, label=f"deep-safe ({len(safe)})")
    ax.scatter(expert_safe[:, 0], expert_safe[:, 1], s=2,
               c="tab:blue", alpha=0.20, label=f"expert-safe ({len(expert_safe)})")
    ax.scatter(unsafe[:, 0], unsafe[:, 1], s=10,
               c="tab:red", alpha=0.85, label=f"unsafe ring ({len(unsafe)})")
    ax.add_patch(Circle((obs.x, obs.y), R_IN,
                        facecolor="0.85", edgecolor="k", lw=1.2))
    ax.add_patch(Circle((obs.x, obs.y), R_OUT,
                        facecolor="none", edgecolor="k", lw=1.0, ls="--"))
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Three-set split — {TAG}")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "preview.png", dpi=120)
    print(f"Saved {OUT / 'preview.png'}")


if __name__ == "__main__":
    main()
