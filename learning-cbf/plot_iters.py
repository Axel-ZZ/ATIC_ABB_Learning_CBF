"""
Plot iteration-comparison + level-sets for selected RFF runs.

Usage:
    python plot_iters.py --tags tag1 tag2 ... --dataset <ds_tag> [--out out.png]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
import pipeline_io as pio  # noqa: E402


def phi(x, W, b):
    L = W.shape[0]
    return np.sqrt(2.0 / L) * np.cos(x @ W.T + b)


def eval_h(x, theta, W, b):
    return phi(x, W, b) @ theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="rff run tags in order")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="iters_comparison.png")
    ap.add_argument("--theta-slice", type=float, default=np.pi / 4)
    ap.add_argument("--grid-n", type=int, default=140)
    args = ap.parse_args()

    ds_dir = pio.resolve_existing(pio.STAGE_DATA, args.dataset)
    X_safe = pd.read_csv(ds_dir / "safe.csv").to_numpy()
    X_unsafe = pd.read_csv(ds_dir / "unsafe.csv").to_numpy()
    expert_df = pd.read_csv(ds_dir / "expert_safe.csv")
    X_expert = expert_df[["x", "y", "theta"]].to_numpy()
    U_expert = expert_df[["v", "omega"]].to_numpy()

    def _dphi(x, W, b):
        L = W.shape[0]
        z = x @ W.T + b
        return -np.sqrt(2.0 / L) * np.sin(z)[:, :, None] * W[None, :, :]

    def _viol(theta, W, b, cfg):
        h_safe = eval_h(X_safe, theta, W, b)
        h_unsafe = eval_h(X_unsafe, theta, W, b)
        h_exp = eval_h(X_expert, theta, W, b)
        # Lie deriv: ∇h · (g(x) u) for unicycle.
        J = _dphi(X_expert, W, b)
        # xdot for unicycle: [v cos th, v sin th, omega]
        v = U_expert[:, 0]; om = U_expert[:, 1]; th = X_expert[:, 2]
        xdot = np.column_stack([v * np.cos(th), v * np.sin(th), om])
        grad_h = np.einsum("ilk,l->ik", J, theta)
        lie = np.sum(grad_h * xdot, axis=1)
        q = lie + h_exp
        gs, gu, gd = cfg["gamma_safe"], cfg["gamma_unsafe"], cfg["gamma_dyn"]
        return dict(
            safe_pct=100.0 * float((h_safe - gs < 0).mean()),
            unsafe_pct=100.0 * float((-h_unsafe - gu < 0).mean()),
            dyn_pct=100.0 * float((q - gd < 0).mean()),
        )

    # Load runs.
    runs = []
    for tag in args.tags:
        d = pio.resolve_existing(pio.STAGE_CBF, tag, method="rff")
        npz = np.load(d / "model.npz")
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        runs.append(dict(tag=tag, dir=d,
                         theta=npz["theta"], W=npz["W"], b=npz["b"],
                         cfg=cfg, met=met))

    n = len(runs)

    # Figure: top row level-sets, bottom row metrics bar.
    fig = plt.figure(figsize=(5.0 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1.2])

    # Workspace from the data.
    all_xy = np.vstack([X_safe[:, :2], X_unsafe[:, :2], X_expert[:, :2]])
    xmin, ymin = all_xy.min(axis=0) - 0.3
    xmax, ymax = all_xy.max(axis=0) + 0.3
    gx = np.linspace(xmin, xmax, args.grid_n)
    gy = np.linspace(ymin, ymax, args.grid_n)
    XX, YY = np.meshgrid(gx, gy)

    th_slice = args.theta_slice
    pts = np.column_stack([XX.ravel(), YY.ravel(),
                           np.full(XX.size, th_slice)])

    def _near(arr, th):
        dth = np.abs(((arr[:, 2] - th + np.pi) % (2 * np.pi)) - np.pi)
        return arr[dth < np.pi / 8]

    s = _near(X_safe, th_slice)
    u = _near(X_unsafe, th_slice)

    for i, r in enumerate(runs):
        ax = fig.add_subplot(gs[0, i])
        h_vals = eval_h(pts, r["theta"], r["W"], r["b"]).reshape(XX.shape)
        # Symmetric, zero-centered diverging colormap → red = h<0, blue = h>0.
        vabs = max(abs(h_vals.min()), abs(h_vals.max()), 1e-6)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        cf = ax.contourf(XX, YY, h_vals, levels=21, cmap="RdBu",
                         norm=norm, alpha=0.85)
        # Hatched overlay on h<0 region so unsafe pockets are unmistakable.
        ax.contourf(XX, YY, h_vals, levels=[h_vals.min(), 0.0],
                    colors="none", hatches=["xxx"])
        ax.contour(XX, YY, h_vals, levels=[0.0], colors="k", linewidths=2.0)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04,
                     label="h(x)  (red<0, blue>0)")
        if len(s):
            ax.scatter(s[:, 0], s[:, 1], s=3, c="tab:green",
                       alpha=0.4, label="safe")
        if len(u):
            ax.scatter(u[:, 0], u[:, 1], s=10, c="tab:red",
                       alpha=0.9, label="unsafe")
        ax.set_aspect("equal")
        title = (f"iter {i+1}\n{r['tag']}\n"
                 f"L={r['cfg']['n_features']} σ={r['cfg']['sigma']} "
                 f"λu={r['cfg']['lam_unsafe']} λs={r['cfg']['lam_safe']}\n"
                 f"‖θ‖={r['met'].get('theta_norm', 0):.1f}")
        ax.set_title(title, fontsize=8)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # Metrics bar chart spanning the bottom.
    ax = fig.add_subplot(gs[1, :])
    n_safe = X_safe.shape[0]
    n_unsafe = X_unsafe.shape[0]
    n_dyn = X_expert.shape[0]

    labels = [f"iter {i+1}" for i in range(n)]
    safe_pct = []
    unsafe_pct = []
    dyn_pct = []
    for r in runs:
        v = _viol(r["theta"], r["W"], r["b"], r["cfg"])
        safe_pct.append(v["safe_pct"])
        unsafe_pct.append(v["unsafe_pct"])
        dyn_pct.append(v["dyn_pct"])
        print(f"{r['tag']}  safe {v['safe_pct']:5.2f}%  "
              f"unsafe {v['unsafe_pct']:5.2f}%  dyn {v['dyn_pct']:5.2f}%")

    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, safe_pct, w, label="safe", color="tab:green")
    ax.bar(x, unsafe_pct, w, label="unsafe", color="tab:red")
    ax.bar(x + w, dyn_pct, w, label="dyn", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("violations (%)")
    ax.set_title("verification violations per iter")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    out = REPO / "runs" / "cbfs" / "comparisons" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(f"RFF iteration progression on dataset {args.dataset}  "
                 f"(theta slice = {np.degrees(th_slice):.0f}°)",
                 y=1.00, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
