"""
Render the learned h(x, y) over the maze from a saved checkpoint.

The current model takes (x, y, theta, v, omega), so h is sliced at four
fixed headings (v, omega held at nominal values). The zero level set
(black line) should trace the walls: h > 0 in free space, h < 0 inside.

Usage:  python plot_h.py [checkpoint.pt]
        (default: best.pt of the newest learning-cbf/checkpoints/cbf_* run)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "maze_env"))

from maze_config import sketch_maze, plot_maze   # noqa: E402

V_NOMINAL = 0.25
OMEGA_NOMINAL = 0.0
HEADINGS = [0.0, np.pi / 2, np.pi, -np.pi / 2]


class CBFNet(nn.Module):
    """Must mirror train.py's architecture to load its state dict."""

    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    else:
        runs = sorted((REPO / "learning-cbf/checkpoints").glob("cbf_*"))
        if not runs:
            raise SystemExit("no checkpoints found")
        ckpt_path = runs[-1] / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_net = ckpt["config"]
    model = CBFNet(in_dim=cfg_net["in_dim"], hidden=cfg_net["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, acc {ckpt['acc']:.3f})")

    maze = sketch_maze()
    n = 300
    xs = np.linspace(-0.1, 6.1, n)
    ys = np.linspace(-0.1, 6.1, n)
    Xg, Yg = np.meshgrid(xs, ys)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def eval_h(extra_cols):
        inp = torch.tensor(np.column_stack(
            [Xg.ravel(), Yg.ravel()] + extra_cols), dtype=torch.float32)
        with torch.no_grad():
            return model(inp).numpy().reshape(n, n)

    def draw(ax, H, title):
        vmax = np.abs(H).max()
        im = ax.contourf(Xg, Yg, H, levels=30, cmap="RdBu",
                         vmin=-vmax, vmax=vmax)
        ax.contour(Xg, Yg, H, levels=[0.0], colors="k", linewidths=2)
        plot_maze(ax, maze)
        ax.set_title(title)
        return im

    if cfg_net["in_dim"] == 2:
        # State-only position model: one panel says it all.
        fig, ax = plt.subplots(figsize=(10, 10))
        im = draw(ax, eval_h([]), "h(x, y)")
        fig.colorbar(im, ax=ax, shrink=0.85)
    else:
        # Legacy 5-input model: slice at four fixed headings.
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        for ax, th in zip(axes.flat, HEADINGS):
            im = draw(ax, eval_h([np.full(Xg.size, th),
                                  np.full(Xg.size, V_NOMINAL),
                                  np.full(Xg.size, OMEGA_NOMINAL)]),
                      f"h(x, y)  at  θ = {np.degrees(th):.0f}°, "
                      f"v = {V_NOMINAL}, ω = {OMEGA_NOMINAL}")
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Learned CBF — {ckpt_path.parent.name} "
                 f"(zero level set in black)", fontsize=14)
    plt.tight_layout()
    out = ckpt_path.parent / "h_contour.png"
    plt.savefig(out, dpi=100)
    print(f"saved {out}")
