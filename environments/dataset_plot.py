"""Shared plotting for the CBF pipeline.

One generic primitive — scatter labeled 2D point clouds over the workspace
(optionally on top of the environment obstacles) — reused everywhere:

  * build_dataset.py        : 4-set dataset preview (expert/safe/unsafe scatter)
  * models/train_nn_cbf.py  : evaluate_2d() overlays the safe/unsafe samples on
                              top of an h(x) level-set field at a theta slice.

The level-set helpers (`workspace_grid`, `near_theta`, `draw_h_field`) layer an
h(x) contour onto the same axes, so the training plot is just "scatter sets +
field" built from the shared pieces.

matplotlib is imported lazily inside the functions so importing this module is
cheap and side-effect free (callers pick the backend, e.g. "Agg").
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Generic scatter primitive
# --------------------------------------------------------------------------- #
def new_workspace_ax(env=None, bounds=None, figsize=(9, 9),
                     obstacle_color: str = "0.8"):
    """A fresh (fig, ax) with an equal-aspect workspace background.

    Pass `env` to draw the obstacles/walls, and/or `bounds`
    (((xmin, ymin), (xmax, ymax))) to fix the limits. Either or both may be None.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    if env is not None:
        from environments.environment import plot_environment
        plot_environment(ax, env, obstacle_color=obstacle_color)
    if bounds is not None:
        (xmin, ymin), (xmax, ymax) = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def scatter_sets(ax, sets: Iterable[Mapping], rng=None) -> None:
    """Scatter a list of labeled point clouds on `ax`.

    Each entry in `sets` is a dict:
        xy        : (N, >=2) array — only the first two columns are used
        label     : legend label                       (optional)
        color     : matplotlib color                   (optional)
        s         : marker size       (default 2)
        alpha     : marker alpha       (default 0.5)
        zorder    : draw order         (default 6)
        subsample : plot at most this many points       (optional)
    Empty sets are skipped.
    """
    rng = rng or np.random.default_rng()
    for spec in sets:
        xy = np.asarray(spec["xy"])
        if xy.size == 0:
            continue
        k = spec.get("subsample")
        if k and len(xy) > k:
            xy = xy[rng.choice(len(xy), k, replace=False)]
        ax.scatter(xy[:, 0], xy[:, 1], s=spec.get("s", 2),
                   c=spec.get("color"), alpha=spec.get("alpha", 0.5),
                   label=spec.get("label"), zorder=spec.get("zorder", 6))


def add_legend(ax) -> None:
    """Opaque, above-everything legend (no-op if nothing is labeled)."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(markerscale=3, loc="upper right", fontsize=8,
                  framealpha=1.0, facecolor="white",
                  edgecolor="0.5").set_zorder(10)


def preview_dataset(sets: Iterable[Mapping], out_path, *, env=None, bounds=None,
                    title: Optional[str] = None, rng=None,
                    figsize=(9, 9), dpi: int = 110) -> Path:
    """Scatter labeled point clouds over the workspace and save a PNG.

    Returns the written path. Used for the dataset preview and any other
    "show me these point sets" figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = new_workspace_ax(env=env, bounds=bounds, figsize=figsize)
    scatter_sets(ax, sets, rng=rng)
    if title:
        ax.set_title(title)
    add_legend(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# h(x) level-set helpers (training-time evaluation)
# --------------------------------------------------------------------------- #
def workspace_grid(point_sets: Sequence[np.ndarray], theta: float,
                   grid_n: int = 140, pad: float = 0.1):
    """Regular (x, y) grid covering the union of `point_sets`, at fixed heading.

    Returns (XX, YY, pts, bounds) where pts is (grid_n**2, 3) ready to feed the
    CBF, and bounds is ((xmin, ymin), (xmax, ymax)).
    """
    xy = np.concatenate([np.asarray(p)[:, :2] for p in point_sets if len(p)], axis=0)
    (xmin, ymin), (xmax, ymax) = xy.min(0) - pad, xy.max(0) + pad
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, theta)], axis=1)
    return XX, YY, pts, ((xmin, ymin), (xmax, ymax))


def near_theta(X: np.ndarray, theta: float, tol: float = 0.35) -> np.ndarray:
    """Boolean mask of rows whose heading is within `tol` rad of `theta` (wrapped)."""
    X = np.asarray(X)
    if len(X) == 0:
        return np.zeros(0, dtype=bool)
    d = np.abs(np.angle(np.exp(1j * (X[:, 2] - theta))))
    return d < tol


def draw_h_surface_3d(ax, XX, YY, h_grid, *, zero_plane: bool = True):
    """3D surface of h(x,y) on a 3D axis, colored by value with the h=0 plane.

    `ax` must be a 3D axis (``add_subplot(projection="3d")``). Returns the
    surface artist (for a colorbar). Mirrors `draw_h_field` but in 3D: the
    RdBu surface is the CBF, and the translucent grey plane at z=0 marks the
    safe/unsafe boundary (the surface's intersection with it is the 0 level-set).
    """
    import matplotlib.pyplot as plt
    vmax = float(np.nanmax(np.abs(h_grid))) or 1.0
    surf = ax.plot_surface(XX, YY, h_grid, cmap="RdBu", vmin=-vmax, vmax=vmax,
                           linewidth=0, antialiased=True, alpha=0.95)
    if zero_plane:
        ax.plot_surface(XX, YY, np.zeros_like(h_grid), color="0.6",
                        alpha=0.25, linewidth=0, antialiased=False)
        # emphasise the 0 level-set on the floor for reference
        try:
            ax.contour(XX, YY, h_grid, levels=[0.0], colors="k",
                       linewidths=1.5, offset=float(np.nanmin(h_grid)))
        except Exception:
            pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("h(x)")
    return surf


def draw_h_field(ax, XX, YY, h_grid, bounds=None, levels: int = 25):
    """Filled contour of h(x) with the zero level-set emphasised.

    Returns the contourf set (for a colorbar). Draw this BEFORE scattering points
    so the samples sit on top.
    """
    vmax = float(np.nanmax(np.abs(h_grid))) or 1.0
    cf = ax.contourf(XX, YY, h_grid, levels=levels, cmap="RdBu",
                     vmin=-vmax, vmax=vmax, zorder=1)
    ax.contour(XX, YY, h_grid, levels=[0.0], colors="k",
               linewidths=1.5, zorder=2)
    if bounds is not None:
        (xmin, ymin), (xmax, ymax) = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return cf


# --------------------------------------------------------------------------- #
# Plot a saved dataset (CLI entry point)
# --------------------------------------------------------------------------- #
# Per-set styling, matching build_dataset.py's preview. The key is the CSV stem
# (without ".csv") so `--sets expert_safe safe` selects exactly those files.
SET_STYLE = {
    "unsafe":        {"label": "unsafe (sampled, S^c)", "color": "tab:red",
                      "s": 2, "alpha": 0.4, "zorder": 6, "subsample": 8000},
    "expert_safe":   {"label": "expert safe", "color": "tab:blue",
                      "s": 1, "alpha": 0.4, "zorder": 7, "subsample": 12000},
    "safe":          {"label": "safe (sampled)", "color": "tab:green",
                      "s": 2, "alpha": 0.5, "zorder": 7, "subsample": 4000},
    "expert_unsafe": {"label": "expert unsafe (NUTS)", "color": "tab:orange",
                      "s": 4, "alpha": 0.8, "zorder": 8, "subsample": 6000},
}
# Default draw order (back to front), so omitting --sets shows everything.
ALL_SETS = ["unsafe", "expert_safe", "safe", "expert_unsafe"]


def load_set(dataset_dir, name: str) -> np.ndarray:
    """Load one set's (x, y, ...) array from <dataset_dir>/<name>.csv (empty if absent)."""
    import pandas as pd
    csv = Path(dataset_dir) / f"{name}.csv"
    if not csv.exists():
        return np.empty((0, 2))
    return pd.read_csv(csv).to_numpy()


def plot_dataset_dir(dataset_dir, names: Sequence[str] = ALL_SETS, *,
                     out_path=None, rng=None):
    """Scatter the selected sets of a saved dataset over its environment.

    `names` picks which CSVs to draw (subset of SET_STYLE keys), in draw order.
    Returns the written PNG path.
    """
    import json
    dataset_dir = Path(dataset_dir)
    cfg = json.loads((dataset_dir / "config.json").read_text())

    env = None
    try:
        from environments.environment import build_env
        env = build_env(cfg["env"])
    except Exception as e:
        print(f"  (environment not drawn: {e})")

    sets = []
    for name in names:
        if name not in SET_STYLE:
            raise ValueError(f"unknown set '{name}'; choose from {list(SET_STYLE)}")
        sets.append({**SET_STYLE[name], "xy": load_set(dataset_dir, name)})

    out_path = out_path or dataset_dir / ("preview_" + "_".join(names) + ".png")
    return preview_dataset(sets, out_path, env=env, rng=rng,
                           title=f"{cfg['env']} dataset ({', '.join(names)})")


def plot_dataset_dir_3d(dataset_dir, names: Sequence[str] = ALL_SETS, *,
                        out_path=None, rng=None, figsize=(9, 8), dpi: int = 120,
                        elev: float = 22.0, azim: float = -60.0):
    """3D scatter of a saved dataset's sets in (x, y, theta) state space.

    Same sets/styling as `plot_dataset_dir`, but with theta on the vertical axis
    so the heading coverage of each cloud is visible (the sampled safe/unsafe
    sets have theta drawn uniformly; the expert sets carry their demonstrated
    heading). Returns the written PNG path.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = rng or np.random.default_rng()
    dataset_dir = Path(dataset_dir)
    cfg = json.loads((dataset_dir / "config.json").read_text())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    for name in names:
        if name not in SET_STYLE:
            raise ValueError(f"unknown set '{name}'; choose from {list(SET_STYLE)}")
        spec = SET_STYLE[name]
        xyz = load_set(dataset_dir, name)
        if xyz.size == 0 or xyz.shape[1] < 3:
            continue
        k = spec.get("subsample")
        if k and len(xyz) > k:
            xyz = xyz[rng.choice(len(xyz), k, replace=False)]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=spec.get("s", 2),
                   c=spec.get("color"), alpha=spec.get("alpha", 0.5),
                   label=spec.get("label"), depthshade=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("theta (rad)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{cfg['env']} dataset (x, y, theta) — {', '.join(names)}", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(markerscale=3, loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else \
        dataset_dir / ("preview3d_" + "_".join(names) + ".png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Plot a trained CBF as a 3D surface (loads model.eqx from runs/models/)
# --------------------------------------------------------------------------- #
def load_cbf(model_dir):
    """Load a trained CBF (`model.eqx` + `meta.json`) from a runs/models/<dir>.

    Returns (model, meta). jax/equinox are imported lazily (via
    models.train_nn_cbf.load_model) so importing this module stays cheap.
    """
    import json
    model_dir = Path(model_dir)
    model_path = model_dir / "model.eqx"
    if not model_path.exists():
        raise SystemExit(f"no model.eqx in {model_dir}")
    import sys
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from models.train_nn_cbf import load_model
    model = load_model(str(model_path))
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def _eval_cbf_grid(model, bounds, theta: float, grid_n: int):
    """Evaluate h(x, y, theta) on a regular grid over `bounds`. Returns XX, YY, h_grid."""
    import jax
    import jax.numpy as jnp
    (xmin, ymin), (xmax, ymax) = bounds
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, theta)], axis=1)
    h = np.asarray(jax.vmap(model)(jnp.asarray(pts, dtype=jnp.float32)))
    return XX, YY, h.reshape(XX.shape)


def plot_cbf_surface_3d(model_dir, *, theta: float = np.pi / 4, grid_n: int = 120,
                        out_path=None, bounds=None, title=None,
                        figsize=(8, 7), dpi: int = 120, elev: float = 35.0,
                        azim: float = -60.0, clip_pct: float = 99.0) -> Path:
    """Render a trained CBF as a 3D surface h(x, y) at a fixed heading `theta`.

    The environment/bounds are taken from the model's dataset (via `meta.json`)
    unless `bounds` is given explicitly. Saves a PNG and returns its path.

    Learned CBFs often have huge tails (h diving to large negatives deep inside
    obstacles), which flatten the near-boundary structure. `clip_pct` symmetric-
    clips h to its ±`clip_pct` percentile FOR DISPLAY ONLY so the 0 level-set
    stays legible; pass 100 to disable and show the raw surface.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_dir = Path(model_dir)
    model, meta = load_cbf(model_dir)

    if bounds is None:
        # Derive the workspace from the model's dataset environment.
        ds = meta.get("dataset")
        if ds:
            try:
                import json
                from environments.environment import build_env
                ds_dir = Path(__file__).resolve().parents[1] / "runs" / "datasets" / ds
                cfg = json.loads((ds_dir / "config.json").read_text())
                bounds = build_env(cfg["env"]).bounds
            except Exception as e:
                print(f"  (could not load dataset bounds: {e})")
        if bounds is None:
            bounds = ((-1.0, -1.0), (1.0, 1.0))

    XX, YY, h_grid = _eval_cbf_grid(model, bounds, theta, grid_n)
    if clip_pct < 100.0:
        c = float(np.nanpercentile(np.abs(h_grid), clip_pct)) or 1.0
        h_grid = np.clip(h_grid, -c, c)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    surf = draw_h_surface_3d(ax, XX, YY, h_grid)
    fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.1,
                 label="h(x)  (red<0, blue>0)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or (f"{model_dir.name}\nCBF h(x) surface  "
                           f"@ theta = {np.degrees(theta):.0f}°"), fontsize=9)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else model_dir / f"cbf_surface_th{np.degrees(theta):.0f}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _resolve_model_dir(name: str) -> Path:
    """Resolve a model dir (or path) under runs/models/ (prefix match allowed)."""
    p = Path(name)
    if (p / "model.eqx").exists():
        return p
    root = Path(__file__).resolve().parents[1] / "runs" / "models"
    cand = root / name
    if (cand / "model.eqx").exists():
        return cand
    matches = sorted(d for d in root.glob(f"{name}*") if (d / "model.eqx").exists())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        avail = sorted(str(p.parent.relative_to(root)) for p in root.glob("**/model.eqx"))
        raise SystemExit(f"no model matching '{name}' under {root}\navailable:\n  "
                         + "\n  ".join(avail))
    raise SystemExit(f"'{name}' is ambiguous: {[d.name for d in matches]}")


def _resolve_dataset_dir(name: str) -> Path:
    """Resolve a dataset by directory name (or path) under runs/datasets/."""
    p = Path(name)
    if p.is_dir():
        return p
    root = Path(__file__).resolve().parents[1] / "runs" / "datasets"
    cand = root / name
    if cand.is_dir():
        return cand
    # Allow a prefix match so the long hashed tag need not be typed in full.
    matches = sorted(d for d in root.glob(f"{name}*") if d.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no dataset matching '{name}' under {root}\n"
                         f"available: {[d.name for d in sorted(root.iterdir()) if d.is_dir()]}")
    raise SystemExit(f"'{name}' is ambiguous: {[d.name for d in matches]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a saved CBF dataset, or a trained CBF as a 3D surface.")
    parser.add_argument("dataset", nargs="?",
                        help="dataset dir name under runs/datasets/ (prefix ok) or a path")
    parser.add_argument("--model",
                        help="instead of a dataset, render the CBF in this runs/models/ "
                             "dir (prefix ok) as a 3D h(x) surface")
    parser.add_argument("--theta", type=float, default=45.0,
                        help="heading slice (degrees) for the 3D CBF surface (default: 45)")
    parser.add_argument("--grid-n", type=int, default=120,
                        help="grid resolution per axis for the 3D CBF surface")
    parser.add_argument("--clip-pct", type=float, default=99.0,
                        help="display-only symmetric percentile clip on h for the 3D "
                             "surface (100 = raw, unclipped)")
    parser.add_argument("--sets", nargs="+", default=ALL_SETS, choices=list(SET_STYLE),
                        metavar="SET",
                        help=f"which sets to draw, in order (default: all). "
                             f"choices: {', '.join(SET_STYLE)}")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="for a dataset, scatter its sets in 3D (x, y, theta) space")
    parser.add_argument("-o", "--out", default=None, help="output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="subsampling RNG seed")
    args = parser.parse_args()

    if args.model:
        mdir = _resolve_model_dir(args.model)
        path = plot_cbf_surface_3d(mdir, theta=np.radians(args.theta),
                                   grid_n=args.grid_n, clip_pct=args.clip_pct,
                                   out_path=args.out)
        print(f"saved {path}")
    elif args.dataset:
        ddir = _resolve_dataset_dir(args.dataset)
        plot_fn = plot_dataset_dir_3d if args.three_d else plot_dataset_dir
        path = plot_fn(ddir, args.sets, out_path=args.out,
                       rng=np.random.default_rng(args.seed))
        print(f"saved {path}")
    else:
        parser.error("give a dataset positional, or --model <dir> for a 3D CBF surface")
