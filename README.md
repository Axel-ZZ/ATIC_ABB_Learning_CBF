# atic-cbfs

Differential-drive planning + learned Control Barrier Functions.

## Setup

[uv](https://docs.astral.sh/uv/) manages the env.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync                                           # install from pyproject.toml
source .venv/bin/activate                         # or prefix everything with `uv run`
```

## Pipeline

Three stages, each writes `runs/<stage>/<tag>/` with `config.json`, `meta.json`, and a row in `runs/<stage>/index.csv`. Downstream stages reference upstream ones by tag.

```
trajectories  →  datasets  →  cbfs/{rff,nn}
```

### 1. Trajectories — expert demos around an obstacle
```bash
python expert_data_generation/planner.py             # single optimization-based trajectory
python expert_data_generation/dense_trajectories.py  # grid + wall-hugging bulk set
```

### 2. Datasets — three-set split (expert_safe / unsafe / safe)
```bash
python learning-cbf/build_three_sets.py \
    --traj legacy_set_02_src__src \
    --sigma 0.10 --deep-target 4000 --deep-margin 1.2 \
    --synth-unsafe-n 3000 --subsample-expert 5000 --neutral-shell-n 600
```

### 3. CBF training
```bash
# Convex RFF baseline (Robey et al.)
python learning-cbf/optimization_cbf.py \
    --dataset sig0.10_marg1.2_n4000_synU3000_bboxOv_subE5000_neu600__2627ee \
    --mode soft --n-features 400 --sigma 1.3 \
    --gamma-safe 1.5 --gamma-unsafe 1.5 --gamma-dyn 0.05

# NN variant
*to be implemente*
```

Outputs land under `runs/cbfs/{rff,nn}/<tag>/`.

## Notebooks

| Notebook | When to use |
|---|---|
| `expert_data_generation/set-analyisis.ipynb` | **Before training.** Inspect a generated dataset: safe/unsafe split, kNN density, heading coverage, boundary sampling. |
| `learning_cbf.ipynb` | **After training.** Cross-run leaderboard + Pareto front, per-run verification + level-set plot, dataset diagnostics, distribution↔performance scatter. |

`learning_cbf.ipynb` is the main analysis surface:
- **§1 Overview** — every run (caches `verification.json` per run), Pareto front over (`v_total`, runtime, ‖θ‖).
- **§2 Selection** — `TAG=None` auto-picks the Pareto run with lowest `v_total`.
- **§3 Dataset diagnostics** — kNN sparsity, (x,y) density, heading coverage for the selected run's dataset.
- **§4 Run inspector** — `h` level-set at a θ-slice + violation bar chart.
- **§5 Distribution ↔ performance** — scatter grid of dataset stats vs violation rates with Pearson r — tells you which dataset knob to tune next.

### Reading violation rates

| metric | meaning | green | yellow | red |
|---|---|---|---|---|
| `v_safe` | safe pts with `h(x)<γ_safe` — CBF too pessimistic | <1% | 1–5% | >5% |
| `v_unsafe` | unsafe pts with `-h(x)<γ_unsafe` — safety leak | <0.5% | 0.5–2% | >2% |
| `v_dyn` | expert pairs with `⟨∇h,ẋ⟩+h<γ_dyn` — dynamics violated | <2% | 2–10% | >10% |

Only `v_unsafe` is a hard safety failure. `v_safe` and `v_dyn` trade against each other.

## Example plots

The notebooks produce:

- **Three-set overlay** (`set-analyisis.ipynb`) — green deep-safe / blue expert-safe / red unsafe-ring around obstacle + σ-band.
- **Density triptych** (both) — kNN-distance histogram per class, (x,y) sample-count heatmap, per-point sparsity scatter.
- **Run inspector** (`learning_cbf.ipynb §4`) — `h` contour at chosen θ-slice with the 0-level overlaid + safe/unsafe/dyn bar chart.
- **Pareto scatter** (`§1`) — `v_total` vs runtime and vs ‖θ‖, non-dominated runs ringed in black.
- **Distribution↔performance grid** (`§5`) — one dataset metric vs one violation rate per cell, Pearson r in the title.

Pre-rendered iteration comparisons sit in `runs/cbfs/comparisons/` (e.g. `iter10_allfour.png`).

## Implementation notes

- `docs/rff.md` — paper-to-code map for `optimization_cbf.py` (Robey et al., arXiv:2004.03315). Every line tagged `[PAPER]` / `[NOTEBOOK]` / `[ADDED]`.
- `docs/kinodynamic_rrt.md` — trajectory-optimizer formulation (decision vars, cost, constraints).
- `pipeline_io.py` — shared tag/path/index plumbing. Read first when extending the pipeline.

## Dependencies

Add packages with `uv add <pkg>`; they're written to `pyproject.toml`.
