# Repository structure — `wt-clean-up` branch

Differential-drive planning + **learning Control Barrier Functions (CBFs) from
expert demonstrations**. The repo is organized around a modular `environments/`
package — one `Environment` abstraction, one trajectory generator, and one
dataset builder — so the **single-obstacle** world and the **maze** run the exact
same code paths. The learned CBF is later used as a safety filter (CBF-QP).

Pipeline:

```
expert demos (sim)  ->  4 CBF data sets  ->  learn h(x)  ->  CBF-QP safety filter
  trajectories.py        build_dataset.py     models/         (not built yet)
```

> Status legend: ✅ working · 🟡 real but not self-contained in this branch
> (imports glue modules stripped out of the push) · 🚧 stub / TODO · 📦 generated artifacts.

---

## Repository tree

```
ATIC_ABB_Learning_CBF-wt-clean-up/
├── environments/                 ✅ worlds + expert data generation (the working core)
│   ├── environment.py            ✅ Environment dataclass + build_env + plot_environment
│   │                                + nearest_obstacle_distance + ENV_NAMES registry
│   ├── obstacles.py              ✅ CircleObstacle / WallObstacle (shared geometry interface)
│   ├── vehicle_dynamics.py       ✅ NumPy diff-drive unicycle kinematics (continuous + RK4)
│   ├── trajectories.py           ✅ generate_sim_set (expert demos) + Trajectory + export_csv
│   ├── build_dataset.py          ✅ THE entry point: edit CONFIG, run -> the four CBF data sets
│   ├── validate_dataset.py       ✅ CLI: --env validator (clearance / control bounds / heading)
│   ├── single_obstacle/
│   │   ├── config.py             ✅ single_obstacle_env() — 4x4 m box (border walls) + 1 circle
│   │   └── results/              📦 cached <env>_sim.csv + layout/preview PNGs (git-ignored)
│   └── maze/
│       ├── maze_config.py        ✅ sketch_maze() — 6x6 m, 11 interior + 4 border walls
│       └── results/              📦 cached maze_sim.csv + PNGs (git-ignored)
├── models/                       🟡 CBF learners (real code; need glue, see gaps)
│   ├── sos_cbf.py                🟡 convex RFF CBF learner (Robey et al., eq 3.6/3.7)
│   ├── train_nn_cbf.py           🟡 NN CBF learner (JAX/Equinox), descent loss eq 3.7
│   └── train.sh                  shell wrapper
├── docs/
│   └── cbf_construction/rff.md   ✅ line-by-line paper↔code map for sos_cbf.py
├── runs/
│   └── datasets/<tag>/           📦 generated datasets: expert_safe/expert_unsafe/safe/unsafe
│                                    + config.json + meta.json + preview.png
├── main.py                       🚧 sketch of the final entry point (load env+model+filter, sim)
├── README.md                     predates the environments/ unification; partly aspirational
├── pyproject.toml / uv.lock      dependencies
└── info.md                       this file
```

Every world is bounded by **border walls** (drawn and collision-real), so "outside
the box" is part of the unsafe region uniformly for both worlds.

---

## Papers / references

The method follows two papers (PDFs are the basis for the data sets and learners):

- **[Robey 2020]** A. Robey, H. Hu, L. Lindemann, et al., *"Learning Control
  Barrier Functions from Expert Demonstrations,"* arXiv:**2004.03315** (CDC 2020).
  The core method: the data sets **Z_dyn / X_safe / X̄_safe / X_N**, the validity
  constraints (safe value, unsafe value, CBF-derivative), and both the convex-RFF
  (eq 3.6) and NN hinge-loss (eq 3.7) formulations.
- **[Lindemann 2021]** L. Lindemann, H. Hu, A. Robey, et al., *"Learning Hybrid
  Control Barrier Functions from Data,"* CoRL 2020. Its **Appendix C.1, Algorithm 1
  (NUTS / reverse-kNN)** is the sampling method for the unsafe set X_N — used for
  `expert_unsafe.csv`.
- **[Rahimi & Recht 2007]** Random Fourier Features — the RKHS parameterization
  `h(x)=φ(x)ᵀθ` in `sos_cbf.py`.
- **ATIC project proposal** — frames the study: compare the **geometric** vs
  **sampling** construction of the unsafe set, NN learner, single-obstacle then maze.

Where each is used:

| code | paper |
|---|---|
| set definitions in `build_dataset.py` | [Robey 2020] §2–3 |
| geometric X_N (`unsafe.csv`, fills S^c) | proposal §5.2.1 (analytic) |
| sampling X_N (`expert_unsafe.csv`, `nuts_boundary`) | [Lindemann 2021] Alg. 1 |
| `models/sos_cbf.py` (convex RFF) | [Robey 2020] eq 3.6/3.7 + [Rahimi & Recht 2007] |
| `models/train_nn_cbf.py` (NN) | [Robey 2020] eq 3.7 |
| `docs/cbf_construction/rff.md` | line-by-line map to [Robey 2020] |

---

## 1. `environments/` — worlds + data generation (✅ unified)

### `Environment` abstraction (`environment.py`)

A world is just data: `bounds`, a list of `obstacles`, `start`/`goal`,
`robot_radius`. `build_env("maze" | "single_obstacle")` returns one; both worlds
return an `Environment`, so generator / plotting / (later) CBF code never branches
on which world it is. `plot_environment(ax, env)` draws any world by dispatching to
each obstacle's `draw()`; `nearest_obstacle_distance(env, X, Y)` is the vectorized
distance field used by sampling and validation. Add a world by registering its
factory in `build_env` / `ENV_NAMES`.

### Obstacle geometry (`obstacles.py`)

`CircleObstacle` and `WallObstacle` are duck-type compatible — both expose the same
interface, so circles and axis-aligned boxes are handled by identical code:

| method | used by |
|---|---|
| `collides(pt, r)` | sampling collision checks |
| `distance(X, Y)` (vectorized) / `surface_distance(pt)` | dataset sampling + validation |
| `occupied(X, Y, r)` | vectorized A* occupancy-grid construction |
| `draw(ax, **kw)` | plotting |

### Dynamics (`vehicle_dynamics.py`)

`RobotModel` + `DiffDriveKinematics`: the unicycle `x=[x,y,θ]`, `u=[v,ω]` with
`f(x)=0`, `g(x)=[[cosθ,0],[sinθ,0],[0,1]]`, as plain **NumPy** steppers
(continuous + RK4 discrete). This is [Robey 2020]'s control-affine system. Only
forward simulation is needed (no NLP anymore), so there is no CasADi here — the
sim generator just calls the RK4 step in a loop. The CBF learners re-declare
`g(x)` in their own autodiff framework (NumPy for `sos_cbf.py`, JAX for
`train_nn_cbf.py`) because they differentiate through it.

### Trajectory generator (`trajectories.py`)

`generate_sim_set` produces the **expert demonstrations Z_dyn** (collision-free
(state, action) trajectories) — but *sim-based, not optimal control*:
A* → shortcut → pure-pursuit → forward-sim through the true RK4 dynamics, with
control noise; collisions are rejected. Feasible by construction (the recorded
actions are what was fed to the dynamics), biased toward obstacles (mode B) and
corridors (mode C). ~100–250× faster than per-trajectory NLP and the only approach
that scales to the maze's non-convex walls. (This fast sampling of *feasible*
demos is our own method; [Robey 2020] §3.5 only requires safe (x,u) pairs.) Also
holds the `Trajectory` record and `export_csv`.

### Dataset builder (`build_dataset.py`) — the one entry point

Edit the **CONFIG block** at the top (at minimum `ENV`) and run the file (IDE
"Run", or `python environments/build_dataset.py`; CLI flags override CONFIG). It
generates the safe trajectories (cached in `environments/<env>/results/`, so
re-runs that only change sampling are fast) and writes **four sets** to
`runs/datasets/<tag>/`:

| file | cols | paper set | construction |
|---|---|---|---|
| `expert_safe.csv` | `x,y,θ,v,ω` | Z_dyn / X_safe | states of the collision-free expert trajectories |
| `expert_unsafe.csv` | `x,y,θ,v,ω` | X_N (sampling) | reverse-kNN boundary of the expert data — `nuts_boundary` [Lindemann Alg. 1] |
| `safe.csv` | `x,y,θ` | X̄_safe (sampled) | deep-safe points, clearance ≥ `deep_margin` |
| `unsafe.csv` | `x,y,θ` | X_N (geometric) | fills **S^c** — obstacles / border walls **inflated by the robot radius** (the C-space obstacle) |

The two unsafe sets are the project's core comparison: **geometric** (`unsafe.csv`,
fills the walls, proposal §5.2.1) vs **sampling** (`expert_unsafe.csv`,
[Lindemann 2021] NUTS). NUTS grid-thins the expert states to ~uniform density
(`--cell`), builds a KD-tree on (x,y), counts neighbours within `--eta`, and flags
the sparsest `--boundary-pct`% as boundary. The grid-thinning is important: on raw
data the neighbour count tracks how often the expert revisited a spot, so
sparsely-driven interior corridors get flagged instead of the walls; after
thinning the count reflects geometry (the ball is clipped only at the true data
edge), giving a clean shell at the clearance standoff. The geometric set instead
fills the true unsafe region. Because the robot is a disc, that region in
center-coordinates is the **obstacle inflated by the robot radius** (the
configuration-space obstacle): `sample_unsafe` keeps points with clearance <
`robot_radius`, so the fill reaches all the way to the robot's collision standoff
from the walls — no empty collision ring between the walls and the safe sets. Each
run also writes `config.json`, `meta.json`, and a points-only `preview.png` (opaque
legend; obstacles light so the S^c fill shows).

> Note: "fill S^c" is denser than [Robey 2020]'s literal thin σ-net around the
> demonstrated tube D (the paper leans on a Lipschitz bound to extend h<0 off a
> thin net). Filling the unsafe region is a deliberate, more-robust choice.

### Validator (`validate_dataset.py`)

`python environments/validate_dataset.py --env <env> --csv <file>` — reports
obstacle clearance, control bounds, and heading coverage of any generated CSV,
using the world's geometry.

---

## 2. `models/` — CBF learners 🟡

Real implementations of [Robey 2020], but they import a `pipeline_io` glue module
(and `plot_iters`) that was stripped from this branch — so they don't run as-is yet.

| file | what | paper |
|---|---|---|
| `sos_cbf.py` | convex RFF CBF: `min ‖θ‖` s.t. safe/unsafe/derivative constraints (hard, eq 3.6) + hinge relaxation (soft, eq 3.7); CVXPY | [Robey 2020] §3.4 |
| `train_nn_cbf.py` | NN CBF (2×64 MLP, tanh), full descent loss `⟨∇h,f+gu⟩+α(h)`, JAX/Equinox/optax, wandb | [Robey 2020] eq 3.7 |

`train_nn_cbf.py` already expects all four sets, including an `expert_unsafe` set
(weak unsafe bound) and `unsafe` (strong) — matching `build_dataset.py`'s output.
`docs/cbf_construction/rff.md` maps `sos_cbf.py` line-by-line to the paper.

---

## 3. Datasets (`runs/datasets/`)

`build_dataset.py` writes each dataset to `runs/datasets/<tag>/` (config-hashed
tag) with the four CSVs + `config.json` + `meta.json` + `preview.png`. Currently
present: one full dataset per world (single_obstacle, maze). All `results/` and
`runs/` outputs are git-ignored / regenerable. (CBF training runs will also land
under `runs/` once the learners are wired.)

---

## 4. Known gaps / next steps

1. **Data — DONE.** `build_dataset.py` emits the four sets in the learners' schema.
2. **NN framework split.** `train_nn_cbf.py` needs `jax`/`equinox`/`optax`;
   `pyproject.toml` lists `torch` and not those. Pick one stack.
3. **Missing glue.** `models/*` import `pipeline_io` (+ `plot_iters`), not in this
   branch — bring in or re-create so the learners can read `runs/datasets/<tag>/`.
4. **Closed-loop sim + filter not built.** `main.py` sketches loading an env +
   model + CBF-QP filter and rolling out; `safety_filters/` (the CBF-QP) and the
   interactive simulator/plotting are the remaining qualitative-evaluation stage.
5. **NUTS tuning.** Defaults (`--cell 0.05 --eta 0.15 --boundary-pct 20`) give a
   clean wall-hugging shell (median clearance ~0.18 m on the maze). `--cell`
   (grid-thinning) is the key knob; we use a percentile threshold (self-scaling)
   rather than [Lindemann]'s fixed neighbour count `N`. Quality still depends on
   expert coverage — a region the expert barely visited can't yield a clean
   boundary; that limitation of the sampling approach is itself a project finding.

---

## 5. How to run

### Setup (once per terminal)

Run everything **from the repository root** — the folder that directly contains
`environments/`, `models/`, `runs/`. The zip/clone created a double-nested folder,
so the root is one level down:

```bash
cd ".../ATIC_ABB_Learning_CBF-wt-clean-up/ATIC_ABB_Learning_CBF-wt-clean-up"
ls environments        # sanity check: should list build_dataset.py, trajectories.py, …
```

Use the base **Anaconda** `python` (it has numpy + pandas + matplotlib + scipy;
cvxpy for the convex learner). Data generation needs only numpy/pandas/matplotlib/
scipy — **CasADi is no longer used** and can be dropped from `pyproject.toml`.

```bash
python -c "import numpy, pandas, matplotlib, scipy; print('env OK')"
```

### Look at a world (optional)

```bash
python environments/maze/maze_config.py            # -> environments/maze/results/maze_layout.png
python environments/single_obstacle/config.py      # -> environments/single_obstacle/results/single_obstacle_layout.png
```

### Build a CBF dataset — one file

Open `environments/build_dataset.py`, set `ENV = "single_obstacle"` or `"maze"` in
the CONFIG block (tweak counts / `--eta` / `--boundary-pct` if you like), and **run
the file** (IDE "Run" button, or the command below). It generates the expert
trajectories, builds the geometric + sampling sets, and writes the four sets to
`runs/datasets/<tag>/`.

```bash
python environments/build_dataset.py                       # uses the CONFIG block
python environments/build_dataset.py --env maze            # …or override on the CLI
```

Trajectories are cached in `environments/<env>/results/`; re-running after you only
change the sampling reuses them (fast). Pass `--regen` (or set `REGEN = True`) to
force regeneration. Sanity-check any generated CSV:

```bash
python environments/validate_dataset.py --env maze --csv maze_sim.csv
```

### Tips

- **Wrong-directory error** (`can't open file … environments\…`) → you're in the
  outer folder; `cd` one level deeper (see Setup).
- If a script ever prints a `UnicodeEncodeError` on this cp1252 console, prefix
  with `$env:PYTHONIOENCODING="utf-8"` (PowerShell) / `set PYTHONIOENCODING=utf-8`
  (cmd). The shipped scripts already avoid this.
- Outputs under `environments/*/results/` and `runs/` are regenerable (git-ignored).
