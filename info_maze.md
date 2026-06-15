# Maze environment & CBF learning — what we built and why

End goal: learn a neural Control Barrier Function h for a differential-drive
robot in a maze, to be used later in a CBF-QP safety filter. This document
covers the maze environment, the expert-data pipeline, the training sets,
and what the training experiments established.

```text
maze sim data
    ↓
safe set (from trajectories)
unsafe set (sampled collision)
transition set (x, u, x')
    ↓
NN h(x)
    ↓
classification + invariance loss
    ↓
CBF-QP later
```

---

## 1. The maze environment (`maze_config.py`)

- 6×6 m workspace, transcribed from a hand-drawn sketch:
  **11 interior walls + 4 border walls**, all axis-aligned boxes
  (`WallObstacle`, added to `expert_data_generation/rrt_diff_drive.py`
  as a purely additive change — duck-type compatible with
  `CircleObstacle`, so the old circle-world code is untouched).
- Robot: differential drive (unicycle model from
  `expert_data_generation/model.py`), circular footprint,
  **radius 0.1 m**, v ∈ [−0.2, 0.5] m/s, ω ∈ [−2, 2] rad/s, dt = 0.1 s
  (RK4 integration).
- Every border-adjacent wall leaves a passable 0.5 m gap; narrowest
  corridor is 0.5 m. Dead ends are intentional (data diversity).
- `sketch_maze()` builds the config; running the file renders
  `results/maze_layout.png`.

## 2. Expert trajectory generation (`maze_trajectories.py`)

Three modes — we built them in this order while chasing throughput:

| Mode | Pipeline | Speed | Use |
|---|---|---|---|
| `mission` | kinodynamic RRT → CasADi NLP smoothing | ~1 min/traj | start→goal demos |
| `coverage` | random A→B pairs, A* warm start → NLP | ~3 s/traj | NLP-quality figures |
| `sim` (default) | A* → shortcut smoothing → pure-pursuit controller → forward simulation through the true RK4 dynamics, with control noise → collision check | **~15 ms/traj** | the dataset |

Key insight behind `sim` (and the profiling that motivated it): the NLP
was 99.5% of the runtime, but a CBF dataset doesn't need optimal
trajectories — only collision-free, dynamically feasible, diverse ones.
Simulated tracking gives feasibility *by construction* (the recorded
actions are literally what was fed into the dynamics), at 200× the speed.

Sampling is biased toward the decision boundary the CBF must learn:

- **Mode A (70%)**: random A→B pairs, 1.5–8.5 m apart
- **Mode B (20%)**: wall-hugging — both endpoints within 0.25 m of a wall
- **Mode C (10%)**: corridor forcing — only pairs without line of sight
- Control noise (σ_v = 0.02, σ_ω = 0.1) injected before clipping;
  in-simulation collision check on a hard grid (robot radius + 3 cm)

**The dataset** (`results/maze_sim.csv`): 5,000 trajectories,
386,649 (state, action) rows, generated in 74 s. Validated by
`validate_dataset.py`: zero states within the robot radius of a wall
(min distance 0.126 m), controls within bounds, 57% of states within
0.3 m of a wall.

Same CSV schema as the original circle world, so the existing notebook
and training code load it unchanged.

## 3. Training sets (`learning-cbf/build_maze_sets.py` → `generated-sets/set_02/`)

| File | Rows | Content |
|---|---|---|
| `X_safe.csv` | 386,649 | expert states (wall distance ≥ 0.126 m) |
| `N.csv` | 50,000 | **sampled true collision states**: 33% strictly inside walls, the rest in the collision ring d < 0.1 m (40% concentrated in the band [0.07, 0.10) to pin the boundary) |
| `transitions.csv` | 381,649 | exact (x, u, x′) RK4 steps for the invariance loss, verified against the model dynamics to 1e-6 |

Label semantics:

```text
unsafe   d(x) <  0.10 m         true collision (robot disc overlaps wall)
buffer   0.10 ≤ d(x) < 0.126 m  UNLABELED — the h=0 surface lives here
safe     d(x) ≥ 0.126 m         every expert state
```

Two deliberate design points, both learned the hard way:

1. **True negatives, not relabeling.** The old circle-world set_01 made
   its "unsafe" set by relabeling near-obstacle *expert* states — so it
   contained zero actual collisions and h was unconstrained inside the
   obstacle. set_02 samples real collision states instead.
2. **Controls in N are bootstrap-resampled from the safe set.** First
   attempt drew them uniformly; the network then classified almost
   entirely on the control *distribution* (pure pursuit's controls are
   highly peaked) and ignored position — 91.5% "accuracy" with h < 0
   over the entire map. Identical control distributions in both classes
   remove that shortcut.

## 4. Training experiments (placeholder `train.py`, margin classifier)

Chronology — each step changed one thing and the metric is honest
balanced accuracy on safe/unsafe:

| Run | Input | Steps | Acc | Lesson |
|---|---|---|---|---|
| 1 | (x,y,θ,v,ω) | 200 | "0.915" | fake — control-distribution shortcut |
| 2 | (x,y,θ,v,ω), fixed N | 2,000 | 0.844 | now actually learning geometry |
| 3 | + hidden 128, batch 1024 | 15,000 | 0.978 | level set roughly traces walls |
| 4 | + cosine lr decay | 40,000 | 0.991 | tighter, but heading-slice artifacts |
| 5 | **(x, y) only** | 40,000 | **1.000** | perfect separation, clean level set |

x — x-position [m] |
y — y-position [m] |
θ — heading/orientation [rad] |
v — linear speed [m/s] |
ω — angular velocity (turn rate) [rad/s]

The final step is the important conceptual one: *the robot footprint is
circular and the environment is static, so the safety set depends only
on position.* Heading affects the dynamics, not the collision geometry —
θ enters later in the QP through ∇h(x)·f(x, u), not as a network input.
Dropping from 5D to 2D made the same dataset saturation-dense and the
learned h(x, y) is now essentially the inflated signed distance field:
zero level set tracing every wall at the ~0.1 m robot-radius standoff.

**Checkpoints** (`learning-cbf/checkpoints/cbf_<timestamp>/`, git-ignored):
`best.pt` = weights at the best balanced accuracy (updated continuously
during the run), `last.pt` = end of run. Each stores model + optimizer
state + config + accuracy. Saving is atomic-with-retry because OneDrive
intermittently locks frequently-rewritten files.

**Visualization**: `learning-cbf/plot_h.py [ckpt]` renders the h(x, y)
contour over the maze (newest run by default; 4 heading slices for
legacy 5-input checkpoints). Zero level set in black — it should trace
the walls at one robot radius, not the wall faces themselves.

## 5. What's still missing (by design — friend's training code)

- **Invariance loss** `h(x′) ≥ (1 − αΔt)·h(x)` on `transitions.csv` —
  the term that makes h a CBF rather than a classifier
- tanh/softplus activations (smooth ∇h for the QP) instead of ReLU
- **CBF-QP safety filter** (`learning-cbf/optimization_cbf.py`, empty):
  min ‖u − u_des‖² s.t. ∇h(x)·f(x, u) + αh(x) ≥ 0, using the unicycle
  f and the learned h
- Out of scope: recoverability (heading/speed-dependent safety) — we
  model geometric safety only

## 6. How to run everything

```bash
# 1. generate expert data (~75 s for 5000 trajectories)
python maze_env/maze_trajectories.py --trajs 5000
# (modes: --mode sim|coverage|mission; sanity check the output:)
python maze_env/validate_dataset.py maze_sim.csv

# 2. build the training sets
python learning-cbf/build_maze_sets.py --n-unsafe 50000

# 3. train (placeholder classifier; ~7 min CPU at 40k steps)
python learning-cbf/train.py

# 4. inspect the learned h
python learning-cbf/plot_h.py
```

Environment notes (this machine): **base Anaconda** has the full stack
(torch + casadi + wandb + pandas). The `glider` conda env has casadi but
no torch; if running there, prepend `envs\glider\Library\bin` to PATH so
CasADi finds the IPOPT DLL — and avoid `conda run` (it buffers output
and crashes on Unicode under cp1252).

## File map

```text
maze_env/
├── maze_config.py        # MazeConfig + sketch_maze() + plot helper
├── maze_trajectories.py  # generation modes (sim default, coverage, mission)
├── validate_dataset.py   # clearance/bounds checker for generated CSVs
├── info.md               # this file
└── results/              # git-ignored, regenerable
    ├── maze_sim.csv      # the dataset: 5000 trajectories, 386k rows
    ├── maze_sim.png / maze_coverage.* / maze_layout.png

learning-cbf/
├── build_maze_sets.py    # maze_sim.csv -> generated-sets/set_02/
├── train.py              # placeholder margin classifier (h(x,y))
├── plot_h.py             # h contour from a checkpoint
├── optimization_cbf.py   # (empty) future CBF-QP filter
├── generated-sets/set_02/  # X_safe / N / transitions + README
└── checkpoints/          # git-ignored training runs (best.pt / last.pt)
```
