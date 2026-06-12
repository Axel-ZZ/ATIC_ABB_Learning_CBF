from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree

import equinox as eqx
import optax
import wandb

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pipeline_io as pio  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_INPUTS       = 3    # [x, y, theta]
N_HIDDEN       = 64
N_HIDDEN_LAYERS = 2


# --------------------------------------------------------------------------- #
# Hyperparameters + data container
# --------------------------------------------------------------------------- #
class HParams(NamedTuple):
    safe_value:     float = 1.5
    unsafe_value:   float = 0.5
    gamma:          float = 0.05
    lam_constraint: float = 15.0   # paper's lambda_d (derivative term)
    lam_boundary:   float = 1.0
    lam_safe:       float = 2.0
    lam_unsafe:     float = 2.0
    lam_param:      float = 0.1


class Data(NamedTuple):
    x_constraint:    jnp.ndarray
    u_constraint:    jnp.ndarray
    x_expert_unsafe: jnp.ndarray
    x_safe:          jnp.ndarray
    x_unsafe:        jnp.ndarray
    x_boundary:      jnp.ndarray = None


# --------------------------------------------------------------------------- #
# Data loading  (mirrors optimization_cbf.py: three CSVs per dataset tag)
# --------------------------------------------------------------------------- #
def load_expert_safe(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expert (x, u) pairs from the safe interior.

    Returns
    -------
    X : (N, 3)   states  [x, y, theta]
    U : (N, 2)   controls [v, omega]
    """
    df = pd.read_csv(path)
    X = df[["x", "y", "theta"]].to_numpy(dtype=np.float32)
    U = df[["v", "omega"]].to_numpy(dtype=np.float32)
    return X, U


def load_states_only(path: str) -> np.ndarray:
    """Load only the state columns (unsafe and deep-safe samples)."""
    df = pd.read_csv(path)
    return df[["x", "y", "theta"]].to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# Dynamics + class-K function
# --------------------------------------------------------------------------- #
def alpha(h):
    return h ** 3


def dynamics_f(x):
    return jnp.zeros((3,))


def dynamics_g(x):
    th = x[2]
    return jnp.array([[jnp.cos(th), 0.0],
                      [jnp.sin(th), 0.0],
                      [0.0,         1.0]])   # (3, 2)


# --------------------------------------------------------------------------- #
# CBF model  (Equinox MLP — callable pytree, no separate params dict)
# --------------------------------------------------------------------------- #
def make_model(key):
    return eqx.nn.MLP(
        in_size=N_INPUTS, out_size="scalar",
        width_size=N_HIDDEN, depth=N_HIDDEN_LAYERS,
        activation=jax.nn.tanh,
        key=key,
    )


# --------------------------------------------------------------------------- #
# CBF descent quantity  r(x,u) = hdot + alpha(h)
# --------------------------------------------------------------------------- #
def r_with_input(x, u, model):
    dh = jax.grad(model)(x)
    return (jnp.dot(dynamics_f(x), dh)
            + jnp.dot(dynamics_g(x).T @ dh, u)
            + alpha(model(x)))


# --------------------------------------------------------------------------- #
# Loss  (eq. 3.7)
# --------------------------------------------------------------------------- #
@jit
def loss_with_input(model, data: Data, hp: HParams):
    """Returns (total_loss, components_dict)."""
    relu = lambda z: jnp.maximum(z, 0.0)
    h = jax.vmap(model)    # model(x) takes a single x; vmap lifts to batch

    # (1) boundary: push h -> 0 (dormant in the paper run)
    if data.x_boundary is not None:
        boundary_cost = jnp.sum(jnp.square(h(data.x_boundary)))
    else:
        boundary_cost = jnp.asarray(0.0)

    # (2) safe: h >= safe_value on safe samples; weaker margin on expert states
    safe_cost = (jnp.sum(relu(hp.safe_value       - h(data.x_safe)))
               + jnp.sum(relu(hp.safe_value / 10  - h(data.x_constraint))))

    # (3) unsafe: h <= -unsafe_value on unsafe samples; weaker on expert-unsafe states
    if data.x_unsafe is not None:
        unsafe_cost = (jnp.sum(relu(h(data.x_unsafe)        + hp.unsafe_value))
                     + jnp.sum(relu(h(data.x_expert_unsafe)  + hp.unsafe_value / 10)))
    else:
        unsafe_cost = jnp.asarray(0.0)

    # (4) descent: r(x,u) >= gamma along expert pairs + small r^2 tightening
    r = vmap(r_with_input, in_axes=(0, 0, None))(
            data.x_constraint, data.u_constraint, model)
    constraint_cost = jnp.sum(relu(hp.gamma - r)) + 0.001 * jnp.sum(jnp.square(r))

    # (5) weight decay over array leaves only (model also contains non-array fields)
    param_cost = jnp.sum(jnp.square(
        ravel_pytree(eqx.filter(model, eqx.is_array))[0]))

    components = {
        "loss/constraint": hp.lam_constraint * constraint_cost,
        "loss/boundary":   hp.lam_boundary   * boundary_cost,
        "loss/safe":       hp.lam_safe        * safe_cost,
        "loss/unsafe":     hp.lam_unsafe      * unsafe_cost,
        "loss/param":      hp.lam_param       * param_cost,
        "diag/min_r":               jnp.min(r),
        "diag/frac_r_below_gamma":  jnp.mean((r < hp.gamma).astype(jnp.float32)),
    }
    total = (components["loss/constraint"] + components["loss/boundary"]
           + components["loss/safe"]       + components["loss/unsafe"]
           + components["loss/param"])
    return total, components


# --------------------------------------------------------------------------- #
# Training step  (Equinox-aware grad + optax update)
# --------------------------------------------------------------------------- #
def make_step(optimizer, hp: HParams):
    @eqx.filter_jit
    def step(model, opt_state, data: Data):
        (total, components), grads = eqx.filter_value_and_grad(
            loss_with_input, has_aux=True)(model, data, hp)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, total, components
    return step


# --------------------------------------------------------------------------- #
# Evaluation / test step  (2D level-set + violation rates, for wandb)
# --------------------------------------------------------------------------- #
def evaluate_2d(model, data: Data, hp: HParams,
                th_slice: float = np.pi / 4, grid_n: int = 140):
    """
    Returns (metrics, fig):
      metrics  — test/* violation fractions on the safe/unsafe/expert sets
      fig      — matplotlib 2D level-set figure of h(x) at `th_slice`
    Reuses the rendering helpers from plot_iters.py.
    """

    #TODO instead import a function from the plotting class as this is reused. and keept the training code cleaner.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_iters as plot

    h = jax.vmap(model)
    X_safe   = np.asarray(data.x_safe)
    X_unsafe = np.asarray(data.x_unsafe)
    X_expert = np.asarray(data.x_constraint)

    h_safe   = np.asarray(h(data.x_safe))
    h_unsafe = np.asarray(h(data.x_unsafe))
    r        = np.asarray(vmap(r_with_input, in_axes=(0, 0, None))(
        data.x_constraint, data.u_constraint, model))

    metrics = {
        "test/safe_pct":    100.0 * float((h_safe   < hp.safe_value).mean()),
        "test/unsafe_pct":  100.0 * float((-h_unsafe < hp.unsafe_value).mean()),
        "test/dyn_pct":     100.0 * float((r < hp.gamma).mean()),
        "test/min_h_safe":  float(h_safe.min()),
        "test/max_h_unsafe": float(h_unsafe.max()),
        "test/min_r":       float(r.min()),
    }

    XX, YY, pts, bounds = plot.workspace_grid(
        [X_safe, X_unsafe, X_expert], th_slice, grid_n=grid_n)
    h_grid = np.asarray(h(jnp.asarray(pts, dtype=jnp.float32))).reshape(XX.shape)
    s = plot.near_theta(X_safe, th_slice)
    u = plot.near_theta(X_unsafe, th_slice)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    cf = plot.draw_h_field(ax, XX, YY, h_grid, s, u, bounds)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04,
                 label="h(x)  (red<0, blue>0)")
    ax.set_title(f"theta slice = {np.degrees(th_slice):.0f}°   "
                 f"safe {metrics['test/safe_pct']:.1f}% / "
                 f"unsafe {metrics['test/unsafe_pct']:.1f}% / "
                 f"dyn {metrics['test/dyn_pct']:.1f}% off-spec", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return metrics, fig


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(data: Data, hp: HParams = HParams(),
          num_epochs: int = 50_000, lr: float = 0.1, seed: int = 5433,
          log_every: int = 1000, use_wandb: bool = True,
          eval_every: int = 10_000, eval_theta_slice: float = np.pi / 4,
          wandb_project: str = "neural-cbf", wandb_run_name: str = None,
          wandb_mode: str = None):

    model    = make_model(jax.random.PRNGKey(seed))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=num_epochs)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step      = make_step(optimizer, hp)

    if use_wandb:
        wandb.init(
            project=wandb_project, name=wandb_run_name, mode=wandb_mode,
            config={**hp._asdict(),
                    "num_epochs": num_epochs, "lr": lr, "seed": seed,
                    "n_hidden": N_HIDDEN, "n_hidden_layers": N_HIDDEN_LAYERS,
                    "n_safe":   int(data.x_safe.shape[0]),
                    "n_unsafe": int(data.x_unsafe.shape[0]),
                    "n_expert": int(data.x_constraint.shape[0])},
        )

    for epoch in range(num_epochs):
        model, opt_state, total, components = step(model, opt_state, data)

        if log_every and (epoch % log_every == 0 or epoch == num_epochs - 1):
            lr_now  = float(schedule(epoch))
            metrics = {k: float(v) for k, v in components.items()}
            metrics.update({"loss/total": float(total), "lr": lr_now})
            print(f"epoch {epoch:6d} | loss {metrics['loss/total']:12.4f} "
                  f"| min_r {metrics['diag/min_r']:+.3f} | lr {lr_now:.5f}")
            if use_wandb:
                fig = None
                if eval_every and (epoch % eval_every == 0
                                   or epoch == num_epochs - 1):
                    test_metrics, fig = evaluate_2d(
                        model, data, hp, th_slice=eval_theta_slice)
                    metrics.update(test_metrics)
                    metrics["test/levelset"] = wandb.Image(fig)
                wandb.log(metrics, step=epoch)
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

    if use_wandb:
        wandb.finish()
    return model


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="stage-2 dataset tag (reads runs/datasets/<tag>/"
                        "{expert_safe,unsafe,safe}.csv)")
    p.add_argument("--num-epochs", type=int,   default=50_000)
    p.add_argument("--lr",         type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=5433)
    p.add_argument("--log-every",  type=int,   default=1000)
    p.add_argument("--eval-every", type=int,   default=10_000,
                   help="log a test/ step (2D level-set + violations) every N epochs")
    p.add_argument("--eval-theta-slice", type=float, default=np.pi / 4,
                   help="theta value (rad) for the 2D level-set slice")
    p.add_argument("--wandb-project",  default="neural-cbf")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode",     default="online")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def save_model(out_dir: Path, model, meta: dict) -> None:
    """Serialise the trained Equinox MLP + a small meta sidecar."""
    eqx.tree_serialise_leaves(str(out_dir / "model.eqx"), model)
    pio.save_meta(out_dir, meta)


def load_model(model_path: str, key=None):
    """Rebuild the MLP skeleton and load saved leaves from `model_path`."""
    if key is None:
        key = jax.random.PRNGKey(0)          # leaves get overwritten on load
    skeleton = make_model(key)
    return eqx.tree_deserialise_leaves(str(model_path), skeleton)




def main():
    args = parse_args()

    ds_dir = pio.resolve_existing(pio.STAGE_DATA, args.dataset)

    # X_N: X_sampled_unsafe U X_sampled_unsafe. These are sampled from the continous set N which is sigma offset of the continous set D. D is the defined as the union of n-dimentional balls of radius ε centered at the expert trajectories. The sampled states are labeled as safe/unsafe by checking whether they are inside/outside the union of n-dimentional balls. The expert trajectories are also labeled as safe/unsafe by checking whether they are inside/outside the union of n-dimentional balls centered at each state in the expert trajectories. This bridges the X_safe samples and continous space.
    # X_safe: X_expert_safe U X_sampled_safe. X_safe_bar is the set of deep safe states.

    #Load from one file instead.
    print(f"[stage 3/nn] parent dataset = {args.dataset}")
    X_expert_safe, _ = load_expert_safe(str(ds_dir / "expert_safe.csv"))  # discrete Z_dyn: expert (x,u)
    X_expert_unsafe, _ = load_expert_safe(str(ds_dir / "expert_safe.csv"))  # discrete Z_dyn: expert (x,u)
    X_sampled_safe               = load_states_only(str(ds_dir / "safe.csv")) # X_N:   ε̄-net of N
    X_sampled_unsafe             = load_states_only(str(ds_dir / "unsafe.csv")) 
    

    print(f"  Union of expert trajectories X={X_expert_safe.shape}")
    print(f"  unsafe      X={X_sampled_safe.shape}")
    print(f"  safe        X={X_sampled_unsafe.shape}")

    #Combining safe sampled states and expert trajectories to get the full safe set.
    X_safe = np.concatenate([X_expert_safe, X_sampled_safe], axis=0)
    X_unsafe = X_sampled_unsafe


    data = Data(
        x_constraint    = jnp.asarray(X_expert_safe),    # state - trajectory (expert safe)
        x_safe          = jnp.asarray(X_safe),   # states - sampled from safe interior
        x_unsafe        = jnp.asarray(X_unsafe), # states - sampled from unsafe exterior
        x_expert_unsafe = jnp.asarray(X_expert_unsafe), # expert unsafe states
        x_boundary      = None,
    )

    hp = HParams()
    model = train(data, hp=hp, num_epochs=args.num_epochs, lr=args.lr,
                  seed=args.seed, log_every=args.log_every,
                  eval_every=args.eval_every,
                  eval_theta_slice=args.eval_theta_slice,
                  use_wandb=not args.no_wandb,
                  wandb_project=args.wandb_project,
                  wandb_run_name=args.wandb_run_name, wandb_mode=args.wandb_mode)

    #TODO save the model and meta (dataset tag + hp dict)


if __name__ == "__main__":
    main()