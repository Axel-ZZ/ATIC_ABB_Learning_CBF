"""Minibatched variant of train_nn_cbf.py.

Same model, loss and dataset interface — the only difference is the training
step samples a random minibatch from each of the (large) data sets every
iteration instead of using the full batch. On CPU this cuts per-step cost
roughly in proportion to the subsample size.

Shared pieces (model, loss, eval, data loading) are imported from
train_nn_cbf so the two scripts can't drift apart.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb

sys.path.append(str(Path(__file__).resolve().parent))
from train_nn_cbf import (  # noqa: E402
    Data, HParams, N_HIDDEN, N_HIDDEN_LAYERS, MODELS_DIR,
    make_model, loss_with_input, evaluate_2d, save_model,
    resolve_dataset, load_expert, load_states_only,
)


# --------------------------------------------------------------------------- #
# Minibatched training step
# --------------------------------------------------------------------------- #
def make_minibatch_step(optimizer, hp: HParams, batch_size: int):
    """Equinox-aware grad + optax update on a freshly sampled minibatch.

    Each set is sampled independently with replacement; (x_constraint,
    u_constraint) share indices so the expert (x, u) pairs stay aligned.
    """
    @eqx.filter_jit
    def step(model, opt_state, data: Data, key):
        kc, ks, ku, ke = jax.random.split(key, 4)

        def take(arr, k):
            idx = jax.random.randint(k, (batch_size,), 0, arr.shape[0])
            return arr[idx]

        idx_c = jax.random.randint(kc, (batch_size,), 0, data.x_constraint.shape[0])
        batch = Data(
            x_constraint    = data.x_constraint[idx_c],
            u_constraint    = data.u_constraint[idx_c],
            x_safe          = take(data.x_safe, ks),
            x_unsafe        = take(data.x_unsafe, ku),
            x_expert_unsafe = take(data.x_expert_unsafe, ke),
            x_boundary      = None,
        )
        (total, components), grads = eqx.filter_value_and_grad(
            loss_with_input, has_aux=True)(model, batch, hp)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, total, components
    return step


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(data: Data, hp: HParams = HParams(), batch_size: int = 4096,
          num_epochs: int = 50_000, lr: float = 0.1, seed: int = 5433,
          log_every: int = 100, use_wandb: bool = True,
          eval_every: int = 10_000, eval_theta_slice: float = np.pi / 4,
          wandb_project: str = "neural-cbf", wandb_run_name: str = None,
          wandb_mode: str = None):

    key       = jax.random.PRNGKey(seed)
    key, mkey = jax.random.split(key)
    model     = make_model(mkey)
    schedule  = optax.cosine_decay_schedule(init_value=lr, decay_steps=num_epochs)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step      = make_minibatch_step(optimizer, hp, batch_size)

    if use_wandb:
        wandb.init(
            project=wandb_project, name=wandb_run_name, mode=wandb_mode,
            config={**hp._asdict(),
                    "batch_size": batch_size,
                    "num_epochs": num_epochs, "lr": lr, "seed": seed,
                    "n_hidden": N_HIDDEN, "n_hidden_layers": N_HIDDEN_LAYERS,
                    "n_safe":   int(data.x_safe.shape[0]),
                    "n_unsafe": int(data.x_unsafe.shape[0]),
                    "n_expert": int(data.x_constraint.shape[0])},
        )

    for epoch in range(num_epochs):
        key, skey = jax.random.split(key)
        model, opt_state, total, components = step(model, opt_state, data, skey)

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
                    # eval on the FULL sets for honest metrics
                    test_metrics, fig = evaluate_2d(
                        model, data, hp, th_slice=eval_theta_slice)
                    metrics.update(test_metrics)
                    if fig is not None:
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
                   help="dataset tag or unique prefix; reads runs/datasets/<tag>/"
                        "{expert_safe,expert_unsafe,safe,unsafe}.csv")

    # optimisation
    p.add_argument("--batch-size", type=int,   default=4096,
                   help="samples drawn (with replacement) from each set per step")
    p.add_argument("--num-epochs", type=int,   default=50_000)
    p.add_argument("--lr",         type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=5433)

    # hyperparameters (HParams)
    defaults = HParams()
    p.add_argument("--safe-value",     type=float, default=defaults.safe_value)
    p.add_argument("--unsafe-value",   type=float, default=defaults.unsafe_value)
    p.add_argument("--gamma",          type=float, default=defaults.gamma)
    p.add_argument("--lam-constraint", type=float, default=defaults.lam_constraint)
    p.add_argument("--lam-boundary",   type=float, default=defaults.lam_boundary)
    p.add_argument("--lam-safe",       type=float, default=defaults.lam_safe)
    p.add_argument("--lam-unsafe",     type=float, default=defaults.lam_unsafe)
    p.add_argument("--lam-param",      type=float, default=defaults.lam_param)

    # logging / eval
    p.add_argument("--log-every",  type=int,   default=100)
    p.add_argument("--eval-every", type=int,   default=10_000,
                   help="log a test/ step (2D level-set + violations) every N epochs")
    p.add_argument("--eval-theta-slice", type=float, default=np.pi / 4,
                   help="theta value (rad) for the 2D level-set slice")
    p.add_argument("--wandb-project",  default="neural-cbf")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode",     default="online")
    p.add_argument("--no-wandb", action="store_true")

    # output
    p.add_argument("--out-dir", default=None,
                   help="where to save model.eqx + meta.json "
                        "(default: runs/models/<dataset-tag>_minibatch/)")
    return p.parse_args()


def main():
    args = parse_args()

    ds_dir = resolve_dataset(args.dataset)
    tag    = ds_dir.name

    print(f"[stage 3/nn-minibatch] dataset = {tag}  batch_size = {args.batch_size}")
    X_expert_safe, U_expert_safe = load_expert(str(ds_dir / "expert_safe.csv"))
    X_expert_unsafe              = load_states_only(str(ds_dir / "expert_unsafe.csv"))
    X_sampled_safe               = load_states_only(str(ds_dir / "safe.csv"))
    X_sampled_unsafe             = load_states_only(str(ds_dir / "unsafe.csv"))

    print(f"  expert_safe   X={X_expert_safe.shape}  U={U_expert_safe.shape}")
    print(f"  expert_unsafe X={X_expert_unsafe.shape}")
    print(f"  safe          X={X_sampled_safe.shape}")
    print(f"  unsafe        X={X_sampled_unsafe.shape}")

    X_safe = np.concatenate([X_expert_safe, X_sampled_safe], axis=0)

    data = Data(
        x_constraint    = jnp.asarray(X_expert_safe),
        u_constraint    = jnp.asarray(U_expert_safe),
        x_safe          = jnp.asarray(X_safe),
        x_unsafe        = jnp.asarray(X_sampled_unsafe),
        x_expert_unsafe = jnp.asarray(X_expert_unsafe),
        x_boundary      = None,
        x_deep_safe     = jnp.asarray(X_sampled_safe),    # sampled deep-safe only (eval)
    )

    hp = HParams(
        safe_value     = args.safe_value,
        unsafe_value   = args.unsafe_value,
        gamma          = args.gamma,
        lam_constraint = args.lam_constraint,
        lam_boundary   = args.lam_boundary,
        lam_safe       = args.lam_safe,
        lam_unsafe     = args.lam_unsafe,
        lam_param      = args.lam_param,
    )
    model = train(data, hp=hp, batch_size=args.batch_size,
                  num_epochs=args.num_epochs, lr=args.lr,
                  seed=args.seed, log_every=args.log_every,
                  eval_every=args.eval_every,
                  eval_theta_slice=args.eval_theta_slice,
                  use_wandb=not args.no_wandb,
                  wandb_project=args.wandb_project,
                  wandb_run_name=args.wandb_run_name, wandb_mode=args.wandb_mode)

    out_dir = Path(args.out_dir) if args.out_dir else MODELS_DIR / f"{tag}_minibatch"
    meta = {
        "dataset": tag,
        "trainer": "minibatch",
        "batch_size": args.batch_size,
        "hparams": hp._asdict(),
        "num_epochs": args.num_epochs, "lr": args.lr, "seed": args.seed,
        "n_hidden": N_HIDDEN, "n_hidden_layers": N_HIDDEN_LAYERS,
        "n_safe": int(X_safe.shape[0]),
        "n_unsafe": int(X_sampled_unsafe.shape[0]),
        "n_expert": int(X_expert_safe.shape[0]),
    }
    save_model(out_dir, model, meta)
    print(f"[done] saved model + meta to {out_dir}")


if __name__ == "__main__":
    main()
