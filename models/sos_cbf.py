"""
cbf_optim.py
============
Optimisation-based CBF learning from expert demonstrations.

Implements the convex programme from §3.4 of Robey et al. (2020)
"Learning Control Barrier Functions from Expert Demonstrations"
(arXiv 2004.03315), equation (3.6) and the hinge-loss relaxation (3.7).

System: single-agent unicycle
    state    x = [px, py, theta]^T            (n_x = 3)
    control  u = [v, omega]^T                  (n_u = 2)
    dynamics  xdot = f(x) + g(x) u   with  f(x) = 0  and
                       [cos(theta)  0]
              g(x) = [ sin(theta)  0]
                       [    0       1]

Parameterisation: Random Fourier Features (RFF) approximating a Gaussian-RBF
RKHS, which keeps the whole problem convex when alpha(h) = h.

CSV layout (header row required, single agent):
    x,y,theta,v,omega
    2.129...,1.431...,0.678...,0.5,0.0953...
    ...

Three separate CSVs are passed via argparse:
    --expert-safe   expert (x,u) pairs inside the safe set    (uses all 5 cols)
    --unsafe        states only, sampled inside the N-ring    (uses cols 0..2)
    --safe          states only, sampled deep inside safe set (uses cols 0..2)

Run:
    python cbf_optim.py \
        --expert-safe data/expert_safe.csv \
        --unsafe       data/unsafe.csv \
        --safe         data/safe.csv \
        --mode hard \
        --solver SCS
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Any, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pipeline_io as pio  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 0.  Constants
# ──────────────────────────────────────────────────────────────────────

N_X = 3        # state dimension  [px, py, theta]
N_U = 2        # control dimension [v, omega]


# ──────────────────────────────────────────────────────────────────────
# 1.  CSV loading
# ──────────────────────────────────────────────────────────────────────

def load_expert_safe(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expert (x, u) pairs from the safe interior.

    Returns
    -------
    X : (N, 3)   states  [x, y, theta]
    U : (N, 2)   controls [v, omega]
    """
    df = pd.read_csv(path)
    X = df[["x", "y", "theta"]].to_numpy(dtype=np.float64)
    U = df[["v", "omega"]].to_numpy(dtype=np.float64)
    return X, U


def load_states_only(path: str) -> np.ndarray:
    """Load only the state columns (unsafe and deep-safe samples)."""
    df = pd.read_csv(path)
    return df[["x", "y", "theta"]].to_numpy(dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# 2.  Safety specification  (PLACEHOLDER — fill this in)
# ──────────────────────────────────────────────────────────────────────

def geometric_safety_value(x: np.ndarray) -> np.ndarray:
    """
    Geometric safe-set indicator  s(x).
    Convention: s(x) > 0  ⇔ safe,  s(x) < 0 ⇔ unsafe.

    NOTE: This function is only used inside `verify_conditions` for
    sanity-checking — the actual learning uses the labels implicit in
    which CSV each sample comes from.  Edit it to match your obstacle
    geometry if you want the verification to be meaningful.

    Example (single circular obstacle at origin with radius Ds):
        return np.linalg.norm(x[..., :2], axis=-1) - Ds

    Parameters
    ----------
    x : (N, 3) or (3,)
    """
    # TODO: define your safe set here
    raise NotImplementedError("geometric_safety_value() — fill in your obstacle geometry")


# ──────────────────────────────────────────────────────────────────────
# 3.  Dynamics  (control-affine:  xdot = f(x) + g(x) u )
# ──────────────────────────────────────────────────────────────────────

def dynamics_f(x: np.ndarray) -> np.ndarray:
    """Drift term f(x).  Zero for the unicycle."""
    return np.zeros_like(x)


def dynamics_g(x: np.ndarray) -> np.ndarray:
    """
    Input matrix g(x) for the unicycle.

        g(x) = [[cos(theta), 0],
                [sin(theta), 0],
                [    0,      1]]

    Parameters
    ----------
    x : (N, 3) or (3,)

    Returns
    -------
    G : (N, 3, 2) or (3, 2)
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]

    N = x.shape[0]
    G = np.zeros((N, N_X, N_U))
    G[:, 0, 0] = np.cos(x[:, 2])
    G[:, 1, 0] = np.sin(x[:, 2])
    G[:, 2, 1] = 1.0

    return G[0] if single else G


# ──────────────────────────────────────────────────────────────────────
# 4.  Random Fourier Features
# ──────────────────────────────────────────────────────────────────────

def build_rff(n_features: int, sigma: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw W ~ N(0, sigma^2 I) and b ~ Uniform(0, 2π) for the feature map
        phi(x) = sqrt(2/L) * cos(W x + b).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, sigma, size=(n_features, N_X))
    b = rng.uniform(0, 2 * np.pi, size=(n_features,))
    return W, b


def phi(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RFF map.  x: (N, n_x) → Phi: (N, L)."""
    L = W.shape[0]
    z = x @ W.T + b
    return np.sqrt(2.0 / L) * np.cos(z)


def dphi(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Jacobian of phi w.r.t. x.

        d phi_j / d x_k  =  -sqrt(2/L) * sin(W_j · x + b_j) * W_{j,k}

    Returns
    -------
    J : (N, L, n_x)
    """
    L = W.shape[0]
    z = x @ W.T + b
    coeff = -np.sqrt(2.0 / L)
    sin_z = np.sin(z)
    return coeff * sin_z[:, :, None] * W[None, :, :]


# ──────────────────────────────────────────────────────────────────────
# 5.  Convex programme  (paper eq 3.6, hard constraints)
# ──────────────────────────────────────────────────────────────────────

def learn_cbf_rff(
    x_safe: np.ndarray,
    u_safe: np.ndarray,
    x_unsafe: np.ndarray,
    x_interior: np.ndarray,
    *,
    n_features: int = 200,
    sigma: float = 1.2,
    gamma_safe: float = 1.5,
    gamma_unsafe: float = 0.5,
    gamma_dyn: float = 0.05,
    solver: str = "SCS",
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Hard-constraint convex programme:

        min  ||theta||_2
        s.t. h(x_i) >= gamma_safe        for x_i in x_interior
             h(x_i) <= -gamma_unsafe     for x_i in x_unsafe
             ∇h(x_i)·(f+g u_i) + h(x_i) >= gamma_dyn   for (x_i,u_i) in expert
             h(x_i) >= gamma_safe/10     for x_i in x_safe (expert)

    With alpha(h) = h  and  h(x) = phi(x)^T theta, every constraint is
    affine in theta — this is a Second-Order Cone Programme.
    """
    W_rff, b_rff = build_rff(n_features, sigma, seed)

    Phi_safe     = phi(x_safe, W_rff, b_rff)
    Phi_unsafe   = phi(x_unsafe, W_rff, b_rff)
    Phi_interior = phi(x_interior, W_rff, b_rff)
    J_safe       = dphi(x_safe, W_rff, b_rff)         # (N1, L, n_x)

    f_vals = dynamics_f(x_safe)                        # (N1, 3)
    G_vals = dynamics_g(x_safe)                        # (N1, 3, 2)
    xdot   = f_vals + np.einsum("ijk,ik->ij", G_vals, u_safe)
    A_dyn  = np.einsum("ilk,ik->il", J_safe, xdot)    # (N1, L)

    theta = cp.Variable(n_features, name="theta")
    constraints = [
        Phi_interior @ theta >= gamma_safe,                       # deep-safe
        Phi_unsafe   @ theta <= -gamma_unsafe,                    # N-ring
        (A_dyn + Phi_safe) @ theta >= gamma_dyn,                  # CBF deriv.
        Phi_safe     @ theta >= gamma_safe / 10.0,                # expert lb
    ]

    problem = cp.Problem(cp.Minimize(cp.norm(theta, 2)), constraints)

    if verbose:
        print(f"  features     : {n_features}")
        print(f"  safe interior: {x_interior.shape[0]} rows")
        print(f"  unsafe       : {x_unsafe.shape[0]} rows")
        print(f"  expert safe  : {x_safe.shape[0]} rows")
        print(f"  solver       : {solver}")

    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"  WARNING: solver status = {problem.status}")

    return {
        "theta": theta.value, "W": W_rff, "b": b_rff,
        "problem": problem, "status": problem.status,
    }


# ──────────────────────────────────────────────────────────────────────
# 6.  Hinge-loss relaxation  (paper eq 3.7, soft constraints)
# ──────────────────────────────────────────────────────────────────────

def learn_cbf_rff_soft(
    x_safe: np.ndarray,
    u_safe: np.ndarray,
    x_unsafe: np.ndarray,
    x_interior: np.ndarray,
    *,
    n_features: int = 200,
    sigma: float = 1.2,
    gamma_safe: float = 1.5,
    gamma_unsafe: float = 0.5,
    gamma_dyn: float = 0.05,
    lam_safe: float = 2.0,
    lam_unsafe: float = 2.0,
    lam_dyn: float = 15.0,
    lam_param: float = 0.1,
    x_neutral: np.ndarray | None = None,
    gamma_neutral: float = 0.3,
    lam_neutral: float = 5.0,
    solver: str = "SCS",
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Soft penalty formulation.  If x_neutral is given, also adds a far-field
    cap term enforcing |h(x_neutral)| <= gamma_neutral as a soft hinge.
    """
    W_rff, b_rff = build_rff(n_features, sigma, seed)

    Phi_safe     = phi(x_safe, W_rff, b_rff)
    Phi_unsafe   = phi(x_unsafe, W_rff, b_rff)
    Phi_interior = phi(x_interior, W_rff, b_rff)
    J_safe       = dphi(x_safe, W_rff, b_rff)

    f_vals = dynamics_f(x_safe)
    G_vals = dynamics_g(x_safe)
    xdot   = f_vals + np.einsum("ijk,ik->ij", G_vals, u_safe)
    A_dyn  = np.einsum("ilk,ik->il", J_safe, xdot)

    theta = cp.Variable(n_features)

    safe_hinge   = cp.sum(cp.pos(gamma_safe - Phi_interior @ theta))
    unsafe_hinge = cp.sum(cp.pos(Phi_unsafe @ theta + gamma_unsafe))
    dyn_hinge    = cp.sum(cp.pos(gamma_dyn - (A_dyn + Phi_safe) @ theta))
    param_reg    = cp.sum_squares(theta)

    terms = (lam_param  * param_reg
             + lam_safe   * safe_hinge
             + lam_unsafe * unsafe_hinge
             + lam_dyn    * dyn_hinge)

    if x_neutral is not None and x_neutral.shape[0] > 0:
        Phi_neutral = phi(x_neutral, W_rff, b_rff)
        # |h| - gamma_neutral hinge: penalise |h(x)| > gamma_neutral.
        h_neu = Phi_neutral @ theta
        neutral_hinge = (cp.sum(cp.pos(h_neu - gamma_neutral))
                         + cp.sum(cp.pos(-h_neu - gamma_neutral)))
        terms = terms + lam_neutral * neutral_hinge

    objective = cp.Minimize(terms)

    problem = cp.Problem(objective)
    problem.solve(solver=solver, verbose=verbose)

    return {
        "theta": theta.value, "W": W_rff, "b": b_rff,
        "problem": problem, "status": problem.status,
    }


# ──────────────────────────────────────────────────────────────────────
# 7.  Evaluation helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_h(x: np.ndarray, theta: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """h(x) = phi(x)^T theta.  Returns (N,)."""
    return phi(x, W, b) @ theta


def evaluate_grad_h(x: np.ndarray, theta: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """∇h(x) = J_phi(x)^T theta.  Returns (N, n_x)."""
    J = dphi(x, W, b)
    return np.einsum("ilk,l->ik", J, theta)


def verify_conditions(
    x_safe: np.ndarray, u_safe: np.ndarray,
    x_unsafe: np.ndarray, x_interior: np.ndarray,
    theta: np.ndarray, W: np.ndarray, b: np.ndarray,
    gamma_safe: float, gamma_unsafe: float, gamma_dyn: float,
) -> Dict[str, Any]:
    """Post-hoc Theorem 3.4 check.  All slacks should be ≥ 0."""
    h_interior = evaluate_h(x_interior, theta, W, b)
    h_unsafe   = evaluate_h(x_unsafe,   theta, W, b)
    h_safe_exp = evaluate_h(x_safe,     theta, W, b)

    grad_h = evaluate_grad_h(x_safe, theta, W, b)
    f_vals = dynamics_f(x_safe)
    G_vals = dynamics_g(x_safe)
    xdot   = f_vals + np.einsum("ijk,ik->ij", G_vals, u_safe)
    lie_deriv = np.sum(grad_h * xdot, axis=1)
    q_vals = lie_deriv + h_safe_exp            # alpha(h) = h

    safe_slack   = h_interior - gamma_safe
    unsafe_slack = -h_unsafe   - gamma_unsafe
    dyn_slack    = q_vals      - gamma_dyn

    print("\n=== Theorem 3.4 verification ===")
    print(f"  safe   slack min = {safe_slack.min():+.4f}   "
          f"violations = {int((safe_slack < 0).sum())} / {len(safe_slack)}")
    print(f"  unsafe slack min = {unsafe_slack.min():+.4f}   "
          f"violations = {int((unsafe_slack < 0).sum())} / {len(unsafe_slack)}")
    print(f"  dyn    slack min = {dyn_slack.min():+.4f}   "
          f"violations = {int((dyn_slack < 0).sum())} / {len(dyn_slack)}")

    return {
        "safe_slack_min":   float(safe_slack.min()),
        "unsafe_slack_min": float(unsafe_slack.min()),
        "dyn_slack_min":    float(dyn_slack.min()),
    }


# ──────────────────────────────────────────────────────────────────────
# 8.  CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   help="stage-2 dataset tag (reads runs/datasets/<tag>/{expert_safe,unsafe,safe}.csv)")
    p.add_argument("--mode", choices=["hard", "soft"], default="hard")
    p.add_argument("--n-features", type=int, default=200)
    p.add_argument("--sigma",      type=float, default=1.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--gamma-safe",   type=float, default=1.5)
    p.add_argument("--gamma-unsafe", type=float, default=0.5)
    p.add_argument("--gamma-dyn",    type=float, default=0.05)
    p.add_argument("--lam-safe",   type=float, default=2.0)
    p.add_argument("--lam-unsafe", type=float, default=2.0)
    p.add_argument("--lam-dyn",    type=float, default=15.0)
    p.add_argument("--lam-param",  type=float, default=0.1)
    p.add_argument("--gamma-neutral", type=float, default=0.3,
                   help="far-field cap: |h(x_neutral)| <= this")
    p.add_argument("--lam-neutral",   type=float, default=5.0,
                   help="weight on the neutral-cap hinge (0 disables)")
    p.add_argument("--solver", default="SCS",
                   choices=["SCS", "MOSEK", "CLARABEL", "ECOS"])
    p.add_argument("--quiet",  action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="Run Theorem 3.4 verification after solving")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    ds_dir = pio.resolve_existing(pio.STAGE_DATA, args.dataset)
    print(f"[stage 3/rff] parent dataset = {args.dataset}")
    X_safe, U_safe = load_expert_safe(str(ds_dir / "expert_safe.csv"))
    X_unsafe       = load_states_only(str(ds_dir / "unsafe.csv"))
    X_interior     = load_states_only(str(ds_dir / "safe.csv"))
    X_neutral = None
    neutral_csv = ds_dir / "neutral.csv"
    if neutral_csv.exists():
        X_neutral = load_states_only(str(neutral_csv))
    print(f"  expert_safe X={X_safe.shape}  U={U_safe.shape}")
    print(f"  unsafe      X={X_unsafe.shape}")
    print(f"  safe        X={X_interior.shape}")
    if X_neutral is not None:
        print(f"  neutral     X={X_neutral.shape}")

    CFG = dict(
        method="rff", dataset_tag=args.dataset, mode=args.mode,
        n_features=args.n_features, sigma=args.sigma, seed=args.seed,
        gamma_safe=args.gamma_safe, gamma_unsafe=args.gamma_unsafe,
        gamma_dyn=args.gamma_dyn,
        lam_safe=args.lam_safe, lam_unsafe=args.lam_unsafe,
        lam_dyn=args.lam_dyn, lam_param=args.lam_param,
        gamma_neutral=args.gamma_neutral, lam_neutral=args.lam_neutral,
        has_neutral=X_neutral is not None,
        solver=args.solver,
    )
    human = (f"{args.mode}_L{args.n_features}_sig{args.sigma:g}_"
             f"gs{args.gamma_safe:g}")
    TAG = pio.make_tag(human, CFG)
    OUT = pio.run_dir(pio.STAGE_CBF, TAG, method="rff")
    print(f"  tag = {TAG}\n  out = {OUT}")

    t0 = time.monotonic()
    if args.mode == "hard":
        result = learn_cbf_rff(
            X_safe, U_safe, X_unsafe, X_interior,
            n_features=args.n_features, sigma=args.sigma, seed=args.seed,
            gamma_safe=args.gamma_safe, gamma_unsafe=args.gamma_unsafe,
            gamma_dyn=args.gamma_dyn, solver=args.solver, verbose=verbose,
        )
    else:
        result = learn_cbf_rff_soft(
            X_safe, U_safe, X_unsafe, X_interior,
            n_features=args.n_features, sigma=args.sigma, seed=args.seed,
            gamma_safe=args.gamma_safe, gamma_unsafe=args.gamma_unsafe,
            gamma_dyn=args.gamma_dyn,
            lam_safe=args.lam_safe, lam_unsafe=args.lam_unsafe,
            lam_dyn=args.lam_dyn,   lam_param=args.lam_param,
            x_neutral=X_neutral,
            gamma_neutral=args.gamma_neutral, lam_neutral=args.lam_neutral,
            solver=args.solver, verbose=verbose,
        )
    runtime_s = time.monotonic() - t0
    print(f"\n[done] status={result['status']}")

    metrics: Dict[str, Any] = dict(status=result["status"], runtime_s=runtime_s)
    if result["theta"] is not None:
        theta_norm = float(np.linalg.norm(result["theta"]))
        metrics["theta_norm"] = theta_norm
        print(f"  ||theta||_2 = {theta_norm:.4f}")

        np.savez(OUT / "model.npz",
                 theta=result["theta"], W=result["W"], b=result["b"],
                 sigma=float(args.sigma),
                 n_features=int(args.n_features),
                 seed=int(args.seed))

        if args.verify:
            slacks = verify_conditions(
                X_safe, U_safe, X_unsafe, X_interior,
                result["theta"], result["W"], result["b"],
                args.gamma_safe, args.gamma_unsafe, args.gamma_dyn,
            )
            metrics.update(slacks)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=repr))
    pio.save_config(OUT, CFG)
    pio.save_meta(OUT, metrics)
    pio.append_index_row(pio.STAGE_CBF, dict(
        tag=TAG, method="rff", dataset_tag=args.dataset,
        mode=args.mode, n_features=args.n_features, sigma=args.sigma,
        gamma_safe=args.gamma_safe, gamma_unsafe=args.gamma_unsafe,
        gamma_dyn=args.gamma_dyn, solver=args.solver,
        status=result["status"], theta_norm=metrics.get("theta_norm", ""),
        runtime_s=f"{runtime_s:.1f}", path=str(OUT),
    ))


if __name__ == "__main__":
    main()