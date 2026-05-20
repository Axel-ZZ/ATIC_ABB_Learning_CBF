# `cbf_optim.py` — Paper-to-Code Reference

Implementation of CBF learning from Robey et al., *"Learning Control Barrier
Functions from Expert Demonstrations"* (arXiv:2004.03315, CDC 2020).

Every design decision is tagged:

- **[PAPER]** — directly from arXiv:2004.03315.
- **[NOTEBOOK]** — from the authors' airplane notebook (`unstable-zeros/learning-cbfs`), not stated in the paper.
- **[ADDED]** — engineering choice in neither source; flagged for stripping in strict reproduction.

---

## 1. System dynamics

### Paper (eq. 1)

The paper assumes a control-affine nonlinear system:

$$\dot{x}(t) = f(x(t)) + g(x(t))\,u(t), \qquad x(0) \in \mathbb{R}^n$$

where $f : \mathbb{R}^n \to \mathbb{R}^n$ and $g : \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are locally Lipschitz. The theory is agnostic to the specific system — it applies to any control-affine plant satisfying these regularity conditions.

### Implementation

The implemented system is a single-agent unicycle **[ADDED]** (the paper's examples are a 2-D planar system and a 6-D two-airplane system):

$$x = \begin{bmatrix} p_x \\ p_y \\ \theta \end{bmatrix} \in \mathbb{R}^3, \qquad u = \begin{bmatrix} v \\ \omega \end{bmatrix} \in \mathbb{R}^2$$

$$f(x) = \mathbf{0}, \qquad g(x) = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix}$$

The zero drift $f(x) = 0$ is **[NOTEBOOK]** — the airplane notebook also uses zero drift. The unicycle satisfies the paper's Lipschitz requirements ($g$ is continuously differentiable, hence locally Lipschitz), so Theorem III.4 applies.

```python
N_X = 3                                        # state dimension
N_U = 2                                        # control dimension

def dynamics_f(x):
    """f(x) = 0. Shape: same as x."""
    return np.zeros_like(x)

def dynamics_g(x):
    """g(x) for unicycle. x: (N,3) → G: (N,3,2)."""
    N = x.shape[0]
    G = np.zeros((N, N_X, N_U))
    G[:, 0, 0] = np.cos(x[:, 2])              # dp_x/dt  = v cos(θ)
    G[:, 1, 0] = np.sin(x[:, 2])              # dp_y/dt  = v sin(θ)
    G[:, 2, 1] = 1.0                           # dθ/dt    = ω
    return G
```

**Why the dynamics model is needed at all:** the CBF derivative constraint (§4, constraint C3) requires computing $\dot{x}_i = f(x_i) + g(x_i)\,u_i$ at each expert data point. The model provides exact $\dot{x}$ from each $(x, u)$ pair independently, without requiring sequential trajectory data or finite differencing.

---

## 2. Function class: RKHS via Random Fourier Features

### Paper (§III-C, Convexity paragraph)

The paper defines the hypothesis space as a linear-in-parameters family:

$$\mathcal{H} = \big\{\, h_\theta(\cdot) = \langle \varphi(\cdot),\, \theta \rangle \;:\; \theta \in \Theta \,\big\}$$

For this family, $h(x) = \varphi(x)^\top \theta$ is linear in $\theta$, and the RKHS norm reduces to $\|h\|_{\mathcal{H}} = \|\theta\|_2$. The paper uses Random Fourier Features (Rahimi & Recht, 2007) to approximate the Gaussian RBF kernel's RKHS with a finite feature map.

### Paper (§III-C, RFF definition)

Given $\ell$ random features, the map is:

$$\varphi(x) = \sqrt{\frac{2}{\ell}} \begin{pmatrix} \cos(\langle x, w_1 \rangle + b_1) \\ \vdots \\ \cos(\langle x, w_\ell \rangle + b_\ell) \end{pmatrix}, \qquad w_i \sim \mathcal{N}(0, \sigma^2 I), \quad b_i \sim \mathrm{Uniform}(0, 2\pi)$$

and the Jacobian (used by the paper to derive the Lipschitz bound, §III-C) has $i$-th row:

$$\frac{\partial \varphi_i}{\partial x} = -\sqrt{\frac{2}{\ell}}\,\sin(\langle x, w_i \rangle + b_i)\, w_i^\top$$

### Implementation **[PAPER]**

```python
def build_rff(n_features, sigma, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, sigma, size=(n_features, N_X))     # W ∈ ℝ^{ℓ × n}, rows are wᵢ
    b = rng.uniform(0, 2 * np.pi, size=(n_features,))    # b ∈ ℝ^ℓ,       entries are bᵢ
    return W, b                                           # FROZEN — not decision variables
```

$W$ and $b$ are sampled once and frozen. Only $\theta$ is optimised. The `seed` argument makes the basis (and therefore $h$) reproducible; changing it gives a different random approximation of the same underlying RKHS.

```python
def phi(x, W, b):
    """φ(x) = √(2/ℓ) cos(Wx + b).   x:(N,n) → Φ:(N,ℓ)"""
    L = W.shape[0]
    z = x @ W.T + b                                      # z[i,j] = ⟨xᵢ, wⱼ⟩ + bⱼ
    return np.sqrt(2.0 / L) * np.cos(z)
```

| Math | Code | Shape |
|---|---|---|
| $z_{ij} = \langle x_i, w_j \rangle + b_j$ | `z = x @ W.T + b` | $(N, \ell)$ |
| $\varphi_j(x_i) = \sqrt{2/\ell}\,\cos(z_{ij})$ | `np.sqrt(2.0/L) * np.cos(z)` | $(N, \ell)$ |
| $h(x_i) = \varphi(x_i)^\top \theta$ | `phi(x, W, b) @ theta` | $(N,)$ |

```python
def dphi(x, W, b):
    """Jacobian Dφ(x).   x:(N,n) → J:(N,ℓ,n)"""
    L = W.shape[0]
    z = x @ W.T + b
    coeff = -np.sqrt(2.0 / L)
    sin_z = np.sin(z)                                    # (N, ℓ)
    return coeff * sin_z[:, :, None] * W[None, :, :]     # (N, ℓ, n)
```

| Math | Code | Shape |
|---|---|---|
| $\frac{\partial \varphi_j}{\partial x_k}(x_i) = -\sqrt{2/\ell}\,\sin(z_{ij})\,W_{jk}$ | `coeff * sin_z[:,:,None] * W[None,:,:]` | $(N, \ell, n)$ |
| $\nabla h(x_i) = D\varphi(x_i)^\top \theta$ | `np.einsum("ilk,l->ik", J, theta)` | $(N, n)$ |

**Defaults:** $\ell = 200$, $\sigma = 1.2$ are **[PAPER]** — the planar example's exact settings.

---

## 3. The three data sets

### Paper (§II-B, §III-A)

The paper defines three data collections, each serving a structurally different constraint:

$$Z_{\text{dyn}} = \{(x_i, u_i)\}_{i=1}^{N_1} \quad \text{with } x_i \in \mathrm{int}(\mathcal{S})$$

Expert demonstrations — state-control pairs from safe trajectories. The states define an $\varepsilon$-net of the set $D = \bigcup_{i=1}^{N_1} B_{\varepsilon,p}(x_i)$ (paper eq. 4). These are the only samples with associated controls.

$$X_N = \{x_i\}_{i=1}^{N_2} \quad \text{sampled from } \mathcal{N} = \{\mathrm{bd}(D) \oplus B_{\sigma,p}(0)\} \setminus D$$

The $\mathcal{N}$-ring — a thin band of width $\sigma$ surrounding $D$. States only, no controls. Sampled by gridding or uniform random.

$$\bar{X}_{\text{safe}} = \Big\{ x_i \in X_{\text{safe}} \;:\; \inf_{x \in X_N} \|x - x_i\|_p \;\geq\; \frac{\gamma_{\text{unsafe}} + \gamma_{\text{safe}}}{L_h} \Big\}$$

(paper eq. 9) — the inner subset of expert points that are far enough from $X_N$ to allow $h$ to transition from $+\gamma_{\text{safe}}$ to $-\gamma_{\text{unsafe}}$ without excessive Lipschitz constant. States only (controls not needed for the value constraint).

### Implementation **[ADDED]** data source / **[PAPER]** structure

The code loads three CSVs via argparse, mapping each to the corresponding paper set:

| CLI flag | Paper set | Columns | Constraint served |
|---|---|---|---|
| `--expert-safe` | $Z_{\text{dyn}}$ | `x,y,theta,v,omega` | derivative (C3) |
| `--unsafe` | $X_N$ | `x,y,theta` | unsafe value (C2) |
| `--safe` | $\bar{X}_{\text{safe}}$ | `x,y,theta` | safe value (C1) |

```python
def load_expert_safe(path):
    df = pd.read_csv(path)
    X = df[["x", "y", "theta"]].to_numpy(dtype=np.float64)   # (N₁, 3)
    U = df[["v", "omega"]].to_numpy(dtype=np.float64)         # (N₁, 2)
    return X, U

def load_states_only(path):
    df = pd.read_csv(path)
    return df[["x", "y", "theta"]].to_numpy(dtype=np.float64) # (N, 3)
```

**Caveat:** in the paper, $\bar{X}_{\text{safe}}$ is *derived* from $Z_{\text{dyn}}$ by the eq. 9 distance filter — it is not independently sampled. The code consumes a separate `--safe` CSV for convenience. For strict paper reproduction, $\bar{X}_{\text{safe}}$ should be filtered from the expert set.

---

## 4. Formulation A — Hard-constraint convex programme

### Paper (eq. 11)

$$\min_{h \in \mathcal{H}} \;\|h\| \qquad \text{subject to:}$$

$$\text{(C1)} \quad h(x_i) \geq \gamma_{\text{safe}} \qquad \forall\, x_i \in \bar{X}_{\text{safe}}$$

$$\text{(C2)} \quad h(x_i) \leq -\gamma_{\text{unsafe}} \qquad \forall\, x_i \in X_N$$

$$\text{(C3)} \quad \underbrace{\langle \nabla h(x_i),\, f(x_i) + g(x_i)\,u_i \rangle}_{\text{Lie derivative}} + \;\alpha(h(x_i)) \;\geq\; \gamma_{\text{dyn}} \qquad \forall\, (x_i, u_i) \in Z_{\text{dyn}}$$

(The paper also lists Lipschitz constraints (11a)/(11b) but explicitly prescribes omitting them from the solve and verifying post-hoc by bootstrapping.)

With the RFF parameterisation $h(x) = \varphi(x)^\top\theta$ and $\alpha(h) = h$ (linear class-$\mathcal{K}$), every term becomes affine in $\theta$:

| Paper quantity | Parameterised form |
|---|---|
| $h(x_i)$ | $\varphi(x_i)^\top \theta$ |
| $\nabla h(x_i)$ | $D\varphi(x_i)^\top \theta$ |
| $\langle \nabla h(x_i), \dot{x}_i \rangle$ | $\theta^\top D\varphi(x_i) \dot{x}_i$ |
| $\alpha(h(x_i)) = h(x_i)$ | $\varphi(x_i)^\top \theta$ |
| $\|h\|_{\mathcal{H}}$ | $\|\theta\|_2$ |

This makes the problem a Second-Order Cone Programme (SOCP), solvable to global optimality.

### Implementation — `learn_cbf_rff()`

#### Step 1: Build the RFF basis **[PAPER]**

```python
W_rff, b_rff = build_rff(n_features, sigma, seed)
```

Freezes $(W, b)$. All subsequent quantities are functions of data and these frozen weights.

#### Step 2: Precompute feature matrices **[PAPER]**

```python
Phi_safe     = phi(x_safe, W_rff, b_rff)      # Φ_expert  ∈ ℝ^{N₁ × ℓ}
Phi_unsafe   = phi(x_unsafe, W_rff, b_rff)     # Φ_unsafe  ∈ ℝ^{N₂ × ℓ}
Phi_interior = phi(x_interior, W_rff, b_rff)   # Φ_safe    ∈ ℝ^{N₃ × ℓ}
```

Row $i$ of each matrix is $\varphi(x_i)^\top$, so `Phi @ theta` evaluates $h$ at every point in the set simultaneously.

#### Step 3: Precompute the Lie-derivative matrix **[PAPER]**

This is the most involved step. The goal is to express the Lie derivative $\langle \nabla h(x_i), \dot{x}_i \rangle$ as a matrix-vector product $A_{\text{dyn}} \,\theta$, keeping everything affine in $\theta$.

**3a.** Compute $\dot{x}_i = f(x_i) + g(x_i)\,u_i$ at each expert point:

```python
f_vals = dynamics_f(x_safe)                                # (N₁, 3)  — zeros
G_vals = dynamics_g(x_safe)                                # (N₁, 3, 2)
xdot   = f_vals + np.einsum("ijk,ik->ij", G_vals, u_safe) # (N₁, 3)
```

The einsum contracts $g(x_i) \in \mathbb{R}^{3 \times 2}$ with $u_i \in \mathbb{R}^2$ to produce $\dot{x}_i \in \mathbb{R}^3$. Since $f = 0$, this is just $g(x_i)\,u_i$.

**3b.** Compute the RFF Jacobian at each expert point:

```python
J_safe = dphi(x_safe, W_rff, b_rff)                       # (N₁, ℓ, 3)
```

`J_safe[i]` $= D\varphi(x_i) \in \mathbb{R}^{\ell \times 3}$, where row $j$ is $\frac{\partial \varphi_j}{\partial x}(x_i)$.

**3c.** Contract the Jacobian with $\dot{x}$ to get the Lie-derivative coefficients:

```python
A_dyn = np.einsum("ilk,ik->il", J_safe, xdot)             # (N₁, ℓ)
```

This computes $A_{\text{dyn}}[i, :] = D\varphi(x_i)\,\dot{x}_i \in \mathbb{R}^\ell$, so that:

$$A_{\text{dyn}}[i, :]\,\theta \;=\; \big(D\varphi(x_i)\,\dot{x}_i\big)^\top \theta \;=\; \dot{x}_i^\top\, D\varphi(x_i)^\top\, \theta \;=\; \langle \nabla h(x_i),\, \dot{x}_i \rangle$$

This identity is exact — no approximation. The Lie derivative of $h$ along the dynamics at $x_i$ is a linear function of $\theta$.

#### Step 4: Define the CVXPY constraints

```python
theta = cp.Variable(n_features, name="theta")
```

**Constraint C1 — safe value** **[PAPER]**

Paper: $h(x_i) \geq \gamma_{\text{safe}}$ for $x_i \in \bar{X}_{\text{safe}}$

```python
Phi_interior @ theta >= gamma_safe
```

`Phi_interior @ theta` evaluates $h$ at every deep-safe point. This ensures the safe set $\mathcal{C} = \{h \geq 0\}$ has non-empty interior (Corollary III.2).

**Constraint C2 — unsafe value** **[PAPER]**

Paper: $h(x_i) \leq -\gamma_{\text{unsafe}}$ for $x_i \in X_N$

```python
Phi_unsafe @ theta <= -gamma_unsafe
```

Forces $h < 0$ on the ring $\mathcal{N}$. Combined with C1, this pins the zero level set between the expert tube and the ring, guaranteeing $\mathcal{C} \subset D \subseteq \mathcal{S}$ (Proposition III.1).

**Constraint C3 — CBF derivative** **[PAPER]** (given $\alpha(h) = h$)

Paper: $\langle \nabla h(x_i), f(x_i) + g(x_i)u_i \rangle + \alpha(h(x_i)) \geq \gamma_{\text{dyn}}$

```python
(A_dyn + Phi_safe) @ theta >= gamma_dyn
```

`A_dyn @ theta` $= \langle \nabla h(x_i), \dot{x}_i \rangle$ (the Lie derivative) and `Phi_safe @ theta` $= h(x_i) = \alpha(h(x_i))$. The second equality **only holds because $\alpha$ is the identity** — this is the load-bearing assumption that keeps the constraint affine in $\theta$ (see §6).

This is the constraint that turns a classifier into a CBF: it ensures that when $h$ is near zero (near the boundary), the dynamics push $h$ back positive — the system is steered inward. This makes $\{h \geq 0\}$ forward invariant (Proposition III.3).

**Constraint C4 — expert lower bound** **[NOTEBOOK]**

```python
Phi_safe @ theta >= gamma_safe / 10.0
```

A secondary, weaker lower bound on the expert points ($h(x_i) \geq \gamma_{\text{safe}} / 10$ for $x_i \in Z_{\text{dyn}}$). This exists in the airplane notebook's loss function (`safe_value/10` term) but **is not in paper eq. 11**. It prevents $h$ from dipping low on expert points that are close to the boundary but not quite in the unsafe ring. **Delete this line for a strict paper reproduction.**

#### Step 5: Objective **[PAPER]**

Paper: $\min_{h \in \mathcal{H}} \|h\|$

```python
problem = cp.Problem(cp.Minimize(cp.norm(theta, 2)), constraints)
```

$\|h\|_{\mathcal{H}} = \|\theta\|_2$ for RFF (the RKHS norm equals the Euclidean norm of the coefficient vector). Minimising this picks the smoothest $h$ consistent with the constraints, directly reducing the Lipschitz constants $L_h$ and $L_q$ that appear in Theorem III.4.

#### Step 6: Solve **[PAPER]**

```python
problem.solve(solver=solver, verbose=verbose)
```

The entire problem is an SOCP (norm objective + linear constraints). Default solver is SCS; the paper's planar example uses MOSEK.

#### Complete constraint summary

| # | Paper equation | Code | Shape check |
|---|---|---|---|
| C1 | $h(x_i) \geq \gamma_{\text{safe}}$, $x_i \in \bar{X}_{\text{safe}}$ | `Phi_interior @ theta >= gamma_safe` | $(N_3, \ell) \cdot (\ell,) \geq$ scalar → $N_3$ constraints |
| C2 | $h(x_i) \leq -\gamma_{\text{unsafe}}$, $x_i \in X_N$ | `Phi_unsafe @ theta <= -gamma_unsafe` | $(N_2, \ell) \cdot (\ell,) \leq$ scalar → $N_2$ constraints |
| C3 | $\langle \nabla h, \dot{x}_i \rangle + h(x_i) \geq \gamma_{\text{dyn}}$ | `(A_dyn + Phi_safe) @ theta >= gamma_dyn` | $(N_1, \ell) \cdot (\ell,) \geq$ scalar → $N_1$ constraints |
| C4 | *(not in paper)* | `Phi_safe @ theta >= gamma_safe / 10` | $(N_1, \ell) \cdot (\ell,) \geq$ scalar → $N_1$ constraints |

---

## 5. Formulation B — Unconstrained hinge-loss relaxation

### Paper (eq. 12)

$$\min_{\theta \in \Theta} \;\|\theta\|^2 \;+\; \lambda_s \!\sum_{x_i \in \bar{X}_{\text{safe}}} [\gamma_{\text{safe}} - h_\theta(x_i)]_+ \;+\; \lambda_u \!\sum_{x_i \in X_N} [h_\theta(x_i) + \gamma_{\text{unsafe}}]_+ \;+\; \lambda_d \!\sum_{(x_i,u_i) \in Z_{\text{dyn}}} [\gamma_{\text{dyn}} - q_\theta(x_i, u_i)]_+$$

where $[z]_+ = \max(z, 0)$ is the hinge function and:

$$q_\theta(x_i, u_i) = \langle \nabla h_\theta(x_i),\, f(x_i) + g(x_i)\,u_i \rangle + \alpha(h_\theta(x_i))$$

Each hard constraint from eq. 11 is replaced by a penalty that is zero when the constraint is satisfied and linear in the violation magnitude when it is not.

### Implementation — `learn_cbf_rff_soft()`

Steps 1–3 (RFF basis, feature matrices, Lie-derivative matrix) are identical to Formulation A. The difference is entirely in the objective.

#### Term 1: Parameter regularisation

Paper: $\|\theta\|^2$

```python
param_reg = cp.sum_squares(theta)                     # ||θ||²₂
```

**[PAPER]** form / **[NOTEBOOK]** weight. The paper writes this with implicit coefficient 1. The code multiplies by `lam_param` (default `0.1`, from the **[NOTEBOOK]** airplane training loop). Set `lam_param = 1.0` for the literal paper equation.

#### Term 2: Safe hinge

Paper: $\lambda_s \sum_{x_i \in \bar{X}_{\text{safe}}} [\gamma_{\text{safe}} - h_\theta(x_i)]_+$

```python
safe_hinge = cp.sum(cp.pos(gamma_safe - Phi_interior @ theta))
```

**[PAPER]**. `cp.pos(z)` $= \max(z, 0) = [z]_+$. Penalises any deep-safe point where $h < \gamma_{\text{safe}}$ — the penalty is proportional to the shortfall.

#### Term 3: Unsafe hinge

Paper: $\lambda_u \sum_{x_i \in X_N} [h_\theta(x_i) + \gamma_{\text{unsafe}}]_+$

```python
unsafe_hinge = cp.sum(cp.pos(Phi_unsafe @ theta + gamma_unsafe))
```

**[PAPER]**. Penalises any ring point where $h > -\gamma_{\text{unsafe}}$ (i.e. $h$ is not sufficiently negative).

#### Term 4: Derivative hinge

Paper: $\lambda_d \sum_{(x_i,u_i) \in Z_{\text{dyn}}} [\gamma_{\text{dyn}} - q_\theta(x_i, u_i)]_+$

```python
dyn_hinge = cp.sum(cp.pos(gamma_dyn - (A_dyn + Phi_safe) @ theta))
```

**[PAPER]** (given $\alpha(h) = h$). Same `(A_dyn + Phi_safe) @ theta` construction as Formulation A. Penalises any expert point where the CBF derivative condition is violated.

#### Combined objective

$$\mathcal{L}(\theta) = \lambda_p\|\theta\|^2 + \lambda_s \cdot \text{safe\_hinge} + \lambda_u \cdot \text{unsafe\_hinge} + \lambda_d \cdot \text{dyn\_hinge}$$

```python
objective = cp.Minimize(
    lam_param  * param_reg
    + lam_safe   * safe_hinge
    + lam_unsafe * unsafe_hinge
    + lam_dyn    * dyn_hinge
)
problem = cp.Problem(objective)
problem.solve(solver=solver, verbose=verbose)
```

**Convexity note:** with $\alpha(h) = h$ and RFF, every hinge term is convex (piecewise-linear composed with affine) and $\|\theta\|^2$ is convex, so the full objective is convex. Unlike eq. 11, there are no constraints — this is an unconstrained convex program. With a DNN parameterisation or cubic $\alpha$, the problem becomes non-convex and would use Adam/SGD instead of CVXPY.

#### Default multipliers **[NOTEBOOK]**

| Code parameter | Paper symbol | Default | Source |
|---|---|---|---|
| `lam_param` | (implicit 1 in paper) | `0.1` | **[NOTEBOOK]** airplane training |
| `lam_safe` | $\lambda_s$ | `2.0` | **[NOTEBOOK]** airplane training |
| `lam_unsafe` | $\lambda_u$ | `2.0` | **[NOTEBOOK]** airplane training |
| `lam_dyn` | $\lambda_d$ | `15.0` | **[NOTEBOOK]** airplane training |

The paper leaves all multipliers unspecified ("allow us to trade off the relative importance"). The paper's planar example does not even use eq. 12 — it uses the convex eq. 11 with MOSEK.

#### Default margins

| Code parameter | Paper symbol | Default | Source |
|---|---|---|---|
| `gamma_safe` | $\gamma_{\text{safe}}$ | `1.5` | **[NOTEBOOK]** airplane example |
| `gamma_unsafe` | $\gamma_{\text{unsafe}}$ | `0.5` | **[NOTEBOOK]** airplane example |
| `gamma_dyn` | $\gamma_{\text{dyn}}$ | `0.05` | **[NOTEBOOK]** airplane example |

The paper's planar example uses $\gamma_{\text{safe}} = 0.1$, $\gamma_{\text{unsafe}} = 0.3$, $\gamma_{\text{dyn}} = 0.01$.

---

## 6. The α assumption

### Paper (§II-A, eq. 3)

The CBF validity condition requires an extended class-$\mathcal{K}$ function $\alpha : \mathbb{R} \to \mathbb{R}$ (strictly increasing, $\alpha(0) = 0$) such that:

$$\sup_{u \in \mathcal{U}} \langle \nabla h(x),\, f(x) + g(x)\,u \rangle \;\geq\; -\alpha(h(x))$$

### Paper (§III-C, Convexity paragraph)

The paper states: eq. 11 is convex **if $\alpha$ is linear in its argument**. In the planar example, $\alpha(x) = x$.

### Implementation **[PAPER]**

The code hard-codes $\alpha(h) = h$. This is encoded implicitly — there is no `alpha` function. Instead, wherever $\alpha(h(x_i))$ appears, it is replaced by $h(x_i) = \varphi(x_i)^\top \theta$:

```python
# In the derivative constraint:
(A_dyn + Phi_safe) @ theta >= gamma_dyn
#  ↑                ↑
#  |                └── Φ_safe @ θ = h(xᵢ) = α(h(xᵢ))  ← only valid when α = identity
#  └──────────────────── A_dyn @ θ = ⟨∇h(xᵢ), ẋᵢ⟩
```

With linear $\alpha$, the constraint is affine in $\theta$. With $\alpha(h) = h^3$ (the **[NOTEBOOK]** airplane choice), $\alpha(h) = (\varphi^\top\theta)^3$ is cubic in $\theta$ — non-convex, and unusable as a CVXPY hard constraint. This is a paper-level theoretical limitation, not a code limitation.

**Summary:** the convex path (CVXPY) requires linear $\alpha$. For non-linear $\alpha$, the only option is the gradient-based non-convex relaxation (eq. 12 via Adam/SGD with a DNN, as in the notebook).

---

## 7. Evaluation and verification

### Evaluating $h(x)$ and $\nabla h(x)$

```python
def evaluate_h(x, theta, W, b):
    """h(x) = φ(x)ᵀθ"""
    return phi(x, W, b) @ theta                           # (N,)

def evaluate_grad_h(x, theta, W, b):
    """∇h(x) = Dφ(x)ᵀθ"""
    J = dphi(x, W, b)                                     # (N, ℓ, n)
    return np.einsum("ilk,l->ik", J, theta)                # (N, n)
```

These directly implement the parameterised forms from §2.

### Post-hoc verification — `verify_conditions()`

**[ADDED]** / partial **[PAPER]**. The function checks the raw constraint slacks at each data point:

| Check | Computation | What it verifies |
|---|---|---|
| Safe slack | $h(x_i) - \gamma_{\text{safe}}$ for $x_i \in \bar{X}_{\text{safe}}$ | Should be $\geq 0$ at all points |
| Unsafe slack | $-h(x_i) - \gamma_{\text{unsafe}}$ for $x_i \in X_N$ | Should be $\geq 0$ at all points |
| Derivative slack | $q(x_i, u_i) - \gamma_{\text{dyn}}$ for $(x_i, u_i) \in Z_{\text{dyn}}$ | Should be $\geq 0$ at all points |

where $q(x_i, u_i) = \langle \nabla h(x_i), \dot{x}_i \rangle + h(x_i)$ is computed as:

```python
grad_h    = evaluate_grad_h(x_safe, theta, W, b)          # (N₁, 3)
xdot      = f_vals + np.einsum("ijk,ik->ij", G_vals, u_safe)  # (N₁, 3)
lie_deriv = np.sum(grad_h * xdot, axis=1)                 # (N₁,)   ← ⟨∇h, ẋ⟩
q_vals    = lie_deriv + h_safe_exp                         # (N₁,)   ← q = Lie + α(h)
```

**Gap vs. paper:** the paper's Theorem III.4 verification (Fig. 3) is stronger. It folds the estimated Lipschitz constant into each slack:

- Safe (Fig. 3a): checks $h(x_i) - L_h(x_i) \cdot \varepsilon \geq 0$
- Derivative (Fig. 3b): checks $q(x_i, u_i) - L_q(x_i) \cdot \varepsilon \geq 0$
- Unsafe (Fig. 3c): checks $h(x_i) + L_h(x_i) \cdot \bar{\varepsilon} < 0$

The paper provides the closed-form RFF Lipschitz bound:

$$L_h \;\leq\; \sqrt{2}\,\sigma^2 \bigg(1 + \frac{n}{\ell} + \sqrt{\frac{2}{\ell}\log\frac{1}{\delta}}\bigg) \|\theta\|_2$$

The current implementation checks raw slacks (necessary but weaker). A full Theorem III.4 reproduction requires adding the Lipschitz-adjusted slack computation.

---

## 8. Provenance summary

| Element | Source | For strict paper reproduction |
|---|---|---|
| Objective $\|\theta\|_2$ as RKHS norm | **[PAPER]** | Keep |
| Constraint C1: $h \geq \gamma_{\text{safe}}$ on $\bar{X}_{\text{safe}}$ | **[PAPER]** | Keep |
| Constraint C2: $h \leq -\gamma_{\text{unsafe}}$ on $X_N$ | **[PAPER]** | Keep |
| Constraint C3: Lie derivative + $\alpha(h) \geq \gamma_{\text{dyn}}$ | **[PAPER]** | Keep |
| Constraint C4: $h \geq \gamma_{\text{safe}}/10$ on expert | **[NOTEBOOK]** | **Delete** |
| RFF map $\varphi$, Jacobian $D\varphi$, `build_rff` | **[PAPER]** | Keep |
| $\ell = 200$, $\sigma = 1.2$ | **[PAPER]** (planar) | Keep if reproducing planar |
| $\alpha(h) = h$ (linear) | **[PAPER]** (planar) | Keep for convex path |
| Lipschitz bounds omitted from solve | **[PAPER]** (bootstrapping) | Keep; verify post-hoc |
| $f(x) = 0$ (zero drift) | **[NOTEBOOK]** | Keep (true for unicycle) |
| Hinge relaxation structure (eq. 12) | **[PAPER]** | Keep |
| $\lambda$ weights (2, 2, 15, 0.1) | **[NOTEBOOK]** | Re-tune per problem |
| $\gamma$ margins (1.5, 0.5, 0.05) | **[NOTEBOOK]** | Use (0.1, 0.3, 0.01) for planar |
| Three-CSV / argparse pipeline | **[ADDED]** | Optional |
| $\bar{X}_{\text{safe}}$ from separate CSV | **[ADDED]** | Derive via eq. 9 filter |
| Raw-slack verification only | **[ADDED]** | Add Lipschitz-adjusted slacks |
| `geometric_safety_value()` placeholder | **[ADDED]** | Fill in for post-hoc check |