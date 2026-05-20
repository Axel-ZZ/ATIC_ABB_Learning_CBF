# Trajectory Optimization Solver for Differential-Drive Robot

## Problem Formulation

### Decision Variables
- **State trajectory**: $\mathbf{X} \in \mathbb{R}^{3 \times (N+1)}$
  - $\mathbf{X}_{:,k} = [x_k, y_k, \theta_k]^\top$ for $k = 0, \ldots, N$
  - $x_k, y_k$: position in $\mathbb{R}^2$
  - $\theta_k$: heading angle in $\mathbb{R}$

- **Control trajectory**: $\mathbf{U} \in \mathbb{R}^{2 \times N}$
  - $\mathbf{U}_{:,k} = [v_k, \omega_k]^\top$ for $k = 0, \ldots, N-1$
  - $v_k$: linear velocity (m/s)
  - $\omega_k$: angular velocity (rad/s)

### Dynamics Constraints

**Discrete-time dynamics** (RK4 integration):

$$\mathbf{X}_{:,k+1} = f_{\text{RK4}}(\mathbf{X}_{:,k}, \mathbf{U}_{:,k}, \Delta t) \quad \forall k = 0, \ldots, N-1$$

where $f_{\text{RK4}}$ integrates the continuous differential-drive kinematics:

$$
\begin{align}
\dot{x} &= v \cos(\theta) \\
\dot{y} &= v \sin(\theta) \\
\dot{\theta} &= \omega
\end{align}
$$

over timestep $\Delta t$ using the fourth-order Runge-Kutta method.

### Boundary Conditions

**Initial state** (hard constraint):
$$\mathbf{X}_{:,0} = \mathbf{x}_{\text{start}}$$

**Terminal position** (hard constraint):
$$x_N = x_{\text{goal}}, \quad y_N = y_{\text{goal}}$$

**Terminal heading** (soft, in objective):
$$\text{heading error} = \sin\left(\frac{\theta_N - \theta_{\text{goal}}}{2}\right)$$

### Control Bounds

Box constraints on control inputs:
$$
\begin{align}
v_{\min} &\leq v_k \leq v_{\max} \quad \forall k \\
\omega_{\min} &\leq \omega_k \leq \omega_{\max} \quad \forall k
\end{align}
$$

Typical values: $v \in [-0.2, 0.5]$ m/s, $\omega \in [-2.0, 2.0]$ rad/s.

### Obstacle Avoidance Constraints

For each circular obstacle $i$ with center $(c_x^{(i)}, c_y^{(i)})$ and radius $r^{(i)}$:

$$
(x_k - c_x^{(i)})^2 + (y_k - c_y^{(i)})^2 \geq (r^{(i)} + d_{\text{clear}})^2 \quad \forall k, \forall i
$$

where $d_{\text{clear}}$ is the minimum clearance margin.

### Objective Function

Minimize the weighted sum:

$$
J = w_{\text{effort}} \cdot J_{\text{effort}} + w_{\text{smooth}} \cdot J_{\text{smooth}} + w_{\text{heading}} \cdot J_{\text{heading}} + w_{\text{via}} \cdot J_{\text{via}}
$$

**Control effort**:
$$J_{\text{effort}} = \sum_{k=0}^{N-1} \left(v_k^2 + \omega_k^2\right)$$

**Control smoothness** (penalizes jerk):
$$J_{\text{smooth}} = \sum_{k=0}^{N-2} \left[(v_{k+1} - v_k)^2 + (\omega_{k+1} - \omega_k)^2\right]$$

**Terminal heading error**:
$$J_{\text{heading}} = \sin^2\left(\frac{\theta_N - \theta_{\text{goal}}}{2}\right)$$

**Via-point anchor** (for dense trajectory generation):
$$J_{\text{via}} = (x_{k_{\text{mid}}} - v_x)^2 + (y_{k_{\text{mid}}} - v_y)^2$$

where $(v_x, v_y)$ is the target via-point at polar coordinates $(R, \phi)$ around the obstacle center, and $k_{\text{mid}} = \lfloor N/2 \rfloor$.

### Weight Parameters

Default values:
- $w_{\text{effort}} \in [1, 5]$: controls preference for low-energy paths
- $w_{\text{smooth}} \in [2, 20]$: controls preference for smooth control profiles
- $w_{\text{heading}} = 50$: enforces terminal heading alignment
- $w_{\text{via}} = 200$: anchors trajectory to pass through designated via-point (used for dense coverage)

## Solution Method

**Solver**: IPOPT (Interior Point OPTimizer) via CasADi

**Convergence**:
- Maximum iterations: 300
- Tolerance: $10^{-4}$
- Warm start: initialized with piecewise-linear path through via-point

**Output**: Locally optimal trajectory $(\mathbf{X}^*, \mathbf{U}^*)$ satisfying all constraints and minimizing $J$.

## Dense Trajectory Generation Strategy

To generate a corpus of expert trajectories with dense coverage around a single obstacle:

1. **Polar via-point sampling**: For obstacle at $(c_x, c_y)$ with radius $r$:
   - Sample angles: $\phi \in \{0, \frac{2\pi}{n_{\text{angles}}}, \ldots, 2\pi - \frac{2\pi}{n_{\text{angles}}}\}$
   - Sample radii: $R \in \{r + \delta_1, r + \delta_2, \ldots, r + \delta_m\}$
   - Each $(R, \phi)$ pair defines via-point: $(v_x, v_y) = (c_x + R\cos\phi, c_y + R\sin\phi)$

2. **Per via-point optimization**: Solve the above NLP with via-point anchor active.

3. **Deduplication**: Compute discrete Fréchet distance between all trajectory pairs; reject duplicates below threshold $\tau_{\text{dedup}}$.

4. **Feasibility filtering**: Reject trajectories where arc length exceeds $\alpha \cdot \|\mathbf{x}_{\text{goal}} - \mathbf{x}_{\text{start}}\|_2$ (e.g., $\alpha = 2.5$).

**Result**: Dense angular and radial coverage of dynamically feasible, locally optimal trajectories around the obstacle.