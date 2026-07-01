"""
Differential-drive (unicycle) kinematics — plain NumPy.

State   x = [x, y, theta]
Control u = [v, omega]
    x_dot     = v cos(theta)
    y_dot     = v sin(theta)
    theta_dot = omega

This is the control-affine system of Robey et al. (2004.03315): f(x)=0,
g(x)=[[cos θ,0],[sin θ,0],[0,1]]. Only forward simulation is needed here
(the sim trajectory generator integrates one RK4 step at a time), so this
is pure NumPy — no CasADi / autodiff. The CBF learners in `models/`
re-declare g(x) in their own framework (NumPy for the convex solver, JAX
for the NN) because they differentiate through it.
"""
from __future__ import annotations

import numpy as np


class RobotModel:
    """Physical parameters of the differential-drive robot."""
    def __init__(self, wheel_radius: float, wheel_base: float):
        self.wheel_radius = wheel_radius   # r
        self.wheel_base = wheel_base       # L (distance between wheels)


def _state_dot(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Continuous unicycle dynamics  x_dot = f(x) + g(x) u  (f = 0)."""
    theta = state[2]
    v, omega = control[0], control[1]
    return np.array([v * np.cos(theta), v * np.sin(theta), omega])


class DiffDriveKinematics:
    """Unicycle kinematics with continuous and RK4/Euler discrete steppers."""

    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.n_states = 3
        self.n_controls = 2

    def continuous_dynamics(self):
        """Return f(state, control) -> state_dot."""
        def f(state, control):
            return _state_dot(np.asarray(state, float), np.asarray(control, float))
        return f

    def discrete_dynamics(self, dt: float, method: str = "rk4"):
        """
        Return a one-step integrator  step(state, control, h=dt) -> state_next.
        (Signature mirrors the old CasADi Function: callable as f(s, u, dt).)
        """
        if method not in ("rk4", "euler"):
            raise ValueError(f"Unknown method '{method}'. Use 'rk4' or 'euler'.")

        def step(state, control, h: float = dt):
            s = np.asarray(state, float)
            u = np.asarray(control, float)
            if method == "euler":
                return s + h * _state_dot(s, u)
            k1 = _state_dot(s, u)
            k2 = _state_dot(s + 0.5 * h * k1, u)
            k3 = _state_dot(s + 0.5 * h * k2, u)
            k4 = _state_dot(s + h * k3, u)
            return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return step

    def control_matrix(self, state):
        theta = state[2]
        return np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])


# ── Quick demo ──────────────────────────────────────────────
if __name__ == "__main__":
    kin = DiffDriveKinematics(RobotModel(wheel_radius=0.05, wheel_base=0.3))

    f = kin.continuous_dynamics()
    print("Continuous at (0,0,0), v=1, omega=0.5:", f([0, 0, 0], [1.0, 0.5]))

    f_d = kin.discrete_dynamics(dt=0.1, method="rk4")
    print("Discrete RK4 (dt=0.1):", f_d([0, 0, 0], [1.0, 0.5], 0.1))
