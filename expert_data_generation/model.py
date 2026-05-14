import casadi as ca
import numpy as np


class RobotModel:
    """Simple robot model holding physical parameters."""
    def __init__(self, wheel_radius: float, wheel_base: float):
        self.wheel_radius = wheel_radius  # r
        self.wheel_base = wheel_base      # L (distance between wheels)


class DiffDriveKinematics:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.n_states = 3   # x, y, theta
        self.n_controls = 2 # v, omega

    def continuous_dynamics(self) -> ca.Function:
        """
        Returns a CasADi Function: f(state, control) -> state_dot

        State:   [x, y, theta]
        Control: [v, omega]

        x_dot     = v * cos(theta)
        y_dot     = v * sin(theta)
        theta_dot = omega
        """
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        state = ca.vertcat(x, y, theta)

        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")
        control = ca.vertcat(v, omega)

        rhs = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega,
        )

        return ca.Function("f_continuous", [state, control], [rhs],
                           ["state", "control"], ["state_dot"])

    def discrete_dynamics(self, dt: float, method: str = "rk4") -> ca.Function:
        """
        Returns a CasADi Function: f(state, control, dt) -> state_next

        Integrates the continuous dynamics over one time step.
        Methods: 'euler' or 'rk4'.
        """
        state = ca.SX.sym("state", self.n_states)
        control = ca.SX.sym("control", self.n_controls)
        h = ca.SX.sym("dt")

        f_cont = self.continuous_dynamics()

        if method == "euler":
            state_next = state + h * f_cont(state, control)

        elif method == "rk4":
            k1 = f_cont(state, control)
            k2 = f_cont(state + h / 2 * k1, control)
            k3 = f_cont(state + h / 2 * k2, control)
            k4 = f_cont(state + h * k3, control)
            state_next = state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'euler' or 'rk4'.")

        return ca.Function("f_discrete", [state, control, h], [state_next],
                           ["state", "control", "dt"], ["state_next"])

    def wheel_velocities_to_body(self) -> ca.Function:
        """
        Converts wheel velocities [w_left, w_right] (rad/s)
        to body velocities [v, omega].

        v     = r/2 * (w_right + w_left)
        omega = r/L * (w_right - w_left)
        """
        r = self.robot.wheel_radius
        L = self.robot.wheel_base

        w_l = ca.SX.sym("w_left")
        w_r = ca.SX.sym("w_right")
        wheel_vel = ca.vertcat(w_l, w_r)

        v = (r / 2.0) * (w_r + w_l)
        omega = (r / L) * (w_r - w_l)
        body_vel = ca.vertcat(v, omega)

        return ca.Function("wheel_to_body", [wheel_vel], [body_vel],
                           ["wheel_vel"], ["body_vel"])


# ── Quick demo ──────────────────────────────────────────────
if __name__ == "__main__":
    robot = RobotModel(wheel_radius=0.05, wheel_base=0.3)
    kin = DiffDriveKinematics(robot)

    # Continuous
    f = kin.continuous_dynamics()
    print("Continuous dynamics at (0,0,0), v=1, omega=0.5:")
    print(f([0, 0, 0], [1.0, 0.5]))

    # Discrete (RK4)
    f_d = kin.discrete_dynamics(dt=0.1)
    state_next = f_d([0, 0, 0], [1.0, 0.5], 0.1)
    print("\nDiscrete (RK4, dt=0.1):")
    print(state_next)

    # Wheel velocities → body velocities
    w2b = kin.wheel_velocities_to_body()
    print("\nWheel [10, 12] rad/s → body vel:")
    print(w2b([10, 12]))