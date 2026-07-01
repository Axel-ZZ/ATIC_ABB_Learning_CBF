"""
Load the environment + dynamics (+ CBF safety filter), run the interactive
simulator, and plot the resulting trajectory / CBF values.

NOTE on the CBF: the actual CBF-QP safety filter (`safety_filters/`) and the
learners' load/inference glue (models/sos_cbf.py, models/train_nn_cbf.py)
aren't wired into this branch yet (see info.md, "Known gaps"). `Simulator`
only needs a `cbf` object exposing `.value(state)` (and ideally
`.gradient(state)`) — see simulator/interactive_sim.py's module docstring
for the exact interface. Until that adapter exists, this runs with
`PassthroughCBF()`, which never intervenes, so the rest of the pipeline
(controls, collision/goal detection, plotting, saving) can be exercised now.

Swap in the real filter once it's ready, e.g.:

    from safety_filters.cbf_qp import load_trained_cbf
    cbf = load_trained_cbf("models/maze_cbf_filter.pt")
"""
from environments.environment import build_env
from environments.vehicle_dynamics import RobotModel, DiffDriveKinematics
from sim.qp_filter import QPFilter
from sim.simulator import Simulator2D
from sim.load_CBF import NNCBF

import numpy as np
import matplotlib.pyplot as plt
from environments.environment import plot_environment

def main() -> None:
    environment = build_env("maze")
    model = DiffDriveKinematics(RobotModel(wheel_radius=0.005, wheel_base=0.1))
    
    cbf = NNCBF("runs/models/sweep_maze_separation/ls20_lu20/model.eqx")
    qp_filter = QPFilter(cbf=cbf, model=model, alpha=1.0, u_min=[-1.0, -1.0], u_max=[1.0, 1.0], slack_penalty=1e5)

    sim = Simulator2D(environment, model, qp_filter)
    sim.run()


if __name__ == "__main__":
    main()
