#Loads model, enviroment, runs interactive sim, displays plots.
from safety_filters.neural_cbf import NeuralCBF
from environments.dynamics_models import Unicycle

from environments.maze import MazeEnv
from environments.single_obstacle import SingleObstacleEnv

from sim.interactive_sim import simulator as sim

if MazeEnv == "__main__":
    #Load model and enviroment
    system = Unicycle()
    env = MazeEnv()
    cbf = NeuralCBF.load("models/maze_cbf_filter.pt")
    env = SingleObstacleEnv()

    #Run interactive sim
    obs_traj, action_traj, cbf_traj = sim.simulate(system, cbf, env, render=True)

    #Display plots
    env.plot_trajectories(obs_traj, action_traj, cbf_traj)



