import sys

import numpy as np

from dronesim import constants as conf
from dronesim.controllers.controller import Controller
from dronesim.models.quadrotor import Quadrotor
from dronesim.planning.reference import get_offseted_references, waypoints2timebased
from dronesim.utils import detect_collisions
from dronesim.visualization import (
    plot_position_error,
    plot_trajectories,
    plot_velocity_error,
    generate_animation,
)

np.set_printoptions(
    threshold=sys.maxsize,
    precision=3,
    suppress=True,
)
n_drones = 1
points = waypoints2timebased(conf.waypoints, steps=conf.T)
references = get_offseted_references(points, n_drones, np.array([0, 0, 0]))
trajectories = np.zeros((n_drones, conf.T, conf.Nx))

drones = [Quadrotor(*references[i, 0, :3]) for i in range(n_drones)]
controllers = [Controller(drones[i], conf.obstacles) for i in range(n_drones)]

t_collision = conf.T

for t in range(conf.T):
    for i in range(n_drones):
        controllers[i].step(references[i, t, :])
        trajectories[i, t, :] = np.array(drones[i].state())

    # collisions check
    if detect_collisions(drones, conf.obstacles):
        t_collision = t
        break

for d in range(n_drones):
    plot_position_error(
        references[d],
        trajectories[d, :t_collision],
        filename=f"position_error_drone_{d}",
    )
    plot_velocity_error(
        references[d],
        trajectories[d, :t_collision],
        filename=f"velocity_error_drone_{d}",
    )

plot_trajectories(
    references,
    trajectories[:, :t_collision, :],
    obstacles=conf.obstacles,
    drone_radius=conf.l,
    show=False,
    pause=10,
)
generate_animation(
    trajectories[:, :t_collision, :],
    conf.obstacles,
    drone_radius=conf.l,
    skip_frames=20,
    title="Spring Model Simulation",
)