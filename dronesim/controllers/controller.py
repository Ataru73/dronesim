

import numpy as np
from dronesim import constants as conf
from dronesim.controllers.avoidance import obstacle_avoidance
from dronesim.controllers.position import PositionController
from dronesim.models.quadrotor import Quadrotor


class Controller:

    def __init__(self, drone: Quadrotor, obstacles: np.ndarray):

        self.drone = drone
        self.obstacles = obstacles
        self.position_controller = PositionController()

    def step(self, reference: np.ndarray, obstacles: np.ndarray):

        F_oa = obstacle_avoidance(self.drone.state(), obstacles, conf.l)
        tmp = reference.copy()
        tmp[:3] += F_oa
        _input = self.position_controller.step(self.drone.state(), tmp)
        self.drone.step(_input)
