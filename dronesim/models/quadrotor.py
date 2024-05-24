from math import cos, sin, tan

import numpy as np

from dronesim import constants as conf


class Quadrotor:
    """Dynamical Model of a Quadrotor. This is a simplified model
    of a quadrotor. Taken from:

    - https://folk.ntnu.no/skoge/prost/proceedings/cdc-ecc-2011/data/papers/1143.pdf
    - https://wilselby.com/research/arducopter/modeling/
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        """Initialize the drone and its paramters

        Parameters
        ----------
        x : float
            initial postiion x-axis
        y : float
            initial position y-axis
        z : float
            initial position z-axis
        """

        # ------------------------------
        # the drone state $q \in \mathbb{R}^12$ contains
        # all the information about the drone
        # ------------------------------
        self.x = x  # x position
        self.y = y  # y position
        self.z = z  # z position
        self.dx = 0.0  # linear velocities x-axis
        self.dy = 0.0  # linear velocities y-axis
        self.dz = 0.0  # linear velocities z-axis
        self.phi = 0.0  # roll
        self.theta = 0.0  # pitch
        self.psi = 0.0  # yaw
        self.p = 0.0  # angular velocities (rotation on x-axis)
        self.q = 0.0  # angular velocities (rotation on y-axis)
        self.r = 0.0  # angular velocities (rotation on z-axis)

    def position(self):
        # return drone's position vector
        return np.array([self.x, self.y, self.z])

    def attitude(self):
        # returns drone's attitude vector
        return [self.phi, self.theta, self.psi]

    def state(self):
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.dx,
                self.dy,
                self.dz,
                self.phi,
                self.theta,
                self.psi,
                self.p,
                self.q,
                self.r,
            ]
        )

    def measure(self):
        """Return the measured state, which may be a noisy estimation
        of the true drone's state.

        Returns:
            A noisy drone's state.
        """
        # TODO: to be implemented
        pass

    def reset(self) -> None:
        # return self.__init__()
        pass

    def step(self, inputs) -> None:
        """Updates drone's internal state given the input commands

        Args:
            velocities: input velocities
        """
        # u1 is the total thrust force
        # u2 is \tau_x
        # u3 is \tau_y
        # u4 is \tau_z
        u1, u2, u3, u4 = inputs

        # T is the angular transformation matrix, used to translate
        # the angular velocities from body (B) frame to interial frame (I)
        #
        #                   \omega_I = T * \omega_B
        #
        # where \omega_I = [\dot \phi, \dot \theta, \dot \psi]^T
        # and   \omega_B = [p, q, r]^T
        #
        # linear and angular positions on inertial frame (I):
        #                   [x y z \phi \theta \psi]^T
        #
        # linear and angular velocities on body frame (B):
        #                   [u v w p q r]^T
        #
        # v_I = [\dot x, \dot y, \dot z]
        # v_B = [u, v, w]
        # ω_I = [\dot \phi, \dot \theta, \dot \psi]
        # ω_B = [p, q, r]
        # coordinates ω
        T = np.array(
            [
                [1, sin(self.phi) * tan(self.theta), cos(self.phi) * tan(self.theta)],
                [0, cos(self.phi), -sin(self.phi)],
                [0, sin(self.phi) / cos(self.theta), cos(self.phi) / cos(self.theta)],
            ]
        )

        # here \omega_I = T * \omega_B
        #               = T * [p q r]^T
        # singularity when theta=pi/2
        omega_i = T @ np.array([self.p, self.q, self.r])
        dphi, dtheta, dpsi = omega_i

        ddx = (
            (
                cos(self.phi) * sin(self.theta) * cos(self.psi)
                + sin(self.phi) * sin(self.psi)
            )
            * u1
            / conf.m
        )
        ddy = (
            (
                cos(self.phi) * sin(self.theta) * sin(self.psi)
                - sin(self.phi) * cos(self.psi)
            )
            * u1
            / conf.m
        )
        ddz = -conf.G + (cos(self.phi) * cos(self.theta)) * u1 / conf.m

        dp = (self.q * self.r * (conf.Iyy - conf.Izz) / conf.Ixx) + u2 / conf.Ixx
        dq = (self.p * self.r * (conf.Izz - conf.Ixx) / conf.Iyy) + u3 / conf.Iyy
        dr = (
            self.p * self.q * (conf.Ixx - conf.Iyy) / conf.Izz
        ) + u4 / conf.Izz * conf.d

        # update the state given the inputs
        self.x += conf.DT * self.dx
        self.y += conf.DT * self.dy
        self.z += conf.DT * self.dz
        self.dx += conf.DT * ddx
        self.dy += conf.DT * ddy
        self.dz += conf.DT * ddz
        self.phi += conf.DT * dphi
        self.theta += conf.DT * dtheta
        self.psi += conf.DT * dpsi
        self.p += conf.DT * dp
        self.q += conf.DT * dq
        self.r += conf.DT * dr

    def __str__(self):
        prec = 1
        repr = "[Drone]"
        repr += f"|x {self.x:.{prec}f}"
        repr += f"|y {self.y:.{prec}f}"
        repr += f"|z {self.z:.{prec}f}"
        repr += f"|ẋ {self.dx:.{prec}f}"
        repr += f"|ẏ {self.dy:.{prec}f}"
        repr += f"|ż {self.dz:.{prec}f}"
        repr += f"|φ {self.phi:.{prec}f}"
        repr += f"|θ {self.theta:.{prec}f}"
        repr += f"|ψ {self.psi:.{prec}f}"
        repr += f"|p {self.p:.{prec}f}"
        repr += f"|q {self.q:.{prec}f}"
        repr += f"|r {self.r:.{prec}f}"
        return repr


if __name__ == "__main__":
    d = Quadrotor(30.0, 45.0, 67.0)
    print(d)
