import numpy as np
from dronesim import constants as conf


def xyz_to_pitch_roll_yaw(thrust_direction_xyz):
    # Normalize the thrust direction vector
    thrust_direction_xyz = np.array(thrust_direction_xyz) / np.linalg.norm(
        thrust_direction_xyz
    )

    # Calculate roll and pitch angles
    roll = np.arctan2(thrust_direction_xyz[0], thrust_direction_xyz[2])
    pitch = -np.arcsin(thrust_direction_xyz[1])

    # Calculate yaw angle (since thrust is upwards, yaw can be set to 0)
    yaw = 0.0

    return pitch, roll, yaw


def pitch_roll_yaw_to_xyz(pitch, roll, yaw):
    # Calculate the rotation matrix
    rotation_matrix = np.array(
        [
            [
                np.cos(yaw) * np.cos(roll),
                np.cos(yaw) * np.sin(roll) * np.sin(pitch)
                - np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.sin(roll) * np.cos(pitch)
                + np.sin(yaw) * np.sin(pitch),
            ],
            [
                np.sin(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(roll) * np.sin(pitch)
                + np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.sin(roll) * np.cos(pitch)
                - np.cos(yaw) * np.sin(pitch),
            ],
            [-np.sin(roll), np.cos(roll) * np.sin(pitch), np.cos(roll) * np.cos(pitch)],
        ]
    )

    # Thrust direction in XYZ coordinates
    thrust_direction_xyz = np.dot(rotation_matrix, np.array([0, 0, 1]))

    return thrust_direction_xyz


class NonlinearController:
    def __init__(self) -> None:
        coeff=1.0

        # acceleration PD gains
        self.kphi1 = 0.1
        self.ktheta1 = 0.1
        self.kpsi1 = 0.1
        self.kphi2 = 20
        self.ktheta2 = 20
        self.kpsi2 = 20

        self.prev_phi_d = 0
        self.prev_theta_d = 0
        self.prev_psi_d = 0.0
        self.prev_T_d = 0

        # position PD gains
        self.kx1 = 10*coeff
        self.ky1 = 10*coeff
        self.kz1 = 10*coeff
        self.kx2 = 5
        self.ky2 = 5
        self.kz2 = 5

        self.prev_x_d = 0
        self.prev_y_d = 0
        self.prev_z_d = 0
        self.prev_phi = 0
        self.prev_theta = 0
        self.prev_psi = 0

        # saturation on commanded acceleration
        self.saturation = 8

    def accelerationPDaction(self, state, reference):
        ax_d, ay_d, az_d = reference

        # update the state given the inputs
        x, y, z = state[:3]
        dx, dy, dz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        # find the T_d vector
        Tx_d = conf.m * ax_d
        Ty_d = conf.m * ay_d
        Tz_d = conf.m * az_d + conf.m * conf.G
        T_d = np.sqrt(Tx_d**2 + Ty_d**2 + Tz_d**2)

        # find the T_g vector
        # this vector is supposed to counterbalance gravity along the z axis
        # T_g = conf.m * conf.G / (np.cos(theta) * np.cos(phi))

        # find the phi_d and theta_d
        phi_d, theta_d, _ = xyz_to_pitch_roll_yaw([Tx_d, Ty_d, Tz_d])
        psi_d = psi

        # find the phidot_d, thetadot_d, psidot_d
        phidot_d = (phi_d - self.prev_phi_d) / conf.DT
        thetadot_d = (theta_d - self.prev_theta_d) / conf.DT
        psidot_d = (psi_d - self.prev_psi_d) / conf.DT

        # find the phidot, thetadot, psidot
        phidot = (phi - self.prev_phi) / conf.DT
        thetadot = (theta - self.prev_theta) / conf.DT
        psidot = (psi - self.prev_psi) / conf.DT

        # convert the euler angles to thrust direction vector
        # dot product between the two vectors will tell us if the angle between them is greater or less than 90 degrees
        # thrust_d = pitch_roll_yaw_to_xyz(phi_d, theta_d, psi_d)
        # thrust = pitch_roll_yaw_to_xyz(phi, theta, psi)
        # cosalpha = np.dot(thrust_d, thrust) / (
        #     np.linalg.norm(thrust_d) * np.linalg.norm(thrust)
        # )

        # # policy defined in the file controller for spring model
        # if cosalpha > 0:
        #     u1 = T_g + (T_d - T_g) * cosalpha
        # else:
        #     u1 = T_g

        # find the U1, U2, U3, U4
        # the second term is a decoupling term that lets each angle be controlled independently
        U1 = T_d
        U2 = conf.Ixx * (
            self.kphi1 * (phi_d - phi) + self.kphi2 * (phidot_d - phidot)
        ) - q * r * (conf.Iyy - conf.Izz)
        U3 = conf.Iyy * (
            self.ktheta1 * (theta_d - theta) + self.ktheta2 * (thetadot_d - thetadot)
        ) - p * r * (conf.Izz - conf.Ixx)
        U4 = (
            conf.Izz * (self.kpsi1 * (psi_d - psi) + self.kpsi2 * (psidot_d - psidot))
            - p * q * (conf.Ixx - conf.Iyy) / conf.d
        )

        self.prev_phi_d = phi_d
        self.prev_theta_d = theta_d
        self.prev_psi_d = psi_d
        self.prev_T_d = T_d
        self.prev_phi = phi
        self.prev_theta = theta
        self.prev_psi = psi

        return [U1, U2, U3, U4]

    def positionPDaction(self, state, reference):
        x_d, y_d, z_d = reference

        # update the state given the inputs
        x, y, z = state[:3]
        dx, dy, dz = state[3:6]

        # find the derivative of the reference
        dx_d = (x_d - self.prev_x_d) / conf.DT
        dy_d = (y_d - self.prev_y_d) / conf.DT
        dz_d = (z_d - self.prev_z_d) / conf.DT

        # find the acceleration reference
        ax_d = self.kx1 * (x_d - x) + self.kx2 * (dx_d - dx)
        ay_d = self.ky1 * (y_d - y) + self.ky2 * (dy_d - dy)
        az_d = self.kz1 * (z_d - z) + self.kz2 * (dz_d - dz)

        if np.abs(ax_d)>20: ax_d=np.sign(ax_d)*20
        if np.abs(ay_d)>20: ay_d=np.sign(ay_d)*20
        if np.abs(az_d)>20: az_d=np.sign(az_d)*20

        self.prev_x_d = x_d
        self.prev_y_d = y_d
        self.prev_z_d = z_d

        return [ax_d, ay_d, az_d]

    def positionfilter(self, reference):
        x, y, z = reference[:3]

        a = np.exp(-conf.DT / 0.1)
        filteredx = a * self.prev_x_d + (1 - a) * x
        filteredy = a * self.prev_y_d + (1 - a) * y
        filteredz = a * self.prev_z_d + (1 - a) * z

        self.prev_x_d = filteredx
        self.prev_y_d = filteredy
        self.prev_z_d = filteredz

        return [filteredx, filteredy, filteredz]

    def saturate(self, desired_acceleration):
        ax, ay, az = desired_acceleration
        norm = np.linalg.norm([ax, ay, az])

        if norm > self.saturation:
            ax = ax / norm * self.saturation
            ay = ay / norm * self.saturation
            az = az / norm * self.saturation

        return [ax, ay, az]

    def action(self, state, reference):
        positionaction = self.positionPDaction(state, self.positionfilter(reference))
        accelerationaction = self.accelerationPDaction(
            state, self.saturate(positionaction)
        )

        return accelerationaction


class LinearController:

    def __init__(self) -> None:
        self.kx1 = 1
        self.kx2 = 0.8
        self.ky1 = -1
        self.ky2 = -0.8
        self.kz1 = 2
        self.kz2 = 2
        self.kphi1 = 180
        self.kphi2 = 20
        self.ktheta1 = 180
        self.ktheta2 = 20
        self.kpsi1 = 180
        self.kpsi2 = 20

    def step(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:

        x_d, y_d, z_d = reference[:3]
        psi_d = reference[6]

        # update the state given the inputs
        x, y, z = state[:3]
        dx, dy, dz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        # control laws
        u1 = self.kz1 * (z_d - z) + self.kz2 * (0 - dz)
        theta_d = self.kx1 * (x_d - x) + self.kx2 * (0 - dx)
        phi_d = self.ky1 * (y_d - y) + self.ky2 * (0 - dy)

        # dphi = p * sin(phi)*tan(theta)*q +cos(phi)*tan(theta)*r
        # dtheta = cos(phi)*q - sin(phi)*r
        # dpsi = sin(phi) / cos(theta) * q + cos(phi) / cos(theta) * r

        U1 = conf.m * conf.G + u1
        U2 = conf.Ixx * (self.kphi1 * (phi_d - phi) + self.kphi2 * (0 - p))
        U3 = conf.Iyy * (self.ktheta1 * (theta_d - theta) + self.ktheta2 * (0 - q))
        U4 = conf.Izz * (self.kpsi1 * (psi_d - psi) + self.kpsi2 * (0 - r))

        return np.array([U1, U2, U3, U4])


class PositionController:

    def __init__(self, type: str = "nonlinear"):
        match type:
            case "linear":
                self.controller = LinearController()
            case "nonlinear":
                self.controller = NonlinearController()

    def step(self, state, reference):
        """Implements a low level position controller, given the
        drone actual state x return the input to move the drone to a
        desired state xd. The desired control method can be specified
        via type parameter.

        Parameters
        ----------
        x : np.ndarray
            drone actual state
        xd : np.ndarray
            desired state of the drone

        Returns
        -------
        np.ndarray
            control inputs for the drone a numpy array (4,)
        """
        return self.controller.action(state, reference)