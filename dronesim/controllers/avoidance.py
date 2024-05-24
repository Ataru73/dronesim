import numpy as np


def spring_obstacle_avoidance(
    state: np.ndarray, obstacles: np.ndarray, radius: float
) -> np.ndarray:
    """Obstacle Avoidance controller based on repulsive forces.
    Implementation follows from:
    - Decentralized Hybrid Model Predictive Control of a
    Formation of Unmanned Aerial Vehicles (Bemporad 2011)
    - Modelling of UAV formation flight using 3D potential field (Paul 2008)


    Parameters
    ----------
    drone : Quadrotor
        quadrotor object
    obstacles : np.ndarray
        list of obstacles

    Returns
    -------
    np.ndarray
        resultat repulsive force from obstacles
    """

    K_SAFE = 20.5  # safety distance from obstacles
    K_OA = 1.0  # repulsive coefficient

    F_oa = np.zeros(3)  # obstacle avoidance repulsion force
    pos = state[:3]
    vel = state[3:6]

    for xc, xy, obs_radius in obstacles:
        pos_obs = np.array([xc, xy, pos[2]])
        distance = pos - pos_obs
        d_ki = np.linalg.norm(distance)

        # safe distance changes proportionally to drone's velocity
        r_safe = radius + obs_radius + K_SAFE * np.linalg.norm(vel)
        if d_ki <= r_safe:
            F_oa += ((K_OA / d_ki) - (K_OA / r_safe)) * distance / d_ki

    return F_oa


def obstacle_avoidance(
    state: np.ndarray, obstacles: np.ndarray, radius: float, type: str = "spring"
) -> np.ndarray:
    """Implements an obstacle avoidance controller, given the
    drone actual state x and a obstacles position returns the
    reupulsive force to guide the drone towards a safe position.
    The desired control method can be specified via type parameter.

    Parameters
    ----------
    state : np.ndarray
        drone state
    obstacles : np.ndarray
        list of obstacles represented as an array (n_obs, 3)
        each obstacle is a cylinder (x_center, y_center, radius)
        and the height span from 0 to max value of z.

    Returns
    -------
    np.ndarray
        repulsive force from obstacles
    """
    match type:
        case "spring":
            return spring_obstacle_avoidance(state, obstacles, radius)
