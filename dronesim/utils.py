import numpy as np

from dronesim import constants as conf


def detect_drones_collisions(drones: list) -> np.ndarray:
    """Returns for each drone if a collision between another drone
    happened

    Parameters
    ----------
    drones : List[Quadrotor]
        drones to check

    Returns
    -------
    np.ndarray
        a boolean array, in which each position i is true
        if drone i collided
    """

    collisions = np.zeros(len(drones), dtype=bool)

    for i1, d1 in enumerate(drones):
        for i2, d2 in enumerate(drones):
            if i1 != i2:
                collisions[i1] = (
                    np.linalg.norm(d1.position() - d2.position()) <= conf.l * 2
                )

    return collisions


def detect_obstacles_collisions(
    drones: list, obstacles: np.ndarray
) -> np.ndarray:
    """Check if any drone collided with an obstacle

    Parameters
    ----------
    drones : List[Quadrotor]
        drones
    obstacles : np.ndarray
        obstacles

    Returns
    -------
    np.ndarray
        a boolean array, in which each position i is true
        if drone i collided with obstacle
    """
    collisions = np.zeros(len(drones), dtype=bool)

    for i, d in enumerate(drones):
        for xc, yc, radius in obstacles:
            pos1 = d.position()
            pos2 = np.array([xc, yc, pos1[2]])
            if np.linalg.norm(pos1 - pos2) <= conf.l + radius:
                collisions[i] = True
                break
            else:
                collisions[i] = False
    return collisions


def detect_collisions(drones: list, obstacles: np.ndarray) -> bool:

    d_d_collisions = detect_drones_collisions(drones)
    if any(d_d_collisions):
        print("collision between 2 drones")
        return True
    d_o_collisions = detect_obstacles_collisions(drones, obstacles)
    if any(d_o_collisions):
        print("collision drone-obstacle")
        return True

    return False