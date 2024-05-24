import numpy as np

from dronesim import constants as conf


def get_reference_trajectory(offset: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """Generate a reference trajectory, if offset is specified
    generates a trajectory with an offset.

    Parameters
    ----------
    offset : np.ndarray, optional
        offset of the trajectory, by default np.array([0, 0, 0])

    Returns
    -------
    ref : np.ndarray (4, T num of simulation steps)
        reference trajectory
    """

    ref = np.zeros((conf.T, 4))
    for t in range(conf.T):
        ref[t, :] += np.array([t, 0, 0, 0])
        ref[t, :3] += offset
    return ref


def equally_spaced_points(
    start: np.ndarray, end: np.ndarray, space: float
) -> np.ndarray:
    """Given start and end positions generates an array of equally-spaced points along the line.

    Parameters
    ----------
    start : np.ndarray
        Start point
    end : np.ndarray
        End point
    space : float
        Spacing of each point

    Returns
    -------
    np.ndarray
        Array containing the generated points
    """
    points = []
    dist = np.linalg.norm(end - start)
    n_points = int(np.ceil(dist / space))
    if n_points > 1:
        step = dist / (n_points - 1)
        for i in range(n_points):
            points.append(steer(start, end, i * step))
    return points


def steer(start: np.ndarray, end: np.ndarray, d: float):
    """Return a point in the direction of the goal, that is distance away from start

    Parameters
    ----------
    start : np.ndarray
        Starting point
    end : np.ndarray
        End point
    d : float
        distance

    Returns
    -------
    np.ndarray
        point from start to goal with length d
    """
    v = end - start
    u = v / (np.sqrt(np.sum(v**2)))
    steered_point = start + u * d
    return steered_point


def waypoints2timebased(points: np.ndarray, space: float = 0.1, steps: int = None):
    """Given a set of waypoints return a timebased trajectory of equally
    distant points. The distance is defined by the space or number of steps

    Parameters
    ----------
    points : np.ndarray
        set of waypoints
    space : float, optional
        space betoween each point, by default .1
    steps : int, optional
        if provided, the space between each point is computed based
        on the number of steps, by default None

    Returns
    -------
    np.ndarray
        returns a timebased trajectory
    """

    # if we need a trajectory of exactly n step, we compute the space
    # in function of the number of steps
    if steps:
        tot_dist = 0.0
        for p in range(len(points) - 1):
            tot_dist += np.linalg.norm(points[p + 1] - points[p])
        space = tot_dist / (steps - 1)

    timebased = []
    for p in range(len(points) - 1):
        timebased += equally_spaced_points(points[p], points[p + 1], space)
    return np.array(timebased)


def get_offseted_references(reference: np.ndarray, n_drones: int, offsets: np.ndarray):
    assert reference.shape[0] == conf.T, print(
        f"Provided reference has incorrect shape {reference.shape}, num of steps is {conf.T}"
    )

    ref = np.zeros((n_drones, conf.T, conf.Nx))

    for d in range(n_drones):
        for i, t in enumerate(range(conf.T)):
            ref[d, i, :] += np.hstack(
                (reference[i], np.zeros(3), np.zeros(3), np.zeros(3))
            )
            ref[d, i, :3] += offsets[d]
        for i in range(conf.T - 1):
            ref[d, i, 3:6] = (ref[d, i + 1, :3] - ref[d, i, :3]) / conf.DT
    return ref
