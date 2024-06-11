import os
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from dronesim import constants as conf


def _legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot_position_error(
    reference: np.ndarray,
    trajectory: np.ndarray,
    T:int,
    title: str = "Position Error",
    filename: str = "position_error",
    savepath: str = "imgs/",
) -> None:
    """Generates a triple plot showing the trajectory's
    error on each axis (x, y, z) with respect to a reference
    trajectory.

    Parameters
    ----------
    reference : np.ndarray
        reference trajectory
    trajectories : np.ndarray

    filename : str, optional
        name of the plot, by default "error"
    """
    os.makedirs(savepath, exist_ok=True)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.supxlabel("time[s]")
    titles = ["x", "y", "z"]
    labels = ["x[m]", "y[m]", "z[m]"]

    x = np.cumsum(np.ones(T) * conf.DT)
    for i, (subtitle, label) in enumerate(zip(titles, labels)):
        ax[i].plot(
            x,
            reference[:, i],
            color=conf.colors["reference"],
            label="reference",
            linestyle="--",
            linewidth=0.5,
        )
        ax[i].grid(True, linestyle="--", linewidth=0.5)
        ax[i].set_title(subtitle)
        ax[i].set_ylabel(label)

    for i, (subtitle, label) in enumerate(zip(titles, labels)):
        ax[i].plot(
            x[: trajectory.shape[0]],
            trajectory[:, i],
            color=conf.colors["trajectory"],
            label="drone",
            linewidth=0.8,
        )

    # unify legend, fix multiple repeated labels isses
    hs, ls = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(hs, ls)) if l not in ls[:i]]
    hs = []
    ls = []
    for h, l in unique:
        hs.append(h)
        ls.append(l)

    fig.legend(hs, ls)

    plt.grid(True, ls="--")
    fig.suptitle(title)

    plt.savefig(os.path.join(savepath, filename), dpi=400)
    plt.close("all")


def plot_velocity_error(
    reference: np.ndarray,
    trajectory: np.ndarray,
    T:int,
    title: str = "Velocity error",
    filename: str = "velocity_error",
    savepath: str = "imgs/",
) -> None:
    """Generates a triple plot showing the trajectory's
    error on each axis (x, y, z) with respect to a reference
    trajectory.

    Parameters
    ----------
    reference : np.ndarray
        reference trajectory
    trajectories : np.ndarray

    filename : str, optional
        name of the plot, by default "error"
    """
    os.makedirs(savepath, exist_ok=True)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.supxlabel("time[s]")
    titles = ["x", "y", "z"]
    labels = ["x[m]", "y[m]", "z[m]"]

    x = np.cumsum(np.ones(T) * conf.DT)
    for i, (subtitle, label) in enumerate(zip(titles, labels)):
        ax[i].plot(
            x,
            reference[:, i],
            color=conf.colors["reference"],
            label="reference",
            linestyle="--",
            linewidth=0.5,
        )
        ax[i].grid(True, linestyle="--", linewidth=0.5)
        ax[i].set_title(subtitle)
        ax[i].set_ylabel(label)

    for i, (subtitle, label) in enumerate(zip(titles, labels)):
        ax[i].plot(
            x[: trajectory.shape[0]],
            trajectory[:, i],
            color=conf.colors["trajectory"],
            label="drone",
            linewidth=0.8,
        )

    # unify legend, fix multiple repeated labels isses
    hs, ls = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(hs, ls)) if l not in ls[:i]]
    hs = []
    ls = []
    for h, l in unique:
        hs.append(h)
        ls.append(l)

    fig.legend(hs, ls)

    plt.grid(True, ls="--")
    fig.suptitle(title)

    plt.savefig(os.path.join(savepath, filename), dpi=400)
    plt.close("all")


def plot_trajectories(
    references,
    trajectories,
    plane,
    missile,
    obstacles: list = [],
    drone_radius=0.1,
    elev=30,
    azim=-60,
    filename="trajectory",
    savepath="imgs",
    show: bool = False,
    pause: int = 10,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(savepath, exist_ok=True)

    n_drones, T, Nx = references.shape

    fig3 = plt.figure(figsize=(12, 8))
    ax = fig3.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Inertial frame plot
    x_axis = np.arange(-5, 5)
    y_axis = np.arange(-5, 5)
    z_axis = np.arange(-5, 5)
    ax.plot(x_axis, np.zeros(10), np.zeros(10), "r--", linewidth=0.5)
    ax.plot(np.zeros(10), y_axis, np.zeros(10), "g--", linewidth=0.5)
    ax.plot(np.zeros(10), np.zeros(10), z_axis, "b--", linewidth=0.5)

    # Names
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Quadcopter Simulation")
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    ax.axes.set_xlim3d(*(0,(np.max(plane[0])+(np.max(plane[0])/3))))
    ax.axes.set_ylim3d(*(0,(np.max(plane[1])+(np.max(plane[1])/3))))
    ax.axes.set_zlim3d(*(0,(np.max(plane[2])+(np.max(plane[2])/3))))


    for d in range(n_drones):
        ax.plot(
            references[d, :, 0],
            references[d, :, 1],
            references[d, :, 2],
            color='blue',  # Example color for reference
            linestyle="--",
            linewidth=0.5,
            label=f"reference {d}",
        )

    # Plot obstacles
    for ox, oy, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 50)  # Use plot's Z limits
        U, Z = np.meshgrid(u, z)
        X = radius * np.cos(U) + ox
        Y = radius * np.sin(U) + oy
        ax.plot_surface(X, Y, Z, color='orange', alpha=0.7)

    for d in range(n_drones):
        ax.plot(
            trajectories[d, :, 0],
            trajectories[d, :, 1],
            trajectories[d, :, 2],
            color=conf.colors["drones"][d],  # Example color for drone trajectory
            label=f"drone {d}",
        )

        # Plot terminal state as a small sphere
        terminal_state = [
            np.array(trajectories[d])[-1, 0],
            np.array(trajectories[d])[-1, 1],
            np.array(trajectories[d])[-1, 2],
        ]
        ax.scatter(
            terminal_state[0],
            terminal_state[1],
            terminal_state[2],
            color=conf.colors["drones"][d],  # Example color for drone
        )

        # Plot a sphere around the terminal state
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = drone_radius * np.outer(np.cos(u), np.sin(v)) + terminal_state[0]
        y = drone_radius * np.outer(np.sin(u), np.sin(v)) + terminal_state[1]
        z = drone_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + terminal_state[2]
        ax.plot_surface(x, y, z, color='green', alpha=0.3)

    # Add another line (example)
    ax.plot(plane[0], plane[1], plane[2], color='purple', label='Plane')

    ax.legend()
    ax.grid(True)
    style = {
        "linewidth": 1,
        "linestyle": "--",
    }
    ax.xaxis._axinfo["grid"].update(style)
    ax.yaxis._axinfo["grid"].update(style)
    ax.zaxis._axinfo["grid"].update(style)
    ax.plot(missile[0], missile[1], missile[2], color='yellow', label='Missile')

    ax.legend()
    ax.grid(True)
    style = {
        "linewidth": 1,
        "linestyle": "--",
    }
    ax.xaxis._axinfo["grid"].update(style)
    ax.yaxis._axinfo["grid"].update(style)
    ax.zaxis._axinfo["grid"].update(style)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, filename), bbox_inches="tight")

    if show:
        plt.show(block=False)
        plt.pause(pause)
    plt.close("all")



def generate_animation(
    trajectories: np.ndarray,
    obstacles: np.ndarray,
    drone_radius=0.4,
    elev=80,
    azim=-80,
    title: str = "Drone Simulation",
    filename: str = "simulation.gif",
    savepath: str = "imgs/",
    skip_frames: int = 10,
) -> None:
    """
    Generates a short video of the drones' flight simulation.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    style = {
        "linewidth": 0.5,
        "linestyle": "--",
    }
    ax.xaxis._axinfo["grid"].update(style)
    ax.yaxis._axinfo["grid"].update(style)
    ax.zaxis._axinfo["grid"].update(style)
    # tick options
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.zaxis.set_tick_params(labelsize=6)
    plt.tight_layout()
    # Create a list of line objects for each trajectory
    lines = []
    labels = []

    for id, traj in enumerate(trajectories):
        (line,) = ax.plot([], [], [], lw=1, color=conf.colors["drones"][id])
        lines.append(line)
        labels.append(f"drone {id}")  # Collect labels for legend

    ax.legend(lines, labels)

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

    def update(index, trajectories, lines):
        # Clear previous spheres and triangles
        for coll in ax.collections:
            coll.remove()

        # Plot obstacles
        for obstacle in obstacles:
            obstacle_x, obstacle_y, obstacle_radius = obstacle
            u = np.linspace(0, 2 * np.pi, 100)
            z = np.linspace(ax.get_zlim()[0], 10, 10)  # Use plot's Z limits
            U, Z = np.meshgrid(u, z)
            X = obstacle_radius * np.cos(U) + obstacle_x
            Y = obstacle_radius * np.sin(U) + obstacle_y
            ax.plot_surface(X, Y, Z, color=conf.colors["obstacles"], alpha=0.8)

            # Plot top disk
            top_disk = np.array(
                [
                    [
                        obstacle_x + obstacle_radius * np.cos(u),
                        obstacle_radius * np.sin(u) + obstacle_y,
                        10,
                    ]
                    for u in np.linspace(0, 2 * np.pi, 100)
                ]
            )
            ax.add_collection3d(
                Poly3DCollection([top_disk], color=conf.colors["obstacles"], alpha=0.8)
            )
            # Plot bottom disk
            # bottom_disk = np.array(
            #     [
            #         [
            #             obstacle_radius * np.cos(u) + obstacle_x,
            #             obstacle_radius * np.sin(u) + obstacle_y,
            #             -10,
            #         ]
            #         for u in np.linspace(0, 2 * np.pi, 100)
            #     ]
            # )
            # ax.add_collection3d(Poly3DCollection([bottom_disk], color="r", alpha=0.7))

        # plot trajectories
        for i, line in enumerate(lines):
            line.set_data(trajectories[i, :index, :2].T)
            line.set_3d_properties(trajectories[i, :index, 2].T)

            # Plot terminal state as a small triangle
            terminal_state = trajectories[i, index, :]
            ax.scatter(*terminal_state[:3], color=conf.colors["drones"][i], lw=0.9)

            # Create a meshgrid of phi and theta values
            phi, theta = np.mgrid[0: 2 * np.pi: 100j, 0: np.pi: 50j]

            # Convert spherical coordinates (phi, theta) to Cartesian coordinates (x, y, z)
            x = terminal_state[0] + drone_radius * np.sin(theta) * np.cos(phi)
            y = terminal_state[1] + drone_radius * np.sin(theta) * np.sin(phi)
            z = terminal_state[2] + drone_radius * np.cos(theta)
            ax.plot_surface(x, y, z, color=conf.colors["drones"][i], alpha=0.3)

    frames = tqdm(
        np.arange(0, trajectories[0].shape[0], skip_frames),
        file=sys.stdout,
        desc="Animation",
    )
    an = FuncAnimation(
        fig, update, init_func=init, frames=frames, fargs=(trajectories, lines)
    )

    os.makedirs(savepath, exist_ok=True)
    an.save(os.path.join(savepath, filename))