import numpy as np

# ------------------------------
# Constants
G: float = 9.81  # gravity
DT: float = 0.033  # timestep duration (frequency)
T: int = 1000  # simulation timesteps

# ------------------------------
# Quadrotor configuration
coeff=8
m: float = 0.5*coeff  # drone mass
Ixx: float = 5e-3*coeff  # drone inertia x
Iyy: float = 5e-3*coeff  # drone inertia y
Izz: float = 9e-3*coeff  # drone inertia z
l: float = 0.25  # drone radius (center of mass-motor distance)
d: float = 1.1e-5  # drag factor
b: float = 7.2e-5  # thrust factor
Jm: float = 3.4e-5  # motor inertia
x0: np.ndarray = np.hstack(
    (
        np.zeros(3),  # position
        np.zeros(3),  # velocity
        np.zeros(3),  # angular positions
        np.zeros(3),  # angular velociyy
    )
)  # initial state
Nx: int = 12  # number of states
Nu: int = 4  # number of inputs

# ------------------------------
# Map configuration
x_lim: (float, float) = (-3, 20)  # map limit for x axis
y_lim: (float, float) = (-3, 20)  # map limit for y axis
z_lim: (float, float) = (-3, 20)  # map limit for z axis
obstacles: np.ndarray = np.array(
    [
        [5.9, 4.7, 1.2],
        # [5.9, 5.7, 1.7],
        [9.5, 12.5, 0.8],
    ]
)  # obstacles
start: np.ndarray = np.zeros(3)  # start point
goal: np.ndarray = np.array([18, 18, 6])  # end point
waypoints: np.ndarray = np.vstack(
    [
        start,
        np.array([4.5, 5.7, 4.0]),
        goal,
    ]
)  # trajectory waypoints

# ------------------------------
# Visualization colors

colors = {
    "obstacles": "#d90429",
    "reference": "black",
    "trajectory": "#0096c7",
    "drones": [
        "#0096c7",
        "#90a955",
        "#fb8500",
        "#c1121f",
        "y",
        "orange",
    ],
}