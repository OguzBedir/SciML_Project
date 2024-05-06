import numpy as np
from modules.domain_utils import get_initial_points
from modules.utils import initial_condition

# >>> START <<<
"""
Quick and dirty solution.
Can be refactored in the future.
"""
import os
import yaml
# Get the current directory
current_dir = os.path.dirname(__file__)
# Navigate to the parent directory
parent_dir = os.path.dirname(current_dir)
# Specify the path to your YAML file in the parent directory
yaml_file_path = os.path.join(parent_dir, "config.yaml")
# Open the YAML file and load its contents
with open(yaml_file_path, "r") as file:
    yaml_data = yaml.safe_load(file)
# Now you can access elements from the loaded YAML data
LENGTH = yaml_data["LENGTH"]
TOTAL_TIME = yaml_data["TOTAL_TIME"]
N_POINTS = yaml_data["N_POINTS"]
VELOCITY = yaml_data["VELOCITY"]

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]
n_points = N_POINTS
# >>> END <<<

class WaveEqn2D:
    def __init__(self, nx: int = n_points, ny: int = n_points, 
                 c: float = VELOCITY, h: float = 1e-3, dt: float = 1e-3,
                 damping_coeff: float = 1.0):
        """Initialize the simulation:

        nx and ny are the dimensions of the domain;
        c is the wave speed;
        h and dt are the space and time grid spacings.
        """

        self.nx, self.ny = nx, ny
        self.c = c
        self.h, self.dt = h, dt
        self.alpha = np.zeros((nx, ny)) # Wave propagation velocities of the entire simulation domain
        self.alpha[0:nx, 0:ny] = ((c * dt) / h)**2

        self.u = np.zeros((3, ny, nx))
        self.t = 0 # Initialize the current time
        self.damping_coeff = damping_coeff

    def place_single_drop(self):
        """Place a single 'drop' at the center of the simulation domain."""
        initial_points = get_initial_points(x_domain, y_domain, t_domain, n_points)
        x_initial, y_initial, _ = initial_points
        self.u[0, :, :] = initial_condition(x_initial, y_initial).detach().cpu().numpy().reshape(n_points, n_points)

    def update(self):
        """Update the simulation by one time tick at the specified point (ix, iy)."""

        # The three planes of u correspond to the time points
        # k+1, k and k-1; i.e. we calculate the next frame
        # of the simulation (k+1) in u[0,...].
        u, nx, ny = self.u, self.nx, self.ny
        u[2] = u[1]     # old k -> new k-1
        u[1] = u[0]     # old k+1 -> new k

        # Calculate the new k+1 at the specified point (ix, iy)
        # Version 2: Much faster by eliminating loops
        u[0, 1:nx-1, 1:ny-1]  = self.alpha[1:nx-1, 1:ny-1] * (
                                u[1, 0:nx-2, 1:ny-1] +
                                u[1, 2:nx,   1:ny-1] +
                                u[1, 1:nx-1, 0:ny-2] +
                                u[1, 1:nx-1, 2:ny] - 4 * u[1, 1:nx-1, 1:ny-1]) \
                                + 2 * u[1, 1:nx-1, 1:ny-1] - u[2, 1:nx-1, 1:ny-1]
        
        # Calculate damping coefficient based on current amplitude
        damping_factor = np.exp(-self.damping_coeff * self.dt * self.t)
        # Clip damping factor to be within range [0, 1]
        damping_factor = np.clip(damping_factor, 0, 1)
        # Apply damping
        u[0] *= damping_factor

        # Apply zero boundary conditions
        u[0, 0, :] = 0  # bottom boundary
        u[0, -1, :] = 0  # top boundary
        u[0, :, 0] = 0  # left boundary
        u[0, :, -1] = 0  # right boundary

        # Update time
        self.t += self.dt
