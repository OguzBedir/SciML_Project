import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from modules.domain_utils import get_initial_points
from modules.utils import initial_condition
import os
import yaml

# Load configuration from YAML file
current_dir = os.path.dirname(__file__)
yaml_file_path = os.path.join(current_dir, "config.yaml")
with open(yaml_file_path, "r") as file:
    yaml_data = yaml.safe_load(file)

LENGTH = yaml_data["LENGTH"]
TOTAL_TIME = yaml_data["TOTAL_TIME"]
N_POINTS = yaml_data["N_POINTS"]

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]
n_points = N_POINTS

# Generate grid points for the initial condition
initial_points = get_initial_points(x_domain, y_domain, t_domain, n_points)

# Unpack the grid points
x_initial, y_initial, t_initial = initial_points

# Evaluate the initial condition function
initial_values = initial_condition(x_initial, y_initial)

# Reshape the values for surface plot
X = x_initial.cpu().detach().numpy().reshape(n_points, n_points)
Y = y_initial.cpu().detach().numpy().reshape(n_points, n_points)
Z = initial_values.cpu().detach().numpy().reshape(n_points, n_points)

# Plot the initial condition as a 3D surface plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Customize plot
plt.colorbar(surf, label='Initial Condition')
ax.set_title('Initial Condition')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Initial Condition')

# Show plot
plt.show()
