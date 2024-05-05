import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from modules.model import PINN
from modules.pinn_utils import f

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
# >>> END <<<

def plot_solution(pinn: PINN, x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)

    def animate(i):
        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)

def plot_color(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, limit=0.2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axes.set_zlim3d(bottom=-limit, top=limit)

    c = ax.plot_surface(X, Y, Z)

    x_floor = torch.linspace(0.0, LENGTH, 50)
    y_floor = torch.linspace(0.0, LENGTH, 50)
    z_floor = torch.zeros((50, 50))
    for x_idx, x_coord in enumerate(x_floor):
        for y_idx, y_coord in enumerate(y_floor):
            # z_floor[x_idx, y_idx] = floor(x_coord, y_coord)
            z_floor[x_idx, y_idx] = 0
    x_floor = torch.tile(x_floor, (50, 1))
    y_floor = torch.tile(y_floor, (50, 1)).T
    f = ax.plot_surface(x_floor, y_floor, z_floor, color='green', alpha=0.7)

    return fig