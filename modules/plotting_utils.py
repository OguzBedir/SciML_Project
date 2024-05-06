import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules.model import PINN
from modules.pinn_utils import f

def plot_2D(pinn: PINN, x_domain: list, y_domain: list, t: int, device: str,
            num_points: int = 100, figsize=(8, 6), cmap='viridis', 
            fname: str='2D_solution_plot.png'):
    """
        Plot the solution of the PINN in 2D.

        Args:
            pinn (PINN): The physics-informed neural network model.
            x_domain (list): List containing the spatial domain along the x-axis [xmin, xmax].
            y_domain (list): List containing the spatial domain along the y-axis [ymin, ymax].
            t (int): Time coordinate.
            num_points (int): Number of points along each axis for plotting (default is 100).
            figsize (tuple): Size of the figure (default is (8, 6)).
            cmap (str): Colormap for the plot (default is 'viridis').
            fname (str): File name for saving the plot (default is '2D_solution_plot.png').
            device (str): Device to use for computation.
    """
    # Generate meshgrid for 2D plotting
    x_grid, y_grid = torch.meshgrid(torch.linspace(x_domain[0], x_domain[1], num_points),
                                     torch.linspace(y_domain[0], y_domain[1], num_points))
    t_grid = torch.ones_like(x_grid) * t  # Plot for given time t

    # Concatenate spatial coordinates and time
    xyt = torch.stack([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()], dim=1).to(device)

    # Compute solution using PINN model
    with torch.no_grad():
        solution = f(pinn, 
                     xyt[:, 0].reshape(-1, 1), 
                     xyt[:, 1].reshape(-1, 1), 
                     xyt[:, 2].reshape(-1, 1)).reshape(num_points, num_points)

    # Plot solution
    plt.figure(figsize=figsize)
    plt.imshow(solution.detach().cpu().numpy(), 
               extent=(x_domain[0], x_domain[1],
                        y_domain[0], y_domain[1]), cmap=cmap, origin='lower')
    plt.colorbar(label='Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"PINN Approximation at time t={t}")
    plt.grid(False)

    # Define the directory path for saving the figure
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
    # Ensure the directory exists, create it if it doesn't
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


def plot_3D(pinn: PINN, x_domain: list, y_domain: list, t: int, device: str,
            num_points: int = 100, figsize=(8, 6), cmap='viridis', 
            fname: str='3D_solution_plot.png'):
    """
    Plot the solution of the PINN in 3D.

    Args:
        pinn (PINN): The physics-informed neural network model.
        x_domain (list): List containing the spatial domain along the x-axis [xmin, xmax].
        y_domain (list): List containing the spatial domain along the y-axis [ymin, ymax].
        t (int): Time coordinate.
        num_points (int): Number of points along each axis for plotting (default is 100).
        figsize (tuple): Size of the figure (default is (8, 6)).
        cmap (str): Colormap for the plot (default is 'viridis').
        fname (str): File name for saving the plot (default is '3D_solution_plot.png').
        device (str): Device to use for computation (default is 'cuda').
    """
    # Generate meshgrid for 3D plotting
    x_grid = torch.linspace(x_domain[0], x_domain[1], num_points)
    y_grid = torch.linspace(y_domain[0], y_domain[1], num_points)
    X, Y = torch.meshgrid(x_grid, y_grid)
    t_grid = torch.full_like(X, t)

    # Concatenate spatial coordinates and time
    xy = torch.stack([X.flatten(), Y.flatten(), t_grid.flatten()], dim=1).to(device)

    # Compute solution using PINN model
    with torch.no_grad():
        solution = f(pinn, 
                     xy[:, 0].reshape(-1, 1), 
                     xy[:, 1].reshape(-1, 1), 
                     xy[:, 2].reshape(-1, 1)).reshape(num_points, num_points)

    # Plot solution
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = torch.meshgrid(x_grid, y_grid)
    surf = ax.plot_surface(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), 
                           solution.detach().cpu().numpy(), cmap=cmap)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'PINN approximation at time t={t}')
    
    # Save the plot
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()