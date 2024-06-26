"""
Generates boundary, initial, and interior points for solving PDEs with PINNs in PyTorch.
"""

import torch

def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    """
         .+------+
       .' |    .'|
      +---+--+'  |
      |   |  |   |
    y |  ,+--+---+
      |.'    | .' t
      +------+'
         x
    """
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

    x_grid, t_grid = torch.meshgrid( x_linspace, t_linspace, indexing="ij")
    y_grid, _      = torch.meshgrid( y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t_grid = t_grid.reshape(-1, 1).to(device)
    t_grid.requires_grad = requires_grad

    x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
    x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
    y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
    y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)

    down    = (x_grid, y0,     t_grid)
    up      = (x_grid, y1,     t_grid)
    left    = (x0,     y_grid, t_grid)
    right   = (x1,     y_grid, t_grid)

    return down, up, left, right
     

def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)
     

def get_interior_points(x_domain, y_domain, t_domain, n_points, mode="mid", 
                        device=torch.device("cpu"), requires_grad=True):
    if mode == "init":
        # For "init" mode, restrict x, y, and t ranges
        x_raw = torch.linspace((x_domain[1] - x_domain[0])//2 - 0.25, 
                               (x_domain[1] - x_domain[0])//2 + 0.25, 
                               steps=n_points, requires_grad=requires_grad)
        y_raw = torch.linspace((y_domain[1] - y_domain[0])//2 - 0.25, 
                               (y_domain[1] - y_domain[0])//2 + 0.25, 
                               steps=n_points, requires_grad=requires_grad)
        t_raw = torch.linspace(t_domain[0], t_domain[1]//2, steps=n_points, requires_grad=requires_grad)
    elif mode == "mid":
        # Implement the function for "mid" mode here
        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points, requires_grad=requires_grad)
        y_raw = torch.linspace(y_domain[0], y_domain[1], steps=n_points, requires_grad=requires_grad)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points, requires_grad=requires_grad)
    elif mode == "end":
        # For "end" mode, randomly sample points from the domain
        x_raw = (x_domain[1] - x_domain[0]) * torch.rand(n_points, requires_grad=requires_grad) + x_domain[0]
        y_raw = (y_domain[1] - y_domain[0]) * torch.rand(n_points, requires_grad=requires_grad) + y_domain[0]
        t_raw = t_domain[1] * torch.rand(n_points, requires_grad=requires_grad)
    else:
        raise ValueError("Invalid mode. Choose from 'init', 'mid', or 'end'.")

    grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")
    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)

    return x, y, t

def get_data_points(x_domain, y_domain, t_domain, n_points, device=torch.device("cpu"), requires_grad=True):
    # Define the raw points with linspace
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points, requires_grad=requires_grad)
    y_raw = torch.linspace(y_domain[0], y_domain[1], steps=n_points, requires_grad=requires_grad)
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points, requires_grad=requires_grad)

    # Create copies of the raw points and add noise
    x = x_raw
    y = y_raw
    t = t_raw

    # Create the meshgrid of points
    grids = torch.meshgrid(x, y, t, indexing="ij")

    # Reshape the grids to obtain the points
    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)

    return x, y, t