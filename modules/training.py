import time
import numpy as np
import torch
from typing import Callable
from modules.model import PINN
from modules.plotting_utils import plot_2D, plot_3D

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
# >>> END <<<

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]

def train_model(
    compute_device: str,
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
    plot_solution: bool = False,
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    residual_loss_values = []
    initial_loss_values = []
    boundary_loss_values = []

    start_time = time.time()

    for epoch in range(max_epochs):
        try:
            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            loss_values.append(loss[0].item())
            residual_loss_values.append(loss[1].item())
            initial_loss_values.append(loss[2].item())
            boundary_loss_values.append(loss[3].item())

            if plot_solution and (epoch + 1) % 10_000 == 0:
                # TODO: 
                # - ADD AN IF STATEMENT FOR SELECTING 2D OR 3D PLOTTING OPTION
                # - CHANGE STATIC 500 TO SOMETHING THAT CAN BE PASSED AS AN ARGUMENT
                plot_2D(nn_approximator, x_domain, y_domain, t=0,
                        fname=f"{epoch+1}, t={0}.png", device=compute_device)
                plot_2D(nn_approximator, x_domain, y_domain, t=0.25,
                        fname=f"{epoch+1}, t={0.25}.png", device=compute_device)
                plot_2D(nn_approximator, x_domain, y_domain, t=0.5,
                        fname=f"{epoch+1}, t={0.5}.png", device=compute_device)

            if (epoch + 1) % 1000 == 0:
                epoch_time = time.time() - start_time
                start_time = time.time()

                print(f"Epoch: {epoch + 1} - "
                    f"Loss: {float(loss[0].item()):>7f}, "
                    f"Residual Loss: {float(loss[1].item()):>7f}, "
                    f"Initial Loss: {float(loss[2].item()):>7f}, "
                    f"Boundary Loss: {float(loss[3].item()):>7f}, "
                    f"Epoch Time: {epoch_time:.4f} seconds")

        except KeyboardInterrupt:
            break

    return (
        nn_approximator,
        np.array(loss_values),
        np.array(residual_loss_values),
        np.array(initial_loss_values),
        np.array(boundary_loss_values)
    )
