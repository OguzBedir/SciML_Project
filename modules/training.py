import time
import numpy as np
import torch
from typing import Callable
from modules.model import PINN

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
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
            if (epoch + 1) % 1000 == 0:
                epoch_time = time.time() - start_time
                start_time = time.time()

                print(f"Epoch: {epoch + 1} - Loss: {float(loss[0].item()):>7f}, Residual Loss: {float(loss[1].item()):>7f}, Initital Loss: {float(loss[2].item()):>7f}, Boundary Loss: {float(loss[3].item()):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values), np.array(residual_loss_values), np.array(initial_loss_values), np.array(boundary_loss_values)