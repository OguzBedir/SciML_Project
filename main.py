import torch
import torch.nn as nn

from modules.model import PINN
from modules.loss import Loss
from modules.training import train_model
from modules.utils import initial_condition

import os
import yaml

current_dir = os.path.dirname(__file__)
yaml_file_path = os.path.join(current_dir, "config.yaml")
with open(yaml_file_path, "r") as file:
    yaml_data = yaml.safe_load(file)

LAYERS = yaml_data["LAYERS"]
NEURONS_PER_LAYER = yaml_data["NEURONS_PER_LAYER"]
LENGTH = yaml_data["LENGTH"]
TOTAL_TIME = yaml_data["TOTAL_TIME"]
N_POINTS = yaml_data["N_POINTS"]
N_POINTS_PLOT = yaml_data["N_POINTS_PLOT"]
WEIGHT_RESIDUAL = yaml_data["WEIGHT_RESIDUAL"]
WEIGHT_INITIAL =  yaml_data["WEIGHT_INITIAL"]
WEIGHT_BOUNDARY = yaml_data["WEIGHT_BOUNDARY"]
LEARNING_RATE = yaml_data["LEARNING_RATE"]
EPOCHS = yaml_data["EPOCHS"]

# Set the compute device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

# train the PINN
loss_fn = Loss(
    x_domain,
    y_domain,
    t_domain,
    N_POINTS,
    initial_condition,
    # floor,
    WEIGHT_RESIDUAL,
    WEIGHT_INITIAL,
    WEIGHT_BOUNDARY
)

pinn_trained, loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = train_model(
    nn_approximator=pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS, 
    plot_solution=True, compute_device=device)

pinn = pinn.cpu()
losses = loss_fn.verbose(pinn)
print(f'Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})')
print(f'Boundary loss: \t{losses[3]:.5f} ({losses[3]:.3E})')