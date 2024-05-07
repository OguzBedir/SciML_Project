import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from modules.model import PINN
from modules.loss import Loss
from modules.training import train_model
from modules.utils import initial_condition, running_average

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

pinn_trained, loss_values, residual_loss_values, initial_loss_values, boundary_loss_values, data_loss_values = train_model(
    nn_approximator=pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS, 
    plot_solution=True, compute_device=device)

pinn = pinn.cpu()
losses = loss_fn.verbose(pinn, mode='mid')
print(f'Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})')
print(f'Boundary loss: \t{losses[3]:.5f} ({losses[3]:.3E})')
print(f'Data loss: \t{losses[3]:.5f} ({losses[3]:.3E})')

# Define the directory path for saving the figure
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img/loss_functions')
# Ensure the directory exists, create it if it doesn't
os.makedirs(save_dir, exist_ok=True)

# Plotting
# Loss function
average_loss = running_average(loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(average_loss)
ax.set_title("Loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, "total_loss.png"))
plt.close()

average_loss = running_average(residual_loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(average_loss)
ax.set_title("Residual loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, "residual_loss.png"))
plt.close()

average_loss = running_average(initial_loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(average_loss)
ax.set_title("Initial loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, "initial_loss.png"))
plt.close()

average_loss = running_average(boundary_loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(average_loss)
ax.set_title("Boundary loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, "boundary_loss.png"))
plt.close()

average_loss = running_average(data_loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(average_loss)
ax.set_title("Loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, "data_loss.png"))
plt.close()
