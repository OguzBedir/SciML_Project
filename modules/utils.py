import numpy as np
import torch

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

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

# Initial condition
def initial_condition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt((x-LENGTH/2)**2 + (y-LENGTH/2)**2) # * 0
    # r[190, 0] = 1
    # res = r
    # res = 2 * torch.exp(-(r)**2 * 30) + 2
    # res = 2 * torch.exp(-(r)**2 * 30)
    res = torch.exp((-(x-1.0).pow(2) - (y-1.0).pow(2)) * 30)
    return res

# def floor(x, y):
#     """Get the sea floor value"""
#     return 0