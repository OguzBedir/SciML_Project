import torch
import numpy as np
from typing import Tuple, Callable
from modules.model import PINN
from modules.domain_utils import get_interior_points, get_initial_points, get_boundary_points, get_data_points
from modules.pinn_utils import f, dfdt, dfdx, dfdy

# >>> START <<<
"""
Quick and dirty solution.
SHOULD be refactored in the future.
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
c = yaml_data["VELOCITY"]
# >>> END <<<

class Loss:
    def __init__(
        self,
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        initial_condition: Callable,
        # floor: Callable,
        weight_r: float = 1.0,
        weight_b: float = 1.0,
        weight_i: float = 1.0
        # verbose: bool = False,
    ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.initial_condition = initial_condition
        # self.floor = floor
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.data = torch.from_numpy(np.load("data.npy"))

    def data_loss(self,pinn):
        x, y, t = get_data_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        preds = f(pinn, x, y, t)
        loss =(torch.flatten(self.data.to(pinn.device())) - preds)
        return loss.pow(2).mean()

    def residual_loss(self, pinn: PINN, mode: str):
        x, y, t = get_interior_points(self.x_domain, self.y_domain, self.t_domain, self.n_points + 5,
                                      mode=mode, device=pinn.device())
        loss = dfdt(pinn, x, y, t, order=2) - \
                c**2 * (
                    dfdx(pinn, x, y, t, order=2) + 
                    dfdy(pinn, x, y, t, order=2)
                )

        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points//4, pinn.device())
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, 
                                                    pinn.device())
        x_down,  y_down,  t_down    = down
        x_up,    y_up,    t_up      = up
        x_left,  y_left,  t_left    = left
        x_right, y_right, t_right   = right

        loss_down  = dfdy( pinn, x_down,  y_down,  t_down  )
        loss_up    = dfdy( pinn, x_up,    y_up,    t_up    )
        loss_left  = dfdx( pinn, x_left,  y_left,  t_left  )
        loss_right = dfdx( pinn, x_right, y_right, t_right )

        return loss_down.pow(2).mean()  + \
            loss_up.pow(2).mean()    + \
            loss_left.pow(2).mean()  + \
            loss_right.pow(2).mean()

    def verbose(self, pinn: PINN, mode: str):
        """
        Returns all parts of the loss function
        """
        residual_loss = self.residual_loss(pinn, mode)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)
        data_loss = self.data_loss(pinn)

        final_loss = \
            self.weight_r * residual_loss + \
            self.weight_i * initial_loss + \
            self.weight_b * boundary_loss + \
            data_loss

        return final_loss, residual_loss, initial_loss, boundary_loss, data_loss

    def __call__(self, pinn: PINN, mode: str = 'mid'):
        """
        Allows you to use instance of this class as if it was a function:
        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn, mode)