import torch

import matplotlib.pyplot as plt
from modules.plotting_utils import plot_2D, plot_3D_subplot

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = torch.load('saved_models/149000.pt')
model.eval()

fig = plt.figure(figsize=(20, 16))  # Adjust figsize as needed
axs = fig.subplots(4, 5, subplot_kw={'projection': '3d'})

for i, ax in enumerate(axs.flatten()):
    time_step = i * 2 / 20
    X, Y, solution = plot_3D_subplot(model, [0, 1.0], [0, 1.0], t=time_step,
                          device=device, figsize=(4, 3))
    surf = ax.plot_surface(X, Y, solution)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'PINN approximation at time t={time_step}')
    ax.set_zlim(0, 1)
    
plt.tight_layout()
plt.savefig('150_000.png')
plt.show()