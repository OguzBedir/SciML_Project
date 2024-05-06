import numpy as np
import matplotlib.pyplot as plt

from wave_eqn.wave_eqn import WaveEqn2D

wave_eqn = WaveEqn2D()
wave_eqn.place_single_drop()

TIME_TO_VISUALIZE = 0
LENGTH = 2.0
N_POINTS = 15

# Simulate until the desired time
while wave_eqn.t < TIME_TO_VISUALIZE:
    wave_eqn.update()

# Get the solution at the specified time
solution = np.copy(wave_eqn.u[0])

# Plot the solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, LENGTH, N_POINTS)
y = np.linspace(0, LENGTH, N_POINTS)
X, Y = np.meshgrid(x, y)
Z = solution
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title(f"Wave Equation at t = {TIME_TO_VISUALIZE}")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Amplitude')

plt.show()