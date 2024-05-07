import numpy as np
import matplotlib.pyplot as plt

from wave_eqn.wave_eqn import WaveEqn2D

wave_eqn = WaveEqn2D()
wave_eqn.place_single_drop()

TIME_TO_VISUALIZE = 1
LENGTH = 1.0
N_POINTS = 20

data = np.zeros(shape=(N_POINTS, N_POINTS, N_POINTS))

time_step_idx = 0
# Simulate until the desired time
while wave_eqn.t < TIME_TO_VISUALIZE:
    wave_eqn.update()
    data[time_step_idx] = wave_eqn.u[0]
    time_step_idx += 1

# Get the solution at the specified time
solution = np.copy(wave_eqn.u[0])
np.save('data.npy', data)

# Plot the solution
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# x = np.linspace(0, LENGTH, N_POINTS)
# y = np.linspace(0, LENGTH, N_POINTS)
# X, Y = np.meshgrid(x, y)
# Z = solution
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_title(f"Wave Equation at t = {TIME_TO_VISUALIZE}")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Amplitude')
# # ax.set_zlim(0, 4)

# plt.show()

# # Plot the solution in 2D
# # plt.figure(figsize=(8, 6))
# # plt.contourf(X, Y, Z, cmap='viridis', levels=50)  # Use contourf for filled contours
# # plt.colorbar(label='Amplitude')  # Add color bar indicating amplitude
# # plt.title(f"Wave Equation at t = {TIME_TO_VISUALIZE}")
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.grid(True)  # Add grid lines for better visualization
# # plt.show()
