import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
a = 10.0
b = 28.0
c = 2.667

# Initial conditions
x0 = 0.0
y0 = 1.0
z0 = 1.05

# Time parameters
t_max = 50.0   # Total time
dt = 0.01      # Time step
num_steps = int(t_max / dt)

# Arrays to store the trajectory
t = np.linspace(0, t_max, num_steps)
x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)

# Set initial conditions
x[0], y[0], z[0] = x0, y0, z0

# Define the Lorenz system derivatives
def lorenz(state, a, b, c):
    x, y, z = state
    dx_dt = a * (y - x)
    dy_dt = b * x - y - x * z
    dz_dt = x * y - c * z
    return np.array([dx_dt, dy_dt, dz_dt])

# RK4 integration step
def rk4_step(state, dt, a, b, c):
    k1 = lorenz(state, a, b, c)
    k2 = lorenz(state + 0.5 * dt * k1, a, b, c)
    k3 = lorenz(state + 0.5 * dt * k2, a, b, c)
    k4 = lorenz(state + dt * k3, a, b, c)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Integrate the system using RK4
state = np.array([x0, y0, z0])
for i in range(1, num_steps):
    state = rk4_step(state, dt, a, b, c)
    x[i], y[i], z[i] = state

# Plotting the 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, lw=0.5, color="blue", label="Bee's path")

# Mark the starting point
ax.scatter([x0], [y0], [z0], color="red", s=100, label="Start (t=0)")

# Set labels and title
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Z position")
ax.set_title("3D Trajectory of the Bee (Lorenz System)")
ax.legend()

# === Save the plot ===
output_path = r"C:\Users\krish\OneDrive\Desktop\Hackathon\src\Problem_2\results\result.png"
plt.savefig(output_path, dpi=300)

print(f"Plot saved successfully at: {output_path}")

# Show the plot
plt.show()
