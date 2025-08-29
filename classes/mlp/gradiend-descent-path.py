import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the objective function (a simple paraboloid for illustration)
def f(x, y):
    return x**2 + y**2

# Define the gradient of the function
def grad_f(x, y):
    return 2 * x, 2 * y

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 50
start_x, start_y = 5.0, 5.0  # Starting point away from the minimum

# Perform gradient descent and record the path
path_x = [start_x]
path_y = [start_y]
path_z = [f(start_x, start_y)]

for _ in range(num_iterations):
    gx, gy = grad_f(path_x[-1], path_y[-1])
    next_x = path_x[-1] - learning_rate * gx
    next_y = path_y[-1] - learning_rate * gy
    path_x.append(next_x)
    path_y.append(next_y)
    path_z.append(f(next_x, next_y))

# Create the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate the surface mesh
x_range = np.arange(-6, 6, 0.25)
y_range = np.arange(-6, 6, 0.25)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, rstride=5, cstride=5)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('3D Simulation of Gradient Descent')

# Initialize the path line (red line with markers)
line, = ax.plot([], [], [], 'r-', marker='o', markersize=4, label='Descent Path')
ax.legend()

# Animation update function
def update(frame):
    line.set_data(path_x[:frame+1], path_y[:frame+1])
    line.set_3d_properties(path_z[:frame+1])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(path_x), interval=200, blit=False)

# Display the plot (run this in a Python environment with GUI support, like Jupyter or a local script)
ani.save('gradient_descent.gif', writer=PillowWriter(fps=10))