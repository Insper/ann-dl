import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the loss function: f(x, y) = x^2 + y^2 + 0.5
def loss_function(x, y):
    return x**2 + y**2 + 0.5

# Gradient of the loss function
def gradient(x, y):
    return np.array([2*x, 2*y])

# Optimization algorithms
def gradient_descent(pos, learning_rate=0.1):
    grad = gradient(pos[0], pos[1])
    return pos - learning_rate * grad

def stochastic_gradient_descent(pos, learning_rate=0.1, noise_scale=0.1):
    grad = gradient(pos[0], pos[1])
    noise = np.random.normal(0, noise_scale, size=2)
    return pos - learning_rate * (grad + noise)

def momentum(pos, velocity, learning_rate=0.1, momentum=0.9):
    grad = gradient(pos[0], pos[1])
    velocity = momentum * velocity - learning_rate * grad
    return pos + velocity, velocity

def rmsprop(pos, cache, learning_rate=0.1, decay_rate=0.9, epsilon=1e-8):
    grad = gradient(pos[0], pos[1])
    cache = decay_rate * cache + (1 - decay_rate) * (grad**2)
    update = learning_rate * grad / (np.sqrt(cache) + epsilon)
    return pos - update, cache

def adam(pos, m, v, t, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grad = gradient(pos[0], pos[1])
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return pos - update, m, v

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create mesh for the surface plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# Plot the surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Loss')
ax.set_title('3D Optimization Algorithms Comparison')

# Initialize starting point and parameters
start_pos = np.array([4.0, 4.0])
iterations = 50
paths = {
    'GD': [start_pos.copy()],
    'SGD': [start_pos.copy()],
    'Momentum': [start_pos.copy()],
    'RMSProp': [start_pos.copy()],
    'Adam': [start_pos.copy()]
}

# Initialize algorithm-specific variables
velocity_momentum = np.zeros(2)
cache_rmsprop = np.zeros(2)
m_adam = np.zeros(2)
v_adam = np.zeros(2)

# Compute paths for all algorithms
for i in range(iterations):
    # Gradient Descent
    paths['GD'].append(gradient_descent(paths['GD'][-1]))
    
    # Stochastic Gradient Descent
    paths['SGD'].append(stochastic_gradient_descent(paths['SGD'][-1]))
    
    # Momentum
    next_pos, velocity_momentum = momentum(paths['Momentum'][-1], velocity_momentum)
    paths['Momentum'].append(next_pos)
    
    # RMSProp
    next_pos, cache_rmsprop = rmsprop(paths['RMSProp'][-1], cache_rmsprop)
    paths['RMSProp'].append(next_pos)
    
    # Adam
    next_pos, m_adam, v_adam = adam(paths['Adam'][-1], m_adam, v_adam, i + 1)
    paths['Adam'].append(next_pos)

# Convert paths to numpy arrays and compute z values
for key in paths:
    paths[key] = np.array(paths[key])
    
# Initialize plot lines for each algorithm
lines = {
    'GD': ax.plot([], [], [], 'o-', label='Gradient Descent', color='red')[0],
    'SGD': ax.plot([], [], [], 'o-', label='SGD', color='blue')[0],
    'Momentum': ax.plot([], [], [], 'o-', label='Momentum', color='green')[0],
    'RMSProp': ax.plot([], [], [], 'o-', label='RMSProp', color='purple')[0],
    'Adam': ax.plot([], [], [], 'o-', label='Adam', color='orange')[0]
}

ax.legend()
ax.view_init(elev=50) 


# Animation update function
def update(frame):
    for key in lines:
        x = paths[key][:frame+1, 0]
        y = paths[key][:frame+1, 1]
        z = loss_function(x, y)
        lines[key].set_data_3d(x, y, z)
    return lines.values()

# Create and store animation
ani = FuncAnimation(fig, update, frames=iterations, interval=200, blit=True)
# writer = PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('gradient-comparation.gif', writer=writer)

print(ani.to_jshtml())

plt.close()
