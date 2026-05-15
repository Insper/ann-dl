from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def activation(sigma):
    return 1 if sigma >= 0 else 0

fig, ax = plt.subplots(figsize=(8, 8))

# creating a plot
lines_plotted = plt.plot([])  
line_plotted = lines_plotted[0]

x_lim = (-2.2, 2.2)
ax.set_xlim(x_lim)

tx_epoch = ax.text(-2, 2, '')

X = np.array([
    [1, 1],
    [2, 2],
    [-1, -1],
    [-2, -1]
])
Y = np.array([
    [1], [1],
    [0], [0]
])

x1 = [1, 2, -1, -2]
x2 = [1, 2, -1, -1]
plt.plot(
    x1[:2], x2[:2], 'or',
    x1[2:], x2[2:], 'ob',
)

# weights initialization
w = np.array([-1, 0])

# bias initialization
b = 0

eta = 0.1

xp = np.linspace(x_lim[0], x_lim[1])
yp = []

max_epochs = 10
epoch = 0
stop = False
while not stop:
    has_error = False
    for x, y in zip(X, Y):
        sigma = np.dot(w, x) + b
        y_hat = activation(sigma)
        error = y - y_hat
        if error != 0:
            b = b + eta * error
            w = w + eta * error * x
            has_error = True
        # print(x, w, y, y_hat, error)
    epoch += 1
    yp.append(-w[0]/w[1]*xp - b/w[1] if w[1] != 0 else 0*xp)
    stop = not has_error or epoch >= max_epochs

def animate(i):
    line_plotted.set_data((xp, yp[i]))
    tx_epoch.set_text(f'epoch: {i+1}')
    return ([plt, tx_epoch])

ani = FuncAnimation(fig, animate, repeat=True, frames=epoch-1, interval=1000)

print(ani.to_jshtml())

plt.close()
