from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def activation(sigma):
    return 1 if sigma >= 0 else 0

fig, ax = plt.subplots(figsize=(8, 8))

# creating a plot
lines_plotted = plt.plot([])  
line_plotted = lines_plotted[0]

ax.set_title('')

N = 500

X = np.concat((
    np.random.multivariate_normal(
        [1.5, 1.5],
        [[.5, 0],[0, .5]],
        size=N
    ),
    np.random.multivariate_normal(
        [5, 5],
        [[.5, 0],[0, .5]],
        size=N
    )
))
Y = np.concat((np.zeros(N), np.ones(N)))

plt.plot(
    X[Y == 0, 0], X[Y == 0, 1], 'ob',
    X[Y == 1, 0], X[Y == 1, 1], 'or',
)

x_lim = ax.get_xlim()
xp = np.linspace(x_lim[0], x_lim[1])
yp = []
wrongs = []


# weights initialization
w = np.array([-1, 0])

# bias initialization
b = 0

eta = 0.1

max_epochs = 100
epoch = 0
stop = False
while not stop:
    wrong = 0
    for x, y in zip(X, Y):
        sigma = np.dot(w, x) + b
        y_hat = activation(sigma)
        error = y - y_hat
        if error != 0:
            b = b + eta * error
            w = w + eta * error * x
            wrong += 1
        # print(x, w, y, y_hat, error)
    epoch += 1
    wrongs.append(wrong)
    yp.append(-w[0]/w[1]*xp - b/w[1] if w[1] != 0 else 0*xp)
    stop = wrong == 0 or epoch >= max_epochs

def animate(i):
    line_plotted.set_data((xp, yp[i]))
    ax.set_title(f'epoch: {i+1} - wrong: {wrongs[i]}')
    return line_plotted

ani = FuncAnimation(fig, animate, repeat=False, frames=epoch, interval=200, blit=False)
ani.save(Path.cwd() / 'perceptron_classification.gif')

# print(ani.to_jshtml())

# plt.show()
plt.close()
