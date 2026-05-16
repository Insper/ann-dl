import numpy as np

def activation(sigma):
    return 1 if sigma >= 0 else 0

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

eta = 0.1

# weights initialization
w = np.array([0, 0])

# bias initialization
b = 0

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
        print(x, w, y, y_hat, error)
    epoch += 1
    stop = not has_error or epoch >= max_epochs
