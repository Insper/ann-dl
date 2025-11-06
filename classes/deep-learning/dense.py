import numpy as np

def dense_forward(x, W, b):
    y_linear = np.dot(W, x) + b
    y = np.maximum(0, y_linear)  # ReLU
    return y, y_linear  # Cache linear for backprop

def dense_backward(dy_post_act, x, W, y_linear):
    # dy_post_act: ∂L/∂y (post-ReLU)
    dy_linear = dy_post_act * (y_linear > 0)  # ReLU derivative
    dx = np.dot(W.T, dy_linear)
    dW = np.outer(dy_linear, x)
    db = dy_linear
    return dx, dW, db

# Example
x = np.array([2, 3])
W = np.array([[1, 2], [0, -1]])
b = np.array([1, -1])
y, y_linear = dense_forward(x, W, b)
dy_post_act = np.array([0.5, -0.2])
dx, dW, db = dense_backward(dy_post_act, x, W, y_linear)
print("Forward y:", y)  # [9, 0]
print("dx:", dx)  # [0.5, 1.0]
print("dW:", dW)  # [[1, 1.5], [0, 0]]
print("db:", db)  # [0.5, 0]