import numpy as np

def dropout_forward(x, p, training=True):
    if training:
        mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
        y = x * mask
        return y, mask
    return x, None

def dropout_backward(dy, mask):
    if mask is None:
        return dy
    return dy * mask

# Example
np.random.seed(0)
x = np.array([1,2,3,4.])
p = 0.5
y, mask = dropout_forward(x, p)
dy = np.array([0.1,0.2,0.3,0.4])
dx = dropout_backward(dy, mask)
print("dx:", dx)