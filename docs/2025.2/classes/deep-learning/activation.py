import numpy as np

def relu_forward(x):
    return np.maximum(0, x), x

def relu_backward(dy, x_cache):
    return dy * (x_cache > 0)

# Example
x = np.array([-1,0,2,-3])
y, x_cache = relu_forward(x)
dy = np.array([0.5,-0.5,1,0])
dx = relu_backward(dy, x_cache)
print("dx:", dx)
