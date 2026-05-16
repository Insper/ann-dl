import numpy as np

def flatten_forward(x):
    return x.flatten(), x.shape

def flatten_backward(dy, orig_shape):
    return dy.reshape(orig_shape)

# Example
x = np.array([[1,2],[3,4]])
y, shape = flatten_forward(x)
dy = np.array([0.1,0.2,0.3,0.4])
dx = flatten_backward(dy, shape)
print("dx:\n", dx)
