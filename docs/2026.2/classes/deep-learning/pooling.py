import numpy as np

def max_pool_forward(X, pool_size=2, stride=2):
    H, W = X.shape
    out_H, out_W = H // stride, W // stride
    Y = np.zeros((out_H, out_W))
    max_idx = np.zeros_like(X, dtype=bool)  # For backprop
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            slice = X[i:i+pool_size, j:j+pool_size]
            max_val = np.max(slice)
            Y[i//stride, j//stride] = max_val
            max_idx[i:i+pool_size, j:j+pool_size] = (slice == max_val)
    return Y, max_idx

def max_pool_backward(dY, max_idx, pool_size=2, stride=2):
    dX = np.zeros_like(max_idx, dtype=float)
    for i in range(dY.shape[0]):
        for j in range(dY.shape[1]):
            dX[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size] = dY[i,j] * max_idx[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
    return dX

# Example
X = np.arange(1,17).reshape(4,4)
Y, max_idx = max_pool_forward(X)
dY = np.array([[0.5, -0.5],[1, 0]])
dX = max_pool_backward(dY, max_idx)
print("Forward Y:\n", Y)
print("dX:\n", dX)