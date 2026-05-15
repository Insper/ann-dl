import numpy as np
from scipy.signal import correlate2d, convolve2d

def conv_forward(X, K, b):
    Y = correlate2d(X, K, mode='valid') + b  # SciPy correlate for conv
    return Y, X  # Cache X

def conv_backward(dY, X, K):
    # Rotate kernel 180 degrees for full conv
    K_rot = np.rot90(K, 2)
    # Pad dY to match X shape for dx
    pad_h, pad_w = K.shape[0]-1, K.shape[1]-1
    dY_padded = np.pad(dY, ((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))
    dX = convolve2d(dY_padded, K_rot, mode='valid')
    dK = correlate2d(X, dY, mode='valid')
    db = np.sum(dY)
    return dX, dK, db

# Example
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
K = np.array([[1,0],[-1,1]])
b = 1
Y, _ = conv_forward(X, K, b)
dY = np.array([[0.5, -0.5],[1, 0]])
dX, dK, db = conv_backward(dY, X, K)
print("Forward Y:\n", Y)
print("dX:\n", dX)
print("dK:\n", dK)
print("db:", db)