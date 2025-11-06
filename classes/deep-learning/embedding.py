import numpy as np

def embedding_forward(index, E):
    return E[index]

def embedding_backward(dy, index, E_shape):
    dE = np.zeros(E_shape)
    dE[index] = dy
    return dE

# Example
E = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
index = 1
y = embedding_forward(index, E)
dy = np.array([0.1, -0.1, 0.2])
dE = embedding_backward(dy, index, E.shape)
print("dE:\n", dE)
