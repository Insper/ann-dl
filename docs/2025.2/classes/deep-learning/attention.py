import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention_forward(Q, K, V):
    d = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d)
    weights = softmax(scores)
    attn = np.dot(weights, V)
    return attn, (scores, weights, K, V)  # Cache

def attention_backward(dattn, cache):
    scores, weights, K, V = cache
    dweights = np.dot(dattn, V.T)
    dscores = weights * (dweights - np.sum(weights * dweights, axis=-1, keepdims=True))
    dQ = np.dot(dscores, K) / np.sqrt(K.shape[-1])
    dK = np.dot(dscores.T, Q) / np.sqrt(K.shape[-1])
    dV = np.dot(weights.T, dattn)
    return dQ, dK, dV

# Example
Q = K = V = np.array([[1.,0.],[0.,1.]])
attn, cache = attention_forward(Q, K, V)
dattn = np.array([[0.1,0.2],[ -0.1,0.3]])
dQ, dK, dV = attention_backward(dattn, cache)
print("dQ:\n", dQ)