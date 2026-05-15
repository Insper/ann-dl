import numpy as np

def batch_norm_forward(x, gamma, beta, epsilon=1e-4):
    mu = np.mean(x)
    var = np.var(x)
    x_hat = (x - mu) / np.sqrt(var + epsilon)
    y = gamma * x_hat + beta
    return y, (x_hat, mu, var)

def batch_norm_backward(dy, cache, gamma):
    x_hat, mu, var = cache
    N = dy.shape[0]
    dx_hat = dy * gamma
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + epsilon)**(-1.5), axis=0)
    dmu = np.sum(dx_hat * -1 / np.sqrt(var + epsilon), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    dx = dx_hat / np.sqrt(var + epsilon) + dvar * 2 * (x - mu) / N + dmu / N
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)
    return dx, dgamma, dbeta

# Example
x = np.array([1,2,3,4.])
gamma, beta = 1, 0
y, cache = batch_norm_forward(x, gamma, beta)
dy = np.array([0.1,0.2,-0.1,0.3])
dx, dgamma, dbeta = batch_norm_backward(dy, cache, gamma)
print("dx:", dx)
print("dgamma:", dgamma)
print("dbeta:", dbeta)