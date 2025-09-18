import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y**2

# Forward pass
def lstm_forward(x_t, h_prev, C_prev, W_f, W_i, W_C, W_o, b_f, b_i, b_C, b_o):
    concat = np.concatenate((h_prev, x_t), axis=0)
    f_t = sigmoid(np.dot(W_f, concat) + b_f)
    i_t = sigmoid(np.dot(W_i, concat) + b_i)
    C_tilde = tanh(np.dot(W_C, concat) + b_C)
    C_t = f_t * C_prev + i_t * C_tilde
    o_t = sigmoid(np.dot(W_o, concat) + b_o)
    h_t = o_t * tanh(C_t)
    cache = (concat, f_t, i_t, C_tilde, o_t, C_t)
    return h_t, C_t, cache

# Backward pass
def lstm_backward(dh_next, dC_next, cache, W_f, W_i, W_C, W_o):
    concat, f_t, i_t, C_tilde, o_t, C_t = cache
    
    # Derivatives
    dC_t = dh_next * o_t * dtanh(tanh(C_t)) + dC_next
    do_t = dh_next * tanh(C_t) * dsigmoid(o_t)
    dC_tilde = dC_t * i_t * dtanh(C_tilde)
    di_t = dC_t * C_tilde * dsigmoid(i_t)
    df_t = dC_t * C_prev * dsigmoid(f_t)
    dC_prev = dC_t * f_t
    
    # Gradients for gates
    dconcat_o = np.dot(W_o.T, do_t)
    dconcat_C = np.dot(W_C.T, dC_tilde)
    dconcat_i = np.dot(W_i.T, di_t)
    dconcat_f = np.dot(W_f.T, df_t)
    dconcat = dconcat_f + dconcat_i + dconcat_C + dconcat_o
    
    # Split for h_prev and x_t
    hidden_size = h_prev.shape[0]
    dh_prev = dconcat[:hidden_size]
    dx_t = dconcat[hidden_size:]
    
    # Parameter gradients
    dW_f = np.outer(df_t, concat)
    db_f = df_t
    dW_i = np.outer(di_t, concat)
    db_i = di_t
    dW_C = np.outer(dC_tilde, concat)
    db_C = dC_tilde
    dW_o = np.outer(do_t, concat)
    db_o = do_t
    
    return dx_t, dh_prev, dC_prev, dW_f, db_f, dW_i, db_i, dW_C, db_C, dW_o, db_o

# Numerical example (hidden size = 1)
x_t = np.array([0.5])
h_prev = np.array([0.1])
C_prev = np.array([0.2])
W_f = np.array([[0.5, 0.5]])
W_i = np.array([[0.4, 0.4]])
W_C = np.array([[0.3, 0.3]])
W_o = np.array([[0.2, 0.2]])
b_f = np.array([0.0])
b_i = np.array([0.0])
b_C = np.array([0.0])
b_o = np.array([0.0])

# Forward
h_t, C_t, cache = lstm_forward(x_t, h_prev, C_prev, W_f, W_i, W_C, W_o, b_f, b_i, b_C, b_o)
print("Forward h_t:", h_t)  # Output: [0.11199714]
print("Forward C_t:", C_t)  # Output: [0.2145628]

# Backward example: assume dh_next = [0.1], dC_next = [0.05]
dh_next = np.array([0.1])
dC_next = np.array([0.05])
dx_t, dh_prev, dC_prev, dW_f, db_f, dW_i, db_i, dW_C, db_C, dW_o, db_o = lstm_backward(dh_next, dC_next, cache, W_f, W_i, W_C, W_o)

print("Backward dx_t:", dx_t)  # Output: [0.02164056]
print("Backward dh_prev:", dh_prev)  # Output: [0.02164056]
print("Backward dC_prev:", dC_prev)  # Output: [0.05780591]
print("Backward dW_f:", dW_f)  # Output: [[0.00049199 0.00245997]]
print("Backward db_f:", db_f)  # Output: [0.00491995]
print("Backward dW_i:", dW_i)  # Output: [[0.00044162 0.00220808]]
print("Backward db_i:", db_i)  # Output: [0.00441615]
print("Backward dW_C:", dW_C)  # Output: [[0.00545376 0.02726878]]
print("Backward db_C:", db_C)  # Output: [0.05453756]
print("Backward dW_o:", dW_o)  # Output: [[0.00052643 0.00263213]]
print("Backward db_o:", db_o)  # Output: [0.00526427]