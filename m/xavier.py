import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# Xavier initialization
def xavier(n_in, n_out):
    limit = math.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit)

# Inputs
x1, x2 = 0.05, 0.10
t = 0.01
lr = 0.5

# Xavier weights
w1 = xavier(2, 2)
w2 = xavier(2, 2)
w3 = xavier(2, 1)
b1, b2 = 0.0, 0.0

# Forward pass
h1 = sigmoid(x1*w1 + x2*w2 + b1)
o = sigmoid(h1*w3 + b2)

# Backward pass
delta_o = (t - o) * sigmoid_derivative(o)
delta_h = delta_o * w3 * sigmoid_derivative(h1)

# Update
w3 += lr * delta_o * h1
w1 += lr * delta_h * x1
w2 += lr * delta_h * x2

print("Output with Xavier Initialization:", o)