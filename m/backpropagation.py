import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

x1 = 0.05
x2 = 0.10
target = 0.01

w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45

b1 = 0.35
b2 = 0.60

learning_rate = 0.5

net_h1 = x1*w1 + x2*w2 + b1
net_h2 = x1*w3 + x2*w4 + b1

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

net_o = out_h1*w5 + out_h2*w6 + b2
output = sigmoid(net_o)

error = 0.5 * (target - output)**2

print("FORWARD PASS")
print("Hidden outputs:", out_h1, out_h2)
print("Final output:", output)
print("Error:", error)

delta_o = (target - output) * sigmoid_derivative(output)

delta_h1 = delta_o * w5 * sigmoid_derivative(out_h1)
delta_h2 = delta_o * w6 * sigmoid_derivative(out_h2)

w5 += learning_rate * delta_o * out_h1
w6 += learning_rate * delta_o * out_h2

w1 += learning_rate * delta_h1 * x1
w2 += learning_rate * delta_h1 * x2
w3 += learning_rate * delta_h2 * x1
w4 += learning_rate * delta_h2 * x2

b2 += learning_rate * delta_o
b1 += learning_rate * (delta_h1 + delta_h2)

print("\nBACKWARD PASS (Updated Weights)")
print("w1:", w1, "w2:", w2)
print("w3:", w3, "w4:", w4)
print("w5:", w5, "w6:", w6)
print("b1:", b1, "b2:", b2)