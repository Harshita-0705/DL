import numpy as np
from sklearn.neural_network import MLPClassifier

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

lr = 0.5
epochs = 10000

np.random.seed(1)
W = np.random.rand(2,1)
b = np.random.rand(1)

for epoch in range(epochs):
    # Forward
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    # Backward
    error = y_pred - y
    delta = error * sigmoid_derivative(y_pred)

    W -= lr * np.dot(X.T, delta)
    b -= lr * np.sum(delta)

print("Random Init Output:")
print(y_pred.round())

limit = np.sqrt(6 / (2 + 1))
W = np.random.uniform(-limit, limit, (2,1))
b = 0

for epoch in range(epochs):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    error = y_pred - y
    delta = error * sigmoid_derivative(y_pred)

    W -= lr * np.dot(X.T, delta)
    b -= lr * np.sum(delta)

print("\nXavier Init Output:")
print(y_pred.round())

mlp = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.5,
    max_iter=10000
)

mlp.fit(X, y.ravel())
print("\nMLPClassifier Output:")
print(mlp.predict(X))