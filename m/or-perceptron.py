training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]
# nor target -> 1, 0, 0, 0

w1 = 0
w2 = 0
bias = 0
learning_rate = 1

# Step activation function
def step(net):
    return 1 if net >= 0 else 0

max_epochs = 10

for epoch in range(1, max_epochs + 1):
    print(f"\nEpoch {epoch}")
    total_error = 0

    for x1, x2, target in training_data:
        net = (x1 * w1) + (x2 * w2) + bias
        output = step(net)
        error = target - output
        total_error += abs(error)

        print(f"Input: ({x1},{x2}) "
              f"Net={net} Output={output} Target={target} Error={error}")

        # Update weights and bias
        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2
        bias = bias + learning_rate * error

        print(f"Updated weights -> w1={w1}, w2={w2}, bias={bias}")

    if total_error == 0:
        print("\n✅ Converged!")
        print(f"✔ Correct output achieved in Epoch {epoch}")
        break

print("\nFinal OR Gate Output:")
for x1, x2, _ in training_data:
    net = (x1 * w1) + (x2 * w2) + bias
    print(f"{x1} OR {x2} = {step(net)}")