import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
G = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
)
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(3):
    for x, _ in loader:
        x = x.view(-1, 784)

        # Train Discriminator
        real_labels = torch.ones(x.size(0), 1)
        fake_labels = torch.zeros(x.size(0), 1)

        z = torch.randn(x.size(0), 100)
        fake = G(z)

        loss_real = loss_fn(D(x), real_labels)
        loss_fake = loss_fn(D(fake.detach()), fake_labels)
        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        loss_G = loss_fn(D(fake), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print("GAN Epoch:", epoch+1)
def add_noise(x, t):
    noise = torch.randn_like(x)
    return x + noise * t

model = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 784)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(3):
    for x, _ in loader:
        x = x.view(-1, 784)
        noisy_x = add_noise(x, 0.3)
        output = model(noisy_x)

        loss = loss_fn(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Diffusion Epoch:", epoch+1)