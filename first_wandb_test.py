# train.py
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Initialize Weights & Biases
wandb.login()
wandb.init(project="my-awesome-project", config={"learning_rate": 0.01, "epochs": 10, "batch_size": 32})

# Hyperparameters
epochs = wandb.config.epochs
lr = wandb.config.learning_rate
batch_size = wandb.config.batch_size

# Simulated dataset
X = torch.randn(1000, 10)
y = (torch.sum(X, dim=1) > 0).float()

# DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Logging
    accuracy = ((outputs > 0.5) == targets).float().mean().item()
    wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Finish logging
wandb.finish()
