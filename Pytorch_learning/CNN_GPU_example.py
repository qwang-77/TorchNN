# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# Determine the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transforms
transform = ToTensor()

# Load the training data
training_data = datasets.FashionMNIST(
    root="D:/77/temp",
    train=True,
    download=True,
    transform=transform
)

# Move the training data to the specified device
training_data = [(data.to(device), torch.tensor(label, dtype=torch.long).to(device)) for data, label in training_data]

# Load the test data
test_data = datasets.FashionMNIST(
    root="D:/77/temp",
    train=False,
    download=True,
    transform=transform
)

# Move the test data to the specified device
test_data = [(data.to(device), torch.tensor(label, dtype=torch.long).to(device)) for data, label in test_data]

# reshaped_training_data = [(data.unsqueeze(0), target) for data, target in training_data]
# reshaped_test_data = [(data.unsqueeze(0), target) for data, target in test_data]
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            # Adjust the input size based on the output size of the last convolutional layer
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_stack(x)
        # Flatten the output before feeding it into the fully connected layers
        x = torch.flatten(x, 1)
        # Apply fully connected layers
        logits = self.fc_stack(x)
        return logits


model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")