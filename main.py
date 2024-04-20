import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from NNCollection import CNN

# Simulate the data, images of 1 and 2
one_image = [
    [0,0, 0, 0, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 1, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,0, 0, 1, 0, 0,0],
    [0,1, 1, 1, 1, 1,0],
    [0,0, 0, 0, 0, 0,0]
]

two_image = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

# Convert the list to a NumPy array
two_image = np.array(two_image)
# Convert the list to a NumPy array
one_image = np.array(one_image)
num_samples = 10000
one_samples_with_noise = []
two_samples_with_noise = []

for _ in range(num_samples):
    # Generate the random noise
    noise = np.random.randn(*one_image.shape)
    noisy_image = one_image + noise*2
    one_samples_with_noise.append(noisy_image)
    noise = np.random.randn(*two_image.shape)
    noisy_image = two_image + noise*2
    two_samples_with_noise.append(noisy_image)

one_samples_with_noise = np.array(one_samples_with_noise)
two_samples_with_noise = np.array(two_samples_with_noise)

x_sim_dat = np.concatenate((one_samples_with_noise, two_samples_with_noise), axis=0)
y_sim_dat = np.ones(20000)
y_sim_dat[10000:] = 2

x_train, x_test, y_train, y_test = train_test_split(x_sim_dat, y_sim_dat, test_size=0.2)
y_train = np.where(y_train == 1, 1, 0)  # Convert labels to 0 and 1
y_test = np.where(y_test == 1, 1, 0)  # Convert labels to 0 and 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xtr_tensor = torch.tensor(x_train,dtype=torch.float32).unsqueeze(1).to(device)
ytr_tensor = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1).to(device)
xte_tensor = torch.tensor(x_test,dtype=torch.float32).unsqueeze(1).to(device)
yte_tensor = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(xtr_tensor, ytr_tensor)
test_dataset = TensorDataset(xte_tensor, yte_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)

# Define a CNN
model = CNN(num_rgb=1, num_filter=16, kernel_size=3, stride_step=1, padding_size=1).to(device)
# Define optimizers and criterias
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Begin training the model
num_epochs = 100
for epochs in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in test_loader:
            outputs = model(data)
            # Convert outputs to binary predictions
            predictions = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

        accuracy = 100 * correct / total

        print(f'Epoch {epochs + 1}/{num_epochs}, Loss: {loss.item()}, Test Accuracy: {accuracy}%')