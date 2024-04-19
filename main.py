import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from NNCollection import FNN
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
# Generate a sequence from 0 to 1 with a step of 0.01 using arange() function
x = np.arange(0, 10.01, 0.001)  # End value (1.01) is exclusive, so we use 1.01 to include 1.0
# print(x.shape)
print(x[0:2])
y = pow(x,0.5) + np.sin(x*math.pi/2) + np.random.normal(loc=0, scale=1, size=x.shape[0])
# plt.plot(x,y)
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.title('Title of the Plot')
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Convert data to PyTorch tensors
x_tr_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for single feature
y_tr_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_te_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for single feature
y_te_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


input_dim = 1  # Dimension of input features
hidden_dim = 100  # Number of units in the hidden layer
output_dim = 1  # Dimension of output
model = FNN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tr_tensor)
    loss = criterion(outputs, y_tr_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Compute training error
        train_outputs = model(x_tr_tensor)
        train_loss = criterion(train_outputs, y_tr_tensor)

        # Compute test error
        test_outputs = model(x_te_tensor)
        test_loss = criterion(test_outputs, y_te_tensor)

    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(x_te_tensor)

y_pred = test_outputs.numpy()
y_test_reshaped = y_test[:, np.newaxis]
x_test_reshaped = x_test[:, np.newaxis]
things_to_plot = np.concatenate((x_test_reshaped,y_test_reshaped,y_pred), axis = 1)
sorted_things = np.argsort(things_to_plot[:, 0])
things_to_plot = things_to_plot[sorted_things]

plt.figure(figsize=(12, 6))
# Plot predicted y as a time series
plt.plot(things_to_plot[:,0], things_to_plot[:,1], color='green', label='Test Y')
plt.plot(things_to_plot[:,0], things_to_plot[:,2], color='red', label='Predicted Y')
# Plot actual y as a time series
# plt.plot(x_test, y_test, color='red', label='Actual y (Test Data)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()