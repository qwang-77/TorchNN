import torch.nn as nn
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # input_dim: the input dimension of the data
        # hidden_dim: how many dimensions are there in each hidden layer, we assume all layers have the same number of nodes
        # output_dim: the output dimension of the FNN, 1 for numerical and bi-classification, more for multivariate
        super(FNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.nonlinear1 = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Non-linearity
        self.nonlinear2 = nn.Sigmoid()

        # Linear function (readout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.nonlinear1(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)

        out = self.nonlinear2(out)

        out = self.fc3(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_rgb, num_filter, kernel_size, stride_step = 1, padding_size = 0):
        # num_rgb: the number of input color channel
        # num_filter: number of filters, we choose them to be the same in all layers
        # kernel_size: the dimension of each kernel
        # stride_step: how far the kernel moves step by step, typically 1
        # padding_size: to explore edge effects, zero's are added to the image on the edge, how many more dims do we want to add

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_rgb, num_filter, kernel_size=kernel_size, stride=stride_step, padding=padding_size)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=kernel_size, stride=stride_step, padding=padding_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x