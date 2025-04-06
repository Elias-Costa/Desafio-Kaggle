import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64*12*12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, input):
        output = self.relu(self.conv1(input)) #26x26
        output = self.relu(self.conv2(output)) #24x24
        output = self.maxPool(output) #12x12

        output = self.relu(self.fc1(torch.flatten(output, start_dim=1)))
        output = self.dropout1(output)

        output = self.relu(self.fc2(output))
        output = self.dropout2(output)

        output = self.fc3(output)

        return output
