import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(5 * 5 * 128, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.batchnorm2(x)
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
