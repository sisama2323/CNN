import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *


# defind nn, example model
class Net_exp(nn.Module):
    def __init__(self):
        super(Net_exp, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model 1
class Alex_net(nn.Module):
    def __init__(self, n_hid1=2048, n_hid2=2048):
        super(Alex_net, self).__init__()
        # paramter
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        # convolution
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.conv1 = nn.Conv2d(3, n_cov1, n_kernel, padding = 1, stride=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(n_cov1, n_cov2, n_kernel, padding = 1, stride=1)

        # neurall network
        # torch.nn.Linear(in_features, out_features, bias=True)
        # self.fc1 = nn.Linear(n_cov2 * n_kernel * n_kernel, n_hid1)
        # self.fc2 = nn.Linear(n_hid1, n_hid2)
        # self.fc3 = nn.Linear(n_hid2, 10)

        self.arch = nn.Sequential(
                nn.Conv2d(3, 48, 5, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(48, 128, 5, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128, 192, 3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(192, 192, 3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(192, 128, 3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
                )
        
        self.fc1 = nn.Linear(128*9, self.n_hid1)
        self.fc2 = nn.Linear(self.n_hid1, self.n_hid2)
        self.fc3 = nn.Linear(self.n_hid2, 10)


    def forward(self, x):
        x = self.arch(x)
        # tensor.view is similar to np.reshape
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# method 2
# net2 = nn.Sequential(
#     nn.Linear(2, 10),
#     nn.ReLU(),
#     nn.Linear(10,2),
# )

# model 2
class Net_2(nn.Module):
    def __init__(self, n_cov1=6, n_cov2=16, n_cov3=32, n_kernel1=3, n_kernel2=3, n_kernel3=3, n_hid1=64, n_hid2=128):
        super(Net_2, self).__init__()
        self.n_cov1 = n_cov1
        self.n_cov2 = n_cov2
        self.n_cov3 = n_cov3
        self.n_kernel1 = n_kernel1
        self.n_kernel2 = n_kernel2
        self.n_kernel3 = n_kernel3
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.feature = numfeature([n_kernel1, n_kernel2, n_kernel3])

        self.pool = nn.MaxPool2d(2, 2)

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, self.n_cov1, self.n_kernel1, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(self.n_cov1)
        
        self.conv2 = nn.Conv2d(self.n_cov1, self.n_cov2, self.n_kernel2, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(self.n_cov2)

        self.conv3 = nn.Conv2d(self.n_cov2, self.n_cov3, self.n_kernel3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(self.n_cov3)

        
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(self.feature * self.feature * self.n_cov3, self.n_hid1)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm2d(self.n_hid1)

        self.fc2 = nn.Linear(self.n_hid1, self.n_hid2)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm2d(self.n_hid2)

        self.fc3 = nn.Linear(self.n_hid2, 10)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))

        # leak relu
        # x = self.pool(F.leaky_relu(self.conv1_bn(self.conv1(x))))
        # x = self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x))))
        # x = self.pool(F.leaky_relu(self.conv3_bn(self.conv3(x))))

        x = x.view(-1, self.feature * self.feature * self.n_cov3)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

