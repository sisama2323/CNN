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
        # x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        # x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        # x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))

        # leak relu
        x = self.pool(F.leaky_relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.conv3_bn(self.conv3(x))))

        x = x.view(-1, self.feature * self.feature * self.n_cov3)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

# model 3
class Inception(nn.Module):
    '''
    Code reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
    Paper referece: https://arxiv.org/pdf/1409.4842.pdf
    '''
    def __init__(self, n_input, n_b1=6, n1_b2=3, n2_b2=6, n1_b3=3, n2_b3=6, n1_b4=6):
        super(Inception, self).__init__()
    
        # first branch
        self.b1 = nn.Sequential(
            nn.Conv2d(n_input, n_b1, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_b1),
            nn.ReLU(True)
        )

        # 2nd branch
        self.b2 = nn.Sequential(
            nn.Conv2d(n_input, n1_b2, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1_b2),
            nn.ReLU(True),
            nn.Conv2d(n1_b2, n2_b2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n2_b2),
            nn.ReLU(True)
        )

        # 3rd branch
        self.b3 = nn.Sequential(
            nn.Conv2d(n_input, n1_b3, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1_b3),
            nn.ReLU(True),
            nn.Conv2d(n1_b3, n2_b3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )

        # 4th branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(n_input, n1_b4, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1_b4),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        # print(x.size())
        x1 = self.b1(x)
        # print(x1.size())
        x2 = self.b2(x)
        # print(x2.size())
        x3 = self.b3(x)
        # print(x3.size())
        x4 = self.b4(x)
        # print(x4.size())
        return torch.cat([x1,x2,x3,x4], 1)

class GoogLeNet(nn.Module):
    '''
    Code reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
    Paper referece: https://arxiv.org/pdf/1409.4842.pdf
    '''
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.a1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        # after a1, size = [4, 192, 13, 13]
        # n_b1 + n2_b2 + n2_b3 + n1_b4
        self.a2 = Inception(n_input=192, n_b1=64, n1_b2=96, n2_b2=128, n1_b3=16, n2_b3=32, n1_b4=32)
        self.a3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.a5 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.a6 = Inception(512, 128, 128, 256, 24,  64,  64)
        
        self.avgpool = nn.AvgPool2d(4, stride=1)

        self.fc = nn.Linear(8192, 10)

    def forward(self, x):
        x = self.a1(x)
        x = self.a2(x)
        x = self.a3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.a5(x)
        x = self.a6(x)
        # print(x.size())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
