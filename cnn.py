'''
Author: Siwei Guo
Reference:  http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn 
'''


import numpy as np
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import *

use_GPU = False
max_iter = 1000
holdF = 0.1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# import dataset
trainset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR-10', train=True, download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR-10', train=False, download=True, transform = transform)
val_idx, test_idx = splittrain(testset, holdF)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2, sampler=test_idx)
validationloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2, sampler=val_idx)

# validation set and training set

# train_input = trainset.train_data[train_idx, :, :, :]
# train_labels = [trainset.train_labels[j] for j in train_idx]

# val_input = trainset.train_data[val_idx, :, :, :]
# val_labels = [trainset.train_labels[j] for j in val_idx]
# # prepare for training set and validation set
# train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_input.reshape(44998, 3, 32, 32)), torch.from_numpy(np.array(train_labels)))
# val_set = torch.utils.data.TensorDataset(torch.from_numpy(val_input.reshape(5000, 3, 32, 32)), torch.from_numpy(np.array(val_labels)))

# trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
# validationloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)


#%%

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
    
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# initalize nn
# Net1(n_cov1, n_cov2, n_kernel, n_hid1, n_hid2)
# net = Net1(6, 12, 3, 100, 50)
net = Net_2(n_cov1=32, n_cov2=64, n_kernel1=3, n_kernel2=3, n_hid1=32, n_hid2=20)

# use GPU
if use_GPU:
    net.cuda()

# define lose function
criterion = nn.CrossEntropyLoss()

# define stochastic gradient descent, learning rate = 0.001, momentum = 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch = 0
stop = False
val_acc = []
# train dataset
#%%
while epoch <= max_iter & ~stop:  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # normalize input
        
        if use_GPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        # calculate loss function
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # gradient descentc
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # test on validation set to prevent overfitting
    correct = testnet(validationloader, net, use_GPU)
    print('Accy.: %f' % (correct))

    # validation set's accuracy decrease for three consequtive epoch, stop the training
    val_acc.append(correct)
    if epoch >= 4:
        if (val_acc[epoch] < val_acc[epoch-1]) and (val_acc[epoch-1] < val_acc[epoch-2]) and (val_acc[epoch-2] < val_acc[epoch-3]):

            stop = True

    epoch += 1




print('Finished Training')


# test dataepoch
correct_test = testnet(testloader, net, use_GPU)

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct))


# save network
# torch.save(net, 'net_alex.pkl') # save entire net
# torch.save(net.state_dict(), 'net_param.pkl')

# load network
# net1 = torch.load('net_alex.pkl')


# what class is predicted well
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data
#     # Input = torch.autograd.Variable(input) plug into net get prediction
#     outputs = net(Variable(images))
#     # outputs.data is a n x c tensor, n is number of dataset and c is number of class
#     # torch.max(outputs.data,1) produce the most likely class for each test data
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1


# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))outputs.data