import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

# function plot images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# split the data into training set and validation set
def splittrain(trainset, holdF):
    val_offset = int(len(trainset)*holdF)
    rand_idx = np.random.permutation(len(trainset))
    val_idx = rand_idx[0:val_offset]
    train_idx = rand_idx[val_offset+1:-1]
    return val_idx, train_idx


def testnet(testdata, net, use_GPU = False):

    top1 = AverageMeter()
    top5 = AverageMeter()
    # correct = 0
    # total = 0
    for data in testdata:
        images, labels = data

        # GPU: pass variable to GPU
        if use_GPU:
            outputs = net(Variable(images).cuda())
            labels = labels.cuda()
        else:
            # wrap them in Variable
            outputs = net(Variable(images))

        # outputs.data is a n x c tensor, n is number of dataset and c is number of class
        # torch.max(outputs.data,1) produce the most likely class for each test data
        # _, predicted = torch.max(outputs.data, 1)
        
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1,5))

        top1.update(prec1[0], labels.size(0))
        top5.update(prec5[0], labels.size(0))

        # calculate the accuracy of prediction
        # print('Target:' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        # print('Target:' + ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

        # total += float(labels.size(0))
        # correct += float(sum(predicted == labels))

    return top1.avg, top5.avg


def numfeature(k):
    l = 32
    for i in k:
        l = (l+2-i+1)/2
    return l

# calculate 
class AverageMeter(object):
    '''
    Code Reference:https://github.com/pytorch/examples/blob/master/imagenet/main.py
    '''

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    '''
    Code Referece: ImageNet
    '''
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res