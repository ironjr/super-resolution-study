"""
util.py

Jaerin Lee
Seoul National University
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def get_mean_std():
    transform = T.Compose([
        T.ToTensor()
    ])

    cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
        transform=transform)
    cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
        transform=transform)

    # Training data
    print(cifar10_train.train_data.shape)
    print(cifar10_train.train_data.mean(axis=(0, 1, 2)) / 255)
    print(cifar10_train.train_data.std(axis=(0, 1, 2)) / 255)

    # Test data
    print(cifar10_test.test_data.shape)
    print(cifar10_test.test_data.mean(axis=(0, 1, 2)) / 255)
    print(cifar10_test.test_data.std(axis=(0, 1, 2)) / 255)


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + \
        len(group_no_decay)
    groups = [dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=.0)]

    return groups
