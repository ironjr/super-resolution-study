###
# data.py
#
# Jaerin Lee
# ECE, Seoul National University
###

import torch
import torch.utils.data as data

class SRDataset(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0