###
# vdsr.py
#
# Jaerin Lee
# ECE, Seoul National University
###

import torch
import torch.nn as nn
import torch.nn.functional as F


class VDSR(nn.Module):

    def __init__(self, imchannel=3, imsize=32, nfilter=64, nlayer=20):
        """
        Args:

        """
        super().__init__()

        if nlayer <= 2:
            print("VDSR: nlayer should be larger than 2")
            exit(0)

        # First layer operates on the input image
        self.first = nn.Sequential(
            nn.Conv2d(imchannel, nfilter, 3, stride=1, padding=1),
            nn.ReLU()
        )

        # Middle layers consists of identical filters
        self.vgg = nn.ModuleList()
        for _ in range(nlayer - 2):
            self.vgg.append(nn.Sequential(
                nn.Conv2d(nfilter, nfilter, 3, stride=1, padding=1),
                nn.ReLU()
            ))

        # Last layer has a single filter
        self.last = nn.Conv2d(imchannel, nfilter, 3, stride=1, padding=1)

        # Initialization based on Kaiming normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = self.first(x)
        for layer in self.vgg:
            res = layer(res)
        res = self.last(res)
        out = x + res
        return out

