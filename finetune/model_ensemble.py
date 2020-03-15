'''
model ensemble for cifar10  // input size(32,32)
'''

import torch
import torchvision

import copy
import torch.nn as nn

from resnetv2 import ResNet50 as resnet50v2
from advertorch.utils import NormalizeByChannelMeanStd
from functools import reduce


class SelfEnsembleModel_all(nn.Module):

    def __init__(self, num_of_branches, num_classes = 10):

        super(SelfEnsembleModel_all, self).__init__()
        self.num_of_branches = num_of_branches
        self.branch1 = resnet50v2()
        self.branch2 = resnet50v2()
        self.branch3 = resnet50v2()
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load(self, files):
        state_dict = self.branch1.state_dict()
        new_state_dict = torch.load(files[0])['state_dict']
        state_dict.update(new_state_dict)
        self.branch1.load_state_dict(state_dict)

        state_dict = self.branch2.state_dict()
        new_state_dict = torch.load(files[1])['state_dict']
        state_dict.update(new_state_dict)
        self.branch2.load_state_dict(state_dict)

        state_dict = self.branch3.state_dict()
        new_state_dict = torch.load(files[2])['state_dict']
        state_dict.update(new_state_dict)
        self.branch3.load_state_dict(state_dict)

    def forward(self, x):

        out1 = self.branch1(self.normalize(x[0]))
        out2 = self.branch2(self.normalize(x[1]))
        out3 = self.branch3(self.normalize(x[2]))

        logit1 = torch.softmax(out1, dim=1)
        logit2 = torch.softmax(out2, dim=1)
        logit3 = torch.softmax(out3, dim=1)

        output = (logit1+logit2+logit3)/3
        
        return output

