'''
model ensemble for cifar10  // input size(32,32)
'''


import torch
import torchvision

import copy
import torch.nn as nn
from resnetv2 import ResNet50 as resnet50v2


def split_resnet50(model):
    return nn.Sequential(
        model.conv1,
        model.layer1,
        model.layer2,
        model.layer3
    )


class PretrainEnsembleModel(nn.Module):

    def __init__(self):

        super(PretrainEnsembleModel, self).__init__()

        self.blocks = split_resnet50(resnet50v2())
        self.layer4_rotation = resnet50v2().layer4 
        self.layer4_jigsaw = resnet50v2().layer4

        self.fc_rotation = nn.Linear(2048, 4)
        self.fc_jigsaw = nn.Linear(2048, 31)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))


    def _Normal(self,x):
        mean=torch.Tensor([0.485, 0.456, 0.406])
        mean=mean[None,:,None,None].cuda()
        std = torch.Tensor([0.229, 0.224, 0.225])
        std = std[None,:,None,None].cuda()
        return x.sub(mean).div(std)

    def forward(self, x):

        feature_map = self.blocks(self._Normal(x))

        return feature_map
