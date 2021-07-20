from typing import List, Any, Tuple

import torch.nn as nn
import torch

from pytorch_resnet_cifar10.resnet import ResNet, BasicBlock  # type: ignore


def weights_init_normal(m: 'nn.Module[torch.Tensor]') -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def resnet20(num_classes, num_channels):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, n_channels=num_channels)


def resnet20_prob_square(num_classes, num_channels):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, prob_square=True, n_channels=num_channels)


def resnet32(num_classes, num_channels):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, n_channels=num_channels)


def resnet44(num_classes, num_channels):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, n_channels=num_channels)

