import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight.to(device), bias.to(device), stride, padding, dilation, groups)


def relu(input):
    return F.threshold(input, 0, 0, inplace=True)


def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)


def bilinear_upsample(in_, factor):
    return F.upsample(in_, None, factor, 'bilinear')


def log_softmax(input):
    return F.log_softmax(input)


def prepare_network(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for layer_value in cfg:
        if layer_value == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, layer_value, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(layer_value), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = layer_value
    return nn.Sequential(*layers)
