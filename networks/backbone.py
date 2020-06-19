"""
Script for CSRNet
Code borrowed from https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/model.py
"""

import torchvision

from .functional_layers import *


class CSRMetaNetwork(nn.Module):
    def __init__(self, loss_function, pre_trained=True):
        super(CSRMetaNetwork, self).__init__()

        self.loss_function = loss_function
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = prepare_network(self.frontend_feat)
        self.backend = prepare_network(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        if pre_trained:
            mod = torchvision.models.vgg16(pretrained=True)
            for i in range(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, weights=None):
        x = self.frontend(x)
        if weights is None:
            x = self.backend(x)
            x = self.output_layer(x)
        else:
            # x = conv2d(x, weights['frontend.0.weight'], weights['frontend.0.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.2.weight'], weights['frontend.2.bias'], padding=1)
            # x = relu(x)
            # x = maxpool(x, kernel_size=2, stride=2)
            # x = conv2d(x, weights['frontend.5.weight'], weights['frontend.5.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.7.weight'], weights['frontend.7.bias'], padding=1)
            # x = relu(x)
            # x = maxpool(x, kernel_size=2, stride=2)
            # x = conv2d(x, weights['frontend.10.weight'], weights['frontend.10.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.12.weight'], weights['frontend.12.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.14.weight'], weights['frontend.14.bias'], padding=1)
            # x = relu(x)
            # x = maxpool(x, kernel_size=2, stride=2)
            # x = conv2d(x, weights['frontend.17.weight'], weights['frontend.17.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.19.weight'], weights['frontend.19.bias'], padding=1)
            # x = relu(x)
            # x = conv2d(x, weights['frontend.21.weight'], weights['frontend.21.bias'], padding=1)
            # x = relu(x)
            x = conv2d(x, weights['backend.0.weight'], weights['backend.0.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['backend.2.weight'], weights['backend.2.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['backend.4.weight'], weights['backend.4.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['backend.6.weight'], weights['backend.6.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['backend.8.weight'], weights['backend.8.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['backend.10.weight'], weights['backend.10.bias'], dilation=2, padding=2)
            x = relu(x)
            x = conv2d(x, weights['output_layer.weight'], weights['output_layer.bias'])
        x = F.upsample(x, scale_factor=8)
        return x

    def network_forward(self, x, weights=None):
        return self.forward(x, weights)

    def copy_weights(self, network):
        for m_from, m_to in zip(network.modules(), self.modules()):
            if isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
