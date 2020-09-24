# this entire page was copied from torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/models
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, dropout_p=0):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3,batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, in_channels=3,**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg11'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(pretrained=False, in_channels=3,**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels, batch_norm=True), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg13(pretrained=False, in_channels=3,**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], in_channels=in_channels), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg13'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg13_bn(pretrained=False,in_channels=3,**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], in_channels=in_channels,batch_norm=True), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg13_bn'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, in_channels=3,**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'],in_channels=in_channels), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg16'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg16_bn(pretrained=False,in_channels=3,**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'],in_channels=in_channels,batch_norm=True), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained=False,in_channels=3, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'],in_channels=in_channels), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg19'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def vgg19_bn(pretrained=False, in_channels=3,**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], in_channels=in_channels,batch_norm=True), **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        if in_channels !=3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels,1,1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model