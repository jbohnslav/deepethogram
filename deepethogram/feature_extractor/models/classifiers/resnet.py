# this entire page was edited from torchvision:
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
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from deepethogram.utils import load_state_from_dict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=1000, dropout_p=0, compress_to: int = 512):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = torch.jit.script(FastGlobalAvgPool2d(flatten=True))
        # self.adaptive_max = nn.AdaptiveMaxPool2d(1)


        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if compress_to is not None:
            # example: if you have a ResNet50, it has 2048 features before the fully connected layer
            # saving these features to disk for action detection would be extremely large in size
            # instead, reduce the features to 512
            if compress_to < 512 * block.expansion:
                self.compression_fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, compress_to),
                    nn.ReLU(inplace=True)
                )
                fc_infeatures = compress_to
                print('Altered from standard resnet50: instead of {} inputs to the fc layer, it has {}'.format(
                    512 * block.expansion, compress_to))
            else:
                self.compression_fc = nn.Identity()
                fc_infeatures = 512 * block.expansion
        else:
            self.compression_fc = nn.Identity()
            fc_infeatures = 512 * block.expansion

        self.fc = nn.Linear(fc_infeatures, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # uncomment below two lines for Resnet-D
                # nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=1, bias=False),
                # uncomment below line for vanilla resnet
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # should be in all model types V 
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.compression_fc(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        if in_channels != 3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels, 1, 1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        # model.load_state_dict(state_dict)
        load_state_from_dict(model, state_dict)
    return model


def resnet34(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        if in_channels != 3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels, 1, 1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        load_state_from_dict(model, state_dict)
    return model


def resnet50(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        if in_channels != 3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels, 1, 1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        load_state_from_dict(model, state_dict)
        # model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        if in_channels != 3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels, 1, 1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model


def resnet152(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, **kwargs)
    if pretrained:
        # from Wang et al. 2015: Towards good practices for very deep two-stream convnets
        state_dict = model_zoo.load_url(model_urls['resnet152'])
        if in_channels != 3:
            rgb_kernel_key = list(state_dict.keys())[0]
            rgb_kernel = state_dict[rgb_kernel_key]
            flow_kernel = rgb_kernel.mean(dim=1).unsqueeze(1).repeat(1, in_channels, 1, 1)
            state_dict[rgb_kernel_key] = flow_kernel
            state_dict.update(state_dict)
        model.load_state_dict(state_dict)
    return model
