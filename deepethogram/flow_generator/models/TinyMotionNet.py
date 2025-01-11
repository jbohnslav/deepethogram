""" Re-implementation of the TinyMotionNet architecture

References
-------
.. [1]: Zhu, Lan, Newsam, and Hauptman. Hidden Two-stream convolutional networks for action recognition.
        https://arxiv.org/abs/1704.00389

Based on code from Nvidia's FlowNet2
Copyright 2017 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Changes:  changed filter sizes, number of input images, number of layers, added cropping or interpolation for
non-power-of-two shaped images, and multiplication... only kept their naming convention and overall structure
"""
import logging
# import warnings

from .components import *

log = logging.getLogger(__name__)


# modified from https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetSD.py
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/submodules.py
class TinyMotionNet(nn.Module):
    def __init__(self, num_images=11, input_channels=None, batchNorm=True, output_channels=None, flow_div=1):
        super().__init__()
        self.num_images = num_images
        if input_channels is None:
            self.input_channels = self.num_images * 3
        else:
            self.input_channels = int(input_channels)
        if output_channels is None:
            self.output_channels = int((num_images - 1) * 2)
        else:
            self.output_channels = int(output_channels)

        # self.out_channels = int((num_images-1)*2)
        self.batchNorm = batchNorm
        log.debug("ignoring flow div value of {}: setting to 1 instead".format(flow_div))
        self.flow_div = 1

        self.conv1 = conv(self.batchNorm, self.input_channels, 64, kernel_size=7)
        self.conv2 = conv(self.batchNorm, 64, 128, stride=2, kernel_size=5)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv4 = conv(self.batchNorm, 256, 128, stride=2)

        self.deconv3 = deconv(128, 128)
        self.deconv2 = deconv(128, 64)

        self.xconv3 = i_conv(self.batchNorm, 384 + self.output_channels, 128)
        self.xconv2 = i_conv(self.batchNorm, 192 + self.output_channels, 64)

        self.predict_flow4 = predict_flow(128, out_planes=self.output_channels)
        self.predict_flow3 = predict_flow(128, out_planes=self.output_channels)
        self.predict_flow2 = predict_flow(64, out_planes=self.output_channels)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 4, 2, 1)

        self.concat = CropConcat(dim=1)
        self.interpolate = Interpolate

    def forward(self, x):
        N, C, H, W = x.shape
        out_conv1 = self.conv1(x)  # 1 -> 1
        out_conv2 = self.conv2(out_conv1)  # 1 -> 1/2
        out_conv3 = self.conv3(out_conv2)  # 1/2 -> 1/4
        out_conv4 = self.conv4(out_conv3)  # 1/4 -> 1/8

        flow4 = self.predict_flow4(out_conv4) * self.flow_div
        # see motionnet.py for explanation of multiplying by 2
        flow4_up = self.upsampled_flow4_to_3(flow4) * 2
        out_deconv3 = self.deconv3(out_conv4)

        concat3 = self.concat((out_conv3, out_deconv3, flow4_up))
        out_interconv3 = self.xconv3(concat3)
        flow3 = self.predict_flow3(out_interconv3) * self.flow_div
        flow3_up = self.upsampled_flow3_to_2(flow3) * 2
        out_deconv2 = self.deconv2(out_interconv3)

        concat2 = self.concat((out_conv2, out_deconv2, flow3_up))
        out_interconv2 = self.xconv2(concat2)
        flow2 = self.predict_flow2(out_interconv2) * self.flow_div

        # flow1 = F.interpolate(flow2, (H, W), mode='bilinear', align_corners=False) * 2
        # flow2*=self.flow_div
        # flow3*=self.flow_div
        # flow4*=self.flow_div
        # import pdb
        # pdb.set_trace()

        # if self.training:
        #     return flow1, flow2, flow3, flow4
        # else:
        #     return flow1,
        return flow2, flow3, flow4
