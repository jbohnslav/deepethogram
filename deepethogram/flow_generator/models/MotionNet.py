"""Re-implementation of the MotionNet architecture

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

import torch.nn as nn
from torch.nn import init

from .components import CropConcat, conv, deconv, i_conv, predict_flow

log = logging.getLogger(__name__)


class MotionNet(nn.Module):
    def __init__(self, num_images=11, batchNorm=True, flow_div=1):
        super(MotionNet, self).__init__()

        self.num_images = num_images
        self.out_channels = int((num_images - 1) * 2)
        self.batchNorm = batchNorm

        log.debug("ignoring flow div value of {}: setting to 1 instead".format(flow_div))
        self.flow_div = 1

        self.conv1 = conv(self.batchNorm, self.num_images * 3, 64)
        self.conv1_1 = conv(self.batchNorm, 64, 64)

        self.conv2 = conv(self.batchNorm, 64, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)

        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)

        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)

        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)

        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1024 + self.out_channels, 256)
        self.deconv3 = deconv(768 + self.out_channels, 128)
        self.deconv2 = deconv(384 + self.out_channels, 64)

        self.xconv5 = i_conv(self.batchNorm, 1024 + self.out_channels, 512)
        self.xconv4 = i_conv(self.batchNorm, 768 + self.out_channels, 256)
        self.xconv3 = i_conv(self.batchNorm, 384 + self.out_channels, 128)
        self.xconv2 = i_conv(self.batchNorm, 192 + self.out_channels, 64)

        self.predict_flow6 = predict_flow(1024, out_planes=self.out_channels)
        self.predict_flow5 = predict_flow(512, out_planes=self.out_channels)
        self.predict_flow4 = predict_flow(256, out_planes=self.out_channels)
        self.predict_flow3 = predict_flow(128, out_planes=self.out_channels)
        self.predict_flow2 = predict_flow(64, out_planes=self.out_channels)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 4, 2, 1)
        self.concat = CropConcat(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        N, C, H, W = x.shape
        # 1 -> 1
        out_conv1 = self.conv1_1(self.conv1(x))
        # 1 -> 1/2
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        # 1/2 -> 1/4
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        # 1/4 -> 1/8
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # 1/8 -> 1/16
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # 1/16 -> 1/32
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6) * self.flow_div
        # EXPLANATION FOR MULTIPLYING BY 2
        # when reconstructing t0_est from t1 and the flow, our reconstructor normalizes by the width of the flow.
        # a 256x256 image / 32 = 8x8
        # a value of 1 in flow6 will be divided by (WIDTH / 2), because the top-left corner is (-1, -1) and the
        # top-right is (1, -1). so a value of 2 is 100% of the image. dividing by (WIDTH / 2) ensures that a raw pixel
        # value of IMAGE_WIDTH gets mapped to 2!
        # So if flow6 has a value of 8, the calculation is: 8 / (8/2) = 2, so a flow that moves all the way across the
        # image will be mapped to a value of 2, as expected above. SO
        # a value of 1 in flow6: 1 / (8 / 2) = 0.25, which corresponds to 1/8 of the image, or one pixel.
        # in the next line of code, we will upsample flow6 by 2, to a size of 16x16
        # a value of 1 in flow6 will naively be mapped to a value of 1 in flow5. now, this movement of 1 pixel no
        # longer means 1/8 of the image, it will only move 1/16 of the image. So to correct for this, we multiply
        # the upsampled version by 2.
        flow6_up = self.upsampled_flow6_to_5(flow6) * 2
        out_deconv5 = self.deconv5(out_conv6)

        # if the image sizes are not divisible by 8, there will be rounding errors in the size
        # between the downsampling and upsampling phases
        concat5 = self.concat((out_conv5, out_deconv5, flow6_up))
        out_interconv5 = self.xconv5(concat5)
        flow5 = self.predict_flow5(out_interconv5) * self.flow_div

        flow5_up = self.upsampled_flow5_to_4(flow5) * 2
        out_deconv4 = self.deconv4(concat5)

        concat4 = self.concat((out_conv4, out_deconv4, flow5_up))
        out_interconv4 = self.xconv4(concat4)
        flow4 = self.predict_flow4(out_interconv4) * self.flow_div
        flow4_up = self.upsampled_flow4_to_3(flow4) * 2
        out_deconv3 = self.deconv3(concat4)

        # if the image sizes are not divisible by 8, there will be rounding errors in the size
        # between the downsampling and upsampling phases
        concat3 = self.concat((out_conv3, out_deconv3, flow4_up))
        out_interconv3 = self.xconv3(concat3)
        flow3 = self.predict_flow3(out_interconv3) * self.flow_div
        flow3_up = self.upsampled_flow3_to_2(flow3) * 2
        out_deconv2 = self.deconv2(concat3)

        concat2 = self.concat((out_conv2, out_deconv2, flow3_up))
        out_interconv2 = self.xconv2(concat2)
        flow2 = self.predict_flow2(out_interconv2) * self.flow_div

        return flow2, flow3, flow4
