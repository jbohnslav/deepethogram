"""
modified from here
# https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
# MIT License
#
# Copyright (c) 2017 Clément Pinard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

References
-------
.. [1]: Fischer et al. FlowNet: Learning optical flow with convolutional networks. ICCV 2015
        https://arxiv.org/abs/1504.06852
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .components import conv, deconv, get_hw, predict_flow


class FlowNetS(nn.Module):
    def __init__(self, num_images=2, batchNorm=True, flow_div=1):
        super(FlowNetS, self).__init__()
        self.flow_div = flow_div
        input_channels = num_images * 3
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

        self.upsample1 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6) * self.flow_div

        # EXPLANATION FOR MULTIPLYING BY 2
        # when reconstructing t0_est from t1 and the flow, our reconstructor normalizes by the width of the flow.
        # a 256x256 image / 32 = 8x8
        # a value of 1 in flow6 will be divided by (WIDTH / 2), because the top-left corner is (-1, -1) and the
        # top-right is (1, -1). so a value of 2 is 100% of the image. dividing by (WIDTH / 2) ensures that a value of
        # So if flow6 has a value of 8, the calculation is: 8 / (8/2) = 2, so a flow that moves all the way across the
        # image will be mapped to a value of 2, as expected above. SO
        # a value of 1 in flow6: 1 / (8 / 2) = 0.25, which corresponds to 1/8 of the image, or one pixel.
        # in the next line of code, we will upsample flow6 by 2, to a size of 16x16
        # a value of 1 in flow6 will naively be mapped to a value of 1 in flow5. now, this movement of 1 pixel no
        # longer means 1/8 of the image, it will only move 1/16 of the image. So to correct for this, we multiply
        # the upsampled version by 2.
        flow6_up = self.upsampled_flow6_to_5(flow6) * 2
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5) * self.flow_div
        flow5_up = self.upsampled_flow5_to_4(flow5) * 2
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4) * self.flow_div
        flow4_up = self.upsampled_flow4_to_3(flow4) * 2
        out_deconv3 = self.deconv3(concat4)

        if get_hw(out_conv3) != get_hw(out_deconv3):
            out_conv3 = F.interpolate(out_conv3, size=get_hw(out_deconv3), mode="bilinear", align_corners=False)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3) * self.flow_div
        flow3_up = self.upsampled_flow3_to_2(flow3) * 2
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2) * self.flow_div

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return (flow2,)
