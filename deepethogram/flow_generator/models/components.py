from typing import Union

import torch
import torch.nn as nn


def conv(batchNorm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
    """Convenience function for conv2d + optional BN + leakyRELU"""
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def crop_like(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Crops input to target's H,W"""
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def deconv(in_planes: int, out_planes: int, bias: bool = True):
    """Convenience function for ConvTranspose2d + leakyRELU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=bias),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Interpolate(nn.Module):
    """Wrapper to be able to perform interpolation in a nn.Sequential

    Modified from the PyTorch Forums:
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2
    """

    def __init__(self, size=None, scale_factor=None, mode: str = "bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        assert mode in ["nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]
        self.mode = mode
        if self.mode == "nearest":
            self.align_corners = None
        else:
            self.align_corners = False

    def forward(self, x):
        x = self.interp(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


def i_conv(batchNorm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
    """Convenience function for conv2d + optional BN + no activation"""
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias
            ),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias
            ),
        )


def predict_flow(in_planes: int, out_planes: int = 2, bias: bool = False):
    """Convenience function for 3x3 conv2d with same padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


def get_hw(tensor):
    """Convenience function for getting the size of the last two dimensions in a tensor"""
    return tensor.size(-2), tensor.size(-1)


class CropConcat(nn.Module):
    """Module for concatenating 2 tensors of slightly different shape."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: tuple) -> torch.Tensor:
        assert type(tensors) == tuple
        hs, ws = [tensor.size(-2) for tensor in tensors], [tensor.size(-1) for tensor in tensors]
        h, w = min(hs), min(ws)

        return torch.cat(tuple([tensor[..., :h, :w] for tensor in tensors]), dim=self.dim)


def conv3d(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, tuple] = 3,
    stride: Union[int, tuple] = 1,
    bias: bool = True,
    batchnorm: bool = True,
    act: bool = True,
    padding: tuple = None,
):
    """3D convolution

    Expects inputs of shape N, C, D/F/T, H, W.
    D/F/T is frames, depth, time-- the extra axis compared to 2D convolution.
    Returns output of shape N, C_out, D/F/T_out, H_out, W_out.
    Out shape will be determined by input parameters. for more information see PyTorch docs
    https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html

    Args:
        in_planes: int
            Number of channels in input tensor.
        out_planes: int
            Number of channels in output tensor
        kernel_size: int, tuple
            Size of 3D convolutional kernel. in order of (D/F/T, H, W). If int, size is repeated 3X
        stride: int, tuple
            Stride of convolutional kernel in D/F/T, H, W order
        bias: bool
            if True, adds a bias parameter
        batchnorm: bool
            if True, adds batchnorm 3D
        act: bool
            if True, adds LeakyRelu after (optional) batchnorm
        padding: int, tuple
            padding in T, H, W. If int, repeats 3X. if none, "same" padding, so that the inputs are the same shape
            as the outputs (assuming stride 1)
    Returns:
        nn.Sequential with conv3d, (batchnorm), (activation function)
    """
    modules = []
    if padding is None and type(kernel_size) == int:
        padding = (kernel_size - 1) // 2
    elif padding is None and type(kernel_size) == tuple:
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
    else:
        raise ValueError("Unknown padding type {} and kernel_size type: {}".format(padding, kernel_size))

    modules.append(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if batchnorm:
        modules.append(nn.BatchNorm3d(out_planes))
    if act:
        modules.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*modules)


def deconv3d(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 4,
    stride: int = 2,
    bias: bool = True,
    batchnorm: bool = True,
    act: bool = True,
    padding: int = 1,
):
    """Convenience function for ConvTranspose3D. Optionally adds batchnorm3d, leakyrelu"""
    modules = [
        nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    ]
    if batchnorm:
        modules.append(nn.BatchNorm3d(out_planes))
    if act:
        modules.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*modules)


def predict_flow_3d(in_planes: int, out_planes: int):
    """Convenience function for conv3d, 3x3, no activation or batchnorm"""
    return conv3d(in_planes, out_planes, kernel_size=3, stride=1, bias=True, act=False, batchnorm=False)
