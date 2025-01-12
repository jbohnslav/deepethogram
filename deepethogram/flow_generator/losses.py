"""Loss functions for optical flow estimation and motion prediction.

This module provides various loss functions used in optical flow estimation and motion prediction,
including SSIM loss, gradient-based losses, and smoothness terms. The main class MotionNetLoss
combines these components for training motion prediction networks.
"""

import logging
from math import exp

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def gaussian(window_size, sigma):
    """Create a 1D Gaussian window.

    Implementation based on: https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create a 2D Gaussian window for SSIM calculation.

    Implementation based on: https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor((_2D_window.expand(channel, 1, window_size, window_size).contiguous()))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculate SSIM between two images.

    Implementation based on: https://github.com/Po-Hsun-Su/pytorch-ssim

    Args:
        img1: First image tensor
        img2: Second image tensor
        window: Gaussian window for SSIM calculation
        window_size: Size of the Gaussian window
        channel: Number of channels in the images
        size_average: If True, average SSIM across spatial dimensions

    Returns:
        Tensor containing SSIM value(s)
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class SSIMLoss(torch.nn.Module):
    """SSIM loss module for comparing structural similarity between images.

    Args:
        window_size: Size of the Gaussian window for SSIM calculation
        size_average: If True, average loss across spatial dimensions
        denominator: Scaling factor for the final loss value
    """

    def __init__(self, window_size=11, size_average=True, denominator=2):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.denominator = denominator

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        similarity = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return (1 - similarity) / self.denominator


# PyTorch is NCHW
def gradient_x(img, mode="constant"):
    """Calculate horizontal gradients of an image.

    Args:
        img: Input image tensor in NCHW format
        mode: Padding mode for boundary handling

    Returns:
        Tensor containing horizontal gradients
    """
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return F.pad(gx, (0, 1, 0, 0), mode=mode)


def gradient_y(img, mode="constant"):
    """Calculate vertical gradients of an image.

    Args:
        img: Input image tensor in NCHW format
        mode: Padding mode for boundary handling

    Returns:
        Tensor containing vertical gradients
    """
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return F.pad(gy, (0, 0, 0, 1), mode=mode)


def get_gradients(img):
    """Calculate both horizontal and vertical gradients of an image."""
    gx = gradient_x(img)
    gy = gradient_y(img)
    return gx + gy


def SSIM_simple(x, y):
    """Calculate a simplified version of SSIM using average pooling instead of Gaussian kernels."""
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)

    sigma_x = F.avg_pool2d(x**2, kernel_size=3, stride=1, padding=0) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, kernel_size=3, stride=1, padding=0) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM_full = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM_full), min=0, max=1)


def total_generalized_variation(image, flow):
    """Calculate total generalized variation between image and flow fields."""
    flowx = flow[:, 0:1, ...]
    flowy = flow[:, 1:, ...]

    gx2_image = gradient_x(gradient_x(image))
    gy2_image = gradient_y(gradient_y(image))

    gx2_flowx = gradient_x(gradient_x(flowx))
    gy2_flowx = gradient_y(gradient_y(flowx))

    gx2_flowy = gradient_x(gradient_x(flowy))
    gy2_flowy = gradient_y(gradient_y(flowy))

    TGV = (
        torch.abs(gx2_flowx) * torch.exp(-torch.abs(gx2_image))
        + torch.abs(gy2_flowx) * torch.exp(-torch.abs(gy2_image))
        + torch.abs(gx2_flowy) * torch.exp(-torch.abs(gx2_image))
        + torch.abs(gy2_flowy) * torch.exp(-torch.abs(gy2_image))
    )
    return TGV


def smoothness_firstorder(image, flow):
    """Calculate first-order smoothness term weighted by image gradients."""
    flow_gradients_x = gradient_x(flow)
    flow_gradients_y = gradient_y(flow)

    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)

    # take the absolute of the image gradients
    # take the mean across channels, so now you have a [N,1,H,W] tensor
    # negative sign
    # raise e^(result)
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(flow_gradients_x) * weights_x
    smoothness_y = torch.abs(flow_gradients_y) * weights_y
    return smoothness_x, smoothness_y


def charbonnier(tensor, alpha=0.4, eps=1e-4):
    """Apply Charbonnier penalty function."""
    return (tensor * tensor + eps * eps) ** alpha


def charbonnier_smoothness(flows, alpha=0.3, eps=1e-7):
    """Calculate smoothness term using Charbonnier penalty on flow gradients."""
    return charbonnier(gradient_x(flows), alpha=alpha, eps=eps) + charbonnier(gradient_y(flows), alpha=alpha, eps=eps)


class MotionNetLoss(torch.nn.Module):
    """Combined loss function for motion prediction networks.

    Combines reconstruction loss (L1 + SSIM), smoothness terms, and optional flow sparsity.
    Supports multi-scale predictions with different weights at each scale.

    Args:
        regularization_criterion: Loss function for model weight regularization
        is_multiscale: Whether to compute loss at multiple scales
        smooth_weights: List of weights for smoothness terms at each scale
        highres: Whether to add an additional high-resolution scale
        calculate_ssim_full: Whether to compute SSIM at full resolution
        flow_sparsity: Whether to add sparsity penalty on flow predictions
        sparsity_weight: Weight for the flow sparsity term
        smooth_weight_multiplier: Global multiplier for smoothness weights
    """

    def __init__(
        self,
        regularization_criterion,
        is_multiscale=True,
        smooth_weights=[0.01, 0.02, 0.04, 0.08, 0.16],
        highres: bool = False,
        calculate_ssim_full: bool = False,
        flow_sparsity: bool = False,
        sparsity_weight: float = 1.0,
        smooth_weight_multiplier: float = 1.0,
    ):
        super(MotionNetLoss, self).__init__()
        self.smooth_weights = [i * smooth_weight_multiplier for i in smooth_weights]
        if highres:
            self.smooth_weights.insert(0, 0.005)
        self.is_multiscale = is_multiscale
        self.ssim = SSIMLoss(size_average=False, denominator=1)
        self.calculate_ssim_full = calculate_ssim_full
        self.flow_sparsity = flow_sparsity
        self.sparsity_weight = sparsity_weight
        log.info(
            "Using MotionNet Loss with settings: smooth_weights: {} flow_sparsity: {} sparsity_weight: {}".format(
                self.smooth_weights, flow_sparsity, sparsity_weight
            )
        )
        self.regularization_criterion = regularization_criterion

    def forward(self, originals, images, reconstructed, outputs, model: torch.nn.Module):
        """Compute the combined loss.

        Args:
            originals: Original input images
            images: Target images at multiple scales
            reconstructed: Reconstructed images at multiple scales
            outputs: Flow predictions at multiple scales
            model: The network model for regularization

        Returns:
            tuple: (total_loss, dict of individual loss components)
        """
        if type(images) is not tuple:
            images = images
        if type(reconstructed) is not tuple:
            reconstructed = reconstructed

        if outputs[0].size(0) != images[0].size(0):
            raise ValueError("Image shape: ", images[0].shape, "Flow shape:", outputs[0].shape)
        if self.is_multiscale:
            # handle validation case where you only output one scale
            if len(images) == 1:
                weights = [self.smooth_weights[0]]
            elif len(images) == len(self.smooth_weights):
                weights = self.smooth_weights
            elif len(images) < len(self.smooth_weights):
                weights = self.smooth_weights[0 : len(images)]
            else:
                raise ValueError(
                    "Incorrect number of multiscale outputs: %d. Expected %d" % (len(images), len(self.smooth_weights))
                )
        else:
            weights = [1]
        # Components of image loss!
        # Calculate SSIM loss, L1 loss, and gradient loss for each image in the scale pyramid
        # Weight each of these 3 by the values in 'unary_weights'
        # Weight each of them by the first value image_smooth_weights
        # That way, we can know the real value of each component for every image in batch, rather
        # than weight at the end while we're summing all the components up

        L1s = [charbonnier(real - estimated, alpha=0.4) for real, estimated in zip(images, reconstructed)]
        # shape: ([batch_size], [batch_size], batch_size,...) per scale
        L1_mean = [torch.mean(i, dim=[1, 2, 3]) for i in L1s]

        smooths = [charbonnier_smoothness(output) for output in outputs]
        smooth_mean = [torch.mean(i, dim=[1, 2, 3]) * weight for i, weight in zip(smooths, weights)]

        SSIMs = [self.ssim(real, estimated) for real, estimated in zip(images, reconstructed)]
        SSIM_mean = [torch.mean(i, dim=[1, 2, 3]) for i in SSIMs]

        if self.flow_sparsity:
            # use the same smoothness loss
            flow_l1s = [torch.mean(torch.abs(i), dim=[1, 2, 3]) * self.sparsity_weight for i in outputs]

        # Note: adding a full-size SSIM for metrics only!
        if self.calculate_ssim_full:
            with torch.no_grad():
                N, C, H, W = originals.shape
                num_images = int(C / 3) - 1
                recon_h, recon_w = reconstructed[0].size(-2), reconstructed[0].size(-1)

                if H != recon_h or W != recon_w:
                    t0 = originals[:, : num_images * 3, ...].contiguous().view(N * num_images, 3, H, W)
                    recon = reconstructed[0]
                    recon_fullsize = F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    t0 = images[0]
                    recon_fullsize = reconstructed[0]
                SSIM_full = self.ssim(t0, recon_fullsize)
                SSIM_full_mean = SSIM_full.mean(dim=[1, 2, 3])
        else:
            SSIM_full_mean = torch.from_numpy(np.array([np.nan]))

        # Sum across pyramid scales for each loss component!
        # shape: (batch_size,)
        SSIM_per_image = torch.stack(SSIM_mean).sum(dim=0)
        L1_per_image = torch.stack(L1_mean).sum(dim=0)
        smoothness_per_image = torch.stack(smooth_mean).sum(dim=0)

        regularization_loss = self.regularization_criterion(model)

        loss_components = {
            "reg_loss": regularization_loss.detach(),
            "SSIM": SSIM_per_image.detach(),
            "L1": L1_per_image.detach(),
            "smoothness": smoothness_per_image.detach(),
            "SSIM_full": SSIM_full_mean.detach(),
        }
        # mean across batch elements
        loss = (
            torch.mean(SSIM_per_image)
            + torch.mean(L1_per_image)
            + torch.mean(smoothness_per_image)
            + regularization_loss
        )

        if loss != loss:
            msg = "Loss is NaN, re-run training with pytorch debug mode enabled"
            raise ValueError(msg)

        if self.flow_sparsity:
            # sum across scales
            flow_sparsity = torch.stack(flow_l1s).sum(dim=0)
            # mean across batch
            loss += torch.mean(flow_sparsity)
            loss_components["flow_sparsity"] = flow_sparsity.detach()
            del flow_l1s

        del (SSIMs, SSIM_mean, L1s, L1_mean, smooths, smooth_mean)
        return loss, loss_components
