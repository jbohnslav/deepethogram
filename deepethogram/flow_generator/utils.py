import warnings
from typing import Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from deepethogram.utils import Normalizer


def flow_to_rgb(flow: np.ndarray, maxval: Union[int, float] = 20) -> np.ndarray:
    """Convert optic flow to RGB by linearly mapping X to red and Y to green

    255 in the resulting image will correspond to `maxval`. 0 corresponds to -`maxval`.
    Parameters
    ----------
    flow: np.ndarray. shape: (H, W, 2)
        optic flow. dX is the 0th channel, dy the 1st
    maxval: float
        the maximum representable value of the optic flow.

    Returns
    -------
    flow_map: np.ndarray. shape: (H, W, 3)
        RGB image
    """
    H, W, C = flow.shape
    assert C == 2
    flow = (flow + maxval) * (255 / 2 / maxval)
    flow = flow.clip(min=0, max=255)
    flow_map = np.ones((H, W, 3), dtype=np.uint8) * 127
    # X -> red channel
    flow_map[..., 0] = flow[..., 0]
    # Y -> green channel
    flow_map[..., 1] = flow[..., 1]
    # Z is all ones
    return flow_map


def rgb_to_flow(image: np.ndarray, maxval: Union[int, float] = 20):
    """Converts an RGB image to an optic flow by linearly mapping R -> X and G -> Y. Opposite of `flow_to_rgb`

    Parameters
    ----------
    image: np.ndarray. shape: (H, W, 3)
        uint8 RGB image
    maxval: float
        the maximum representable value of the optic flow.

    Returns
    -------
    image: optic flow. shape: (H, W, 2)
    """

    def denormalize(arr: np.ndarray):
        arr = arr / (255 / 2 / maxval) - maxval
        return arr

    H, W, C = image.shape
    assert C == 3
    image = image.astype(np.float32)
    image = denormalize(image)
    return image[..., 0:2]


def flow_to_rgb_polar(flow: np.ndarray, maxval: Union[int, float] = 20) -> np.ndarray:
    """Converts flow to RGB by mapping angle -> hue and magnitude -> saturation.

    Converts the flow map to polar coordinates: dX, dY -> angle, magnitude.
    Uses a HSV representation: Hue = angle, saturation = magnitude, value = 1
    Converts HSV to RGB

    Parameters
    ----------
    flow: np.ndarray. shape: (H, W, 2)
        optic flow. dX is the 0th channel, dy the 1st
    maxval: float
        the maximum representable value of the optic flow.

    Returns
    -------
    flow_map: np.ndarray. shape: (H, W, 3)
        RGB image
    """
    # check for float16
    flow = flow.astype(np.float32)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    # value is all 255
    hsv[:, :, 2] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag[np.isinf(mag)] = 0
    # angle -> hue
    hsv[..., 0] = ang * 180 / np.pi / 2
    # magnitue -> saturation
    color = (mag.astype(np.float32) / maxval).clip(0, 1)
    color = (color * 255).clip(0, 255).astype(np.uint8)
    # hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[..., 1] = color
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def rgb_to_flow_polar(image: np.ndarray, maxval: Union[int, float] = 20):
    """Converts rgb to flow by mapping hue -> angle and saturation -> magnitude.

    Inverse of `flow_to_rgb_polar`
    Parameters
    ----------
    image: np.ndarray. shape: (H, W, 3)
        RGB image
    maxval: float
        the maximum representable value of the optic flow.

    Returns
    -------
    flow: np.ndarray. shape: (H, W, 2)
        optic flow
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    mag = hsv[..., 1]
    mag = mag / 255
    mag *= maxval

    ang = hsv[..., 0]
    ang = ang * 2 * np.pi / 180
    # x,y = cv2.polarToCart(mag, ang)
    x = mag * np.cos(ang)
    y = mag * np.sin(ang)
    flow = np.stack((x, y), axis=2)
    return flow


# def flow_to_rgb_lrcn(flow, max_flow=10):
#     # input: flow, can be positive or negative
#     # ranges from -20 to 20, but only 10**-5 pixels are > 10
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     mag[np.isinf(mag)] = 0
#
#     img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
#     half_range = (255 - 128) / max_flow
#     img[:, :, 0] = flow[..., 0] * half_range + 128
#     img[:, :, 1] = flow[..., 1] * half_range + 128
#     # maximum magnitude is if x and y are both maxed
#     max_magnitude = np.sqrt(max_flow ** 2 + max_flow ** 2)
#     img[:, :, 2] = mag * 255 / max_magnitude
#     img = img.clip(min=0, max=255).astype(np.uint8)
#     return (img)


class Resample2d(torch.nn.Module):
    """Module to sample tensors using Spatial Transformer Networks. Caches multiple grids in GPU VRAM for speed.

    Examples
    -------
    model = MyOpticFlowNetwork()
    resampler = Resample2d(device='cuda:0')
    # flows map t0 -> t1
    flows = model(images)
    t0 = resampler(images, flows)

    model = MyStereoNetwork()
    resampler = Resample2d(device='cuda:0', horiz_only=True)
    disparity = model(left_images, right_images)
    """

    def __init__(
        self,
        size: Union[tuple, list] = None,
        fp16: bool = False,
        device: Union[str, torch.device] = None,
        horiz_only: bool = False,
        num_grids: int = 5,
    ):
        """Constructor for resampler.

        Parameters
        ----------
        size: tuple, list. shape: (2,)
            height and width of input tensors to sample
        fp16: bool
            if True, makes affine matrices in half precision
        device: str, torch.device
            device on which to store affine matrices and grids
        horiz_only: bool
            if True, only resample in the X dimension. Suitable for stereo matching problems
        num_grids: int
            Number of grids of different sizes to cache in GPU memory.

            A "Grid" is a tensor of shape H, W, 2, with (-1, -1) in the top-left to (1, 1) in the bottom-right. This
            is the location at which we will sample the input images. If we don't do anything to this grid, we will
            just return the original image. If we add our optic flow, we will sample the input image at at the locations
            specified by the optic flow.

            In many flow and stereo matching networks, images are warped at multiple resolutions, e.g.
            1/2, 1/4, 1/8, and 1/16 of the original resolution. To avoid making a new grid for sampling 4 times every
            time this is called, we will keep these 4 grids in memory. If there are more than `num_grids` resolutions,
            it will calculate and discard the top `num_grids` most frequently used grids.
        """
        super().__init__()
        if size is not None:
            assert type(size) == tuple or type(size) == list
        self.size = size

        # identity matrix
        self.base_mat = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        if fp16:
            # self.base_mat = self.base_mat.half()
            pass
        self.fp16 = fp16
        self.device = device
        self.horiz_only = horiz_only
        self.num_grids = num_grids
        self.sizes = []
        self.grids = []
        self.uses = []

    def forward(self, images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """resample `images` according to `flow`

        Parameters
        ----------
        images: torch.Tensor. shape (N, C, H, W)
            images
        flow: torch.Tensor. shape (N, 2, H, W) in the flow case or (N, 1, H, W) in the stereo matching case
            should be in ABSOLUTE PIXEL COORDINATES.
        Returns
        -------
        resampled: torch.Tensor (N, C, H, W)
            images sampled at their original locations PLUS the input flow
        """
        # for MPI-sintel, assumes t0 = Resample2d()(t1, flow)
        if self.size is not None:
            H, W = self.size
        else:
            H, W = flow.size(2), flow.size(3)
        # print(H,W)
        # images: NxCxHxW
        # flow: Bx2xHxW
        grid_size = [flow.size(0), 2, flow.size(2), flow.size(3)]
        if not hasattr(self, "grids") or grid_size not in self.sizes:
            if len(self.sizes) >= self.num_grids:
                min_uses = min(self.uses)
                min_loc = self.uses.index(min_uses)
                del (self.uses[min_loc], self.grids[min_loc], self.sizes[min_loc])
            # make the affine mat match the batch size
            self.affine_mat = self.base_mat.repeat(images.size(0), 1, 1)

            # function outputs N,H,W,2. Permuted to N,2,H,W to match flow
            # 0-th channel is x sample locations, -1 in left column, 1 in right column
            # 1-th channel is y sample locations, -1 in first row, 1 in bottom row
            this_grid = (
                F.affine_grid(self.affine_mat, images.shape, align_corners=False).permute(0, 3, 1, 2).to(self.device)
            )
            this_size = [i for i in this_grid.size()]
            self.sizes.append(this_size)
            self.grids.append(this_grid)
            self.uses.append(0)
            # print(this_grid.shape)
        else:
            grid_loc = self.sizes.index(grid_size)
            this_grid = self.grids[grid_loc]
            self.uses[grid_loc] += 1

        # normalize flow
        # input should be in absolute pixel coordinates
        # this normalizes it so that a value of 2 would move a pixel all the way across the width or height
        # horiz_only: for stereo matching, Y values are always the same
        if self.horiz_only:
            # flow = flow[:, 0:1, :, :] / ((W - 1.0) / 2.0)
            flow = torch.cat(
                [flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), torch.zeros((flow.size(0), flow.size(1), H, W))], 1
            )
        else:
            # for optic flow matching: can be displaced in X or Y
            flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        # sample according to grid + flow
        return F.grid_sample(
            input=images,
            grid=(this_grid + flow).permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class Reconstructor:
    def __init__(self, cfg: DictConfig):
        device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
        self.resampler = Resample2d(device=device, fp16=cfg.compute.fp16)
        if "normalization" in list(cfg.augs.keys()):
            mean = list(cfg.augs.normalization.mean)
            std = list(cfg.augs.normalization.std)
        else:
            mean = None
            std = None

        self.normalizer = Normalizer(mean=mean, std=std)

    def reconstruct_images(self, image_batch: torch.Tensor, flows: Union[tuple, list]) -> Tuple[tuple, tuple, tuple]:
        # SSIM DOES NOT WORK WITH Z-SCORED IMAGES
        # requires images in the range [0,1]. So we have to denormalize for it to work!
        image_batch = self.normalizer.denormalize(image_batch)
        if image_batch.ndim == 4:
            N, C, H, W = image_batch.shape
            num_images = int(C / 3) - 1
            t0 = image_batch[:, : num_images * 3, ...].contiguous().view(N * num_images, 3, H, W)
            t1 = image_batch[:, 3:, ...].contiguous().view(N * num_images, 3, H, W)
        elif image_batch.ndim == 5:
            N, C, T, H, W = image_batch.shape
            num_images = T - 1
            t0 = image_batch[:, :, :num_images, ...]
            t0 = t0.transpose(1, 2).reshape(N * num_images, C, H, W)
            t1 = image_batch[:, :, 1:, ...]
            t1 = t1.transpose(1, 2).reshape(N * num_images, C, H, W)
        else:
            raise ValueError("unexpected batch shape: {}".format(image_batch))

        reconstructed = []
        t1s = []
        t0s = []
        flows_reshaped = []
        for flow in flows:
            # upsampled_flow = F.interpolate(flow, (h,w), mode='bilinear', align_corners=False)
            if flow.ndim == 4:
                n, c, h, w = flow.size()
                flow = flow.view(N * num_images, 2, h, w)
            else:
                n, c, t, h, w = flow.shape
                flow = flow.transpose(1, 2).reshape(n * t, c, h, w)

            downsampled_t1 = F.interpolate(t1, (h, w), mode="bilinear", align_corners=False)
            downsampled_t0 = F.interpolate(t0, (h, w), mode="bilinear", align_corners=False)
            t0s.append(downsampled_t0)
            t1s.append(downsampled_t1)
            reconstructed.append(self.resampler(downsampled_t1, flow))
            del (downsampled_t1, downsampled_t0)
            flows_reshaped.append(flow)

        return tuple(t0s), tuple(reconstructed), tuple(flows_reshaped)

    def __call__(self, image_batch: torch.Tensor, flows: Union[tuple, list]) -> Tuple[tuple, tuple, tuple]:
        return self.reconstruct_images(image_batch, flows)


def stacked_to_sequence(tensor: torch.Tensor, num_channels: int = 3) -> torch.Tensor:
    if tensor.ndim > 4:
        warnings.warn("called stacked_to_sequence on a sequence of shape {}".format(tensor.shape))
        return tensor
    N, C, H, W = tensor.shape
    assert (C % num_channels) == 0
    num_channels = 3
    starts = range(0, C, num_channels)
    ends = range(num_channels, C + 1, num_channels)
    return torch.stack([tensor[:, start:end, ...] for start, end in zip(starts, ends)], dim=2)


def rgb_to_hsv_torch(image: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of RGB to HSV color conversion"""
    # https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv
    # https://en.wikipedia.org/wiki/HSL_and_HSV#General_approach
    # https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
    # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

    # assume tensors are N, C, (t optional), H, W
    r = image[:, 0, ...]
    g = image[:, 1, ...]
    b = image[:, 2, ...]

    M = image.max(1)[0]
    m = image.min(1)[0]

    V = M  # brightness
    C = M - m

    eps = 1e-8

    H = torch.zeros_like(r)
    H[b == M] = (240 + 60 * (r - g) / (C + eps))[b == M]
    H[g == M] = (120 + 60 * (b - r) / (C + eps))[g == M]
    H[r == M] = (60 * (g - b) / (C + eps))[r == M]
    H[C == 0] = 0

    H[H < 0] += 360
    # NOTE: I think this is wrong! But this is what OpenCV does, and I wanted to mimic that here
    # No idea why they wouldn't divide by only 255
    H = H / 2 / 255

    S = torch.zeros_like(r)
    S[V > 0] = (C / V)[V > 0]
    # hsv = torch.stack([H,S,V], dim=-3)

    return torch.stack([H, S, V], dim=1)


def flow_to_rgb_polar_torch(image: torch.Tensor, maxval: Union[int, float] = 10):
    hsv = rgb_to_hsv_torch(image)
    mag = hsv[:, 1, ...] * maxval

    ang = hsv[:, 0, ...]
    ang = ang * 2 * np.pi / 180 * 255

    x = mag * torch.cos(ang)
    y = mag * torch.sin(ang)
    flow = torch.stack((x, y), dim=1)
    return flow


def stacked_rgb_to_flow(flow_rgb: torch.Tensor, maxval: int = 10) -> torch.Tensor:
    with torch.no_grad():
        if flow_rgb.ndim > 4:
            already_sequence = True
            sequence = flow_rgb
        else:
            sequence = stacked_to_sequence(flow_rgb)
            already_sequence = False
        flow = flow_to_rgb_polar_torch(sequence, maxval=maxval)
        if not already_sequence:
            N, C, T, H, W = flow.shape
            flow = torch.cat([flow[:, :, i, ...] for i in range(T)], axis=1)
    return flow
