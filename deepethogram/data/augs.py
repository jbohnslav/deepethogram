import logging
from pprint import pformat

import cv2
import numpy as np
import torch
from kornia import augmentation as K, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, pi
from omegaconf import DictConfig
from opencv_transforms import transforms
from torch import nn

cv2.setNumThreads(0)

log = logging.getLogger(__name__)


def get_normalization_layer(mean: list, std: list, num_images: int = 1, mode: str = '2d'):
    """Get Z-scoring layer from config
    If RGB frames are stacked into tensor N, num_rgb*3, H, W, we need to repeat the mean and std num_rgb times
    """
    # if mode == '2d':
    mean = mean.copy() * num_images
    std = std.copy() * num_images

    return transforms.Normalize(mean=mean, std=std)


class Transpose:
    """Module to transpose image stacks.
    """
    def __call__(self, images: np.ndarray) -> np.ndarray:
        shape = images.shape
        if len(shape) == 4:
            # F x H x W x C -> C x F x H x W
            return images.transpose(3, 0, 1, 2)
        elif len(shape) == 3:
            # H x W x C -> C x H x W
            return images.transpose(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeVideo(nn.Module):
    """Z-scores input video sequences
    """
    def __init__(self, mean, std):
        super().__init__()

        mean = np.asarray(mean)
        std = np.asarray(std)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        self.mean = mean.reshape(1, -1, 1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1, 1)

        self.normalize = K.Normalize(mean=mean, std=std)

    #     def forward(self, tensor):
    #         if self.mean.device != tensor.device:
    #             self.mean = self.mean.to(tensor.device)
    #         if self.std.device != tensor.device:
    #             self.std = self.std.to(tensor.device)

    #         return (tensor - self.mean) / self.std
    def forward(self, tensor):
        return self.normalize(tensor.transpose(1, 2)).transpose(1, 2)


class DenormalizeVideo(nn.Module):
    """Un-z-scores input video sequences
    """
    def __init__(self, mean, std):
        super().__init__()

        mean = np.asarray(mean)
        std = np.asarray(std)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        self.mean = mean.reshape(1, -1, 1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1, 1)

        self.normalize = K.Denormalize(mean=mean, std=std)

        #     def forward(self, tensor):
        #         if self.mean.device != tensor.device:
        #             self.mean = self.mean.to(tensor.device)
        #         if self.std.device != tensor.device:
        #             self.std = self.std.to(tensor.device)

        #         return torch.clamp( tensor*self.std + self.mean , 0, 1)

    def forward(self, tensor):
        return self.normalize(tensor.transpose(1, 2)).transpose(1, 2)


class ToFloat(nn.Module):
    """Module for converting input uint8 tensors to floats, dividing by 255
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomColorJitterVideo(nn.Module):
    """Wrapper for applying random color jitter to video clips
    """
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, return_transform=False,
                 same_on_batch=False, p=0.5):
        """Constructor for gpu augmentations for color jittering

        Parameters
        ----------
        brightness : float, optional
            Strength of brightness augmentation, by default 0.1
        contrast : float, optional
            Strength of contrast augmentation, by default 0.1
        saturation : float, optional
            Strength of saturation augmentation, by default 0.1
        hue : float, optional
            Strength of hue augmentation, by default 0.1
        return_transform : bool, optional
            If True, returns transformation matrix, which will always be identity, by default False
        same_on_batch : bool, optional
            If True, applies the same augmentation to every batch element, by default False
        p : float, optional
            Probability of applying any augmentation to a given batch example, by default 0.5
        """
        super().__init__()
        self.p = p
        self.jitter_2d = K.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                       hue=hue, return_transform=return_transform,
                                       same_on_batch=same_on_batch, p=p)
        
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        self.aug_values = {0: self.brightness, 
                           1: self.contrast, 
                           2: self.saturation, 
                           3: self.hue}
        
        self.transform_list = [
            self.adjust_brightness,
            self.adjust_contrast,
            self.adjust_saturation,
            self.adjust_hue
        ]

    def adjust_brightness(self, input, params):
        # https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/functional/functional.html#apply_adjust_brightness
        transformed = adjust_brightness(input, params['brightness_factor'].to(input.dtype) - 1)
        return transformed

    def adjust_contrast(self, input, params):
        transformed = adjust_contrast(input, params['contrast_factor'].to(input.dtype))
        return transformed

    def adjust_saturation(self, input, params):
        transformed = adjust_saturation(input, params['saturation_factor'].to(input.dtype))
        return transformed

    def adjust_hue(self, input, params):
        transformed = adjust_hue(input, params['hue_factor'].to(input.dtype) * 2 * pi)
        return transformed

    def forward(self, batch):
        params = self.jitter_2d.__forward_parameters__(batch.shape, self.jitter_2d.p,
                                                       self.jitter_2d.p_batch, self.jitter_2d.same_on_batch)
        should_aug = params['batch_prob']
        if should_aug.sum() == 0:
            return batch
        # we shouldn't need this context manager here-- but trying it due to VRAM overflow issues
        with torch.no_grad():
            # N C T H W -> N T C H W
            outputs = batch.transpose(1, 2).contiguous().detach()
            # print(outputs.shape)
            for idx in params['order'].tolist():
                if self.aug_values[idx] > 0:
                    t = self.transform_list[idx]
                    outputs[should_aug] = t(outputs[should_aug], params).detach()
            outputs = outputs.transpose(1, 2).contiguous().detach()
        return outputs


class RandomGrayscaleVideo(nn.Module):
    """Wrapper to apply grayscaling to video clips on GPU
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def generate_parameters(self, batch_size: int):
        p = torch.rand(batch_size)
        should_aug = p < self.p
        return should_aug

    def rgb_to_gray(self, batch):
        # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/gray.html#rgb_to_grayscale
        r: torch.Tensor = batch[..., 0:1, :, :, :]
        g: torch.Tensor = batch[..., 1:2, :, :, :]
        b: torch.Tensor = batch[..., 2:3, :, :, :]

        gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray.repeat(1, 3, 1, 1, 1)
        return gray

    def forward(self, batch):
        should_aug = self.generate_parameters(batch.shape[0])
        outputs = batch# .clone()
        outputs[should_aug] = self.rgb_to_gray(outputs[should_aug]).detach()
        return outputs


class StackClipInChannels(nn.Module):
    """Module to convert image from N,C,T,H,W -> N,C*T,H,W
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, T, H, W = tensor.shape
        tensor = tensor.transpose(1, 2)
        stacked = torch.cat([tensor[:, i, ...] for i in range(T)], dim=1)
        return stacked


class UnstackClip(nn.Module):
    """Module to convert image from N,C*T,H,W -> N,C,T,H,W
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, H, W = tensor.shape
        T = C // 3

        return torch.stack(torch.chunk(tensor, T, dim=1), dim=2)


def get_cpu_transforms(augs: DictConfig) -> dict:
    """Makes CPU augmentations from the aug section of a configuration. 

    Parameters
    ----------
    augs : DictConfig
        augmentation parameters

    Returns
    -------
    xform : dict
        keys: ['train', 'val', 'test']. Values: a composed OpenCV augmentation pipeline callable. 
        Example: auged_images = xform['train'](images)
    """
    train_transforms = []
    val_transforms = []
    # order here matters a lot!!
    if augs.crop_size is not None:
        train_transforms.append(transforms.RandomCrop(augs.crop_size))
        val_transforms.append(transforms.CenterCrop(augs.crop_size))
    if augs.resize is not None:
        train_transforms.append(transforms.Resize(augs.resize))
        val_transforms.append(transforms.Resize(augs.resize))
    if augs.pad is not None:
        pad = tuple(augs.pad)
        train_transforms.append(transforms.Pad(pad))
        val_transforms.append(transforms.Pad(pad))

    train_transforms.append(Transpose())
    val_transforms.append(Transpose())

    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    xform = {'train': train_transforms,
             'val': val_transforms,
             'test': val_transforms}
    log.info('CPU transforms: {}'.format(xform))
    return xform


def get_gpu_transforms(augs: DictConfig, mode: str = '2d') -> dict:
    """Makes GPU augmentations from the augs section of a configuration.

    Parameters
    ----------
    augs : DictConfig
        Augmentation parameters
    mode : str, optional
        If '2d', stacks clip in channels. If 3d, returns 5-D tensor, by default '2d'

    Returns
    -------
    xform : dict
        keys: ['train', 'val', 'test']. Values: a nn.Sequential with Kornia augmentations. 
        Example: auged_images = xform['train'](images)
    """
    # input is a tensor of shape N x C x F x H x W
    train_transforms = [ToFloat()]
    val_transforms = [ToFloat()]
    if augs.LR > 0:
        train_transforms.append(K.RandomHorizontalFlip3D(p=augs.LR,
                                                         same_on_batch=False,
                                                         return_transform=False))
    if augs.UD > 0:
        train_transforms.append(K.RandomVerticalFlip3D(p=augs.UD,
                                                       same_on_batch=False, return_transform=False))
    if augs.degrees > 0:
        train_transforms.append(K.RandomRotation3D((1e-7, 1e-7, augs.degrees)))

    if augs.brightness > 0 or augs.contrast > 0 or augs.saturation > 0 or augs.hue > 0:
        train_transforms.append(RandomColorJitterVideo(brightness=augs.brightness,
                                                       contrast=augs.contrast,
                                                       saturation=augs.saturation,
                                                       hue=augs.hue,
                                                       p=augs.color_p,
                                                       same_on_batch=False,
                                                       return_transform=False))

    if augs.grayscale > 0:
        train_transforms.append(RandomGrayscaleVideo(p=augs.grayscale))
    
    norm = NormalizeVideo(mean=augs.normalization.mean,
                          std=augs.normalization.std)
    train_transforms.append(norm)
    val_transforms.append(norm)

    denormalize = []
    if mode == '2d':
        train_transforms.append(StackClipInChannels())
        val_transforms.append(StackClipInChannels())
        denormalize.append(UnstackClip())
    denormalize.append(DenormalizeVideo(mean=augs.normalization.mean,
                                        std=augs.normalization.std))

    train_transforms = nn.Sequential(*train_transforms)
    val_transforms = nn.Sequential(*val_transforms)
    denormalize = nn.Sequential(*denormalize)

    gpu_transforms = dict(train=train_transforms,
                val=val_transforms,
                test=val_transforms,
                denormalize=denormalize)
    log.info('GPU transforms: {}'.format(gpu_transforms))
    return gpu_transforms


def get_gpu_transforms_inference(augs: DictConfig, mode: str = '2d') -> dict:
    """Gets GPU transforms needed for inference

    Parameters
    ----------
    augs : DictConfig
        Augmentation parameters
    mode : str, optional
        If '2d', stacks clip in channels. If 3d, returns 5-D tensor, by default '2d'

    Returns
    -------
    xform : dict
        keys: ['train', 'val', 'test']. Values: a nn.Sequential with Kornia augmentations. 
        Example: auged_images = xform['val'](images)
    """
    # sequential iterator already handles casting to float, dividing by 255, and stacking in channel dimension
    # import pdb; pdb.set_trace()
    # norm = get_normalization_layer(np.array(augs.normalization.mean), np.array(augs.normalization.std),
    #                                num_images, mode)
    xform = [NormalizeVideo(mean=augs.normalization.mean,
                          std=augs.normalization.std)]
    if mode == '2d':
        xform.append(StackClipInChannels())
    xform = nn.Sequential(*xform)
    gpu_transforms = dict(val=xform,
                          test=xform)
    return gpu_transforms


def get_empty_gpu_transforms() -> dict:
    """Placeholder for GPU transforms that actually does nothing. Useful for debugging

    Returns
    -------
    xform : dict
        keys: ['train', 'val', 'test']. Values: a nn.Sequential with Kornia augmentations. 
        Example: auged_images = xform['train'](images)
    """
    gpu_transforms = dict(train=nn.Identity(),
                          val=nn.Identity(),
                          test=nn.Identity(),
                          denormalize=nn.Identity())
    return gpu_transforms
