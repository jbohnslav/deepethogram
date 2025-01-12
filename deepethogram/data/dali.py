"""NVIDIA DALI pipeline and loader implementations for video processing.

This module provides DALI-based data loading functionality for video datasets,
with a focus on the Kinetics dataset format. It includes GPU-accelerated video loading
and augmentation capabilities.
"""

import os

try:
    import nvidia.dali.ops as ops  # noqa: F401
    import nvidia.dali.types as types  # noqa: F401
    from nvidia.dali.backend import TensorListCPU  # noqa: F401
    from nvidia.dali.pipeline import Pipeline  # noqa: F401
    from nvidia.dali.plugin import pytorch  # noqa: F401
except ImportError:
    dali = False
else:
    dali = True


class KineticsDALIPipe(Pipeline):
    """DALI Pipeline for processing video data in Kinetics format.

    This pipeline handles video loading, augmentation, and preprocessing using NVIDIA DALI.
    It supports both supervised and unsupervised modes, with configurable augmentations
    including brightness, contrast, and spatial transformations.

    Args:
        directory (str): Root directory containing the video files
        supervised (bool): Whether to return labels with the data
        sequence_length (int): Number of frames to load per sequence
        batch_size (int): Number of sequences per batch
        num_workers (int): Number of parallel workers
        gpu_id (int): ID of GPU to use
        shuffle (bool): Whether to shuffle the data
        crop_size (tuple): Size of output crops (H, W)
        resize (tuple, optional): Size to resize frames to before cropping
        brightness (float): Maximum brightness adjustment factor
        contrast (float): Maximum contrast adjustment factor
        mean (list): Mean values for normalization per channel
        std (list): Standard deviation values for normalization per channel
        conv_mode (str): Convolution mode ('2d' or '3d')
        image_shape (tuple): Base image dimensions (H, W)
        validate (bool): Whether this pipeline is for validation (reduces augmentation)
    """

    def __init__(
        self,
        directory,
        supervised: bool = True,
        sequence_length: int = 11,
        batch_size: int = 1,
        num_workers: int = 1,
        gpu_id: int = 0,
        shuffle: bool = True,
        crop_size: tuple = (256, 256),
        resize: tuple = None,
        brightness: float = 0.25,
        contrast: float = 0.1,
        mean: list = [0.5, 0.5, 0.5],
        std: list = [0.5, 0.5, 0.5],
        conv_mode="3d",
        image_shape=(256, 256),
        validate: bool = False,
    ):
        super().__init__(batch_size, num_workers, gpu_id, prefetch_queue_depth=1)
        self.input = ops.VideoReader(
            additional_decode_surfaces=1,
            channels=3,
            device="gpu",
            dtype=types.FLOAT,
            enable_frame_num=False,
            enable_timestamps=False,
            file_root=directory,
            image_type=types.RGB,
            initial_fill=1,
            lazy_init=False,
            normalized=True,
            num_shards=1,
            pad_last_batch=False,
            prefetch_queue_depth=1,
            random_shuffle=shuffle,
            sequence_length=sequence_length,
            skip_vfr_check=True,
            step=-1,
            shard_id=0,
            stick_to_shard=False,
            stride=1,
        )

        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.cmn = ops.CropMirrorNormalize(device="gpu", crop=crop_size, mean=mean, std=std, output_layout=types.NFHWC)

        self.coin = ops.CoinFlip(probability=0.5)
        self.brightness_val = ops.Uniform(range=[1 - brightness, 1 + brightness])
        self.contrast_val = ops.Uniform(range=[1 - contrast, 1 + contrast])
        self.supervised = supervised
        self.half = ops.Constant(fdata=0.5)
        self.zero = ops.Constant(idata=0)
        self.cast_to_long = ops.Cast(device="gpu", dtype=types.INT64)
        if crop_size is not None:
            H, W = crop_size
        else:
            # default
            H, W = image_shape
        if conv_mode == "3d":
            self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])
            self.reshape = None
        elif conv_mode == "2d":
            self.transpose = ops.Transpose(device="gpu", perm=[0, 3, 1, 2])
            self.reshape = ops.Reshape(device="gpu", shape=[-1, H, W])
        self.validate = validate

    def define_graph(self):
        images, labels = self.input(name="Reader")
        # custom brightness contrast operator
        if not self.validate:
            images = self.brightness_val() * (0.5 + self.contrast_val() * (images - 0.5))

        if self.validate:
            x, y = self.half(), self.half()
            mirror = self.zero()
        else:
            x, y = self.uniform(), self.uniform()
            mirror = self.coin()

        images = self.cmn(images, crop_pos_x=x, crop_pos_y=y, mirror=mirror)
        images = self.transpose(images)
        if self.reshape is not None:
            images = self.reshape(images)

        if self.supervised:
            return images, self.cast_to_long(labels)
        else:
            return images


class DALILoader:
    """Data loader wrapper for DALI pipeline.

    Provides an iterator interface to the DALI pipeline for easy integration
    with PyTorch training loops.

    Args:
        directory (str): Root directory containing the video files
        supervised (bool): Whether to return labels with the data
        sequence_length (int): Number of frames to load per sequence
        batch_size (int): Number of sequences per batch
        num_workers (int): Number of parallel workers
        gpu_id (int): ID of GPU to use
        shuffle (bool): Whether to shuffle the data
        crop_size (tuple): Size of output crops (H, W)
        mean (list): Mean values for normalization per channel
        std (list): Standard deviation values for normalization per channel
        conv_mode (str): Convolution mode ('2d' or '3d')
        validate (bool): Whether this is for validation
        distributed (bool): Whether to use distributed training mode

        https://github.com/NVIDIA/DALI/blob/cde7271a840142221273f8642952087acd919b6e/docs/examples/use_cases/video_superres/dataloading/dataloaders.py
    """

    def __init__(
        self,
        directory,
        supervised: bool = True,
        sequence_length: int = 11,
        batch_size: int = 1,
        num_workers: int = 1,
        gpu_id: int = 0,
        shuffle: bool = True,
        crop_size: tuple = (256, 256),
        mean: list = [0.5, 0.5, 0.5],
        std: list = [0.5, 0.5, 0.5],
        conv_mode: str = "3d",
        validate: bool = False,
        distributed: bool = False,
    ):
        self.pipeline = KineticsDALIPipe(
            directory=directory,
            batch_size=batch_size,
            supervised=supervised,
            sequence_length=sequence_length,
            num_workers=num_workers,
            gpu_id=gpu_id,
            crop_size=crop_size,
            mean=mean,
            std=std,
            conv_mode=conv_mode,
            validate=validate,
        )
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        names = ["images", "labels"] if supervised else ["images"]
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline, names, self.epoch_size, auto_reset=True)

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()


def get_dataloaders_kinetics_dali(
    directory,
    rgb_frames=1,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    supervised=True,
    conv_mode="2d",
    gpu_id: int = 0,
    crop_size: tuple = (256, 256),
    mean: list = [0.5, 0.5, 0.5],
    std: list = [0.5, 0.5, 0.5],
    distributed: bool = False,
):
    """Create DALI dataloaders for train and validation sets.

    Args:
        directory (str): Root directory containing train and val subdirectories
        rgb_frames (int): Number of RGB frames per sequence
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle training data
        num_workers (int): Number of worker processes
        supervised (bool): Whether to return labels
        conv_mode (str): Convolution mode ('2d' or '3d')
        gpu_id (int): GPU device ID
        crop_size (tuple): Output crop dimensions
        mean (list): Normalization mean values
        std (list): Normalization standard deviation values
        distributed (bool): Whether to use distributed training

    Returns:
        dict: Dictionary containing train and validation dataloaders
    """
    shuffles = {"train": shuffle, "val": True, "test": False}
    dataloaders = {}
    for split in ["train", "val"]:
        splitdir = os.path.join(directory, split)
        dataloaders[split] = DALILoader(
            splitdir,
            supervised=supervised,
            batch_size=batch_size,
            gpu_id=gpu_id,
            shuffle=shuffles[split],
            crop_size=crop_size,
            mean=mean,
            std=std,
            validate=split == "val",
            num_workers=num_workers,
            sequence_length=rgb_frames,
            conv_mode=conv_mode,
            distributed=distributed,
        )

    dataloaders["split"] = None
    return dataloaders
