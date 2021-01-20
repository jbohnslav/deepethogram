import bisect
from collections import deque
import logging
import os
import pprint
import random
import warnings
from functools import partial
from typing import Union, Tuple

import h5py
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from opencv_transforms import transforms
from torch.utils import data
from vidio import VideoReader

# from deepethogram.dataloaders import log
from deepethogram import projects
from deepethogram.data.augs import get_transforms, get_cpu_transforms
from deepethogram.data.utils import purge_unlabeled_videos, get_video_metadata, extract_metadata, find_labelfile, \
    read_all_labels, get_split_from_records, remove_invalid_records_from_split_dictionary, \
    make_loss_weight
from deepethogram.file_io import read_labels

log = logging.getLogger(__name__)


# https://pytorch.org/docs/stable/data.html
class VideoIterable(data.IterableDataset):
    def __init__(self, videofile, transform, sequence_length=11, num_workers: int = 0,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0]):
        super().__init__()

        assert os.path.isfile(videofile) or os.path.isdir(videofile)
        self.readers = {i: 0 for i in range(num_workers)}
        self.videofile = videofile
        # self.reader = VideoReader(videofile)
        self.transform = transform

        self.start = 0
        self.sequence_length = sequence_length
        with VideoReader(self.videofile) as reader:
            self.N = len(reader)

        self.blank_start_frames = self.sequence_length // 2
        self.cnt = 0

        self.mean_by_channels = self.parse_mean_by_channels(mean_by_channels)
        # NOTE: not great practice, but I want each dataset to know when to stop
        self.num_workers = num_workers
        self.buffer = deque([], maxlen=self.sequence_length)

        self.reset_counter = self.num_workers
        self._zeros_image = None
        self._image_shape = None
        self.get_image_shape()

    def __len__(self):
        return self.N

    def get_image_shape(self):
        with VideoReader(self.videofile) as reader:
            im = reader[0]
        im = self.transform(im)
        self._image_shape = im.shape

    def get_zeros_image(self, ):
        if self._zeros_image is None:
            if self._image_shape is None:
                raise ValueError('must set shape before getting zeros image')
            # ALWAYS ASSUME OUTPUT IS TRANSPOSED
            self._zeros_image = np.zeros(self._image_shape, dtype=np.uint8)
            for i in range(3):
                self._zeros_image[i, ...] = self.mean_by_channels[i]
        return self._zeros_image

    def parse_mean_by_channels(self, mean_by_channels):
        if isinstance(mean_by_channels[0], (float, np.floating)):
            return np.clip(np.array(mean_by_channels) * 255, 0, 255).astype(np.uint8)
        elif isinstance(mean_by_channels[0], (int, np.integer)):
            assert np.array_equal(np.clip(mean_by_channels, 0, 255), np.array(mean_by_channels))
            return np.array(mean_by_channels).astype(np.uint8)
        else:
            raise ValueError('unexpected type for input channel mean: {}'.format(mean_by_channels))

    def my_iter_func(self, start, end):
        for i in range(start, end):
            self.buffer.append(self.get_current_item())
            yield {'images': np.stack(self.buffer, axis=1),
                   'framenum': self.cnt - 1 - self.sequence_length // 2}

    def get_current_item(self):
        worker_info = data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # blank_start_frames =
        # print(self.cnt)
        if self.cnt < 0:
            im = self.get_zeros_image()
        elif self.cnt >= self.N:
            im = self.get_zeros_image()
        else:
            try:
                im = self.readers[worker_id][self.cnt]
            except Exception as e:
                print(f'problem reading frame {self.cnt}')
                raise
            im = self.transform(im)
        self.cnt += 1
        return im

    def fill_buffer_init(self, iter_start):
        self.cnt = iter_start
        # hack for the first one: don't quite fill it up
        for i in range(iter_start, iter_start + self.sequence_length - 1):
            self.buffer.append(self.get_current_item())

    def __iter__(self):
        worker_info = data.get_worker_info()
        # print(worker_info)
        iter_end = self.N - self.sequence_length // 2
        if worker_info is None:
            iter_start = - self.blank_start_frames
            self.readers[0] = VideoReader(self.videofile)
        else:
            per_worker = self.N // self.num_workers
            remaining = self.N % per_worker
            nums = [per_worker for i in range(self.num_workers)]
            nums = [nums[i] + 1 if i < remaining else nums[i] for i in range(self.num_workers)]
            # print(nums)
            nums.insert(0, 0)
            starts = np.cumsum(nums[:-1])  # - self.blank_start_frames
            starts = starts.tolist()
            ends = starts[1:] + [iter_end]
            starts[0] = -self.blank_start_frames

            # print(starts, ends)

            iter_start = starts[worker_info.id]
            iter_end = min(ends[worker_info.id], self.N)
            # print(f'worker: {worker_info.id}, start: {iter_start} end: {iter_end}')
            self.readers[worker_info.id] = VideoReader(self.videofile)
        # FILL THE BUFFER TO START
        # print('iter start: {}'.format(iter_start))
        self.fill_buffer_init(iter_start)
        return self.my_iter_func(iter_start, iter_end)

    def close(self):
        for k, v in self.readers.items():
            if isinstance(v, int):
                continue
            try:
                v.close()
            except Exception as e:
                print(f'error destroying reader {k}')
            else:
                print(f'destroyed {k}')

    def __exit__(self):
        self.close()

    def __del__(self):
        self.close()


class TwoStreamIterator:
    """Sequential iterator for loading two-stream videos from disk for inference.
    Uses two SequentialIterators, one for the flow video and one for the RGB video.
    """

    def __init__(self, rgbfile: Union[str, os.PathLike], flowfile: Union[str, os.PathLike],
                 rgb_frames: int, flow_frames: int, flow_style: str, flow_max: Union[int, float], device,
                 spatial_transform=None, stack_channels: bool = True,
                 batch_size: int = 1, supervised: bool = False, debug: bool = False):
        """
        Constructor for TwoStreamIterator
        Args:
            rgbfile: absolute path to videofile containing RGB frames
            flowfile: absolute path to videofile containing stored optical flow frames
            rgb_frames: number of RGB frames to input to the model. typically 1
            flow_frames: number of optical flows to input to the model. typically 10
            flow_style:
                rgb: do not do any conversion of optic flow (rarely used)
                linear: linearly map the range of RGB values to flow values (see flow_utils.flow_to_rgb)
                polar: see flow_utils.flow_to_rgb_polar
            flow_max: the maximum possible optic flow value
            device: str or torch.device, which GPU to put the data on
            spatial_transform: augmentations that affect the spatial extent of the frames, typically cropping, rotation,
                resizing, etc. should be from opencv_transforms. applied to both rgb and flow frames
            stack_channels: if True, returns Tensor of shape (num_images*3, H, W). if False, returns Tensor of shape
                (num_images, 3, H, W)
            batch_size: currently only 1 is supported
            supervised: if True, return labels
            debug: if True, prints lots of statements
        """
        assert (os.path.isfile(rgbfile))
        assert (os.path.isfile(flowfile))

        self.rgb_iterator = SequentialIterator(rgbfile, num_images=rgb_frames, transform=spatial_transform,
                                               supervised=supervised, device=device)

        if flow_style == 'linear':
            convert = partial(flow_utils.rgb_to_flow, maxval=flow_max)
        elif flow_style == 'polar':
            convert = partial(flow_utils.rgb_to_flow_polar, maxval=flow_max)
        elif flow_style == 'rgb':
            convert = self.no_conversion
        else:
            raise ValueError('Unknown optical flow style: {}'.format(flow_style))
        self.debug = debug
        if self.debug:
            spatial_transform = None

            def convert(inputs):
                return (inputs.astype(np.float32) / 255)
        if spatial_transform is not None:
            spatial_transform = transforms.Compose([convert, spatial_transform])
        else:
            spatial_transform = convert
        # print(spatial_transform)
        self.flow_iterator = SequentialIterator(flowfile, num_images=flow_frames, transform=spatial_transform,
                                                supervised=False, device=device)
        assert (len(self.flow_iterator) == len(self.rgb_iterator))
        self.N = len(self.flow_iterator)
        self.index = 0
        self.supervised = supervised
        # self.totensor = transforms.ToTensor()

    # Python 2 compatibility:
    def next(self):
        return self.__next__()

    def no_conversion(self, inputs):
        return inputs

    def __next__(self):
        """Loads the next rgb frame and flow frame, rolls the current batches, and appends the next frame to the
        batches"""
        # print(self.index)
        if self.index > self.N:
            # print('stopping')
            self.end()
            raise StopIteration
        rgb = next(self.rgb_iterator)

        if self.supervised:
            rgb, label = rgb
        flow = next(self.flow_iterator)

        self.index += 1
        if self.supervised:
            return rgb, flow, label
        else:
            return rgb, flow

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def end(self):
        """release file objects"""
        if hasattr(self, 'rgb_iterator'):
            del self.rgb_iterator
        if hasattr(self, 'flow_iterator'):
            del self.flow_iterator

    def __del__(self):
        self.end()


class MultiMovieIterator:
    """Reads short clips of images sequentially from a list of videos

    Examples:
        iterator = MultiMovieIterator(['movie1.avi', 'movie2.avi'])
        for batch in iterator:
            outputs = model(batch)
    """

    def __init__(self, videofiles: list, *args, **kwargs) -> None:
        """Initializes a MultiMovieIterator object.

        Args:
            videofiles: list of videos. Can be avis, mp4s, or jpg-encoded hdf5 files
            *args, **kwargs: see SequentialIterator
        """
        for video in videofiles:
            assert (os.path.isfile(video))

        self.videos = videofiles
        self.N = len(videofiles)
        self.args = args
        self.kwargs = kwargs
        self.N_batches = self.get_num_batches()
        print('Total batches: {}'.format(self.N_batches))
        self.count = 0

    def initialize_iterator(self, videofile: Union[str, bytes, os.PathLike]):
        """Returns a single movie iterator from a video file"""
        iterator = iter(SequentialIterator(videofile, *self.args, **self.kwargs))
        return (iterator)

    def get_num_batches(self) -> int:
        """Gets total number of batches"""
        total = 0
        for video in self.videos:
            iterator = self.initialize_iterator(video)
            total += len(iterator)
            iterator.end()
        return total

    def __len__(self):
        return (self.N_batches)

    def __next__(self):
        """Gets the next clip of images (and possibly labels) from list of movies"""
        if not hasattr(self, 'iterator'):
            self.iterator = iter(SequentialIterator(self.videos[0], *self.args, **self.kwargs))
        try:
            batch = next(self.iterator)
        except StopIteration:
            if self.count >= self.N:
                raise
            else:
                self.count += 1
                self.iterator.end()
                self.iterator = iter(SequentialIterator(self.videos[self.count], *self.args, **self.kwargs))
                batch = next(self.iterator)
        return (batch)

    def __iter__(self):
        return self


class SingleVideoDataset(data.Dataset):
    """PyTorch Dataset for loading a set of sequential frames and one-hot labels for Action Detection.

    Features:
        - Loads a set of sequential frames and sequential one-hot labels
        - Adds zero frames at beginning or end so that every label has a corresponding clip
        - Applies the same augmentations to every frame in the clip
        - Automatically finds label files with similar names to the list of movies
        - Stacks all channels together for input into a CNN

    Example:
        dataset = VideoDataset(['movie1.avi', 'movie2.avi'], frames_per_clip=11, reduce=False)
        images, labels = dataset(np.random.randint(low=0, high=len(dataset))
        print(images.shape)
        # 33 x 256 x 256
        print(labels.shape)
        # assuming there are 5 classes in dataset
        # ~5 x 11
    """

    def __init__(self, videofile: Union[str, os.PathLike], labelfile: Union[str, os.PathLike] = None,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
                 frames_per_clip: int = 1, transform=None,
                 reduce: bool = True, conv_mode: str = '2d', keep_reader_open: bool = False):
        """Initializes a VideoDataset object.

        Args:
            video_list: a list of strings or paths to movies
            frames per clip: how many sequential images to load
            transform: either None or a TorchVision.transforms object or opencv_transforms object
            supervised: whether or not to return a label. False: for self-supervision
            reduce: whether or not to change a set of one-hot labels to integers denoting the class that equals one.
                Applicable for multiclass, not multi-label cases, using a softmax activation and NLLloss
            conv_mode: if 2d, returns a tensor of shape C, H, W. Multiple frames are stacked in C dimension. if 3d,
                returns a tensor of shape C, T, H, W
        Returns:
            VideoDataset object
        """

        self.videofile = videofile
        self.labelfile = labelfile
        self.mean_by_channels = self.parse_mean_by_channels(mean_by_channels)
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.reduce = reduce
        self.conv_mode = conv_mode
        self.keep_reader_open = keep_reader_open

        self.supervised = self.labelfile is not None

        assert os.path.isfile(videofile) or os.path.isdir(videofile)
        assert self.conv_mode in ['2d', '3d']

        # find labels given the filename of a video, load, save as an attribute for fast reading
        if self.supervised:
            assert os.path.isfile(labelfile)
            # self.video_list, self.label_list = purge_unlabeled_videos(self.video_list, self.label_list)
            labels, class_counts, num_labels, num_pos, num_neg = read_all_labels([self.labelfile])
            self.labels = labels
            self.class_counts = class_counts
            self.num_labels = num_labels
            self.num_pos = num_pos
            self.num_neg = num_neg
            log.debug('label shape: {}'.format(self.labels.shape))

        metadata = {}
        ret, width, height, framecount = get_video_metadata(self.videofile)
        if ret:
            metadata['name'] = videofile
            metadata['width'] = width
            metadata['height'] = height
            metadata['framecount'] = framecount
        else:
            raise ValueError('error loading video: {}'.format(videofile))
        self.metadata = metadata
        self.N = self.metadata['framecount']
        self._zeros_image = None

    def get_zeros_image(self, c, h, w):
        if self._zeros_image is None:
            # ALWAYS ASSUME OUTPUT IS TRANSPOSED
            self._zeros_image = np.zeros((c, h, w), dtype=np.uint8)
            for i in range(3):
                self._zeros_image[i, ...] = self.mean_by_channels[i]
        return self._zeros_image

    def parse_mean_by_channels(self, mean_by_channels):
        if isinstance(mean_by_channels[0], (float, np.floating)):
            return np.clip(np.array(mean_by_channels) * 255, 0, 255).astype(np.uint8)
        elif isinstance(mean_by_channels[0], (int, np.integer)):
            assert np.array_equal(np.clip(mean_by_channels, 0, 255), np.array(mean_by_channels))
            return np.array(mean_by_channels).astype(np.uint8)
        else:
            raise ValueError('unexpected type for input channel mean: {}'.format(mean_by_channels))

    def __len__(self):
        return self.N

    def prepend_with_zeros(self, stack, blank_start_frames):
        if blank_start_frames == 0:
            return stack
        for i in range(blank_start_frames):
            stack.insert(0, self.get_zeros_image(*stack[0].shape))
        return stack

    def append_with_zeros(self, stack, blank_end_frames):
        if blank_end_frames == 0:
            return stack
        for i in range(blank_end_frames):
            stack.append(self.get_zeros_image(*stack[0].shape))
        return stack

    def __getitem__(self, index: int):
        """Used for reading frames and possibly labels from disk.

        Args:
            index: integer from 0 to number of total clips in dataset
        Returns:
            np.ndarray of shape (H,W,C), where C is 3* frames_per_clip
                Could also be torch.Tensor of shape (C,H,W), depending on the augmentation applied
        """

        images = []
        # if frames per clip is 11, dataset[0] would have 5 blank frames preceding, with the 6th-11th being real frames
        blank_start_frames = max(self.frames_per_clip // 2 - index, 0)

        framecount = self.metadata['framecount']
        # cap = cv2.VideoCapture(self.movies[style][movie_index])
        start_frame = index - self.frames_per_clip // 2 + blank_start_frames
        blank_end_frames = max(index - framecount + self.frames_per_clip // 2 + 1, 0)
        real_frames = self.frames_per_clip - blank_start_frames - blank_end_frames

        seed = np.random.randint(2147483647)
        with VideoReader(self.videofile, assume_writer_style=True) as reader:
            for i in range(real_frames):
                image = reader[i + start_frame]
                if self.transform:
                    random.seed(seed)
                    image = self.transform(image)
                    images.append(image)

        images = self.prepend_with_zeros(images, blank_start_frames)
        images = self.append_with_zeros(images, blank_end_frames)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('idx: {} st: {} blank_start: {} blank_end: {} real: {} total: {}'.format(index,
                                                                                               start_frame,
                                                                                               blank_start_frames,
                                                                                               blank_end_frames,
                                                                                               real_frames, framecount))

        # images are now numpy arrays of shape 3, H, W
        # stacking in the first dimension changes to 3, T, H, W, compatible with Conv3D
        images = np.stack(images, axis=1)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('images shape: {}'.format(images.shape))
        # print(images.shape)
        outputs = {'images': images}
        if self.supervised:
            label = self.labels[index]
            if self.reduce:
                label = np.where(label)[0][0].astype(np.int64)
            outputs['labels'] = label
        return outputs


class VideoDataset(data.Dataset):
    """ Simple wrapper around SingleVideoDataset for smoothly loading multiple videos """

    def __init__(self, videofiles: list, labelfiles: list, *args, **kwargs):
        datasets = []
        for i in range(len(videofiles)):
            labelfile = None if labelfiles is None else labelfiles[i]
            # for i, (videofile, labelfile) in enumerate(zip(videofiles, labelfiles)):
            dataset = SingleVideoDataset(videofiles[i], labelfile, *args, **kwargs)
            datasets.append(dataset)

            if labelfiles is not None:
                if i == 0:
                    class_counts = dataset.class_counts
                    num_pos = dataset.num_pos
                    num_neg = dataset.num_neg
                    num_labels = dataset.num_labels
                else:
                    class_counts += dataset.class_counts
                    num_pos += dataset.num_pos
                    num_neg += dataset.num_neg
                    num_labels += dataset.num_labels

        self.dataset = data.ConcatDataset(datasets)
        if labelfiles is not None:
            self.class_counts = class_counts
            self.num_pos = num_pos
            self.num_neg = num_neg
            self.num_labels = num_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class DeprecatedVideoDataset(data.Dataset):
    """PyTorch Dataset for loading a set of sequential frames and one-hot labels for Action Detection.

    Features:
        - Loads a set of sequential frames and sequential one-hot labels
        - Adds zero frames at beginning or end so that every label has a corresponding clip
        - Applies the same augmentations to every frame in the clip
        - Automatically finds label files with similar names to the list of movies
        - Stacks all channels together for input into a CNN

    Example:
        dataset = VideoDataset(['movie1.avi', 'movie2.avi'], frames_per_clip=11, reduce=False)
        images, labels = dataset(np.random.randint(low=0, high=len(dataset))
        print(images.shape)
        # 33 x 256 x 256
        print(labels.shape)
        # assuming there are 5 classes in dataset
        # ~5 x 11
    """

    def __init__(self, video_list: list, frames_per_clip: int = 1, transform=None, label_list: list = None,
                 reduce: bool = True, conv_mode: str = '2d'):
        """Initializes a VideoDataset object.

        Args:
            video_list: a list of strings or paths to movies
            frames per clip: how many sequential images to load
            transform: either None or a TorchVision.transforms object or opencv_transforms object
            supervised: whether or not to return a label. False: for self-supervision
            reduce: whether or not to change a set of one-hot labels to integers denoting the class that equals one.
                Applicable for multiclass, not multi-label cases, using a softmax activation and NLLloss
            conv_mode: if 2d, returns a tensor of shape C, H, W. Multiple frames are stacked in C dimension. if 3d,
                returns a tensor of shape C, T, H, W
        Returns:
            VideoDataset object
        """
        assert (len(video_list) > 0)
        self.video_list = video_list
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.label_list = label_list
        self.supervised = self.label_list is not None
        # self.supervised = supervised
        self.reduce = reduce
        assert conv_mode in ['2d', '3d']
        self.conv_mode = conv_mode
        # self.verbose = verbose

        # find labels given the filename of a video, load, save as an attribute for fast reading
        if self.supervised:
            self.video_list, self.label_list = purge_unlabeled_videos(self.video_list, self.label_list)
            labels, class_counts, num_labels, num_pos, num_neg = read_all_labels(self.label_list)
            self.labels = labels
            self.class_counts = class_counts
            self.num_labels = num_labels
            self.num_pos = num_pos
            self.num_neg = num_neg
            log.debug('label shape: {}'.format(self.labels.shape))
        metadata = {'name': [], 'width': [], 'height': [], 'framecount': []}
        total_sequences = 0
        start_indices = []
        # Many datasets have videos with variable height, width, numbers of frames, etc.
        for i, video in enumerate(self.video_list):
            ret, width, height, framecount = get_video_metadata(video)
            if ret:
                metadata['name'].append(video)
                metadata['width'].append(width)
                metadata['height'].append(height)
                metadata['framecount'].append(framecount)
                total_sequences += framecount
                start_indices.append(total_sequences)
            else:
                raise ValueError('error loading video: {}'.format(video))
            # if supervised:
        if self.supervised:
            assert (len(metadata['name']) == self.num_labels)
        self.start_indices = start_indices
        self.metadata = pd.DataFrame(metadata)
        self.N = total_sequences

    def __len__(self):
        return self.N - 1

    def __getitem__(self, index: int):
        """Used for reading frames and possibly labels from disk.

        Args:
            index: integer from 0 to number of total clips in dataset
        Returns:
            np.ndarray of shape (H,W,C), where C is 3* frames_per_clip
                Could also be torch.Tensor of shape (C,H,W), depending on the augmentation applied
        """
        # inspired by NVIDIA's NVVL library
        movie_index = bisect.bisect_right(self.start_indices, index)
        frame_index = index - self.start_indices[movie_index - 1] if movie_index > 0 else index

        images = []
        # if frames per clip is 11, dataset[0] would have 5 blank frames preceding, with the 6th-11th being real frames
        blank_start_frames = max(self.frames_per_clip // 2 - frame_index, 0)
        if blank_start_frames > 0:
            for i in range(blank_start_frames):
                h = self.metadata['height'][movie_index]
                w = self.metadata['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))

        framecount = self.metadata['framecount'][movie_index]
        # cap = cv2.VideoCapture(self.movies[style][movie_index])
        start_frame = frame_index - self.frames_per_clip // 2 + blank_start_frames
        blank_end_frames = max(frame_index - framecount + self.frames_per_clip // 2 + 1, 0)
        real_frames = self.frames_per_clip - blank_start_frames - blank_end_frames
        with VideoReader(self.video_list[movie_index], assume_writer_style=True) as reader:
            for i in range(real_frames):
                images.append(reader[i + start_frame])

        if log.isEnabledFor(logging.DEBUG):
            log.debug('idx: {} st: {} blank_start: {} blank_end: {} real: {} total: {}'.format(frame_index,
                                                                                               start_frame,
                                                                                               blank_start_frames,
                                                                                               blank_end_frames,
                                                                                               real_frames, framecount))
        if blank_end_frames > 0:
            for i in range(blank_end_frames):
                h = self.metadata['height'][movie_index]
                w = self.metadata['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))

        if self.transform:
            images_transformed = []
            seed = np.random.randint(2147483647)
            for image in images:
                random.seed(seed)
                images_transformed.append(self.transform(image))
            images = images_transformed
        # images are now numpy arrays of shape 3, H, W
        # stacking in the first dimension changes to 3, T, H, W, compatible with Conv3D
        images = np.stack(images, axis=1)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('images shape: {}'.format(images.shape))

        # # it's faster to stack the images, perform the augmentation on the stack, then unstack!
        # # assumes 3-channel RGB frames
        # if self.conv_mode == '3d':
        #     images = torch.stack([images[i * 3:i * 3 + 3, ...]
        #                           for i in range(self.frames_per_clip)], dim=1)
        # if log.isEnabledFor(logging.DEBUG):
        #     log.debug('images shape after 3d -> 4d: {}'.format(images.shape))
        # print(images.shape)
        outputs = {'images': images}
        if self.supervised:
            label = self.labels[index]
            if self.reduce:
                label = np.where(label)[0][0].astype(np.int64)
            outputs['labels'] = label
        return outputs


class SingleSequenceDataset(data.Dataset):
    """PyTorch Dataset for loading a set of saved 1d features and one-hot labels for Action Detection.

        Features:
            - Loads a set of sequential frames and sequential one-hot labels
            - loads by indexing from an HDF5 dataset, given a dataset name (latent_name)
            - Pads beginning or end so that every label has a corresponding clip
            - Optionally loads two-stream features

        Example:
            dataset = SequenceDataset(['features1.h5', 'features2.h5'], label_files=['labels1.csv', 'labels2.csv',
                h5_key='CNN_features', sequence_length=180, is_two_stream=True)
            features, labels = dataset(np.random.randint(low=0, high=len(dataset))
            print(features.shape)
            # 180 x 1024
            print(labels.shape)
            # assuming there are 5 classes in dataset
            # ~5 x 180
        """

    def __init__(self, h5file: Union[str, os.PathLike], labelfile: Union[str, os.PathLike],
                 h5_key: str, sequence_length: int = 60,
                 dimension=None, nonoverlapping: bool = True,
                 store_in_ram: bool = True, is_two_stream: bool = False, return_logits=False,
                 reduce: bool = False):
        assert os.path.isfile(h5file)
        self.h5file = h5file
        if labelfile is not None:
            assert os.path.isfile(labelfile)
            self.supervised = True
        else:
            self.supervised = False
        self.labelfile = labelfile

        self.sequence_length = sequence_length
        self.dimension = dimension
        self.nonoverlapping = nonoverlapping
        self.store_in_ram = store_in_ram
        self.is_two_stream = is_two_stream
        if self.supervised:
            self.label = read_labels(self.labelfile)
            # normally we want row vectors, but in this case we want columns
            self.label = self.label.T

        self.key = h5_key
        if self.is_two_stream:
            self.flow_key = self.key + '/flow_features'
            self.image_key = self.key + '/spatial_features'

        self.return_logits = return_logits
        self.logit_key = self.key + '/logits'
        # with h5py.File(h5file, 'r') as f:
        #     print(list(f.keys()))
        #     dataset = f[self.key]
        #     print(list(dataset.keys()))
        #     print(f[self.logit_key][:].shape)
        #     print(self.h5file)
        # print('logit key: {}'.format(self.logit_key))
        self.reduce = reduce
        self.verify_keys()

        if self.store_in_ram:
            self.logits, self.sequence = self.load_sequence()
        self.verify_dataset()

        if self.nonoverlapping:
            self.starts = []
            self.ends = []

            n = self.shape[1]
            # label_n = self.label.shape[1]
            starts = np.arange(n, step=sequence_length)
            ends = np.roll(np.copy(starts), -1)
            ends[-1] = n

            self.starts = starts
            self.ends = ends
            self.N = len(starts)
        else:
            ndim, seq_n = self.shape
            self.N = seq_n
            self.starts = None
            self.ends = None
        if self.supervised:
            self.class_counts = (self.label == 1).sum(axis=1)
            self.num_pos = (self.label == 1).sum(axis=1)
            self.num_neg = np.logical_not((self.label == 1)).sum(axis=1)

    def verify_dataset(self):
        log.debug('verifying dataset')
        # num_features = None

        if self.store_in_ram:
            logits, sequence = self.sequence
        else:
            logits, sequence = self.load_sequence()
        seq_shape = sequence.shape
        self.shape = seq_shape

        num_features = seq_shape[0]
        seq_n = seq_shape[1]
        if self.supervised:
            label_n = self.label.shape[1]
            if seq_n != label_n:
                raise ValueError('sequences and labels have different shape: {}, {}'.format(
                    seq_shape, self.label.shape))

        # making sure loading one sequence works
        log.debug('Testing loading one sequence...')
        sequence = self.load_sequence(start_ind=0, end_ind=1024)
        self.num_features = num_features

    def index_to_onehot(self, label, num_classes):
        new_label = np.zeros((num_classes, label.shape[0]), dtype=label.dtype)
        new_label[label, np.arange(label.shape[0])] = 1
        return new_label

    #     def get_sequence_shape(self):
    #         assert (os.path.isfile(self.h5file))
    #         with h5py.File(self.h5file, 'r') as f:
    #             if self.is_two_stream:
    #                 flow_shape = f[self.flow_key].shape
    #                 image_shape = f[self.image_key].shape
    #                 assert (flow_shape == image_shape)
    #                 if flow_shape[0] > flow_shape[1]:
    #                     flow_shape = flow_shape[::-1]
    #                 if image_shape[0] > image_shape[1]:
    #                     image_shape = image_shape[::-1]
    #                 shape = (image_shape[0] + flow_shape[0], flow_shape[1])
    #             else:
    #                 shape = f[self.key].shape
    #                 if shape[0] > shape[1]:
    #                     shape = shape[::-1]
    #         return shape

    def load_sequence(self, start_ind: int = None, end_ind: int = None):
        with h5py.File(self.h5file, 'r') as f:
            if start_ind is None:
                start_ind = 0
            if self.is_two_stream:
                flow_shape = f[self.flow_key].shape
                image_shape = f[self.image_key].shape
                assert (len(flow_shape) == 2)
                assert (flow_shape == image_shape)
                # we want each timepoint to be one COLUMN
                # assume each video has more frames than dimension of activation vector
                if end_ind is None:
                    end_ind = flow_shape[0]

                flow_feats = f[self.flow_key][start_ind:end_ind, :].T
                image_feats = f[self.image_key][start_ind:end_ind, :].T
                sequence = np.concatenate((image_feats, flow_feats), axis=0)
            else:
                shape = f[self.key].shape
                if shape[0] > shape[1]:
                    if end_ind is None:
                        end_ind = shape[0]
                    sequence = f[self.key][start_ind:end_ind, :].T
                else:
                    if end_ind is None:
                        end_ind = shape[1]
                    sequence = f[self.key][:, start_ind:end_ind]
            logits = f[self.logit_key][start_ind:end_ind, :].T

        # reduce according to dimension
        if self.dimension is not None:
            if self.dimension > sequence.shape[0]:
                warnings.warn('requested dimension is {} but sequence {} shape is only {}'.format(
                    self.dimension, self.h5file, sequence.shape))
            else:
                sequence = sequence[:self.dimension, ...]
        self.num_features = sequence.shape[0]

        return logits, sequence

    def verify_keys(self):
        log.debug('verifying keys')
        with h5py.File(self.h5file, 'r') as f:
            if not self.key in f:
                raise ValueError('Latent key {} not found in file {}. \n Available keys: {}'.format(self.key,
                                                                                                    self.h5file,
                                                                                                    list(f.keys())))
            if self.is_two_stream:
                assert self.flow_key in f, 'flow key {} not found in file {}'.format(self.flow_key,
                                                                                     self.h5file)
                assert self.image_key in f, 'image key {} not found in file {}'.format(self.image_key,
                                                                                       self.h5file)
            if self.return_logits:
                try:
                    logits = f[self.logit_key][:]
                except BaseException as e:
                    print('logits not found in keys: {}'.format(list(f[self.key].keys())))

    def __len__(self):
        return self.N

    def __getitem__(self, index: int) -> dict:
        if self.nonoverlapping:
            start = self.starts[index]
            end = self.ends[index]

            indices = np.arange(start, end)
            if self.supervised:
                labels = self.label[:, indices]

            # if self.store_in_ram:
            #     logits, values = self.logits[:, indices], self.sequence[:, indices]
            # else:
            #     logits, values = self.load_sequence(start, end)
            # if values.shape[1] < self.sequence_length:
            #     pad_right = self.sequence_length - values.shape[1]
            # else:
            #     pad_right = 0
            pad_right = 0
            if len(indices) < self.sequence_length:
                pad_right = self.sequence_length - len(indices)
            pad_left = 0
        else:
            middle = index
            start = middle - self.sequence_length // 2
            end = middle + self.sequence_length // 2
            N = self.shape[1]
            if start < 0:
                pad_left = np.abs(start)
                pad_right = 0
                start = 0
            elif end > N:
                pad_left = 0
                pad_right = end - N
                end = N
            else:
                pad_left = 0
                pad_right = 0
            indices = np.arange(start, end)

        # print(start, middle, end)
        if self.store_in_ram:
            logits, values = self.logits[:, indices], self.sequence[:, indices]
        else:
            logits, values = self.load_sequence(start, end)

        if (len(indices) + pad_left + pad_right) != self.sequence_length:
            import pdb;
            pdb.set_trace()

        if log.isEnabledFor(logging.DEBUG):
            print('start: {} end: {}'.format(start, end))

        logits = np.pad(logits, ((0, 0), (pad_left, pad_right)), mode='constant')
        values = np.pad(values, ((0, 0), (pad_left, pad_right)), mode='constant')
        if self.supervised:
            labels = self.label[:, indices].astype(np.int64)
            labels = np.pad(labels, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-1)
            labels = torch.from_numpy(labels).to(torch.long)
        # for torch dataloaders
        #         values = values.T
        #         labels = labels.T
        logits = torch.from_numpy(logits).float()
        values = torch.from_numpy(values).float()

        out = {'features': values, 'logits': logits}

        if self.supervised:
            out['labels'] = labels

        return out

        # if self.supervised:
        #     if self.return_logits:
        #         return values, logits, labels
        #     else:
        #         return values, labels
        # else:
        #     if self.return_logits:
        #         return values, logits
        #     else:
        #         return values

    def __del__(self):
        if hasattr(self, 'sequences'):
            del self.sequences
        if hasattr(self, 'labels'):
            del self.labels


class SequenceDataset(data.Dataset):
    """ Simple wrapper around SingleSequenceDataset for smoothly loading multiple sequences """

    def __init__(self, h5files: list, labelfiles: list, *args, **kwargs):
        datasets = []
        for i, (h5file, labelfile) in enumerate(zip(h5files, labelfiles)):
            dataset = SingleSequenceDataset(h5file, labelfile, *args, **kwargs)
            datasets.append(dataset)

            if i == 0:
                class_counts = dataset.class_counts
                num_pos = dataset.num_pos
                num_neg = dataset.num_neg
                num_features = dataset.num_features
            else:
                class_counts += dataset.class_counts
                num_pos += dataset.num_pos
                num_neg += dataset.num_neg
                assert dataset.num_features == num_features

        self.dataset = data.ConcatDataset(datasets)
        self.class_counts = class_counts
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_features = num_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class TwoStreamDataset(data.Dataset):
    """PyTorch Dataset for loading pre-computed optic flow. DEPRECATED """

    def __init__(self, rgb_list: list, flow_list: list, label_list=None, rgb_frames: int = 1, flow_frames: int = 10,
                 spatial_transform=None,
                 color_transform=None, reduce: bool = False, verbose: bool = False,
                 flow_style: str = 'linear', flow_max: int = 5):
        # spatial transforms are applied to both image and flow
        # color transform are applied only to RGB videos
        assert (len(rgb_list) > 0)
        assert (len(flow_list) > 0)
        assert (len(rgb_list) == len(flow_list))
        # assert(flow_style in ['polar', 'linear'])
        self.movies = {}
        self.movies['rgb'] = rgb_list
        self.movies['flow'] = flow_list

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.frames_per_clip = {}
        self.frames_per_clip['rgb'] = rgb_frames
        self.frames_per_clip['flow'] = flow_frames
        # self.rgb_frames = rgb_frames
        # self.flow_frames = flow_frames
        self.label_files = label_list
        self.supervised = self.label_files is not None
        # self.supervised = supervised
        self.reduce = reduce
        self.verbose = verbose

        if flow_style == 'linear':
            self.convert = partial(flow_utils.rgb_to_flow, maxval=flow_max)
        elif flow_style == 'polar':
            self.convert = partial(flow_utils.rgb_to_flow_polar, maxval=flow_max)
        elif flow_style == 'rgb':
            self.convert = self.no_conversion
        else:
            raise ValueError('Unknown optical flow style: {}'.format(flow_style))

        # find labels given the filename of a video, load, save as an attribute for fast reading
        if self.supervised:
            self.purge_unlabeled_videos()
            labels, class_counts, num_labels, num_pos, num_neg = read_all_labels(self.label_files)
            self.labels = labels
            self.class_counts = class_counts
            self.num_labels = num_labels
            self.num_pos = num_pos
            self.num_neg = num_neg
            if self.verbose:
                print('label shape: {}'.format(self.labels.shape))

        self.metadata = {}
        self.metadata['rgb'] = self.extract_metadata(self.movies['rgb'])
        self.metadata['flow'] = self.extract_metadata(self.movies['flow'])
        if verbose:
            print('RGB')
            print(self.metadata['rgb'])
            print('flows')
            print(self.metadata['flow'])
        for rgb_n, flow_n in zip(self.metadata['rgb']['framecount'], self.metadata['flow']['framecount']):
            assert (rgb_n == flow_n)
        if self.supervised:
            assert (len(self.metadata['rgb']['name']) == self.num_labels)
        self.N = self.metadata['rgb']['total_sequences']

        self.totensor = transforms.ToTensor()

    def purge_unlabeled_videos(self):
        if not self.supervised:
            return
        valid_videos = []
        valid_labels = []
        valid_flows = []
        for i in range(len(self.label_list)):
            label = read_labels(self.label_list[i])
            has_unlabeled_frames = np.any(label == -1)
            if not has_unlabeled_frames:
                valid_videos.append(self.video_list[i])
                valid_labels.append(self.label_list[i])
                valid_flows.append(self.flow_list[i])
        self.movies['rgb'] = valid_videos
        self.label_list = valid_labels
        self.movies['flow'] = valid_flows

    def no_conversion(self, inputs):
        return (inputs)

    def extract_metadata(self, video_list):
        metadata = {'name': [], 'width': [], 'height': [], 'framecount': []}
        total_sequences = 0
        start_indices = []
        for i, video in enumerate(video_list):
            ret, width, height, framecount = get_video_metadata(video)
            if ret:
                metadata['name'].append(video)
                metadata['width'].append(width)
                metadata['height'].append(height)
                metadata['framecount'].append(framecount)
                total_sequences += framecount
                start_indices.append(total_sequences)
            else:
                raise ValueError('error loading video: {}'.format(video))
        metadata['total_sequences'] = total_sequences
        metadata['start_indices'] = start_indices
        return (metadata)

    def __len__(self):
        return (self.N - 1)

    def read_clip(self, movie_index, frame_index, frames_per_clip, style='rgb'):
        assert (style in ['rgb', 'flow'])
        images = []
        blank_start_frames = max(frames_per_clip // 2 - frame_index, 0)
        if blank_start_frames > 0:
            for i in range(blank_start_frames):
                h = self.metadata[style]['height'][movie_index]
                w = self.metadata[style]['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))
        framecount = self.metadata[style]['framecount'][movie_index]
        # cap = cv2.VideoCapture(self.movies[style][movie_index])
        start_frame = frame_index - self.frames_per_clip[style] // 2 + blank_start_frames
        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        blank_end_frames = max(frame_index - framecount + self.frames_per_clip[style] // 2 + 1, 0)
        real_frames = self.frames_per_clip[style] - blank_start_frames - blank_end_frames
        if self.verbose:
            print('idx: {} st: {} blank_start: {} blank_end: {} real: {} total: {}'.format(frame_index,
                                                                                           start_frame,
                                                                                           blank_start_frames,
                                                                                           blank_end_frames,
                                                                                           real_frames, framecount))

        with VideoReader(self.movies[style][movie_index]) as reader:
            for i in range(real_frames):
                images.append(reader[i + start_frame])

        # for i in range(real_frames):
        #     if self.verbose:
        #         print('pos_frames:{}'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))))
        #     # print(start_frame+i)
        #     ret, frame = cap.read()
        #     if not ret:
        #         raise ValueError('Error reading frame {} of movie {}'.format(frame_index,
        #                                                                     self.movies[style][movie_index]))
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     images.append(frame)
        if blank_end_frames > 0:
            for i in range(blank_end_frames):
                h = self.metadata[style]['height'][movie_index]
                w = self.metadata[style]['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))

        # cap.release()
        return (images)

    def __getitem__(self, index: int):
        movie_index = bisect.bisect_right(self.metadata['rgb']['start_indices'], index)
        frame_index = index - self.metadata['rgb']['start_indices'][movie_index - 1] if movie_index > 0 else index
        rgb = self.read_clip(movie_index, frame_index, self.frames_per_clip['rgb'],
                             'rgb')

        flows_raw = self.read_clip(movie_index, frame_index, self.frames_per_clip['flow'],
                                   'flow')
        flow = []
        for f in flows_raw:
            if f is not None:
                flow.append(self.convert(f))
        # list of [H,W,3] -> np.array [H,W,3*images]
        rgb = np.concatenate(rgb, 2)
        # list of [H,W,2] -> np.array [H,W,2*images]
        flow = np.concatenate(flow, 2)

        # NOTE: THE SAME SPATIAL AUGMENTATION MUST BE APPLIED TO BOTH RGB AND FLOW FRAMES
        # e.g. if you're random cropping, you better be taking the same crop!
        if self.spatial_transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            rgb = self.spatial_transform(rgb)
            random.seed(seed)
            flow = self.spatial_transform(flow)

        # colorspace augmentation, brightness, contrast, etc.
        # don't augment the "colors" of a flow image!
        if self.color_transform is not None:
            rgb = self.color_transform(rgb)

        # always assume you want a tensor out
        flow = self.totensor(flow)
        rgb = self.totensor(rgb)

        if self.supervised:
            label = self.labels[index]
            if self.reduce:
                label = np.where(label)[0][0].astype(np.int64)
            return (rgb, flow, label)
        else:
            return (rgb, flow)


class KineticsDataset(data.Dataset):
    """ Dataset for Kinetics. Not recommended. Use DALI instead """

    def __init__(self, kinetics_dir, split, mode, supervised: bool = True,
                 rgb_frames: int = 1, flow_frames: int = 10, spatial_transform=None,
                 color_transform=None, reduce: bool = False,
                 flow_style: str = 'rgb', flow_max: int = 10, conv_mode='2d'):
        assert (mode in ['rgb', 'flow', 'both'])
        assert (conv_mode in ['2d', '3d'])
        self.mode = mode
        self.conv_mode = conv_mode
        splitdir = os.path.join(kinetics_dir, split)
        self.kinetics_dir = kinetics_dir
        self.splitdir = splitdir

        metadata_file = os.path.join(kinetics_dir, split + '_metadata.csv')
        metadata_flow = os.path.join(kinetics_dir, split + '_flow_metadata.csv')

        self.movies = {}
        self.frames_per_clip = {}

        if mode == 'rgb':
            if not os.path.isfile(metadata_file):
                rgb_df = extract_metadata(splitdir)
            else:
                rgb_df = pd.read_csv(metadata_file, index_col=0)
            rgb_list = rgb_df['name'].values
            assert (len(rgb_list) > 0)
            self.movies['rgb'] = rgb_list
            self.frames_per_clip['rgb'] = rgb_frames

        elif mode == 'flow':
            if not os.path.isfile(metadata_flow):
                flow_df = extract_metadata(splitdir, is_flow=True)
            else:
                flow_df = pd.read_csv(metadata_flow, index_col=0)
            flow_list = flow_df['name'].values
            assert (len(flow_list) > 0)
            self.movies['flow'] = flow_list
            self.frames_per_clip['flow'] = flow_frames
        elif mode == 'both':
            if not os.path.isfile(metadata_file) or not os.path.isfile(metadata_flow):
                rgb_df, flow_df = extract_metadata_twostream(splitdir)
            else:
                rgb_df = pd.read_csv(metadata_file, index_col=0)
                flow_df = pd.read_csv(metadata_flow, index_col=0)

            rgb_list = rgb_df['name'].values
            assert (len(rgb_list) > 0)
            self.movies['rgb'] = rgb_list
            flow_list = flow_df['name'].values
            assert (len(flow_list) > 0)
            self.movies['flow'] = flow_list
            assert (len(rgb_list) == len(flow_list))
            self.frames_per_clip = {'rgb': rgb_frames, 'flow': flow_frames}

        # spatial transforms are applied to both image and flow
        # color transform are applied only to RGB videos
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform

        self.supervised = supervised
        self.reduce = reduce
        # self.verbose = verbose

        if flow_style == 'linear':
            self.convert = partial(flow_utils.rgb_to_flow, maxval=flow_max)
        elif flow_style == 'polar':
            self.convert = partial(flow_utils.rgb_to_flow_polar, maxval=flow_max)
        elif flow_style == 'rgb':
            self.convert = self.no_conversion
        else:
            raise ValueError('Unknown optical flow style: {}'.format(flow_style))

        if self.supervised:
            if mode == 'rgb' or mode == 'both':
                self.labels = rgb_df['action_int'].values
            else:
                self.labels = flow_df['action_int'].values

            log.debug('label shape: {}'.format(self.labels.shape))
        self.metadata = {}
        if mode == 'rgb' or mode == 'both':
            self.metadata['rgb'] = rgb_df
        if mode == 'flow' or mode == 'both':
            self.metadata['flow'] = flow_df

        key = 'rgb' if mode == 'rgb' or mode == 'both' else 'flow'
        self.key = key
        total_sequences = 0
        start_indices = []
        for i, row in self.metadata[key].iterrows():
            nframes = row['framecount']
            total_sequences += nframes
            start_indices.append(total_sequences)
        self.metadata[key]['total_sequences'] = total_sequences
        self.metadata[key]['start_indices'] = start_indices

        if mode == 'rgb' or mode == 'both':
            log.debug('dataloader mode: RGB')
            log.debug(self.metadata['rgb'])
        if mode == 'flow' or mode == 'both':
            log.debug('dataloader mode: flows')
            log.debug(self.metadata['flow'])
        if mode == 'both':
            for rgb_n, flow_n in zip(self.metadata['rgb']['framecount'], self.metadata['flow']['framecount']):
                assert (rgb_n == flow_n)
        if self.supervised:
            assert (len(self.metadata[key]['name']) == len(self.labels))
        # have to do this indexing because pandas dataframes replicates the integer value
        # for every row
        self.N = int(self.metadata[key]['total_sequences'].values[0])

    def totensor(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        # handles the case where we've either input a tensor or a uint8 numpy array
        if type(image) == np.ndarray:
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            # backward compatibility
        if isinstance(image, torch.ByteTensor) or image.dtype == torch.uint8:
            return image.float().div(255)
        else:
            return image

    def no_conversion(self, inputs):
        return (inputs)

    def __len__(self):
        return self.N - 1

    def read_clip(self, movie_index, frame_index, frames_per_clip, style='rgb'):
        assert (style in ['rgb', 'flow'])
        images = []
        blank_start_frames = max(frames_per_clip // 2 - frame_index, 0)
        if blank_start_frames > 0:
            for i in range(blank_start_frames):
                h = self.metadata[style]['height'][movie_index]
                w = self.metadata[style]['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))
        framecount = self.metadata[style]['framecount'][movie_index]
        # cap = cv2.VideoCapture(self.movies[style][movie_index])
        start_frame = frame_index - self.frames_per_clip[style] // 2 + blank_start_frames
        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        blank_end_frames = max(frame_index - framecount + self.frames_per_clip[style] // 2 + 1, 0)
        real_frames = self.frames_per_clip[style] - blank_start_frames - blank_end_frames
        # don't want this to affect performance in normal cases. otherwise I'd just always leave the debug statement
        # and control this information using normal logger syntax
        if log.isEnabledFor(logging.DEBUG):
            log.debug('idx: {} st: {} blank_start: {} blank_end: {} real: {} total: {}'.format(frame_index,
                                                                                               start_frame,
                                                                                               blank_start_frames,
                                                                                               blank_end_frames,
                                                                                               real_frames, framecount))

        with VideoReader(self.movies[style][movie_index]) as reader:
            for i in range(real_frames):
                images.append(reader[i + start_frame])

        if blank_end_frames > 0:
            for i in range(blank_end_frames):
                h = self.metadata[style]['height'][movie_index]
                w = self.metadata[style]['width'][movie_index]
                images.append(np.zeros((h, w, 3), dtype=np.uint8))

        return images

    def __getitem__(self, index: int):

        movie_index = bisect.bisect_right(self.metadata[self.key]['start_indices'], index)
        frame_index = index - self.metadata[self.key]['start_indices'][movie_index - 1] if movie_index > 0 else index
        # mostly to stop pycharm from giving me an uninitialized warning
        rgb = None
        flow = None
        if self.mode == 'rgb' or self.mode == 'both':
            rgb = self.read_clip(movie_index, frame_index, self.frames_per_clip['rgb'],
                                 'rgb')
            rgb_channels = rgb[0].shape[-1]
            # list of [H,W,3] -> np.array [H,W,3*images]
            rgb = np.concatenate(rgb, 2)
        else:
            rgb_channels = None

        # print('Dataloader at beginning. mode: {}'.format(self.mode))
        # print('shape: {}'.format(rgb.shape))
        # for i in range(self.frames_per_clip['rgb']):
        #     im = rgb[..., i * 3:i * 3 + 3]
        #     print('{}: mean {:.4f} min {:.4f} max {:.4f}'.format(i, im.mean(), im.min(), im.max()))

        if self.mode == 'flow' or self.mode == 'both':
            flows_raw = self.read_clip(movie_index, frame_index, self.frames_per_clip['flow'],
                                       'flow')

            flow = []
            for f in flows_raw:
                flow.append(self.convert(f))
            flow_channels = flow[0].shape[-1]
            # list of [H,W,2] -> np.array [H,W,2*images]
            flow = np.concatenate(flow, 2)
        else:
            flow_channels = None

        # NOTE: THE SAME SPATIAL AUGMENTATION MUST BE APPLIED TO BOTH RGB AND FLOW FRAMES
        # e.g. if you're random cropping, you better be taking the same crop!
        if self.spatial_transform is not None:
            seed = np.random.randint(2147483647)
            if self.mode == 'rgb' or self.mode == 'both':
                random.seed(seed)
                # print('Dataloader Before augmentation: shape {}'.format(rgb.shape))
                # for i in range(self.frames_per_clip['rgb']):
                #     im = rgb[..., i * 3:i * 3 + 3]
                #     print('{}: mean {:.4f} min {:.4f} max {:.4f} std: {:.4f}'.format(i, im.mean(), im.min(), im.max(),
                #                                                                      im.std()))
                rgb = self.spatial_transform(rgb)
                # print('After augmentation: shape {}'.format(rgb.shape))
                # for i in range(self.frames_per_clip['rgb']):
                #     im = rgb[i * 3:i * 3 + 3, ...]
                #     print('{}: mean {:.4f} min {:.4f} max {:.4f} std: {:.4f}'.format(i, im.mean(), im.min(), im.max(),
                #                                                                      im.std()))
            if self.mode == 'flow' or self.mode == 'both':
                random.seed(seed)
                flow = self.spatial_transform(flow)

        # colorspace augmentation, brightness, contrast, etc.
        # don't augment the "colors" of a flow image!
        if self.color_transform is not None:
            if self.mode == 'rgb' or self.mode == 'both':
                rgb = self.color_transform(rgb)
                # print('After color transform: shape {}'.format(rgb.shape))
                # for i in range(self.frames_per_clip['rgb']):
                #     im = rgb[i * 3:i * 3 + 3, ...]
                #     print('{}: mean {:.4f} min {:.4f} max {:.4f} std: {:.4f}'.format(i, im.mean(), im.min(), im.max(),
                #                                                                      im.std()))

        # always assume you want a tensor out
        if self.mode == 'flow' or self.mode == 'both':
            flow = self.totensor(flow)
            if self.conv_mode == '3d':
                flow = torch.stack([flow[i * flow_channels:i * flow_channels + flow_channels, ...]
                                    for i in range(self.frames_per_clip['flow'])], dim=1)
        # print('Flow: {}'.format(flow))
        # print('RGB shape: {}'.format(rgb))
        if self.mode == 'rgb' or self.mode == 'both':
            # print('before totensor: shape {}'.format(rgb.shape))
            # for i in range(self.frames_per_clip['rgb']):
            #     im = rgb[i * 3:i * 3 + 3, ...]
            #     print('{}: mean {:.4f} min {:.4f} max {:.4f} std: {:.4f}'.format(i, im.mean(), im.min(), im.max(),
            #                                                                      im.std()))
            rgb = self.totensor(rgb)
            if self.conv_mode == '3d':
                rgb = torch.stack([rgb[i * rgb_channels:i * rgb_channels + rgb_channels, ...]
                                   for i in range(self.frames_per_clip['rgb'])], dim=1)

        # print('Dataloader at end: shape {}'.format(rgb.shape))
        # for i in range(self.frames_per_clip['rgb']):
        #     im = rgb[i * 3:i * 3 + 3, ...]
        #     # print('{}: mean {:.4f} min {:.4f} max {:.4f} std: {:.4f}'.format(i, im.mean(), im.min(), im.max(),
        #     #                                                                  im.std()))

        if self.supervised:
            label = torch.tensor(self.labels[movie_index]).long()
            # if self.reduce:
            #     label = np.where(label)[0][0].astype(np.int64)

            if self.mode == 'flow':
                return flow, label
            elif self.mode == 'rgb':
                return rgb, label
            elif self.mode == 'both':
                return rgb, flow, label

        if self.mode == 'flow':
            return flow
        elif self.mode == 'rgb':
            return rgb
        elif self.mode == 'both':
            return rgb, flow


def get_video_datasets(datadir: Union[str, os.PathLike], xform: dict, is_two_stream: bool = False,
                       reload_split: bool = True, splitfile: Union[str, os.PathLike] = None,
                       train_val_test: Union[list, np.ndarray] = [0.8, 0.1, 0.1], weight_exp: float = 1.0,
                       rgb_frames: int = 1, flow_frames: int = 10, supervised=True, reduce=False, flow_max: int = 5,
                       flow_style: str = 'linear', valid_splits_only: bool = True, conv_mode: str = '2d',
                       mean_by_channels: list = [0.5, 0.5, 0.5]):
    """ Gets dataloaders for video-based datasets.

    Parameters
    ----------
    datadir: str, os.PathLike
        absolute path to root directory containing data. e.g. /path/to/DATA
    xform: dict
        Dictionary of augmentations, e.g. from get_transforms
    is_two_stream: bool
        if True, tries to load a two-stream dataloader, with optic flow saved to disk
    reload_split: bool
        if True, tries to reload train/val/test split in splitfile (or tries to find splitfile). if False, makes a new
        train / val / test split
    splitfile: str, os.PathLike
        path to a yaml file containing train, val, test splits
    train_val_test: list, np.ndarray. shape (3,)
        contains fractions or numbers of elements in each split. see train_val_test_split
    weight_exp: float
        loss weights will be raised to this exponent. see DeepEthogram paper
    rgb_frames: int
        number of RGB frames in each training example. for hidden two-stream models, should be 11
    flow_frames: int
        number of optic flows in each training example if using pre-computed optic flow frames. deprecated
    for remaining arguments, see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    batch_size: int
        number of elements in batch
    shuffle: bool
        whether or not to shuffle dataset. should be kept True
    num_workers: int
        number of CPU workers to use to load data
    pin_memory: bool
        if True, copy batch data to GPU pinned memory
    drop_last: bool
        if True, drop the last batch because it might have different numbers of elements. Should be False, as
        VideoDataset pads images at the ends of movies and returns masked labels
    supervised: bool
        if True, return labels and require that files contain labels. if False, use all RGB movies in the dataset
        regardless of label status, and do not return labels
    reduce: bool
        if True, reduce one-hot labels to the index of the positive example. Used with softmax activation and NLL loss
        when there can only be one behavior at a time
    flow_max: int
        Number to divide flow results by for loading flows from disk. Deprecated
    flow_style: str
        one of linear, polar, or rgb. Denotes how pre-computed optic flows are stored on disk. Deprecated
    valid_splits_only: bool
        if True, require that each split has at least one example from each behavior. see train_val_test_split
    conv_mode: str
        one of '2d', '3d'. If 2D, batch will be of shape N, C*T, H, W. if 3D, batch will be of shape N, C, T, H, W

    Returns
    -------
    dataloaders: dict
        each of 'train', 'validation', 'test' will contain a PyTorch DataLoader:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        split contains the split dictionary, for saving
        keys for loss weighting are also added. see make_loss_weight for explanation
    """
    return_types = ['rgb']
    if is_two_stream:
        return_types += ['flow']
    if supervised:
        return_types += ['label']
    # records: dictionary of dictionaries. Keys: unique data identifiers
    # values: a dictionary corresponding to different files. the first record might be:
    # {'mouse000': {'rgb': path/to/rgb.avi, 'label':path/to/labels.csv} }
    records = projects.get_records_from_datadir(datadir)
    # some videos might not have flows yet, or labels. Filter records to only get those that have all required files
    records = projects.filter_records_for_filetypes(records, return_types)
    # returns a dictionary, where each split in ['train', 'val', 'test'] as a list of keys
    # each key corresponds to a unique directory, and has
    split_dictionary = get_split_from_records(records, datadir, splitfile, supervised, reload_split, valid_splits_only,
                                              train_val_test)
    # it's possible that your split has records that are invalid for the current task.
    # e.g.: you've added a video, but not labeled it yet. In that case, it will already be in your split, but it is
    # invalid for current purposes, because it has no label. Therefore, we want to remove it from the current split
    split_dictionary = remove_invalid_records_from_split_dictionary(split_dictionary, records)

    datasets = {}
    for i, split in enumerate(['train', 'val', 'test']):
        rgb = [records[i]['rgb'] for i in split_dictionary[split]]
        flow = [records[i]['flow'] for i in split_dictionary[split]]

        if split == 'test' and len(rgb) == 0:
            datasets[split] = None
            continue

        if supervised:
            labelfiles = [records[i]['label'] for i in split_dictionary[split]]
        else:
            labelfiles = None

        if is_two_stream:
            datasets[split] = TwoStreamDataset(rgb_list=rgb,
                                               flow_list=flow,
                                               rgb_frames=rgb_frames,
                                               flow_frames=flow_frames,
                                               spatial_transform=xform[split]['spatial'],
                                               color_transform=xform[split]['color'],
                                               label_list=labelfiles,
                                               reduce=reduce,
                                               flow_max=flow_max,
                                               flow_style=flow_style
                                               )
        else:
            datasets[split] = VideoDataset(rgb,
                                           labelfiles,
                                           frames_per_clip=rgb_frames,
                                           reduce=reduce,
                                           transform=xform[split],
                                           conv_mode=conv_mode,
                                           mean_by_channels=mean_by_channels)
    data_info = {'split': split_dictionary}

    if supervised:
        data_info['class_counts'] = datasets['train'].class_counts
        data_info['num_classes'] = len(data_info['class_counts'])
        pos_weight, softmax_weight = make_loss_weight(data_info['class_counts'],
                                                      datasets['train'].num_pos,
                                                      datasets['train'].num_neg,
                                                      weight_exp=weight_exp)
        data_info['pos'] = datasets['train'].num_pos
        data_info['neg'] = datasets['train'].num_neg
        data_info['pos_weight'] = pos_weight
        data_info['loss_weight'] = softmax_weight

    return datasets, data_info


def get_sequence_datasets(datadir: Union[str, os.PathLike], latent_name: str, sequence_length: int = 60,
                          is_two_stream: bool = True, nonoverlapping: bool = True, splitfile: str = None,
                          reload_split: bool = True, store_in_ram: bool = False, dimension: int = None,
                          train_val_test: Union[list, np.ndarray] = [0.8, 0.2, 0.0], weight_exp: float = 1.0,
                          supervised=True, reduce=False, valid_splits_only: bool = True,
                          return_logits=False) -> Tuple[dict, dict]:
    """ Gets dataloaders for sequence models assuming DeepEthogram file structure.

    Parameters
    ----------
    datadir: str, os.PathLike
        absolute path to directory. Will have sub-directories where actual data is stored
    latent_name: str
        Key of HDF5 dataset containing extracted features and probabilities
    sequence_length: int
        Number of elements in sequence
    is_two_stream: bool
        If True, look for image_features and flow_features in the HDF5 dataset
    nonoverlapping: bool
        If True, indexing into dataset will return non-overlapping sequences. With a sequence length of 10, for example,
        if nonoverlapping:
            sequence[0] contains data from frames [0:10], sequence[1]: frames [11:20], etc...
        else:
            sequence[0] contains data from frames [0:10], sequence[1]: frames[1:11], etc...
    splitfile: str
        path to a yaml file containing train, validation, test information. If none, make a new one
    reload_split: bool
        if True, try to reload a passed splitfile. Else, make a new split
    store_in_ram: bool
        if True, tries to store all the sequence information in RAM. Note: data IO is not a bottleneck unless you have
        a very slow hard drive. using this = True is not recommended
    dimension: int
        Can be used to reduce the dimensionality of inputs to your sequences. If you have 1024 features, but pass 256,
        will only use the top 256 features. Not recommended
    train_val_test: list, np.ndarray shape (3,)
        Fractions or numbers of elements to use in each split of the data, if creating a new one
    weight_exp: float
        Loss weights will be raised to this exponent. See DeepEthogram paper
    batch_size: int
        Batch size. Can be relatively large for sequences due to their light weight, but they also don't use batch
        normalization, so ¯\_(ツ)_/¯
    shuffle: bool
        if True, shuffle the data. Highly recommend keeping True
    num_workers: int
        Number of CPU workers to use to load data
    pin_memory: bool
        Set to true to reserve memory on the GPU. can improve performance. see
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    drop_last: bool
        If true, drop last batch. However, SequenceDatasets use padding and loss weighting, so you don't need to
        ever drop the last batch
    supervised: bool
        If True, return labels along with features
    reduce: bool
        if True,  return class indices as labels
        if False, return one-hot vectors as labels
    valid_splits_only: bool
        if True, require train, validation, and test to have at least one example from each class
    return_logits: bool
        if True, returns the logits from the feature extractor as well as the feature vectors

    Returns
    -------
    dataloaders: dict with keys:
        train: dataloader for train
        val: dataloader for validation
        test: dataloader for test
        class_counts: number of examples for each class
        pos: ibid
        neg: number of negative examples for each class
        pos_weight: loss weight for sigmoid activation / BCEloss
        loss_weight: loss weight for softmax activation / NLL loss

    """
    return_types = ['output']
    if supervised:
        return_types += ['label']

    # records: dictionary of dictionaries. Keys: unique data identifiers
    # values: a dictionary corresponding to different files. the first record might be:
    # {'mouse000': {'rgb': path/to/rgb.avi, 'label':path/to/labels.csv} }
    records = projects.get_records_from_datadir(datadir)
    # some videos might not have flows yet, or labels. Filter records to only get those that have all required files
    records = projects.filter_records_for_filetypes(records, return_types)
    # returns a dictionary, where each split in ['train', 'val', 'test'] as a list of keys
    # each key corresponds to a unique directory, and has
    split_dictionary = get_split_from_records(records, datadir, splitfile, supervised, reload_split, valid_splits_only,
                                              train_val_test)
    # it's possible that your split has records that are invalid for the current task.
    # e.g.: you've added a video, but not labeled it yet. In that case, it will already be in your split, but it is
    # invalid for current purposes, because it has no label. Therefore, we want to remove it from the current split
    split_dictionary = remove_invalid_records_from_split_dictionary(split_dictionary, records)
    # log.info('~~~~~ train val test split ~~~~~')
    # pprint.pprint(split_dictionary)

    splits = ['train', 'val', 'test']
    datasets = {}
    nonoverlapping = {'train': nonoverlapping, 'val': True, 'test': True}
    for split in splits:
        outputfiles = [records[i]['output'] for i in split_dictionary[split]]

        if split == 'test' and len(outputfiles) == 0:
            datasets[split] = None
            continue
        # h5file, labelfile = outputs[i]
        # print('making dataset:{}'.format(split))

        if supervised:
            labelfiles = [records[i]['label'] for i in split_dictionary[split]]
        else:
            labelfiles = None

        datasets[split] = SequenceDataset(outputfiles, labelfiles, latent_name, sequence_length,
                                          is_two_stream=is_two_stream, nonoverlapping=nonoverlapping[split],
                                          dimension=dimension,
                                          store_in_ram=store_in_ram, return_logits=return_logits, reduce=reduce)

    # figure out what our inputs to our model will be (D dimension)
    data_info = {'split': split_dictionary}
    data_info['num_features'] = datasets['train'].num_features

    if supervised:
        data_info['class_counts'] = datasets['train'].class_counts
        data_info['num_classes'] = len(data_info['class_counts'])
        pos_weight, softmax_weight = make_loss_weight(data_info['class_counts'],
                                                      datasets['train'].num_pos,
                                                      datasets['train'].num_neg,
                                                      weight_exp=weight_exp)
        data_info['pos'] = datasets['train'].num_pos
        data_info['neg'] = datasets['train'].num_neg
        data_info['pos_weight'] = pos_weight
        data_info['loss_weight'] = softmax_weight

    return datasets, data_info


def get_datasets_from_cfg(cfg: DictConfig, model_type: str, input_images: int = 1) -> Tuple[dict, dict]:
    """ Returns dataloader objects using a Hydra-generated configuration dictionary.

    This is the main entry point for getting dataloaders from the command line. it will return the correct dataloader
    with given hyperparameters for either flow, feature extractor, or sequence models.

    Parameters
    ----------
    cfg: DictConfig
        Hydra-generated (or OmegaConf) configuration dictionary
    model_type: str
        one of flow_generator, feature_extractor, sequence
        we need to specify model type and input images because the same config could be used to load any model type.
    input_images: int
        Number of images in each training example.
        input images must be specified because if you're training a feature extractor, the flow extractor might take
        11 images and the spatial model might take 1 image. End to end takes 11 images, because we select out the middle
        image for the spatial model in the end-to-end version

    Returns
    -------
    dataloaders: dict
        train, val, and test will contain a PyTorch dataloader for that data split. Will also contain other useful
        dataset-specific keys, e.g. number of examples in each class, how to weight the loss function, etc. for more
        information see the specific dataloader of the model you're training, e.g. get_video_dataloaders
    """
    #
    supervised = model_type != 'flow_generator'
    if model_type == 'feature_extractor' or model_type == 'flow_generator':
        arch = cfg[model_type].arch
        mode = '3d' if '3d' in arch.lower() else '2d'
        log.info('getting dataloaders: {} convolution type detected'.format(mode))
        xform = get_cpu_transforms(cfg.augs)

        if cfg.project.name == 'kinetics':
            raise NotImplementedError
        else:
            reduce = False
            if cfg.run.model == 'feature_extractor':
                if cfg.feature_extractor.final_activation == 'softmax':
                    reduce = True
            datasets, info = get_video_datasets(datadir=cfg.project.data_path,
                                                xform=xform,
                                                is_two_stream=False,
                                                reload_split=cfg.split.reload,
                                                splitfile=cfg.split.file,
                                                train_val_test=cfg.split.train_val_test,
                                                weight_exp=cfg.train.loss_weight_exp,
                                                rgb_frames=input_images,
                                                supervised=supervised,
                                                reduce=reduce,
                                                valid_splits_only=True,
                                                conv_mode=mode,
                                                mean_by_channels=cfg.augs.normalization.mean)

    elif model_type == 'sequence':
        datasets, info = get_sequence_datasets(cfg.project.data_path, cfg.sequence.latent_name,
                                               cfg.sequence.sequence_length, is_two_stream=True,
                                               nonoverlapping=cfg.sequence.nonoverlapping, splitfile=cfg.split.file,
                                               reload_split=True, store_in_ram=False, dimension=None,
                                               train_val_test=cfg.split.train_val_test,
                                               weight_exp=cfg.train.loss_weight_exp, supervised=True,
                                               reduce=cfg.feature_extractor.final_activation == 'softmax',
                                               valid_splits_only=True, return_logits=False)
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    return datasets, info
