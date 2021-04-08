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
from deepethogram.data.augs import get_cpu_transforms
from deepethogram.data.utils import purge_unlabeled_videos, get_video_metadata, extract_metadata, find_labelfile, \
    read_all_labels, get_split_from_records, remove_invalid_records_from_split_dictionary, \
    make_loss_weight
from deepethogram.data.keypoint_utils import load_dlcfile, interpolate_bad_values, stack_features_in_time, \
    expand_features_sturman
from deepethogram.file_io import read_labels

log = logging.getLogger(__name__)


# https://pytorch.org/docs/stable/data.html
class VideoIterable(data.IterableDataset):
    """Highly optimized Dataset for running inference on videos. 
    
    Features: 
        - Data is only read sequentially
        - Each frame is only read once
        - The input video is divided into NUM_WORKERS segments. Each worker reads its segment in parallel
        - Each clip is read with stride = 1. If sequence_length==3, the first clips would be frames [0, 1, 2], 
            [1, 2, 3], [2, 3, 4], ... etc
    """
    def __init__(self,
                 videofile: Union[str, os.PathLike],
                 transform,
                 sequence_length: int = 11,
                 num_workers: int = 0,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0]):
        """Cosntructor for video iterable

        Parameters
        ----------
        videofile : Union[str, os.PathLike]
            Path to video file
        transform : callable
            CPU transforms (cropping, resizing)
        sequence_length : int, optional
            Number of images in one clip, by default 11
        num_workers : int, optional
            [description], by default 0
        mean_by_channels : Union[list, np.ndarray], optional
            [description], by default [0, 0, 0]
        """
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
            yield {'images': np.stack(self.buffer, axis=1), 'framenum': self.cnt - 1 - self.sequence_length // 2}

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
            iter_start = -self.blank_start_frames
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

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


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
    def __init__(self,
                 videofile: Union[str, os.PathLike],
                 labelfile: Union[str, os.PathLike] = None,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
                 frames_per_clip: int = 1,
                 transform=None,
                 reduce: bool = True,
                 conv_mode: str = '2d',
                 keep_reader_open: bool = False):
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

    def get_zeros_image(self, c, h, w, channel_first: bool = True):
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
                try:
                    image = reader[i + start_frame]
                except Exception as e:
                    image = self._zeros_image.copy().transpose(1, 2, 0)
                    log.warning('Error {} on frame {} of video {}. Is the video corrupted?'.format(
                        e, index, self.videofile))
                if self.transform:
                    random.seed(seed)
                    image = self.transform(image)
                    images.append(image)

        images = self.prepend_with_zeros(images, blank_start_frames)
        images = self.append_with_zeros(images, blank_end_frames)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('idx: {} st: {} blank_start: {} blank_end: {} real: {} total: {}'.format(
                index, start_frame, blank_start_frames, blank_end_frames, real_frames, framecount))

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
        datasets, labels = [], []
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
                labels.append(dataset.labels)

        self.dataset = data.ConcatDataset(datasets)
        if labelfiles is not None:
            self.class_counts = class_counts
            self.num_pos = num_pos
            self.num_neg = num_neg
            self.num_labels = num_labels
            self.labels = np.concatenate(labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


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
    def __init__(self,
                 data_file: Union[str, os.PathLike],
                 labelfile: Union[str, os.PathLike],
                 N: int,
                 sequence_length: int = 60,
                 nonoverlapping: bool = True,
                 store_in_ram: bool = True,
                 reduce: bool = False,
                 stack_in_time: bool = False):

        assert os.path.isfile(data_file)
        if labelfile is not None:
            assert os.path.isfile(labelfile)
            self.supervised = True
            # after transpose, label will be of shape N_behaviors x T
            self.label = read_labels(labelfile).T
            self.class_counts = (self.label == 1).sum(axis=1)
            self.num_pos = (self.label == 1).sum(axis=1)
            self.num_neg = np.logical_not((self.label == 1)).sum(axis=1)
        else:
            self.supervised = False

        self.sequence_length = sequence_length
        self.nonoverlapping = nonoverlapping
        self.reduce = reduce
        self.starts = None
        self.ends = None
        # self.num_features = None
        self.N = N
        self.stack_in_time = stack_in_time

        self.compute_starts_ends()
        self.verify_dataset()

        tmp_sequence = self.__getitem__(0)  # self.read_sequence([0, 1])
        self.num_features = tmp_sequence['features'].shape[0]

    def read_sequence(self, indices):
        raise NotImplementedError

    def verify_dataset(self):
        raise NotImplementedError

    def compute_starts_ends(self):
        if self.nonoverlapping:
            self.starts = []
            self.ends = []

            # label_n = self.label.shape[1]
            starts = np.arange(self.N, step=self.sequence_length)
            ends = np.roll(np.copy(starts), -1)
            # ends[-1] = self.N
            ends[-1] = starts[-1] + self.sequence_length

            self.starts = starts
            self.ends = ends
        else:
            inds = np.arange(self.N)
            self.starts = inds - self.sequence_length // 2
            # if it's odd, should go from
            self.ends = inds + self.sequence_length//2 + \
                self.sequence_length % 2

    def __len__(self):
        return len(self.starts)

    def compute_indices_and_padding(self, index):
        start = self.starts[index]
        end = self.ends[index]

        # sequences close to the 0th frame are padded on the left
        if start < 0:
            pad_left = np.abs(start)
            pad_right = 0
            start = 0
        elif end > self.N:
            pad_left = 0
            pad_right = end - self.N
            end = self.N
        else:
            pad_left = 0
            pad_right = 0

        indices = np.arange(start, end)

        pad = (pad_left, pad_right)

        # stack in time compressed all features to one vector.
        # there's only one label: the one right at the requested index. This is to have temporal features
        # before and after the label as input, e.g. to an MLP
        # therefore, don't pad
        if self.stack_in_time:
            label_indices = index  # (end - start) // 2
            label_pad = (0, 0)
        else:
            label_indices = indices
            label_pad = pad

        assert (len(indices) + pad_left + pad_right) == self.sequence_length, \
                'indices: {} + pad_left: {} + pad_right: {} should equal seq len: {}'.format(
                len(indices), pad_left, pad_right, self.sequence_length)
        # if we are stacking in time, label indices should not be the sequence length
        if not self.stack_in_time:
            assert (len(label_indices) + label_pad[0] + label_pad[1]) == self.sequence_length, \
                    'label indices: {} + pad_left: {} + pad_right: {} should equal seq len: {}'.format(
                    len(label_indices), label_pad[0], label_pad[1], self.sequence_length)
        return indices, label_indices, pad, label_pad

    def __del__(self):
        if hasattr(self, 'sequence'):
            del self.sequence
        if hasattr(self, 'labels'):
            del self.labels

    def __getitem__(self, index: int) -> dict:
        indices, label_indices, pad, label_pad = self.compute_indices_and_padding(index)
        # import pdb; pdb.set_trace()
        # dictionary
        data = self.read_sequence(indices)

        # can be multiple things in "data", like "image features" and "logits" from feature extractors
        # all will be converted to float32
        output = {}
        pad_left, pad_right = pad
        for key, value in data.items():
            value = np.pad(value, ((0, 0), (pad_left, pad_right)), mode='constant')
            if self.stack_in_time:
                value = value.flatten()
            value = torch.from_numpy(value).float()
            output[key] = value

        pad_left, pad_right = label_pad
        # print(index)
        if self.supervised:
            # print(label_indices)
            labels = self.label[:, label_indices].astype(np.int64)
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            labels = np.pad(labels, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-1)
            # if we stack in time, we want to make sure we have labels of shape (N_behaviors,)
            # not (N_behaviors, 1)
            labels = labels.squeeze()
            labels = torch.from_numpy(labels).to(torch.long)
            output['labels'] = labels

            if labels.ndim > 1 and labels.shape[1] != output['features'].shape[1]:
                import pdb
                pdb.set_trace()

        return output


class KeypointDataset(SingleSequenceDataset):
    """Dataset for reading keypoints (e.g. from deeplabcut) and performing basis function expansion. 
    
    Currently, only an edited variant of Sturman et al.'s basis expansion is implemented
    Sturman, O., von Ziegler, L., Schläppi, C. et al. Deep learning-based behavioral analysis reaches human 
        accuracy and is capable of outperforming commercial solutions. Neuropsychopharmacol. 45, 1942–1952 (2020). 
        https://doi.org/10.1038/s41386-020-0776-y
    """
    def __init__(self,
                 data_file: Union[str, os.PathLike],
                 labelfile: Union[str, os.PathLike],
                 videofile: Union[str, os.PathLike],
                 expansion_method: str = 'sturman',
                 confidence_threshold: float = 0.9,
                 *args,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        data_file : Union[str, os.PathLike]
            Path to datafile, e.g. HDF5 or CSV with keypoints
        labelfile : Union[str, os.PathLike]
            CSV file containing binary labels
        videofile : Union[str, os.PathLike]
            Path to raw video. Needed for normalizing keypoints by height and width
        expansion_method : str, optional
            Defines how to perform basis function expansion, by default 'sturman'
        confidence_threshold : float, optional
            Values lower than this will be linearly interpolated, by default 0.9

        Raises
        ------
        NotImplementedError
            For basis function expansion. Currently, only 'sturman' is implemented: 
            
        """
        if expansion_method == 'sturman':
            self.expansion_func = expand_features_sturman
        else:
            raise NotImplementedError

        assert os.path.isfile(data_file)
        assert os.path.exists(videofile)

        keypoints, bodyparts, _ = load_dlcfile(data_file)

        N = keypoints.shape[0]

        with VideoReader(videofile) as reader:
            frame = reader[0]
            H, W = frame.shape[:2]
            # n_frames = len(reader)

        # chop off confidence after this
        keypoints = interpolate_bad_values(keypoints, confidence_threshold)[..., :2]
        features, columns = self.expansion_func(keypoints, bodyparts, H, W)

        # assume sequence is of shape Time x N_features
        # we want to return them in N_features x T, consistent with nn.Conv1d, which requires
        # inputs to be of shape N x C x T

        self.sequence = features.T
        self.shape = self.sequence.shape
        N = self.shape[1]

        # superclass needs to know number of samples, that's why it's down here
        super().__init__(data_file, labelfile, N, *args, **kwargs)

        # log.debug('keypoint features: {}'.format(columns))
        # print('keypoint features: {}'.format(columns))

    def verify_dataset(self):
        if self.supervised:
            assert self.label.shape[1] == self.sequence.shape[1], 'label {} and sequence {} shape do not match!'.format(
                self.label.shape, self.sequence.shape)

        assert self.sequence is not None

    def read_sequence(self, indices):
        data = {}
        data['features'] = self.sequence[:, indices]
        return data


class FeatureVectorDataset(SingleSequenceDataset):
    """Reads image and flow feature vectors from HDF5 files. 
    """
    def __init__(self,
                 data_file,
                 labelfile,
                 h5_key: str,
                 store_in_ram=False,
                 is_two_stream: bool = True,
                 *args,
                 **kwargs):

        self.is_two_stream = is_two_stream
        self.store_in_ram = store_in_ram

        assert os.path.isfile(data_file)
        self.key = h5_key
        if self.is_two_stream:
            self.flow_key = self.key + '/flow_features'
            self.image_key = self.key + '/spatial_features'
        self.logit_key = self.key + '/logits'
        self.data_file = data_file

        self.verify_dataset()
        data = self.read_features_from_disk(None, None)

        features_shape = data['features'].shape
        self.shape = features_shape
        self.N = self.shape[1]
        if self.store_in_ram:
            self.data = data
        else:
            del data

        # superclass needs to know number of samples, that's why it's down here
        super().__init__(data_file, labelfile, self.N, *args, **kwargs)

    def verify_dataset(self):
        with h5py.File(self.data_file, 'r') as f:
            assert self.logit_key in f

            if self.is_two_stream:
                assert self.flow_key in f
                assert self.image_key in f
                flow_shape = f[self.flow_key].shape
                image_shape = f[self.image_key].shape
                assert flow_shape[0] == image_shape[0]
                # self.N = image_shape[0]
            else:
                assert self.key in f
                shape = f[self.key].shape
                # self.N = shape[0]

    def read_features_from_disk(self, start_ind, end_ind):
        inds = slice(start_ind, end_ind)
        with h5py.File(self.data_file, 'r') as f:
            if self.is_two_stream:
                flow_shape = f[self.flow_key].shape
                image_shape = f[self.image_key].shape
                assert len(flow_shape) == 2
                assert (flow_shape == image_shape)
                # we want each timepoint to be one COLUMN
                flow_feats = f[self.flow_key][inds, :].T
                image_feats = f[self.image_key][inds, :].T
                sequence = np.concatenate((image_feats, flow_feats), axis=0)
            else:
                sequence = f[self.key][inds, :].T

            logits = f[self.logit_key][inds, :].T
        return dict(features=sequence, logits=logits)

    def read_sequence(self, indices):
        if self.store_in_ram:
            data = {'features': self.data['features'][:, indices], 'logits': self.data['logits'][:, indices]}
        else:
            # assume indices are in order
            # we use the start and end so that we can slice without knowing the exact size of the dataset
            data = self.read_features_from_disk(indices[0], indices[-1] + 1)
        return data


class SequenceDataset(data.Dataset):
    """ Simple wrapper around SingleSequenceDataset for smoothly loading multiple sequences """
    def __init__(self,
                 datafiles: list,
                 labelfiles: list,
                 videofiles: list = None,
                 is_keypoint: bool = False,
                 *args,
                 **kwargs):
        datasets = []
        for i, (datafile, labelfile) in enumerate(zip(datafiles, labelfiles)):
            if is_keypoint:
                assert videofiles is not None
                dataset = KeypointDataset(datafile, labelfile, videofiles[i], *args, **kwargs)
            else:
                dataset = FeatureVectorDataset(datafile, labelfile, *args, **kwargs)
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


def get_video_datasets(datadir: Union[str, os.PathLike],
                       xform: dict,
                       is_two_stream: bool = False,
                       reload_split: bool = True,
                       splitfile: Union[str, os.PathLike] = None,
                       train_val_test: Union[list, np.ndarray] = [0.8, 0.1, 0.1],
                       weight_exp: float = 1.0,
                       rgb_frames: int = 1,
                       flow_frames: int = 10,
                       supervised=True,
                       reduce=False,
                       flow_max: int = 5,
                       flow_style: str = 'linear',
                       valid_splits_only: bool = True,
                       conv_mode: str = '2d',
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


def get_sequence_datasets(datadir: Union[str, os.PathLike],
                          latent_name: str,
                          sequence_length: int = 60,
                          is_two_stream: bool = True,
                          nonoverlapping: bool = True,
                          splitfile: str = None,
                          reload_split: bool = True,
                          store_in_ram: bool = False,
                          train_val_test: Union[list, np.ndarray] = [0.8, 0.2, 0.0],
                          weight_exp: float = 1.0,
                          supervised=True,
                          reduce=False,
                          valid_splits_only: bool = True,
                          is_keypoint: bool = False,
                          stack_in_time: bool = False) -> Tuple[dict, dict]:
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
    return_types = []
    if is_keypoint:
        log.info('Creating keypoint datasets, with feature expansion. Might take a few minutes')
        return_types.append('keypoint')
    else:
        return_types.append('output')
    if supervised:
        return_types.append('label')

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
    # if stack_in_time, nonoverlapping would make us skip a bunch of labels
    nonoverlapping = {'train': nonoverlapping, 'val': not stack_in_time, 'test': not stack_in_time}
    for split in splits:
        if is_keypoint:
            videofiles = [records[i]['rgb'] for i in split_dictionary[split]]
            datafiles = [records[i]['keypoint'] for i in split_dictionary[split]]
        else:
            videofiles = None
            datafiles = [records[i]['output'] for i in split_dictionary[split]]

        if split == 'test' and len(datafiles) == 0:
            datasets[split] = None
            continue
        # h5file, labelfile = outputs[i]
        # print('making dataset:{}'.format(split))

        if supervised:
            labelfiles = [records[i]['label'] for i in split_dictionary[split]]
        else:
            labelfiles = None

        # todo: figure out a nice way to be able to pass arguments to one subclass that don't exist in the other
        # example: is_two_stream, latent_name
        if is_keypoint:
            datasets[split] = SequenceDataset(datafiles,
                                              labelfiles,
                                              videofiles,
                                              sequence_length=sequence_length,
                                              nonoverlapping=nonoverlapping[split],
                                              store_in_ram=store_in_ram,
                                              reduce=reduce,
                                              is_keypoint=is_keypoint,
                                              stack_in_time=stack_in_time)
        else:
            datasets[split] = SequenceDataset(datafiles,
                                              labelfiles,
                                              videofiles=videofiles,
                                              sequence_length=sequence_length,
                                              h5_key=latent_name,
                                              is_two_stream=is_two_stream,
                                              nonoverlapping=nonoverlapping[split],
                                              store_in_ram=store_in_ram,
                                              reduce=reduce,
                                              is_keypoint=is_keypoint,
                                              stack_in_time=stack_in_time)

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
        datasets, info = get_sequence_datasets(cfg.project.data_path,
                                               cfg.sequence.latent_name,
                                               cfg.sequence.sequence_length,
                                               is_two_stream=True,
                                               nonoverlapping=cfg.sequence.nonoverlapping,
                                               splitfile=cfg.split.file,
                                               reload_split=True,
                                               store_in_ram=False,
                                               train_val_test=cfg.split.train_val_test,
                                               weight_exp=cfg.train.loss_weight_exp,
                                               supervised=True,
                                               reduce=cfg.feature_extractor.final_activation == 'softmax',
                                               valid_splits_only=True,
                                               stack_in_time=cfg.sequence.arch == 'mlp',
                                               is_keypoint=cfg.sequence.input_type == 'keypoints')
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    return datasets, info
