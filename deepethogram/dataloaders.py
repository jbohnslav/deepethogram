import bisect
import glob
import multiprocessing as mp
import os
import pprint
import random
import warnings
from functools import partial
from pprint import pformat
from typing import Union, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
# from deepethogram.flow_generator import utils as flow_utils
# from deepethogram.flow_generator.utils import rgb_to_flow, rgb_to_flow_polar
from omegaconf import DictConfig
from opencv_transforms import transforms
from torch.utils import data
from tqdm import tqdm

from deepethogram import projects
from deepethogram import utils
from deepethogram.file_io import read_labels, VideoReader

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.backend import TensorListCPU
    from nvidia.dali.plugin import pytorch
except ImportError:
    dali = False
    # print('DALI not loaded...')
else:
    dali = True
import logging

log = logging.getLogger(__name__)


class SequentialIterator:
    """Optimized loader to read short clips of videos to disk only using sequential video reads.

    Use case: Your model needs ~11 frames concatenated together as input, and you want to do inference on an entire
        video. Every batch only reads one frame. The newest frame is added to the last position in the stack.
        The now 12th frame is removed from the stack.
        Other dataloaders with this use case would re-load all 11 frames.
    Features:
        - Only uses sequential reads
        - Only reads one frame per new batch
        - Only moves one single frame to GPU per batch
        - Option to return labels as well

    Example:
        iterator = SequentialIterator('movie.avi', num_images=11, device='cuda:0')
        for batch in iterator:
            outputs = model(batch)
            # do something
    """

    def __init__(self, videofile, num_images: int, device, transform=None, stack_channels: bool = True,
                 batch_size: int = 1, supervised: bool = False):
        """Initializes a SequentialIterator object

        Args:
            videofile: path to a video file
            num_images: how many sequential images to load. e.g. 11: resulting tensor will have 33 channels
            device: torch.device object or string. string examples: 'cpu', 'cuda:0'
            transform: either None or a TorchVision.transforms object or opencv_transforms object
            stack_channels: if True, returns Tensor of shape (num_images*3, H, W). if False, returns Tensor of shape
                (num_images, 3, H, W)
            batch_size: 1. Other values not implemented
            supervised: whether or not to return a label. False: for self-supervision
        Returns:
            SequentialIterator object
        """
        assert (os.path.isfile(videofile))
        # self.num_flows = num_flows
        self.num_images = int(num_images)

        self.reader = VideoReader(videofile)

        # self.cap = cv2.VideoCapture(videofile)
        # self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = self.reader.nframes
        if self.num_images == 1:
            self.num_batches = self.nframes
        elif (self.num_images % 2) == 0:
            self.num_batches = self.nframes + self.num_images // 2
        else:
            self.num_batches = self.nframes + self.num_images // 2
            # print(self.num_batches)

        self.index = 0
        self.true_index = 0
        self.transform = transform
        self.device = device
        if self.device is not None:
            assert (type(device) == torch.device)
        self.stack_channels = stack_channels
        if batch_size > 1:
            raise NotImplemented
        self.batch_size = batch_size
        self.supervised = supervised
        if self.supervised:
            labelfile, label_type = find_labelfile(videofile)
            label = read_labels(labelfile)
            H, W = label.shape
            if W > H:
                label = label.T
            self.label = label
        self.seed = np.random.randint(2147483647)

    def process_one_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Processes one frame, including augmentations"""
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # augmentations
            # set the random seed here so that if there's something like randomcropping (not advised!!) the same random
            # crop is performed for every image in the dataset
            random.seed(self.seed)
            frame = self.transform(frame)
        else:
            frame = frame.astype(np.float32) / 255
        if type(frame) != torch.Tensor:
            frame = torch.from_numpy(frame)
        if self.device is not None:
            # move to GPU if necessary
            if frame.device != self.device:
                frame = frame.to(self.device)
        return frame

    def zeros(self):
        """Convenience function for generating a zeros image"""
        return torch.zeros((self.batch_size, self.c, self.h, self.w), dtype=torch.float32)

    def set_batch_image(self, frame, index_in_batch):
        """Puts the input frame into the current batch.

        Args:
            frame: torch Tensor of image data to be inserted into batch
            index_in_batch: integer of which number in batch to put the image into. Must be > 0 and < self.num_images
        """
        assert (index_in_batch < self.num_images)
        if type(frame) != torch.Tensor:
            frame = torch.from_numpy(frame)
        if frame.device != self.device:
            frame = frame.to(self.device)
        # channel first for torch
        if frame.shape[0] != self.c:
            frame = frame.permute(2, 0, 1)
        # now frame should be something like 3,256,256
        if self.stack_channels:
            start = index_in_batch * self.c
            end = index_in_batch * self.c + self.c
            self.batch[0, start:end, ...] = frame
        else:
            self.batch[0, :, index_in_batch, ...] = frame

    def initialize(self):
        """Sets up batch on the first frame of the video"""
        assert (self.index == 0)
        # batch = np.zeros((self.num_images, self.h, self.w, 3), dtype=np.float32)
        # do this so that we can get height, width after whatever transforms
        # ret, frame = self.cap.read()
        frame = next(self.reader)
        if self.supervised:
            label = self.label[self.index, :]
        self.index += 1
        frame = self.process_one_frame(frame)
        # cover the case where you use transforms.ToTensor()
        if frame.shape[0] > frame.shape[2]:
            H, W, C = frame.shape
        else:
            C, H, W = frame.shape
        # H,W,C = frame.shape
        self.h = H
        self.w = W
        self.c = C

        i = 0
        if self.stack_channels:
            batch = np.zeros((self.batch_size, self.num_images * self.c, self.h, self.w), dtype=np.float32)
        else:
            batch = np.zeros((self.batch_size, self.c, self.num_images, self.h, self.w), dtype=np.float32)
        batch = torch.from_numpy(batch).to(self.device)
        self.batch = batch

        num_start_images = self.num_images // 2
        self.set_batch_image(frame, num_start_images)
        for i in range(num_start_images + 1, self.num_images):
            # print(i)
            frame = next(self.reader)
            frame = self.process_one_frame(frame)
            self.set_batch_image(frame, i)
            self.index += 1

        if self.supervised:
            return self.batch, label
        else:
            return self.batch

    def roll_and_append(self, frame: Union[torch.Tensor, np.ndarray]):
        """Makes frame the final image in the batch, removes the first image.

        Args:
            frame: the video frame to be added to the end of the batch
        Example:
            # Current batch has images [0,1,2,3,4,5,6,7,8,9,10]
            self.roll_and_append(frame_11)
            # batch now has images [1,2,3,4,5,6,7,8,9,10,11]
        """
        if type(frame) != torch.Tensor:
            frame = torch.from_numpy(frame)
        if frame.device != self.device:
            frame = frame.to(self.device)
        # channel first for torch
        if frame.ndim < 4:
            frame = frame.unsqueeze(0)
        if frame.shape[1] != self.c:
            frame = frame.permute(0, 3, 1, 2)
        if not self.stack_channels:
            # N, C, H, W -> N, C, T=0, H, W
            frame = frame.unsqueeze(2)

        if self.stack_channels:
            self.batch = torch.cat((self.batch[:, self.c:, ...], frame), dim=1)
        else:
            self.batch = torch.cat((self.batch[:, :, 1:, ...], frame), dim=2)

    # Python 2 compatibility:
    def next(self):
        return self.__next__()

    def __next__(self):
        """Loads the next frame from self.reader, rolls the current batch, and appends the next frame to the batch"""
        if self.index >= self.num_batches:
            self.end()
            raise StopIteration
        if self.index == 0:
            return (self.initialize())
        batch = self.batch
        if self.index < self.nframes:
            frame = next(self.reader)
            frame = self.process_one_frame(frame)
        else:
            frame = self.zeros()
        self.roll_and_append(frame)
        if self.supervised:
            label = self.label[self.true_index, :]
        self.true_index += 1
        self.index += 1
        if self.supervised:
            return self.batch, label
        else:
            return self.batch

    def __len__(self):
        return self.nframes

    def __iter__(self):
        return self

    def totensor(self, batch):
        """Converts a numpy image to a torch Tensor and moves it to self.device"""
        tensor = torch.from_numpy(np.concatenate(batch.transpose(0, 3, 1, 2)).astype(np.float32)).unsqueeze(0)
        # if self.pad is not None:
        # tensor = F.pad(tensor, self.pad)
        tensor = tensor.to(self.device)
        return (tensor)

    def end(self):
        """Closes video file object"""
        if hasattr(self, 'reader'):
            self.reader.close()

    def __del__(self):
        """destructor"""
        self.end()


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


def purge_unlabeled_videos(video_list: list, label_list: list) -> Tuple[list, list]:
    """Get rid of any videos that contain unlabeled frames.
    Goes through all label files, loads them. If they contain any -1 values, remove both the video and the label
    from their respective lists
    """
    valid_videos = []
    valid_labels = []
    for i in range(len(label_list)):
        label = read_labels(label_list[i])
        has_unlabeled_frames = np.any(label == -1)
        if not has_unlabeled_frames:
            valid_videos.append(video_list[i])
            valid_labels.append(label_list[i])
    return video_list, label_list


class VideoDataset(data.Dataset):
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
        with VideoReader(self.video_list[movie_index]) as reader:
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
        images = np.concatenate(images, 2)
        # print(images.shape, images.dtype)
        # print(self.transform)
        # print(images.dtype)
        if self.transform:
            images = self.transform(images)
        if log.isEnabledFor(logging.DEBUG):
            log.debug('images shape: {}'.format(images.shape))

        # it's faster to stack the images, perform the augmentation on the stack, then unstack!
        # assumes 3-channel RGB frames
        if self.conv_mode == '3d':
            images = torch.stack([images[i * 3:i * 3 + 3, ...]
                                  for i in range(self.frames_per_clip)], dim=1)
        if log.isEnabledFor(logging.DEBUG):
            log.debug('images shape after 3d -> 4d: {}'.format(images.shape))
        # print(images.shape)
        if self.supervised:
            label = self.labels[index]
            if self.reduce:
                label = np.where(label)[0][0].astype(np.int64)
            return images, label
        else:
            return images


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
                 store_in_ram: bool = True, is_two_stream: bool = False, return_logits=False):
        assert (os.path.isfile(h5file))
        self.h5file = h5file
        if labelfile is not None:
            assert (os.path.isfile(labelfile))
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
                assert (self.flow_key in f)
                assert (self.image_key in f)
            if self.return_logits:
                try:
                    logits = f[self.logit_key][:]
                except BaseException as e:
                    print('logits not found in keys: {}'.format(list(f[self.key].keys())))

    def __len__(self):
        return self.N

    def __getitem__(self, index: int):
        if self.nonoverlapping:
            start = self.starts[index]
            end = self.ends[index]

            indices = np.arange(start, end)
            if self.supervised:
                labels = self.label[:, indices]

            if self.store_in_ram:
                logits, values = self.logits[:, indices], self.sequence[:, indices]
            else:
                logits, values = self.load_sequence(start, end)
            if values.shape[1] < self.sequence_length:
                pad_right = self.sequence_length - values.shape[1]
            else:
                pad_right = 0
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
            if self.supervised:
                labels = self.label[:, indices]
        if self.supervised:
            labels = labels.astype(np.int64)
        if log.isEnabledFor(logging.DEBUG):
            print('start: {} end: {}'.format(start, end))

        logits = np.pad(logits, ((0, 0), (pad_left, pad_right)), mode='constant')
        values = np.pad(values, ((0, 0), (pad_left, pad_right)), mode='constant')
        if self.supervised:
            labels = np.pad(labels, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-1)

        # for torch dataloaders
        #         values = values.T
        #         labels = labels.T
        logits = torch.from_numpy(logits).float()
        values = torch.from_numpy(values).float()
        if self.supervised:
            labels = torch.from_numpy(labels).to(torch.long)
            if self.return_logits:
                return values, logits, labels
            else:
                return values, labels
        else:
            if self.return_logits:
                return values, logits
            else:
                return values

    def __del__(self):
        if hasattr(self, 'sequences'):
            del (self.sequences)
        if hasattr(self, 'labels'):
            del (self.labels)


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


def get_normalization_layer(mean: list, std: list, num_images: int = 1, mode: str = '2d'):
    """Get Z-scoring layer from config
    If RGB frames are stacked into tensor N, num_rgb*3, H, W, we need to repeat the mean and std num_rgb times
    """
    # if mode == '2d':
    mean = mean.copy() * num_images
    std = std.copy() * num_images

    return transforms.Normalize(mean=mean, std=std)


def get_transforms(augs: DictConfig, input_images: int = 1, mode: str = '2d') -> dict:
    """ Make train, validation, and test transforms from a OmegaConf DictConfig with augmentation parameters

    Parameters
    ----------
    augs: DictConfig
        configuration with augmentation parameters. Example keys
            crop_size: how large to crop
            resize: how to resize after cropping
        for more info, see deepethogram/conf/augs.yaml
    input_images: int
        Number of input images. Used to figure out how to z-score across channels
    mode: str
        either 2d or 3d. Used to figure out how to z-score across channels

    Returns
    -------
    xform: dict
        dictionary of composed Transforms, for train, validation, and test
    """
    # augs = cfg.augs # convenience
    spatial_transforms = []
    common = []
    # order here matters a lot!!
    if augs.crop_size is not None:
        spatial_transforms.append(transforms.RandomCrop(augs.crop_size))
        common.append(transforms.CenterCrop(augs.crop_size))
    if augs.resize is not None:
        spatial_transforms.append(transforms.Resize(augs.resize))
        common.append(transforms.Resize(augs.resize))
    if augs.pad is not None:
        spatial_transforms.append(transforms.Pad(augs.pad))
        common.append(transforms.Pad(augs.pad))
    if augs.LR > 0:
        spatial_transforms.append(transforms.RandomHorizontalFlip(p=augs.LR))
    if augs.UD > 0:
        spatial_transforms.append(transforms.RandomVerticalFlip(p=augs.UD))
    if augs.degrees > 0:
        spatial_transforms.append(transforms.RandomRotation(augs.degrees))

    color_transforms = [transforms.ColorJitter(brightness=augs.brightness, contrast=augs.contrast)]
    xform = {}

    color_transforms.append(transforms.ToTensor())
    common.append(transforms.ToTensor())
    if augs.normalization is not None:
        mean = list(augs.normalization.mean)
        std = list(augs.normalization.std)

        norm_layer = get_normalization_layer(mean, std, input_images, mode)
        color_transforms.append(norm_layer)
        common.append(norm_layer)

    xform['train'] = transforms.Compose(spatial_transforms + color_transforms)
    xform['val'] = transforms.Compose(common)
    xform['test'] = transforms.Compose(common)
    log.info(' ~~~ augmentations ~~~')
    log.info(pformat(xform))
    # pprint.pprint(xform)
    return xform


# def get_transforms_from_config(config: dict):
#     spatial_transforms = []
#     common = []
#     # order here matters a lot!!
#     if config['crop_size'] is not None:
#         spatial_transforms.append(transforms.RandomCrop(config['crop_size']))
#         common.append(transforms.CenterCrop(config['crop_size']))
#     if config['resize'] is not None:
#         spatial_transforms.append(transforms.Resize(config['resize']))
#         common.append(transforms.Resize(config['resize']))
#     if config['pad'] is not None:
#         spatial_transforms.append(transforms.Pad(config['pad']))
#         common.append(transforms.Pad(config['pad']))
#     if config['LR'] > 0:
#         spatial_transforms.append(transforms.RandomHorizontalFlip(p=config['LR']))
#     if config['UD'] > 0:
#         spatial_transforms.append(transforms.RandomVerticalFlip(p=config['UD']))
#     if config['degrees'] > 0:
#         spatial_transforms.append(transforms.RandomRotation(config['degrees']))
#
#     color_transforms = [transforms.ColorJitter(brightness=config['brightness'], contrast=config['brightness'])]
#     xform = {}
#     if config['model'] == 'feature_extractor' and config['arch'] == 'two_stream':
#         # we do our normalization in the postprocessor for two stream due to the fact that we need to use the gpu to
#         # convert our flow images from uint-8 to float32
#         for split in ['train', 'val', 'test']:
#             xform[split] = {}
#             if split == 'train':
#                 xform[split]['spatial'] = transforms.Compose(spatial_transforms)
#                 xform[split]['color'] = transforms.Compose(color_transforms)
#             else:
#                 xform[split]['spatial'] = transforms.Compose(common)
#                 xform[split]['color'] = None
#     else:
#         color_transforms.append(transforms.ToTensor())
#         common.append(transforms.ToTensor())
#         if config['normalization'] is not None:
#             norm_layer = get_normalization_layer(config)
#             color_transforms.append(norm_layer)
#             common.append(norm_layer)
#
#         xform['train'] = transforms.Compose(spatial_transforms + color_transforms)
#         xform['val'] = transforms.Compose(common)
#         xform['test'] = transforms.Compose(common)
#     print(' ~~~ augmentations ~~~')
#     pprint.pprint(xform)
#     return xform


# def get_files_from_datadir(datadir, filetype: str = '.avi'):
#     assert (filetype in ['.avi', '.h5', '.mp4'])
#
#     files = glob.glob(datadir + '/**/*' + filetype, recursive=True)
#     files.sort()
#     excluded = ['flow', 'label', 'output', 'score']
#     movies = []
#     for file in files:
#         base = os.path.basename(file)
#         has_excluded = False
#         for exc in excluded:
#             if exc in base:
#                 has_excluded = True
#         if not has_excluded:
#             movies.append(file)
#
#     flows = [i for i in files if 'flow' in i]
#     flows.sort()
#     return (movies, flows)


def remove_nans_and_infs(array: np.ndarray, set_value: float = 0.0) -> np.ndarray:
    """ Simple function to remove nans and infs from a numpy array """
    bad_indices = np.logical_or(np.isinf(array), np.isnan(array))
    array[bad_indices] = set_value
    return array


def make_loss_weight(class_counts: np.ndarray, num_pos: np.ndarray, num_neg: np.ndarray,
                     weight_exp: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """ Makes weight for different classes in loss function.

    In general, rare classes will be up-weighted and common classes will be down-weighted.

    Parameters
    ----------
    class_counts: np.ndarray, shape (K, )
        Number of positive examples in dataset
    num_pos: np.ndarray, shape (K, )
        number of positive examples in dataset
    num_neg: np.ndarray, shape (K, )
        number of negative examples in dataset
    weight_exp: float
        raise weights to this exponent. See DeepEthogram paper

    Returns
    -------
    pos_weight_transformed: np.ndarray, shape (K, )
        amount to weight each class. Used with sigmoid activation, BCE loss
    softmax_weight_transformed: np.ndarray, shape (K, )
        Amount to weight each class. used with softmax activation, NLL loss

    TODO: remove redundant class_counts, num_pos arguments
    """

    pos_weight = num_neg / num_pos
    pos_weight_transformed = (pos_weight ** weight_exp).astype(np.float32)
    # don't weight losses if there are no examples
    pos_weight_transformed = remove_nans_and_infs(pos_weight_transformed)

    softmax_weight = 1 / (class_counts + 1e-8)
    # we have to get rid of invalid classes here, or else when we normalize below, it will disrupt non-zero classes
    softmax_weight[class_counts == 0] = 0
    # normalize
    softmax_weight = softmax_weight / np.sum(softmax_weight)
    softmax_weight_transformed = (softmax_weight ** weight_exp).astype(np.float32)
    # don't weight losses if there are no examples
    softmax_weight_transformed = remove_nans_and_infs(softmax_weight_transformed)

    np.set_printoptions(suppress=True)
    log.info('Class counts: {}'.format(class_counts))
    log.info('Pos weight: {}'.format(pos_weight))
    log.info('Pos weight, weighted: {}'.format(pos_weight_transformed))
    log.info('Softmax weight: {}'.format(softmax_weight))
    log.info('Softmax weight transformed: {}'.format(softmax_weight_transformed))

    return pos_weight_transformed, softmax_weight_transformed


def get_dataloaders_sequence(datadir: Union[str, os.PathLike], latent_name: str, sequence_length: int = 60,
                             is_two_stream: bool = True, nonoverlapping: bool = True, splitfile: str = None,
                             reload_split: bool = True, store_in_ram: bool = True, dimension: int = None,
                             train_val_test: Union[list, np.ndarray] = [0.8, 0.2, 0.0], weight_exp: float = 1.0,
                             batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=False,
                             supervised=True, reduce=False, valid_splits_only: bool = True,
                             return_logits=False) -> dict:
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
    log.info('~~~~~ train val test split ~~~~~')
    pprint.pprint(split_dictionary)

    datasets = {}
    splits = ['train', 'val', 'test']
    datasets = {}
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
                                          is_two_stream=is_two_stream, nonoverlapping=nonoverlapping,
                                          dimension=dimension,
                                          store_in_ram=store_in_ram, return_logits=return_logits)

    shuffles = {'train': shuffle, 'val': True, 'test': False}

    dataloaders = {split: data.DataLoader(datasets[split], batch_size=batch_size,
                                          shuffle=shuffles[split], num_workers=num_workers,
                                          pin_memory=pin_memory, drop_last=drop_last)
                   for split in ['train', 'val', 'test'] if datasets[split] is not None}

    # figure out what our inputs to our model will be (D dimension)
    dataloaders['num_features'] = datasets['train'].num_features

    if supervised:
        dataloaders['class_counts'] = datasets['train'].class_counts
        dataloaders['num_classes'] = len(dataloaders['class_counts'])
        pos_weight, softmax_weight = make_loss_weight(dataloaders['class_counts'],
                                                      datasets['train'].num_pos,
                                                      datasets['train'].num_neg,
                                                      weight_exp=weight_exp)
        dataloaders['pos'] = datasets['train'].num_pos
        dataloaders['neg'] = datasets['train'].num_neg
        dataloaders['pos_weight'] = pos_weight
        dataloaders['loss_weight'] = softmax_weight
    dataloaders['split'] = split_dictionary
    return dataloaders


def get_video_metadata(videofile):
    """ Simple wrapper to get video availability, width, height, and frame number """
    try:
        with VideoReader(videofile) as reader:
            framenum = reader.nframes
            frame = next(reader)
            width = frame.shape[1]
            height = frame.shape[0]
            ret = True
    except BaseException as e:
        ret = False
        print(e)
        print('Error reading file {}'.format(videofile))
    return ret, width, height, framenum


def extract_metadata(splitdir, allmovies=None, is_flow=False, num_workers=32):
    """ Function to get the video metadata for all videos in Kinetics """
    actions = os.listdir(splitdir)
    actions.sort()

    if allmovies is None:
        allmovies = glob.glob(splitdir + '**/**/**.mp4') + glob.glob(splitdir + '**/**/**.avi')
    allmovies.sort()

    if not is_flow:
        allmovies = [i for i in allmovies if 'flow' not in os.path.basename(i)]
    else:

        allmovies = [i for i in allmovies if 'flow' in os.path.basename(i)]
    widths = []
    heights = []
    framenums = []
    allnames = []
    allactions = []
    action_indices = []

    with mp.Pool(num_workers) as pool:
        for action_index, action in enumerate(tqdm(actions)):
            action_dir = os.path.join(splitdir, action)
            movies = [i for i in allmovies if action_dir in i]

            # movies = glob.glob(action_dir + '**/**.mp4') + glob.glob(action_dir + '**/**.avi')
            movies.sort()
            if not is_flow:
                movies = [i for i in movies if 'flow' not in os.path.basename(i)]
            else:
                movies = [i for i in movies if 'flow' in os.path.basename(i)]
            results = pool.map(get_video_metadata, movies)

            success = []
            for i, row in enumerate(results):
                if row[0]:
                    widths.append(row[1])
                    heights.append(row[2])
                    framenums.append(row[3])
                    success.append(True)
                else:
                    os.remove(movies[i])
                    success.append(False)

            for i, movie in enumerate(movies):
                if success[i]:
                    allnames.append(movie)
                    allactions.append(action)
                    action_indices.append(action_index)

    video_data = {'name': allnames,
                  'action': allactions,
                  'action_int': action_indices,
                  'width': widths,
                  'height': heights,
                  'framecount': framenums}
    df = pd.DataFrame(data=video_data)
    fname = '_metadata.csv'
    if is_flow:
        fname = '_flow' + fname
    df.to_csv(os.path.join(os.path.dirname(splitdir), os.path.basename(splitdir) + fname))
    return df


# def get_twostream_files_kinetics(datadir):
#     assert (os.path.isdir(datadir))
#     movies = glob.glob(datadir + '/**/*.mp4', recursive=True)
#     movies.sort()
#     movies = [i for i in movies if 'flow' not in os.path.basename(i)]
#
#     flows = glob.glob(datadir + '/**/*.avi', recursive=True)
#     flows.sort()
#     flows = [i for i in flows if 'flow' in os.path.basename(i)]
#     rgb_cnt = 0
#     flow_cnt = 0
#     matched = 0
#     matches = {}
#     while rgb_cnt < len(movies):
#         rgb = movies[rgb_cnt]
#         flow = flows[flow_cnt]
#
#         if os.path.basename(rgb)[:10] == os.path.basename(flow)[:10]:
#             matches[rgb] = flow
#             # matched
#             rgb_cnt += 1
#             flow_cnt += 1
#             matched += 1
#         else:
#             # the current rgb does not have a good match
#             print('match not found for {}'.format(rgb))
#             rgb_cnt += 1
#     if matched == 0:
#         raise ValueError('something wrong with reading files from directory: {}'.format(datadir))
#     return (matches)
#
#
# def extract_metadata_twostream(splitdir, num_workers=32):
#     matches = get_twostream_files_kinetics(splitdir)
#     rgbs = list(matches.keys())
#     flows = list(matches.values())
#
#     print('Extracting metadata for directory {}... might take a while'.format(splitdir))
#     rgb_df = extract_metadata(splitdir, allmovies=rgbs, is_flow=False, num_workers=num_workers)
#     flow_df = extract_metadata(splitdir, allmovies=flows, is_flow=True, num_workers=num_workers)
#
#     return rgb_df, flow_df


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


def find_labelfile(video: Union[str, os.PathLike]) -> Tuple[str, str]:
    """ Function for finding a label file for a given a video """
    base = os.path.splitext(video)[0]
    labelfile = base + '_labels.csv'
    if os.path.isfile(labelfile):
        return (labelfile, 'csv')
    labelfile = base + '_labels.h5'
    if os.path.isfile(labelfile):
        return (labelfile, 'h5')
    labelfile = base + '_scores.csv'
    if os.path.isfile(labelfile):
        return (labelfile, 'csv')
    labelfile = base + '_scores.h5'
    if os.path.isfile(labelfile):
        return (labelfile, 'h5')
    basedir = os.path.dirname(video)
    files = os.listdir(basedir)
    files.sort()
    files = [os.path.join(basedir, i) for i in files]
    # handles case where directory contains 'movie.avi', and 'labels.csv'
    files = [i for i in files if 'label' in i or 'score' in i]
    if len(files) == 1:
        if files[0].endswith('csv'):
            return files[0], 'csv'
        elif files[0].endswith('h5'):
            return files[0], 'h5'
    basename = os.path.basename(base).split('_')[:-1]
    basename = '_'.join(basename)
    matching_files = [i for i in files if basename in i]
    if len(matching_files) == 1:
        labelfile = matching_files[0]
        ext = os.path.splitext(labelfile)[1][1:]
        return labelfile, ext
    raise ValueError('no corresponding labels found: {}'.format(video))


def read_all_labels(labelfiles: list):
    """ Function for reading all labels into memory """
    labels = []
    for i, labelfile in enumerate(labelfiles):
        assert (os.path.isfile(labelfile))
        label_type = os.path.splitext(labelfile)[1][1:]
        # labelfile, label_type = find_labelfile(video)
        label = read_labels(labelfile)
        H, W = label.shape
        # labels should be time x num_behaviors
        if W > H:
            label = label.T
        if label.shape[1] == 1:
            # add a background class
            warnings.warn('binary labels found, adding background class')
            label = np.hstack((np.logical_not(label), label))
        labels.append(label)

        label_no_ignores = np.copy(label)
        label_no_ignores[label_no_ignores == -1] = 0
        if i == 0:
            class_counts = label_no_ignores.sum(axis=0)
            num_pos = (label == 1).sum(axis=0)
            num_neg = (label == 0).sum(axis=0)
        else:
            class_counts += label_no_ignores.sum(axis=0)
            num_pos += (label == 1).sum(axis=0)
            num_neg += (label == 0).sum(axis=0)
    num_labels = len(labels)
    labels = np.concatenate(labels)
    class_counts = class_counts
    return labels, class_counts, num_labels, num_pos, num_neg


def parse_split(split: Union[tuple, list, np.ndarray], N: int):
    parsed_split = []
    for item in split:
        # handle when strings might be input due to arg parsing
        num = float(item)
        # https://stackoverflow.com/questions/45865407/python-how-to-convert-string-into-int-or-float
        if int(num) == num:
            num = int(num)
        parsed_split.append(num)
    split = parsed_split
    split = np.array(split)
    # split can either be floats like [0.7, 0.15, 0.15]
    # or it can be ints with numbers of movies for each. -1 means "all other movies"
    # example: [1, -1, 0]: 1 train movie, all rest validation, no test files
    if np.issubdtype(split[0], np.floating):
        assert np.sum(split) == 1
    elif np.issubdtype(split[0], np.integer):
        if -1 in split:
            minus_one = split == -1
            total = split[np.logical_not(minus_one)].sum()

            split[minus_one] = N - total
        total = split.sum()
        assert total <= N
        N = total
        split = split / split.sum()
    else:
        raise ValueError('Unknown split type: {}'.format(split.dtype))
    return split, N



def train_val_test_split(records: dict, split: Union[tuple, list, np.ndarray] = (0.7, 0.15, 0.15)) -> dict:
    """ Split a dict of dicts into train, validation, and test sets.

    Parameters
    ----------
    records: dict of dicts
        E.g. {'animal': {'rgb': path/to/video.mp4, 'label': path/to/label.csv}, 'animal2': ...}
    split: list, np.ndarray. Shape: (3,)
        If they contain floats, assume they are fractions that sum to one
        If they are ints, assume they are number of elements. E.g. [10, 5, -1]: 10 items in training set, 5 in
            validation set, and all the rest in test set

    Returns
    -------
    outputs: dict of lists
        keys: train, val, test
        Each is a list of keys in the record dictionary. e.g.
            {'train': [animal10, animal09], 'val': [animal01, animal00], ...}
    """
    keys = list(records.keys())
    N = len(keys)

    split, N = parse_split(split, N)

    ends = np.floor(N * np.cumsum(split)).astype(np.uint16)
    starts = np.concatenate((np.array([0]), ends))[:-1].astype(np.uint16)

    # in place
    indices = np.random.permutation(N)
    splits = ['train', 'val', 'test']
    keys = np.array(keys)
    outputs = {}
    outputs['metadata'] = {'split': split.tolist()}
    # print(list(split))
    # outputs['metadata']['split'] = split.tolist()
    # handle edge cases
    if len(records) < 4:
        assert len(records) > 1
        warnings.warn('Only {} records found...'.format(len(keys)))
        shuffled = np.random.permutation(keys)
        outputs['train'] = [str(shuffled[0])]
        outputs['val'] = [str(shuffled[1])]
        outputs['test'] = []
        if len(records) == 3:
            shuffled = np.random.permutation(keys)
            outputs['test'] = [str(shuffled[2])]
        return outputs

    for i, spl in enumerate(splits):
        shuffled = keys[indices]
        splitfiles = shuffled[starts[i]:ends[i]]
        outputs[spl] = splitfiles.tolist()

    # print(type(split.tolist()[0]))
    return outputs


def do_all_classes_have_labels(records: dict, split_dict: dict) -> bool:
    """ Helper function to determine if each split has at least one instance of every class """
    labelfiles = []

    for split in ['train', 'val', 'test']:
        if len(split_dict[split]) > 0:
            splitfiles = split_dict[split]
            for f in splitfiles:
                labelfiles.append(records[f]['label'])
            # labelfiles += [records[i]['label'] for i in split_dict[split]]
    _, class_counts, _, _, _ = read_all_labels(labelfiles)
    return np.all(class_counts > 0)


def get_valid_split(records: dict, train_val_test: Union[list, np.ndarray]) -> dict:
    """  Gets a train, val, test split with at least one instance of every class

    Keep doing train_test_split until each split of the data has at least one single example of every behavior
    in the dataset. it would be bad if your train data had class counts: [1000, 0, 0, 10] and your test data had
    class counts: [500, 100, 300, 0]

    Parameters
    ----------
    records: dict of dicts
        See train_val_test_split
    train_val_test: list, np.ndarray
        See train_val_test_split

    Returns
    -------
    split_dict: dict
        See train_val_test_split
    """

    is_wrong = True
    split_dict = None

    while is_wrong:
        split_dict = train_val_test_split(records, train_val_test)
        should_continue = do_all_classes_have_labels(records, split_dict)
        if not should_continue:
            warnings.warn('Not all classes in the dataset have *any* labels!')
            return split_dict
        is_wrong = False
        for split in ['train', 'val', 'test']:
            labelfiles = [records[i]['label'] for i in split_dict[split]]
            if len(labelfiles) > 0:
                _, class_counts, _, _, _ = read_all_labels(labelfiles)
                if not np.all(class_counts > 0):
                    is_wrong = True
    return split_dict


def update_split(records: dict, split_dictionary: dict) -> dict:
    """ Updates existing split if there are new entries in the records dictionary """
    # records: dictionary of dictionaries. Keys: unique data identifiers
    # values: a dictionary corresponding to different files. the first record might be:
    # {'mouse000': {'rgb': path/to/rgb.avi, 'label':path/to/labels.csv} }
    # split_dictionary: {'metadata': ..., 'train':[mouse000, mouse001], 'val':[mouse002,mouse003]... etc}
    old_dictionary = {k: v for (k, v) in split_dictionary.items() if k != 'metadata'}
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    old_keys = [item for sublist in old_dictionary.values() for item in sublist]
    old_keys.sort()

    new_keys = list(records.keys())
    # goes through data dict and looks for items that are not in the unwrapped version
    # of the split dictionary
    new_entries = [i for i in new_keys if i not in old_keys]
    splits = list(split_dictionary.keys())
    splits = [i for i in splits if i != 'metadata']
    if len(splits) == 3:
        # alphabetical order does not work
        splits = ['train', 'val', 'test']
    # goes through new entries, and assigns them to a split based on loaded split_probabilities
    if len(new_entries) > 0:
        split_p = split_dictionary['metadata']['split']
        N = len(new_entries)
        new_splits = np.random.choice(splits, size=(N,), p=split_p).tolist()
        for i, k in enumerate(new_entries):
            split_dictionary[new_splits[i]].append(k)
            log.info('file {} assigned to split {}'.format(k, new_splits[i]))
    return split_dictionary


def get_split_from_records(records: dict, datadir: Union[str, bytes, os.PathLike],
                           splitfile: Union[str, bytes, os.PathLike] = None, supervised: bool = True,
                           reload_split: bool = True, valid_splits_only: bool = True,
                           train_val_test: list = [0.7, 0.15, 0.15]):
    """ Splits the records into train, validation, and test splits

    Parameters
    ----------
    records: dict of dicts
        E.g. {'animal': {'rgb': path/to/video.mp4, 'label': path/to/label.csv}, 'animal2': ...}
    datadir: str, os.PathLike
        absolute path to the base directory containing data. Only used to save split
    splitfile: str, os.PathLike
        absolute path to file containing a pre-made split to load. If none, make a new one from scratch
    supervised: bool
        if True, enables the option to use the valid split function
    reload_split: bool
        if True, tries to load the file in splitfile
    valid_splits_only: bool
        if True and supervised is True, make sure each split has at least 1 instance of each class
    train_val_test: list
        fractions / Ns in each split. see train_val_test_split

    Returns
    -------
    split_dictionary: dict
        see train_val_test_split
    """
    if splitfile is None:
        splitfile = os.path.join(datadir, 'split.yaml')

    if supervised and valid_splits_only:
        # this function makes sure that each split has all classes in the dataset
        split_func = get_valid_split
    else:
        split_func = train_val_test_split

    if reload_split and os.path.isfile(splitfile):
        split_dictionary = utils.load_yaml(splitfile)
        if split_dictionary is None:
            # some malformatting
            split_dictionary = split_func(records, train_val_test)
        # if there are new records, e.g. new records were added to an old splitfile, 
        # assign them to train, val, or test
        split_dictionary = update_split(records, split_dictionary)
    else:
        split_dictionary = split_func(records, train_val_test)

    utils.save_dict_to_yaml(split_dictionary, splitfile)
    return split_dictionary


# def get_dataloaders_kinetics(directory, brightness=0.25,contrast=0.1, LR=0.5,crop_size=(256,256),
#                              sequence_length=2,batch_size=1, shuffle=True,
#                           num_workers=0, pin_memory=False, drop_last=False,
#                           supervised=False, normalization=None, resize=None, random_resize=False):
#
#     transform_list = [transforms.ColorJitter(brightness=brightness,contrast=contrast),
#                       transforms.RandomHorizontalFlip(p=LR),
#                       transforms.ToTensor()]
#     common = [transforms.ToTensor()]
#
#     if not random_resize:
#         if crop_size is not None:
#             transform_list.insert(0, transforms.RandomCrop(crop_size))
#             common.insert(0, transforms.CenterCrop(crop_size))
#         if resize is not None:
#             transform_list.insert(1, transforms.Resize(resize))
#             common.insert(1, transforms.Resize(resize))
#     else:
#         transform_list.insert(0, transforms.RandomResizedCrop(size=crop_size,scale=(0.5,1.0)))
#         common.insert(0, transforms.RandomResizedCrop(size=crop_size,scale=(0.5,1.0)))
#     if normalization is not None:
#         transform_list.append(transforms.Normalize(mean=normalization[0],std=normalization[1]))
#         common.append(transforms.Normalize(mean=normalization[0],std=normalization[1]))
#     # print('Transforms: {}'.format(transforms))
#         # print('Normalizing!')
#
#     xform = {}
#     xform['train'] = transforms.Compose(transform_list)
#     xform['val'] = transforms.Compose(common)
#     xform['test'] = transforms.Compose(common)
#
#     datasets = {split: KineticsDataset(directory,split,
#                                        sequence_length=sequence_length,
#                                        transform=xform[split],
#                                        supervised=supervised)
#                for split in ['train','val','test']}
#     shuffles = {}
#     shuffles['train'] = shuffle
#     shuffles['val'] = True
#     shuffles['test'] = False
#
#     dataloaders = {split: data.DataLoader(datasets[split], batch_size=batch_size,
#                                          shuffle=shuffles[split], num_workers=num_workers,
#                                          pin_memory=pin_memory,drop_last=drop_last)
#                   for split in ['train', 'val','test']}
#     return(dataloaders)

# note: this is a hack that should be refactored. I don't want DALI to be a hard dependency, but if people want to use
# it and install it themselves, it should be an option
if dali:
    class KineticsDALIPipe(Pipeline):
        def __init__(self, directory,
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
                     conv_mode='3d',
                     image_shape=(256, 256),
                     validate: bool = False):
            super().__init__(batch_size, num_workers, gpu_id, prefetch_queue_depth=1)
            self.input = ops.VideoReader(additional_decode_surfaces=1,
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
                                         stride=1)

            self.uniform = ops.Uniform(range=(0.0, 1.0))
            self.cmn = ops.CropMirrorNormalize(device='gpu', crop=crop_size,
                                               mean=mean, std=std,
                                               output_layout=types.NFHWC)

            self.coin = ops.CoinFlip(probability=0.5)
            self.brightness_val = ops.Uniform(range=[1 - brightness, 1 + brightness])
            self.contrast_val = ops.Uniform(range=[1 - contrast, 1 + contrast])
            self.supervised = supervised
            self.half = ops.Constant(fdata=0.5)
            self.zero = ops.Constant(idata=0)
            self.cast_to_long = ops.Cast(device='gpu', dtype=types.INT64)
            if crop_size is not None:
                H, W = crop_size
            else:
                # default
                H, W = image_shape
            # print('CONV MODE!!! {}'.format(conv_mode))
            if conv_mode == '3d':
                self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])
                self.reshape = None
            elif conv_mode == '2d':
                self.transpose = ops.Transpose(device='gpu', perm=[0, 3, 1, 2])
                self.reshape = ops.Reshape(device='gpu', shape=[-1, H, W])
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


    #
    #
    # # https://github.com/NVIDIA/DALI/blob/cde7271a840142221273f8642952087acd919b6e
    # # /docs/examples/use_cases/video_superres/dataloading/dataloaders.py
    class DALILoader:
        def __init__(self, directory,
                     supervised: bool = True,
                     sequence_length: int = 11,
                     batch_size: int = 1,
                     num_workers: int = 1,
                     gpu_id: int = 0,
                     shuffle: bool = True,
                     crop_size: tuple = (256, 256),
                     mean: list = [0.5, 0.5, 0.5],
                     std: list = [0.5, 0.5, 0.5],
                     conv_mode: str = '3d',
                     validate: bool = False,
                     distributed: bool = False):
            self.pipeline = KineticsDALIPipe(directory=directory,
                                             batch_size=batch_size,
                                             supervised=supervised,
                                             sequence_length=sequence_length,
                                             num_workers=num_workers,
                                             gpu_id=gpu_id,
                                             crop_size=crop_size,
                                             mean=mean,
                                             std=std,
                                             conv_mode=conv_mode,
                                             validate=validate)
            self.pipeline.build()
            self.epoch_size = self.pipeline.epoch_size("Reader")
            names = ['images', 'labels'] if supervised else ['images']
            self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                             names,
                                                             self.epoch_size,
                                                             auto_reset=True)

        def __len__(self):
            return int(self.epoch_size)

        def __iter__(self):
            return self.dali_iterator.__iter__()


    def get_dataloaders_kinetics_dali(directory,
                                      rgb_frames=1,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0,
                                      supervised=True,
                                      conv_mode='2d',
                                      gpu_id: int = 0,
                                      crop_size: tuple = (256, 256),
                                      mean: list = [0.5, 0.5, 0.5],
                                      std: list = [0.5, 0.5, 0.5],
                                      distributed: bool = False):
        shuffles = {'train': shuffle, 'val': True, 'test': False}
        dataloaders = {}
        for split in ['train', 'val']:
            splitdir = os.path.join(directory, split)
            dataloaders[split] = DALILoader(splitdir,
                                            supervised=supervised,
                                            batch_size=batch_size,
                                            gpu_id=gpu_id,
                                            shuffle=shuffles[split],
                                            crop_size=crop_size,
                                            mean=mean,
                                            std=std,
                                            validate=split == 'val',
                                            num_workers=num_workers,
                                            sequence_length=rgb_frames,
                                            conv_mode=conv_mode,
                                            distributed=distributed)

        dataloaders['split'] = None
        return dataloaders


    def __len__(self):
        return int(self.epoch_size)


    def __iter__(self):
        return self.dali_iterator.__iter__()


# def get_dataloaders_kinetics(directory, mode='both', xform=None, rgb_frames=1, flow_frames=10,
#                              batch_size=1, shuffle=True,
#                              num_workers=0, pin_memory=False, drop_last=False,
#                              supervised=True,
#                              reduce=True, conv_mode='2d'):
#     datasets = {}
#     for split in ['train', 'val', 'test']:
#         # this is in the two stream case where you can't apply color transforms to an optic flow
#         if type(xform[split]) == dict:
#             spatial_transform = xform[split]['spatial']
#             color_transform = xform[split]['color']
#         else:
#             spatial_transform = xform[split]
#             color_transform = None
#         datasets[split] = KineticsDataset(directory, split, mode, supervised=supervised,
#                                           rgb_frames=rgb_frames, flow_frames=flow_frames,
#                                           spatial_transform=spatial_transform,
#                                           color_transform=color_transform,
#                                           reduce=reduce,
#                                           flow_style='rgb',
#                                           flow_max=10,
#                                           conv_mode=conv_mode)
#
#     shuffles = {'train': shuffle, 'val': True, 'test': False}
#
#     dataloaders = {split: data.DataLoader(datasets[split], batch_size=batch_size,
#                                           shuffle=shuffles[split], num_workers=num_workers,
#                                           pin_memory=pin_memory, drop_last=drop_last)
#                    for split in ['train', 'val', 'test']}
#     dataloaders['split'] = None
#     return (dataloaders)


def remove_invalid_records_from_split_dictionary(split_dictionary: dict, records: dict) -> dict:
    """ Removes records that exist in split_dictionary but not in records.
    Can be useful if you previously had a video in your project and used that to make a train / val / test split,
    but later deleted it.
    """
    valid_records = {}
    record_keys = list(records.keys())
    for split in ['train', 'val', 'test']:
        valid_records[split] = {}
        splitfiles = split_dictionary[split]
        for i, key in enumerate(record_keys):
            if key in splitfiles:
                valid_records[split][key] = records[key]
    return valid_records


def get_video_dataloaders(datadir: Union[str, os.PathLike], xform: dict, is_two_stream: bool = False,
                          reload_split: bool = True, splitfile: Union[str, os.PathLike] = None,
                          train_val_test: Union[list, np.ndarray] = [0.8, 0.1, 0.1], weight_exp: float = 1.0,
                          rgb_frames: int = 1, flow_frames: int = 10, batch_size=1, shuffle=True, num_workers=0,
                          pin_memory=False, drop_last=False, supervised=True, reduce=False, flow_max: int = 5,
                          flow_style: str = 'linear', valid_splits_only: bool = True, conv_mode: str = '2d'):
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
                                           frames_per_clip=rgb_frames,
                                           label_list=labelfiles,
                                           reduce=reduce,
                                           transform=xform[split],
                                           conv_mode=conv_mode)

    shuffles = {'train': shuffle, 'val': True, 'test': False}

    dataloaders = {split: data.DataLoader(datasets[split], batch_size=batch_size,
                                          shuffle=shuffles[split], num_workers=num_workers,
                                          pin_memory=pin_memory, drop_last=drop_last)
                   for split in ['train', 'val', 'test'] if datasets[split] is not None}

    if supervised:
        dataloaders['class_counts'] = datasets['train'].class_counts
        dataloaders['num_classes'] = len(dataloaders['class_counts'])
        pos_weight, softmax_weight = make_loss_weight(dataloaders['class_counts'],
                                                      datasets['train'].num_pos,
                                                      datasets['train'].num_neg,
                                                      weight_exp=weight_exp)
        dataloaders['pos'] = datasets['train'].num_pos
        dataloaders['neg'] = datasets['train'].num_neg
        dataloaders['pos_weight'] = pos_weight
        dataloaders['loss_weight'] = softmax_weight
    dataloaders['split'] = split_dictionary
    return (dataloaders)


# def get_input_images_from_config(config: dict) -> int:
#     if config['model'] == 'flow_generator':
#         # if we're using a flow generator, return num input images, probably 11
#         return config['input_images']
#     elif config['model'] == 'feature_extractor':
#         # if we're training a two-stream model with loaded flows, return num_rgb, probably 1
#         if config['arch'] == 'two_stream':
#             return config['num_rgb']
#         elif config['arch'] == 'hidden_two_stream':
#             # if we're training a hidden two stream network, we could either request ~1 image or ~11 images. We only
#             # want 1 image if we're in the specific part of the curriculum where we're training the spatial only model
#             # for both the flow model and the full spatial + flow, we want ~11 images
#             if config['spatial_only']:
#                 return config['num_rgb']
#             else:
#                 return config['input_images']
#         else:
#             raise ValueError('unknown architecture: {}'.format(config['arch']))
#     else:
#         raise ValueError('unexpected model type: {}'.format(config['model']))


def get_dataloaders_from_cfg(cfg: DictConfig, model_type: str, input_images: int = 1) -> dict:
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
        xform = get_transforms(cfg.augs, input_images, mode)
        if cfg.project.name == 'kinetics':
            if cfg.compute.dali:
                dataloaders = get_dataloaders_kinetics_dali(cfg.project.data_path,
                                                            rgb_frames=input_images,
                                                            batch_size=cfg.compute.batch_size,
                                                            num_workers=cfg.compute.num_workers,
                                                            supervised=supervised,
                                                            conv_mode=mode,
                                                            gpu_id=cfg.compute.gpu_id,
                                                            crop_size=cfg.augs.crop_size,
                                                            mean=list(cfg.augs.normalization.mean),
                                                            std=list(cfg.augs.normalization.std),
                                                            distributed=cfg.compute.distributed)
                # hack, because for DEG projects we'll get the number of positive and negative examples
                # for kinetics, we don't want to weight the loss at all
                dataloaders['pos'] = None
                dataloaders['neg'] = None
            else:
                dataloaders = get_dataloaders_kinetics(cfg.project.data_path,
                                                       mode='rgb',
                                                       xform=xform,
                                                       rgb_frames=input_images,
                                                       batch_size=cfg.compute.batch_size,
                                                       shuffle=True,
                                                       num_workers=cfg.compute.num_workers,
                                                       pin_memory=torch.cuda.is_available(),
                                                       reduce=True,
                                                       supervised=supervised,
                                                       conv_mode=mode)
        else:
            reduce = False
            if cfg.run.model == 'feature_extractor':
                if cfg.feature_extractor.final_activation == 'softmax':
                    reduce = True
            dataloaders = get_video_dataloaders(cfg.project.data_path, xform=xform, is_two_stream=False,
                                                splitfile=cfg.split.file, train_val_test=cfg.split.train_val_test,
                                                weight_exp=cfg.train.loss_weight_exp, rgb_frames=input_images,
                                                batch_size=cfg.compute.batch_size, num_workers=cfg.compute.num_workers,
                                                pin_memory=torch.cuda.is_available(), drop_last=False,
                                                supervised=supervised, reduce=reduce, conv_mode=mode)
    elif model_type == 'sequence':
        dataloaders = get_dataloaders_sequence(cfg.project.data_path, cfg.sequence.latent_name,
                                               cfg.sequence.sequence_length, is_two_stream=True,
                                               nonoverlapping=cfg.sequence.nonoverlapping, splitfile=cfg.split.file,
                                               reload_split=True, store_in_ram=False, dimension=None,
                                               train_val_test=cfg.split.train_val_test,
                                               weight_exp=cfg.train.loss_weight_exp, batch_size=cfg.compute.batch_size,
                                               shuffle=True, num_workers=cfg.compute.num_workers,
                                               pin_memory=torch.cuda.is_available(), drop_last=False, supervised=True,
                                               reduce=cfg.feature_extractor.final_activation == 'softmax',
                                               valid_splits_only=True, return_logits=False)
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    return dataloaders
