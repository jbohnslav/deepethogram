import bisect
import logging
import os
import random
import warnings
from functools import partial
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch
from opencv_transforms import transforms
from torch.utils import data
from vidio import VideoReader

# from deepethogram.dataloaders import log
from deepethogram.data.utils import purge_unlabeled_videos, get_video_metadata, extract_metadata, find_labelfile, \
    read_all_labels
from deepethogram.file_io import read_labels

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
            assert type(device) == torch.device
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
            raise ValueError('must set CPU transforms in SequentialIterator')
        if type(frame) != torch.Tensor:
            frame = torch.from_numpy(frame)
        if frame.dtype != torch.float32:
            frame = frame.float() / 255
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