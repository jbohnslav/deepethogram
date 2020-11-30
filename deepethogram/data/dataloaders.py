import logging
import os
import pprint
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils import data

from deepethogram import projects
from deepethogram.data.augs import get_transforms, get_cpu_transforms
from deepethogram.data.datasets import SequenceDataset, TwoStreamDataset, VideoDataset, KineticsDataset
from deepethogram.data.utils import get_split_from_records, remove_invalid_records_from_split_dictionary, \
    make_loss_weight

try:
    from nvidia.dali.pipeline import Pipeline
    from .dali import get_dataloaders_kinetics_dali
except ImportError:
    get_dataloaders_kinetics_dali = None
# from deepethogram.dataloaders import log

log = logging.getLogger(__name__)


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


def get_dataloaders_kinetics(directory, mode='both', xform=None, rgb_frames=1, flow_frames=10,
                             batch_size=1, shuffle=True,
                             num_workers=0, pin_memory=False, drop_last=False,
                             supervised=True,
                             reduce=True, conv_mode='2d'):
    datasets = {}
    for split in ['train', 'val', 'test']:
        # this is in the two stream case where you can't apply color transforms to an optic flow
        if type(xform[split]) == dict:
            spatial_transform = xform[split]['spatial']
            color_transform = xform[split]['color']
        else:
            spatial_transform = xform[split]
            color_transform = None
        datasets[split] = KineticsDataset(directory, split, mode, supervised=supervised,
                                          rgb_frames=rgb_frames, flow_frames=flow_frames,
                                          spatial_transform=spatial_transform,
                                          color_transform=color_transform,
                                          reduce=reduce,
                                          flow_style='rgb',
                                          flow_max=10,
                                          conv_mode=conv_mode)

    shuffles = {'train': shuffle, 'val': True, 'test': False}

    dataloaders = {split: data.DataLoader(datasets[split], batch_size=batch_size,
                                          shuffle=shuffles[split], num_workers=num_workers,
                                          pin_memory=pin_memory, drop_last=drop_last)
                   for split in ['train', 'val', 'test']}
    dataloaders['split'] = None
    return dataloaders


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
        xform = get_cpu_transforms(cfg.augs)

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
