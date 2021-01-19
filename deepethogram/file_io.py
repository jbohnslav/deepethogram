import os
import warnings
from typing import Union

import h5py
import numpy as np
import pandas as pd
from vidio import VideoReader, VideoWriter


def read_labels(labelfile: Union[str, os.PathLike]) -> np.ndarray:
    """ convenience function for reading labels from a .csv or .h5 file """
    labeltype = os.path.splitext(labelfile)[1][1:]
    if labeltype == 'csv':
        label = read_label_csv(labelfile)
        # return(read_label_csv(labelfile))
    elif labeltype == 'h5':
        label = read_label_hdf5(labelfile)
        # return(read_label_hdf5(labelfile))
    else:
        raise ValueError('Unknown labeltype: {}'.format(labeltype))
    H, W = label.shape
    # labels should be time x num_behaviors
    if W > H:
        label = label.T
    if label.shape[1] == 1:
        # add a background class
        warnings.warn('binary labels found, adding background class')
        label = np.hstack((np.logical_not(label), label))
    return label


def read_label_hdf5(labelfile: Union[str, os.PathLike]) -> np.ndarray:
    """ read labels from an HDF5 file. Must end in .h5

    Assumes that labels are in a dataset with name 'scores' or 'labels'
    Parameters
    ----------
    labelfile

    Returns
    -------

    """
    with h5py.File(labelfile, 'r') as f:
        keys = list(f.keys())
        if 'scores' in keys:
            key = 'scores'
        elif 'labels' in keys:
            key = 'labels'
        else:
            raise ValueError('not sure which dataset in hdf5 contains labels: {}'.format(keys))
        label = f[key][:].astype(np.int64)
    if label.ndim == 1:
        label = label[..., np.newaxis]
    return (label)


def read_label_csv(labelfile):
    df = pd.read_csv(labelfile, index_col=0)
    label = df.values.astype(np.int64)
    if label.ndim == 1:
        label = label[..., np.newaxis]
    return label


def convert_video(videofile: Union[str, os.PathLike], movie_format: str, *args, **kwargs) -> None:
    with VideoReader(videofile) as reader:
        basename = os.path.splitext(videofile)[0]
        if movie_format == 'ffmpeg':
            out_filename = basename + '.mp4'
        elif movie_format == 'opencv':
            out_filename = basename + '.avi'
        elif movie_format == 'hdf5':
            out_filename = basename + '.h5'
        elif movie_format == 'directory':
            out_filename = basename
        else:
            raise ValueError('unexpected value of movie format: {}'.format(movie_format))
        with VideoWriter(out_filename, movie_format=movie_format, *args, **kwargs) as writer:
            for frame in reader:
                writer.write(frame)
