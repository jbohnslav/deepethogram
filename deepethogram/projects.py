import logging
import os
import re
import shutil
import sys
import warnings
from datetime import datetime
from typing import Union

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from tqdm import tqdm

import deepethogram
from deepethogram.utils import get_subfiles, log
from deepethogram.zscore import zscore_video
from . import utils
from .file_io import read_labels, convert_video

log = logging.getLogger(__name__)

required_keys = ['project', 'augs']
projects_file_directory = os.path.dirname(os.path.abspath(__file__))


def initialize_project(directory: Union[str, os.PathLike],
                       project_name: str,
                       behaviors: list = None,
                       make_subdirectory: bool = True,
                       labeler: str = None):
    """Initializes a DeepEthogram project.
    Copies the default configuration file and updates it with the directory, name, and behaviors specified.
    Makes directories where project info, data, and models will live.

    Args:
        directory: str, os.PathLike
            Directory where DeepEthogram data and models will be made / copied. Should be on an SSD. Should
            also have plenty of space.
        project_name: str
            name of the deepethogram project
        behaviors: optional list.
            First should be background.
        make_subdirectory: bool
            if True, make a subdirectory like "/path/to/DATA/project_name_deepethogram"
            if False, keep as the input directory: "/path/to/DATA"
    Example:
        intialize_project('C:\DATA', 'grooming', ['background', 'face_groom', 'body_groom', 'rear'])
    """
    assert os.path.isdir(directory), 'Directory does not exist: {}'.format(
        directory)
    if behaviors is not None:
        assert behaviors[0] == 'background'

    root = os.path.dirname(os.path.abspath(__file__))
    project_config = utils.load_yaml(
        os.path.join(root, 'conf', 'project', 'project_config_default.yaml'))
    project_name = project_name.replace(' ', '_')

    project_config['project']['name'] = project_name

    project_config['project']['class_names'] = behaviors
    if make_subdirectory:
        project_dir = os.path.join(directory,
                                   '{}_deepethogram'.format(project_name))
    else:
        project_dir = directory
    project_config['project']['path'] = project_dir

    project_config['project']['data_path'] = 'DATA'
    project_config['project']['model_path'] = 'models'
    project_config['project']['labeler'] = labeler

    if not os.path.isdir(project_config['project']['path']):
        os.makedirs(project_config['project']['path'])
    os.chdir(project_config['project']['path'])
    if not os.path.isdir(project_config['project']['data_path']):
        os.makedirs(project_config['project']['data_path'])
    if not os.path.isdir(project_config['project']['model_path']):
        os.makedirs(project_config['project']['model_path'])

    fname = os.path.join(project_dir, 'project_config.yaml')
    project_config['project']['config_file'] = fname
    utils.save_dict_to_yaml(project_config, fname)
    return project_config


def add_video_to_project(project: dict,
                         path_to_video: Union[str, os.PathLike],
                         mode: str = 'copy') -> str:
    """
    Adds a video file to a DEG project.

    1. Copies the video file to the project's data directory
    2. initializes a record.yaml file
    3. Computes per-channel image statistics (for input normalization)

    Parameters
    ----------
    project: dict
        pre-loaded configuration dictionary
    path_to_video: str, PathLike
        absolute path to a video file. Filetype must be acceptable to deepethogram.file_io.VideoReader
    mode: str
        if 'copy': copies files to new directory
        if 'symlink': tries to make a symlink from the old location to the new location. NOT RECOMMENDED. if you delete
            the video in its current location, the symlink will break, and we will have errors during training or
            inference
        if 'move': moves the file

    Returns
    -------
    new_path: str
        path to the video file after moving to the DEG project data directory.
    """
    # assert (os.path.isdir(project_directory))
    assert os.path.exists(path_to_video), 'video not found! {}'.format(
        path_to_video)
    if os.path.isdir(path_to_video):
        copy_func = shutil.copytree
    elif os.path.isfile(path_to_video):
        copy_func = shutil.copy
    else:
        raise ValueError('video does not exist: {}'.format(path_to_video))

    assert mode in ['copy', 'symlink', 'move']

    # project = utils.load_yaml(os.path.join(project_directory, 'project_config.yaml'))
    # project = convert_config_paths_to_absolute(project)
    log.debug('configuration file when adding video: {}'.format(project))
    datadir = os.path.join(project['project']['path'],
                           project['project']['data_path'])
    assert os.path.isdir(datadir), 'data path not found: {}'.format(datadir)

    # for speed during training, videos can be saved as directories of PNG / JPEG files.
    if os.path.isdir(path_to_video):
        video_is_directory = True
    else:
        video_is_directory = False

    basename = os.path.basename(path_to_video)
    vidname = os.path.splitext(basename)[0]

    video_directory = os.path.join(datadir, vidname)
    if os.path.isdir(video_directory):
        raise ValueError('Directory {} already exists in your data dir! ' \
                         'Please rename the video to a unique name'.format(vidname))
    os.makedirs(video_directory)
    new_path = os.path.join(video_directory, basename)
    if mode == 'copy':
        if video_is_directory:
            shutil.copytree(path_to_video, new_path)
        else:
            shutil.copy(path_to_video, new_path)
    elif mode == 'symlink':
        os.symlink(path_to_video, new_path)
    elif mode == 'move':
        shutil.move(path_to_video, new_path)
    else:
        raise ValueError('invalid argument to mode: {}'.format(mode))

    record = parse_subdir(video_directory)
    log.debug('New record after adding: {}'.format(record))
    utils.save_dict_to_yaml(record, os.path.join(video_directory,
                                                 'record.yaml'))
    zscore_video(os.path.join(video_directory, basename), project)
    return new_path


def add_label_to_project(path_to_labels: Union[str, os.PathLike],
                         path_to_video) -> str:
    """Adds an externally created label file to the project. Updates record"""
    assert os.path.isfile(path_to_labels)
    assert os.path.isfile(path_to_video)
    assert is_deg_file(path_to_video)
    viddir = os.path.dirname(path_to_video)

    label_dst = os.path.join(viddir, os.path.basename(path_to_labels))

    if os.path.isfile(label_dst):
        warnings.warn(
            'Label already exists in destination {}, overwriting...'.format(
                label_dst))

    df = pd.read_csv(path_to_labels, index_col=0)
    if 'none' in list(df.columns):
        df = df.rename(columns={'none': 'background'})
    if 'background' not in list(df.columns):
        array = df.values
        is_background = np.logical_not(np.any(array == 1, axis=1)).astype(int)
        df2 = pd.DataFrame(data=is_background, columns=['background'])
        df = pd.concat([df2, df], axis=1)

    df.to_csv(label_dst)
    record = parse_subdir(viddir)
    utils.save_dict_to_yaml(record, os.path.join(viddir, 'record.yaml'))
    return label_dst


def add_file_to_subdir(file: Union[str, os.PathLike],
                       subdir: Union[str, os.PathLike]):
    """If you save or move a file into a DEG subdirectory, update the record"""
    if not is_deg_file(subdir):
        raise ValueError('directory is not a DEG subdir: {}'.format(subdir))
    assert (os.path.isfile(file))
    if os.path.dirname(file) != subdir:
        shutil.copy(file, os.path.join(subdir, os.path.basename(file)))
    record = parse_subdir(subdir)
    utils.save_dict_to_yaml(record, os.path.join(subdir, 'record.yaml'))


def change_project_directory(config_file: Union[str, os.PathLike],
                             new_directory: Union[str, os.PathLike]):
    """If you move the project directory to some other location, updates the config file to have the new directories"""
    assert os.path.isfile(config_file)
    assert os.path.isdir(new_directory)
    # make sure that new directory is properly formatted for deepethogram
    datadir = os.path.join(new_directory, 'DATA')
    model_path = os.path.join(new_directory, 'models')
    assert os.path.isdir(datadir)
    assert os.path.isdir(model_path)

    project_config = utils.load_yaml(config_file)
    project_config['project']['path'] = new_directory
    project_config['project']['model_path'] = os.path.basename(model_path)
    project_config['project']['data_path'] = os.path.basename(datadir)
    project_config['project']['config_file'] = os.path.join(
        new_directory, 'project_config.yaml')
    utils.save_dict_to_yaml(project_config,
                            project_config['project']['config_file'])


def remove_video_from_project(config_file,
                              video_file=None,
                              record_directory=None):
    # TODO: remove video from split dictionary, remove mean and std from project statistics
    raise NotImplementedError


def is_deg_file(filename: Union[str, os.PathLike]) -> bool:
    """Quickly assess if a file is part of a well-formatted subdirectory with a records.yaml"""
    if os.path.isdir(filename):
        basedir = filename
        is_directory = True
    elif os.path.isfile(filename):
        basedir = os.path.dirname(filename)
        is_directory = False
    else:
        raise ValueError(
            'submit directory or file to is_deg_file, not {}'.format(filename))

    recordfile = os.path.join(basedir, 'record.yaml')
    record_exists = os.path.isfile(recordfile)

    if is_directory:
        # this is required in case the file passed is a directory full of images; e.g.
        # project/DATA/animal0/images/00000.jpg
        parent_record_exists = os.path.isfile(os.path.join(os.path.dirname(filename), 'record.yaml'))
        return record_exists or parent_record_exists
    else:
        return record_exists


def add_behavior_to_project(config_file: Union[str, os.PathLike],
                            behavior_name: str):
    """ Adds a behavior (class) to the project.

    Adds this behavior to the class_names field of your project configuration.
    Adds -1 column in all labelfiles in current project.
    Saves the altered project_config to disk.

    Parameters
    ----------
    config_file: str, PathLike
        path to the project config file
    behavior_name: str
        behavior to add to the project.
    """
    assert (os.path.isfile(config_file))
    project_config = utils.load_yaml(config_file)
    assert 'class_names' in list(project_config['project'].keys())
    classes = project_config['project']['class_names']
    assert behavior_name not in classes
    classes.append(behavior_name)

    records = get_records_from_datadir(
        os.path.join(project_config['project']['path'],
                     project_config['project']['data_path']))
    for key, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        if os.path.isfile(labelfile):
            df = pd.read_csv(labelfile, index_col=0)
            label = df.values
            N, K = label.shape
            # label = np.concatenate((label, np.ones((N, 1))*-1), axis=1)
            df2 = pd.DataFrame(data=np.ones((N, 1)) * -1,
                               columns=[behavior_name])
            df = pd.concat([df, df2], axis=1)
            df.to_csv(labelfile)
    project_config['project']['class_names'] = classes
    utils.save_dict_to_yaml(project_config, config_file)


def remove_behavior_from_project(config_file: Union[str, os.PathLike],
                                 behavior_name: str):
    """Removes behavior (class) from existing project.

    Removes behavior name from project configuration file.
    Decrements num_behaviors by one.
    Removes this column from existing label files.
    Saves altered project configuration to disk.

    Parameters
    ----------
    config_file: str, PathLike
        path to a deepethogram configuration file
    behavior_name: str
        One of the existing behavior_names.
    """
    if behavior_name == 'background':
        raise ValueError('Cannot remove background class.')
    assert (os.path.isfile(config_file))
    project_config = utils.load_yaml(config_file)
    assert 'class_names' in list(project_config['project'].keys())
    classes = project_config['project']['class_names']
    assert behavior_name in classes

    records = get_records_from_datadir(
        os.path.join(project_config['project']['path'],
                     project_config['project']['data_path']))
    for key, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        if os.path.isfile(labelfile):
            df = pd.read_csv(labelfile, index_col=0)

            if behavior_name not in list(df.columns):
                continue
            df2 = df.drop(behavior_name, axis=1)
            df2.to_csv(labelfile)
    classes.remove(behavior_name)
    project_config['project']['class_names'] = classes
    utils.save_dict_to_yaml(project_config, config_file)


def get_classes_from_project(
        config: Union[dict, str, os.PathLike, DictConfig]) -> list:
    """Loads current class names from current project directory or configuration dictionary.

    Parameters
    ----------
    config: configuration dictionary or *project directory*
        if configuration dictionary: must have a "project_dir" key

    Returns
    -------
    classes: list
        list of behaviors read from project_config.yaml file.
    """

    if type(config) == str or type(config) == os.PathLike:
        config_file = os.path.join(config, 'project_config.yaml')
        assert os.path.isfile(
            config_file
        ), 'Input must be a directory containing a project_config.yaml file'
        config = utils.load_yaml(config_file)

    assert 'project' in list(
        config.keys()), 'Invalid project configuration dictionary: {}'.format(
            config)
    project = config['project']
    assert 'class_names' in list(
        project.keys()), 'Must have class names in project config file'
    return project['project']['class_names']


def exclude_strings_from_filelist(files: list, excluded: list) -> list:
    """Returns a sub-list from a list of strings that do not have any substrings in EXCLUDED.

    Example:
        files = ['movie.avi', 'labels.csv', 'record.yaml']
        non_movies = exclude_strings_from_filelist(files, ['avi', 'mp4'])
    Args:
        files (list of strings): list to subselect from
        excluded (list of strings): list of forbidden substrings

    Returns:
        valid_files: elements from files without any substrings in excluded
    """
    files.sort()
    valid_files = []
    for file in files:
        base = os.path.basename(file)
        if not any([i in base for i in excluded]):
            valid_files.append(file)
    return valid_files


def find_labelfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """ Gets label files from a deepethogram data directory

    Args:
        root (str, pathlike): directory containing labels, movies, etc

    Returns:
        files: list of score or label files
    """
    files = get_subfiles(root, return_type='file')
    files = [i for i in files if 'label' in os.path.basename(i).lower() or 'score' in os.path.basename(i).lower()]
    return files


def find_rgbfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """Finds all possible RGB video files in a deepethogram data directory

    Args:
        root (str, pathlike): deepethogram data directory

    Returns:
        list of absolute paths to RGB videos, or subdirectories containing individual images (framedirs)
    """
    files = get_subfiles(root, return_type='any')
    endings = [os.path.splitext(i)[1] for i in files]
    valid_endings = ['.avi', '.mp4', '.h5', '.mov']
    excluded = ['flow', 'label', 'output', 'score']
    movies = [i for i in files if os.path.splitext(i)[1].lower() in valid_endings]
    movies = exclude_strings_from_filelist(movies, excluded)

    framedirs = get_subfiles(root, return_type='directory')
    framedirs = exclude_strings_from_filelist(framedirs, excluded)
    return movies + framedirs


def find_flowfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """ DEPRECATED.

    Args:
        root ():

    Returns:

    """
    files = get_subfiles(root, return_type='any')
    endings = [os.path.splitext(i)[1] for i in files]
    valid_endings = ['.avi', '.mp4', '.h5']
    movies = [
        files[i] for i in range(len(files))
        if endings[i] in valid_endings and 'flow' in os.path.basename(files[i])
    ]
    framedirs = [
        i for i in get_subfiles(root, return_type='directory')
        if 'frame' in i and 'flow' in os.path.basename(i)
    ]
    return movies + framedirs


def find_outputfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """ Finds deepethogram outputfiles, containing RGB and flow features, along with P(K)

    Args:
        root (str, pathlike): deepethogram data directory

    Returns:
        list of outputfiles. should only have one element
    """
    files = get_subfiles(root, return_type='file')
    files = [i for i in files if 'output' in os.path.basename(i).lower() and os.path.splitext(i)[1].lower() == '.h5']
    return files


def find_keypointfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """ Finds .csv files of DeepLabCut outputs in the data directories

    Args:
        root: (str, pathlike): deepethogram data directory

    Returns:
        list of dlcfiles. should only have one element
    """
    # TODO: support SLEAP, DLC hdf5 files
    files = get_subfiles(root, return_type='file')
    files = [i for i in files if 'dlc' in os.path.basename(i).lower() and os.path.splitext(i)[1] == '.csv']
    return files



def find_statsfiles(root: Union[str, bytes, os.PathLike]) -> list:
    """ Finds normalization statistics in deepethogram data directory

    Args:
        root (str, pathlike)
            deepethogram data directory

    Returns:
        list of stats files, should only have 1 or 0 elements
    """
    files = get_subfiles(root, return_type='file')
    files = [
        i for i in files
        if 'stats' in os.path.basename(i) and os.path.splitext(i)[1] == '.yaml'
    ]
    return files


def get_type_from_file(file: Union[str, bytes, os.PathLike]) -> str:
    """Convenience function. Gets type of VideoReader input file from a path"""
    if os.path.isdir(file):
        if 'frame' in os.path.basename(file):
            return 'directory'
    elif os.path.isfile(file):
        _, ext = os.path.splitext(file)
        return ext
    else:
        raise ValueError('file does not exist: {}'.format(file))


def get_files_by_preferences(files, preference: list = None) -> str:
    """ Given a list of files with different types, return the most-preferred filetype (given by preference)

    Example:
        files = ['movie.mp4', 'movie.avi']
        preference = ['.h5', '.avi', '.mp4']
        file = get_files_by_preference(files, preference)
        # file = 'movie.avi'

    Args:
        files (list of strings, paths): list of files
        preference (list of strings): ordered list of filetypes. first = most preferred, last = least preferred

    Returns:
        If no files with the given ending are present, returns empty list
        If only one file, returns that file
        If multiple files, with endings in preference, returns the first existing file in the order given by preference

    """
    # preference should be an ordered list of file types
    # example: ['directory', '.h5', '.avi', '.mp4']
    if len(files) == 0:
        return []
    if preference is not None:
        types = {get_type_from_file(i): i for i in files}
        keys = list(types.keys())
        for p in preference:
            if p in keys:
                return types[p]
        # if you didn't return in the loop, it's because the file type
        # is not one in the list of preferences at all
        default = files[0]
    else:
        default = files[0]
    return default


def parse_subdir(root: Union[str, bytes, os.PathLike],
                 preference: list = None) -> dict:
    """ Find rgb, flow, label, output, and channel statistics files in a given directory

    Parameters
    ----------
    root: str, os.PathLike
        sub-directory, aka DATA/animal0
    preference: list
        ordered list of filetype preferences for videos. if preference is [.avi, .h5, .mp4], then in the case where
        there are multiple filetypes for a given video, it will return the list ordered by preference, e.g.
        .avi, .h5, then .mp4

    Returns
    -------
    record: dict
        dictionary. keys: file types. values: list of candidate files in that type. e.g. [flow.avi, flow_images/]
        Their order will be given by preference

    Examples
    -------
    record = parse_subdir('/path/to/DATA/animal0')
    """
    if preference is None:
        # determine default here
        # sorted by combination of sequential and random read speeds
        preference = ['directory', '.h5', '.avi', '.mp4']

    find_files = {
        'rgb': find_rgbfiles,
        'flow': find_flowfiles,
        'label': find_labelfiles,
        'output': find_outputfiles,
        'stats': find_statsfiles,
        'keypoint': find_keypointfiles
    }

    record = {}

    for entry in list(find_files.keys()):
        record[entry] = {}
        files = find_files[entry](root)
        if len(files) == 0:
            record[entry]['all'] = []
            record[entry]['default'] = []
        else:
            record[entry]['all'] = [os.path.basename(i) for i in files]
            record[entry]['default'] = os.path.basename(
                get_files_by_preferences(files, preference))
    record['key'] = os.path.basename(root)
    return record


# def write_all_records(root: Union[str, bytes, os.PathLike],
#                       preference: list = None):
#     """ For a given data directory, finds all subdirs and their files. Saves their records as .yaml files

#     Parameters
#     ----------
#     root: str, bytes, os.PathLike
#         data directory. e.g. '/path/to/DATA', which contains 'animal0, animal1, animal2'
#     preference: list
#         list of filetype preferences. see parse_subdir

#     Returns
#     -------
#     None
#     """
#     subdirs = get_subfiles(root, return_type='directory')
#     for subdir in subdirs:
#         record = parse_subdir(subdir, preference=preference)
#         outfile = os.path.join(subdir, 'record.yaml')
#         utils.save_dict_to_yaml(record, outfile)


# def add_key_to_record(record: dict, key: str, value) -> dict:
#     if key in list(record.keys()):
#         raise ValueError('key {} already exists in record :{}'.format(key, record))
#     valtype = type(value)
#     if valtype in [str, bytes, os.PathLike]:
#         if os.path.isfile(value):
#             value = os.path.basename(value)
#     record[key] = {}
#     record[key]['all'] = [value]
#     record[key]['default'] = value
#     return record


def get_record_from_subdir(subdir: Union[str, os.PathLike]) -> dict:
    """ Gets a dictionary of absolute filepaths for each semantic file type, e.g. RGB movies, labels, output files

    Parameters
    ----------
    subdir: str, os.PathLike
        Directory containing raw data, e.g. /path/to/DATA/animal0

    Returns
    -------
    record: dict
        Dictionary containing absolute paths for DeepEthogram files. Keys:
        rgb: RGB movie
        label: csv or hdf5 file containing binary labels
        output: HDF5 file containing feature vectors and probabilities
        flow: deprecated (for saving optic flow to disk)
        stats: yaml file containing that video's RGB channel statistics
        key: basename of this directory. e.g. animal0
    """
    record = parse_subdir(subdir)

    parsed_record = {}
    for key in ['flow', 'label', 'output', 'rgb', 'keypoint']:
        if key in list(record.keys()):
            this_entry = record[key]['default']

            if type(this_entry) == list and len(this_entry) == 0:
                this_entry = None
            else:
                this_entry = os.path.join(subdir, this_entry)
                if not os.path.isfile(this_entry) and not os.path.isdir(
                        this_entry):
                    this_entry = None
            parsed_record[key] = this_entry
    parsed_record['key'] = os.path.basename(subdir)
    return parsed_record


def get_records_from_datadir(datadir: Union[str, bytes, os.PathLike]) -> dict:
    """ Gets a dictionary of record dictionaries from a data directory

    Parameters
    ----------
    datadir: str, bytes, os.PathLike

    Returns
    -------
    records: dict
        e.g.
        {'animal0':
            {'rgb': /path/to/DATA/animal0/rgb_video.mp4,
             'label': /path/to/DATA/animal0/labels.csv,
             'output': /path/to/DATA/animal0/outputs.h5,
             'stats': /path/to/DATA/animal0/stats.yaml
             }
        'animal1': {...}
        ...
        }
    """
    assert os.path.isdir(datadir), 'datadir does not exist: {}'.format(datadir)
    subdirs = get_subfiles(datadir, return_type='directory')
    records = {}
    for subdir in subdirs:
        parsed_record = get_record_from_subdir(os.path.join(datadir, subdir))
        records[parsed_record['key']] = parsed_record
    # write_all_records(datadir)
    return records


def filter_records_for_filetypes(records: dict, return_types: list):
    """ Find the records that have all the requested filetypes. e.g. get all subdirectories with labels """
    valid_records = {}
    for k, v in records.items():
        # k is the key for this record, e.g. experiment00_mouse00
        # v is the dictionary with files found for this record, e.g.
        # {rgb: movie.avi, label: labels.csv, flow: None, output: None}
        all_present = True
        for t in return_types:
            if v[t] is None:
                warnings.warn('No {} file found in record: {}'.format(t, k))
                all_present = False
        if all_present:
            valid_records[k] = v
    return valid_records


# def write_latest_model(model_type: str, model_name: str, model_path: Union[str, os.PathLike], config: dict):
#     latest_models_file = os.path.join(config['project_dir'], 'latest_models.yaml')
#     if not os.path.isfile(latest_models_file):
#         shutil.copy(os.path.join('defaults', 'latest_models.yaml'), latest_models_file)
#     latest_models = utils.load_yaml(latest_models_file)
#     if model_type not in list(latest_models.keys()):
#         latest_models[model_type] = {}
#
#     # we want to keep our model paths relative so that the user can move the model dir around
#     model_path = os.path.relpath(model_path, config['model_path'])
#
#     latest_models[model_type][model_name] = model_path
#     utils.save_dict_to_yaml(latest_models, latest_models_file)


def is_config_dict(config: dict) -> bool:
    """ Tells if a dictionary is a valid project dictionary """
    config_keys = list(config.keys())
    for k in required_keys:
        if k not in config_keys:
            return False
    return True


def get_number_finalized_labels(config: dict) -> int:
    """ Finds the number of label files with no unlabeled frames  """
    records = get_records_from_datadir(
        os.path.join(config['project']['path'],
                     config['project']['data_path']))
    number_valid_labels = 0
    for k, v in records.items():
        for filetype, fileloc in v.items():
            if filetype == 'label':
                if fileloc is None or len(fileloc) == 0:
                    continue
                label = read_labels(fileloc)
                has_unlabeled_frames = np.any(label == -1)
                if not has_unlabeled_frames:
                    number_valid_labels += 1
    return number_valid_labels


def import_outputfile(project_dir: Union[str, os.PathLike],
                      outputfile: Union[str, os.PathLike],
                      class_names: list = None,
                      latent_name: str = None):
    """  Gets the probabilities, thresholds, used HDF5 dataset key, and all dataset keys from an outputfile

    Parameters
    ----------
    project_dir: str, os.PathLike
        absolute filepath to a deepethogram project. Used for loading default latent_names
    outputfile: str, os.PathLike
        Absolute filepath to a deepethogram output file, with probabilities, latents, thresholds
    class_names: list
        list of class names. used for thresholds
    latent_name: str
        Saved latent name in the outputfile. This is where deepethogram.feature_extractor.inference saves the 2 512D
        feature vectors, or where the deepethogram.sequence.inference saves probabilities and thresholds
        These are *keys* to an HDF5 file!

    Returns
    -------
    probabilities: np.ndarray
        (T x K) float32 array of behavior probabilities
    thresholds:
        (K, ) float32 array of thresholds for turning probabilities -> binary predictions
    key: str
        key in the HDF5 file used to load data. Will be latent_name if passed, or inferred if None is passed
    keys: list
        other keys in the HDF5 file
    """
    assert os.path.isfile(outputfile)
    assert os.path.isdir(project_dir)
    # handle edge case
    if latent_name == '' or latent_name == ' ':
        latent_name = None

    # all this tortured logic is to try to figure out what the correct "latent name" is in an HDF5 file. Also includes
    # logic for backwards compatibility
    project_config = load_config(os.path.join(project_dir, 'project_config.yaml'))
    if 'sequence' in project_config.keys() and 'arch' in project_config['sequence'].keys():
        sequence_name = project_config['sequence']['arch']
    else:
        sequence_name = load_default('model/sequence')['sequence']['arch']

    if 'sequence' in project_config.keys() and 'latent_name' in project_config['sequence'].keys():
        sequence_inference_latent_name = project_config['sequence']['latent_name']
    else:
        sequence_inference_latent_name = None
    if 'feature_extractor' in project_config.keys() and 'arch' in project_config['feature_extractor'].keys():
        feature_extractor_arch = project_config['feature_extractor']['arch']
    elif 'preset' in project_config.keys():
        preset = project_config['preset']
        preset_config = load_default('preset/{}'.format(preset))
        feature_extractor_arch = preset_config['feature_extractor']['arch']
    else:
        feature_extractor_arch = load_default(
            'model/feature_extractor')['feature_extractor']['arch']

    with h5py.File(outputfile, 'r') as f:

        keys = list(f.keys())
        if len(keys) == 0:
            raise ValueError(
                'no datasets found in outputfile: {}'.format(outputfile))

        # Order of priority for determining latent name, from high -> low
        # 1. input argument 2. custom latent name from sequence inference 3. the sequence arch name 4. the feature
        # extractor arch name 5. the first one in the HDF5 file.
        if latent_name is not None:
            assert latent_name in keys
            key = latent_name
        elif sequence_inference_latent_name is not None:
            # the project config file specifies a latent name
            key = sequence_inference_latent_name
        elif sequence_name in keys:
            # use the sequence model architecture as a latent name
            key = sequence_name
        elif feature_extractor_arch in keys:
            key = feature_extractor_arch
        else:
            log.warning(
                'No default latent names found, using the first one instead. Keys: {}'
                .format(keys))
            key = keys[0]

        log.info('Key used to load outputfile: {}'.format(key))
        probabilities = f[key]['P'][:]
        negative_probabilities = np.sum(probabilities < 0)
        if negative_probabilities > 0:
            log.warning('N={} negative probabilities found in file {}'.format(
                negative_probabilities, os.path.basename(outputfile)))
            probabilities[probabilities < 0] = 0

        thresholds = f[key]['thresholds'][:]
        if thresholds.ndim == 2:
            # this should not happen
            thresholds = thresholds[-1, :]
        loaded_class_names = f[key]['class_names'][:]
        if type(loaded_class_names[0]) == bytes:
            loaded_class_names = [
                i.decode('utf-8') for i in loaded_class_names
            ]
    log.debug('probabilities shape: {}'.format(probabilities.shape))

    # if you pass class names, make sure that the order matches the order of the argument. Else, just return it
    # in the order it is in the HDF5 file
    if class_names is None:
        return probabilities, thresholds, latent_name, keys

    log.debug('imported names: {}'.format(loaded_class_names))
    indices = []
    for class_name in class_names:
        ind = [
            i for i in range(len(loaded_class_names))
            if loaded_class_names[i] == class_name
        ]
        if len(ind) == 1:
            indices.append(ind[0])
    indices = np.array(indices).squeeze()
    log.debug('indices: {} type: {} shape: {}'.format(indices, type(indices), indices.shape))
    if not indices.shape:
        raise ValueError(
            'Class names not found in file. Loaded: {} Requested: {}'.format(
                loaded_class_names, class_names))
    if len(indices) == 0:
        raise ValueError(
            'Class names not found in file. Loaded: {} Requested: {}'.format(
                loaded_class_names, class_names))
    probabilities = probabilities[:, indices]
    thresholds = thresholds[indices]

    return probabilities, thresholds, key, keys


def has_outputfile(records: dict) -> list:
    """ Convenience function for finding output files in a dictionary of records"""
    keys, has_outputs = [], []
    # check to see which records have outputfiles
    for key, record in records.items():
        keys.append(key)
        has_outputs.append(record['output'] is not None)
    return has_outputs


def do_outputfiles_have_predictions(data_path: Union[str, os.PathLike],
                                    model_name: str) -> list:
    """ Looks for HDF5 datasets in data_path of name model_name """
    assert os.path.isdir(data_path)
    records = get_records_from_datadir(data_path)
    has_predictions = []
    for key, record in records.items():
        file = records[key]['output']
        if file is None:
            has_predictions.append(False)
            continue
        with h5py.File(file, 'r') as f:
            if model_name in list(f.keys()):
                has_predictions.append(True)
            else:
                has_predictions.append(False)
    return has_predictions


def extract_date(string: str):
    """ Extracts the actual date time from a formatted string. Used for finding most recent models """
    pattern = re.compile('\d{6}_\d{6}')
    match = pattern.search(string)
    if match is not None:
        match = match.group()
        match = datetime.strptime(match, '%y%m%d_%H%M%S')
    return match


def sort_runs_by_date(runs: list) -> list:
    """ Sorts run directories by date using the date string in the directory name """
    runs_and_dates = []
    for run in runs:
        runs_and_dates.append((run, extract_date(run)))
    runs_and_dates = sorted(runs_and_dates, key=lambda index: index[1])

    sorted_runs = [run[0] for run in runs_and_dates]
    return sorted_runs


def get_weights_from_model_path(model_path: Union[str, os.PathLike]) -> dict:
    """ Finds absolute path to weight files for each model type and architecture

    Parameters
    ----------
    model_path: str, os.PathLike
        /path/to/models

    Returns
    -------
    model_weights: dict of dicts of lists
        Easiest to understand by example
        {'flow_generator':
            {'TinyMotionNet': [/path/to/oldest/tinymotionnet_checkpoint.pt, path_to_newest/tinymotionnet_checkpoint.pt],
            'MotionNet': [path/to/oldest/motionnet_checkpoint.pt, path/to/newest/motionnet_checkpoint.pt],
            'TinyMotionNet3D': ...
            },
        'feature_extractor':
            {'resnet18': [path/to/oldest/resnet18.checkpoint, path/to/newest/resnet18.checkpoint],
            'resnet50': ...
            }
        'sequence:
            {tgmj: ...
            }
        }
    """
    rundirs = get_subfiles(model_path, return_type='directory')
    # assume the models are only at most one sub directory underneath
    for rundir in rundirs:
        subdirs = get_subfiles(rundir, return_type='directory')
        rundirs += subdirs
    rundirs.sort()

    # model_weights = defaultdict(list)
    model_weights = {
        'flow_generator': {},
        'feature_extractor': {},
        'sequence': {}
    }
    for rundir in rundirs:
        # for backwards compatibility
        paramfile = os.path.join(rundir, 'hyperparameters.yaml')
        if not os.path.isfile(paramfile):
            paramfile = os.path.join(rundir, 'config.yaml')

            if not os.path.isfile(paramfile):
                continue
        params = utils.load_yaml(paramfile)
        if params is None:
            continue
        # this horrible if-else tree is for backwards compatability with how I used to save config files
        if 'model' in params.keys():
            model_type = params['model']
            if params['model'] in params.keys():
                arch = params[params['model']]
            elif params['model'] == 'feature_extractor':
                arch = params['classifier']
            elif 'arch' in params.keys():
                arch = params['arch']
            else:
                raise ValueError(
                    'Could not find architecture from config: {}'.format(
                        params))

        elif 'run' in params.keys():
            model_type = params['run']['model']
            arch = params[model_type]['arch']
        else:
            continue

        # architecture = params[model_type]['arch']

        weightfile = os.path.join(rundir, 'checkpoint.pt')
        if os.path.isfile(weightfile):
            if arch in model_weights[model_type].keys():
                model_weights[model_type][arch].append(weightfile)
            else:
                model_weights[model_type][arch] = [weightfile]
            # model_weights[model_type].append(weightfile)
            # model_weights[model_type][arch].append(weightfile)
    for model in model_weights.keys():
        for arch, runlist in model_weights[model].items():
            model_weights[model][arch] = sort_runs_by_date(runlist)
    return model_weights


# def overwrite_cfg_with_latest_weights(cfg: DictConfig, model_weights: Union[str, os.PathLike, defaultdict],
#                                       model_type: str) -> DictConfig:
#     if cfg.reload.weights is not None:
#         # user has specified specific model weights, don't overwrite with most recent model
#         return cfg
#     if type(model_weights) == str or type(model_weights) == os.PathLike:
#         model_weights = get_weights_from_model_path(model_weights)
#     if model_type not in model_weights.keys() or len(model_weights[model_type]) == 0:
#         log.warning('No pretrained {} model found. not loading'.format(model_type))
#         return cfg
#     latest_weights = model_weights[model_type][-1]
#     assert os.path.isfile(latest_weights)
#     cfg.reload.weights = latest_weights
#
#     # let's say you're retraining the feature extractors. For this, since we're using hidden two stream networks, we
#     # need to have a pretrained flow generator. This will allow us to find the latest weights of each model type.
#     # in our feature extractor trainer, we can independently load a pretrained feature extractor (e.g. resnet18)
#     # and a separate flow generator using the below weights
#     for model in ['flow_generator', 'feature_extractor', 'sequence']:
#         if model not in model_weights.keys() or len(model_weights[model_type]) == 0:
#             continue
#         # if the user has specified certain model weights, don't overwrite them here
#         if cfg[model].weights is not None:
#             latest_weights = model_weights[model][-1]
#             assert (os.path.isfile(latest_weights))
#             cfg[model].weights = latest_weights
#     return cfg


def get_weight_file_absolute_or_relative(cfg, path_to_weights):
    if os.path.isfile(path_to_weights):
        return path_to_weights
    else:
        abs_path = os.path.join(cfg.project.model_path, path_to_weights)
        assert os.path.isfile(abs_path)
        return abs_path


def get_weightfile_from_cfg(cfg: DictConfig,
                            model_type: str) -> Union[str, None]:
    """ Gets the correct weight files from the configuration.

    The weights are loaded in the following order of priority
    1. cfg.reload.weights: assume a specific pretrained weightfile with all components (flow_generator, spatial, flow)
    2. each component individually:
        cfg.flow_generator.weights
        cfg.feature_extractor.weights
    3. if cfg.reload.latest is True:
        scan the model_path folder for matching weights
        use the most recent model ( might not have best performance, etc)
    ASSUMES ONLY ONE ARCHITECTURE! If you switch architectures, weight reloading will fail. You'll have to manually
    supply pretrained weights with cfg.(model_type).weights=path/to/weights
    Parameters
    ----------
    cfg
    model_type

    Returns
    -------
    weightfile: path to weight file
    """

    # if cfg.reload.weights is not None:
    #     assert os.path.isfile(cfg.reload.weights)
    #     return cfg.reload.weights

    assert model_type in ['flow_generator', 'feature_extractor', 'end_to_end', 'sequence']

    if not os.path.isdir(cfg.project.model_path):
        cfg = convert_config_paths_to_absolute(cfg)

    trained_models = get_weights_from_model_path(cfg.project.model_path)

    architecture = cfg[model_type].arch

    if cfg[model_type].weights is not None and cfg[model_type].weights == 'pretrained':
        assert model_type in ['flow_generator', 'feature_extractor']
        pretrained_models = get_weights_from_model_path(cfg.project.pretrained_path)
        assert len(pretrained_models[model_type][architecture]) > 0
        weights = pretrained_models[model_type][architecture][-1]
        log.info('loading pretrained weights: {}'.format(weights))
        return weights
    
    if model_type == 'end_to_end':
        if cfg.reload.latest:
            assert len(trained_models['feature_extractor'][architecture]) > 0
            return trained_models['feature_extractor'][architecture][-1]
    else:
        if cfg[model_type].weights is not None and cfg[model_type].weights != 'latest':
            path_to_weights = get_weight_file_absolute_or_relative(cfg, cfg[model_type].weights)
            assert os.path.isfile(path_to_weights)
            log.info('loading specified weights')
            return path_to_weights
        elif cfg.reload.latest or cfg[model_type].weights == 'latest':
            # print(trained_models)
            assert len(trained_models[model_type][architecture]) > 0
            log.debug('trained models found: {}'.format(trained_models[model_type][architecture]))
            log.info('loading LATEST weights: {}'.format(trained_models[model_type][architecture][-1]))
            return trained_models[model_type][architecture][-1]
        else:
            log.warning('no {} weights found...'.format(model_type))
            return


def convert_config_paths_to_absolute(project_cfg: DictConfig) -> DictConfig:
    """ Converts relative file paths in a project configuration into absolute paths.

    Example:
        project_cfg['project']['path'] = '/path/to/project'
        project_cfg['project']['model_path'] = 'models'
        project_cfg['project']['data_path'] = 'DATA'

        project_cfg = convert_config_paths_to_absolute(project_cfg)
        print(project_cfg)
        # project_cfg['project']['path'] = '/path/to/project'
        # project_cfg['project']['model_path'] = '/path/to/project/models'
        # project_cfg['project']['data_path'] = '/path/to/project/DATA'

    Args:
        project_cfg (dict): project configuration dictionary

    Returns:
        project_cfg (dict)
    """
    assert 'project' in project_cfg.keys()

    root = project_cfg['project']['path']
    model_path = project_cfg['project']['model_path']
    data_path = project_cfg['project']['data_path']
    # backwards compatibility
    if 'pretrained_path' in project_cfg['project'].keys():
        pretrained_path = project_cfg['project']['pretrained_path']
    else:
        pretrained_path = 'pretrained_models'
    cfg_path = os.path.join(root, project_cfg['project']['config_file'])
    
    if (os.path.isdir(model_path) and os.path.isdir(data_path) and os.path.isfile(cfg_path)
        and os.path.isdir(pretrained_path)):
        # already absolute
        return project_cfg
    
    if not os.path.isdir(model_path):
        model_path = os.path.join(root, model_path)
        assert os.path.isdir(model_path), 'model path does not exist! {}'.format(model_path)
    
    if not os.path.isdir(data_path):
        data_path = os.path.join(root, data_path)
        assert os.path.isdir(data_path), 'data path does not exist! {}'.format(data_path)
        
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(root, cfg_path)
        assert os.path.isdir(cfg_path), 'config file does not exist! {}'.format(cfg_path)
        
    if not os.path.isdir(pretrained_path):
        
        # pretrained_dir can be one of the following locations:
        # my_model_dir/pretrained
        # my_project/pretrained
        # my_project/models/pretrained
        pretrained_options = [os.path.join(i, pretrained_path) for i in 
                              [model_path, root, os.path.join(root, 'models')]]
        
        exists = [os.path.isdir(i) for i in pretrained_options]
        
        try:
            index = exists.index(True)
            pretrained_path = pretrained_options[index]
        except ValueError as e:
            error_string = 'pretrained directory does not exist! {}\nSee instructions '.format(pretrained_path) + \
            'on the project GitHub for downloading weights: https://github.com/jbohnslav/deepethogram'
            print(error_string)
            raise
        
    
    project_cfg['project']['model_path'] = model_path
    project_cfg['project']['data_path'] = data_path
    project_cfg['project']['config_file'] = cfg_path
    project_cfg['project']['pretrained_path'] = pretrained_path
    
    return project_cfg


def load_config(path_to_config: Union[str, os.PathLike]) -> dict:
    """Convenience function to load dictionary from yaml and sort out potentially erroneous paths"""
    assert os.path.isfile(path_to_config), 'configuration file does not exist! {}'.format(path_to_config)

    project = utils.load_yaml(path_to_config)
    project = fix_config_paths(project, path_to_config)
    # project = convert_config_paths_to_absolute(project)
    return project


def load_default(conf_name: str) -> dict:
    """ Loads default configs from deepethogram install path
    DEPRECATED. 
    TODO: replace with configuration.load_config_by_name
    """
    log.debug(
        'project loc for loading default: {}'.format(projects_file_directory))
    defaults_file = os.path.join(projects_file_directory, 'conf',
                                 os.path.relpath(conf_name) + '.yaml')
    assert os.path.isfile(
        defaults_file), 'configuration file does not exist! {}'.format(
            defaults_file)

    defaults = utils.load_yaml(defaults_file)
    return defaults

def convert_all_videos(config_file: Union[str, os.PathLike], movie_format='hdf5') -> None:
    """Converts all videos in a project from one filetype to another. 
    
    Note: If using movie_format other than 'directory' or 'hdf5', will re-compress images!

    Parameters
    ----------
    config_file : Union[str, os.PathLike]
        Path to a project configuration file
    movie_format : str, optional
        See file_io.convert_video, by default 'hdf5'
    """
    assert os.path.isfile(config_file)
    project_config = utils.load_yaml(config_file)

    records = get_records_from_datadir(os.path.join(project_config['project']['path'],
                                                    project_config['project']['data_path']))
    for key, record in tqdm(records.items(), desc='converting videos'):
        videofile = record['rgb']
        try:
            convert_video(videofile, movie_format=movie_format)
        except ValueError as e:
            print(e)


def get_config_from_path(path: Union[str, os.PathLike]) -> str:
    for cfg_path in ['project', 'project_config']:
        cfg_path = os.path.join(path, cfg_path + '.yaml')
        if os.path.isfile(cfg_path):
            return cfg_path
    raise ValueError('No configuration file found in directory! {}'.format(os.listdir(path)))


def fix_config_paths(cfg, path_to_config: Union[str, os.PathLike]):
    error = False
    if cfg['project']['path'] != os.path.dirname(path_to_config):
        log.warning('Erroneous project path in the config file itself, changing...')
        cfg['project']['path'] = os.path.dirname(path_to_config)
        error = True
    if cfg['project']['config_file'] != os.path.basename(path_to_config):
        log.warning('Erroneous name of config file in the config file itself, changing...')
        cfg['project']['config_file'] = os.path.basename(path_to_config)
        error = True
    if error:
        utils.save_dict_to_yaml(cfg, path_to_config)
    return cfg


def get_config_file_from_project_path(project_path):
    assert os.path.isdir(project_path)
    project_path = os.path.abspath(project_path)
    cfg_file = get_config_from_path(project_path)
    project_cfg = OmegaConf.load(cfg_file)
    project_cfg = fix_config_paths(project_cfg, cfg_file)
    return project_cfg


# def parse_cfg_paths(cfg: DictConfig) -> DictConfig:
#     """ Changes config file relative paths to absolute paths. Fixes broken paths, if any """
#     project = cfg.project
#
#     if project.path is None:
#         return cfg
#
#     assert os.path.isdir(project.path)
#     # whatever's in project.path is the canonical location
#     cfg_path = get_config_from_path(project.path)
#     # make sure we save this path to the
#     cfg_path = fix_config_paths(project, cfg_path)
#
#
#     cfg.project.data_path = os.path.join(cfg.project.path, cfg.project.data_path)
#     assert os.path.isdir(cfg.project.data_path), 'Data path not found: {}'.format(cfg.project.data_path)
#     cfg.project.model_path = os.path.join(cfg.project.path, cfg.project.model_path)
#     assert os.path.isdir(cfg.project.model_path)
#     if cfg.reload.weights is not None:
#         # if it's not a file, assume it's a relative path within the model directory
#         if not os.path.isfile(cfg.reload.weights):
#             cfg.reload.weights = os.path.join(cfg.project.model_path, cfg.reload.weights)
#
#         assert os.path.isfile(cfg.reload.weights)
#     return cfg
def get_project_path_from_cl(argv: list) -> str:
    for arg in argv:
        if 'project.config_file' in arg:
            key, path = arg.split('=')
            assert os.path.isfile(path)
            # path is the path to the project directory, not the config file
            path = os.path.dirname(path)
            return path
            
        elif 'project.path' in arg:
            key, path = arg.split('=')
            assert os.path.isdir(path)
            return path
    raise ValueError('project path or file not in args: {}'.format(argv))

def make_config(project_path: Union[str, os.PathLike], config_list: list, run_type: str, model: str) -> DictConfig:
    """DEPRECATED
    TODO: replace with configuration.make_config
    """
    config_path = os.path.join(os.path.dirname(deepethogram.__file__), 'conf')
    
    def config_string_to_path(config_path, string): 
        fullpath = os.path.join(config_path, *string.split('/'))  + '.yaml'
        assert os.path.isfile(fullpath)
        return fullpath
    
    cli = OmegaConf.from_cli()
    
    if project_path is not None:
        user_cfg = get_config_file_from_project_path(project_path)
    
        # order of operations: first, defaults specified in config_list
        # then, if preset is specified in user config or at the command line, load those preset values
        # then, append the user config
        # then, the command line args
        # so if we specify a preset and manually change, say, the feature extractor architecture, we can do that
        if 'preset' in user_cfg:
            config_list += ['preset/' + user_cfg.preset]
            
    if 'preset' in cli:
        config_list += ['preset/' + cli.preset]
        
    config_files = [config_string_to_path(config_path, i) for i in config_list]
    
    cfgs = [OmegaConf.load(i) for i in config_files]    
    
    if project_path is not None:
        # first defaults; then user cfg; then cli
        cfg = OmegaConf.merge(*cfgs, user_cfg, cli)
    else:
        cfg = OmegaConf.merge(*cfgs, cli)

    cfg.run = {'type': run_type, 'model': model}
    return cfg
    
def make_config_from_cli(argv, *args, **kwargs):
    """DEPRECATED
    TODO: replace with configuration.make_config
    """
    project_path = get_project_path_from_cl(argv)
    return make_config(project_path, *args, **kwargs)

def configure_run_directory(cfg: DictConfig) -> str:
    """Makes a run directory from a configuration

    Name: date-time_model-type_run-type_notes
    e.g. 20210311_011800_feature_extractor_train_testing_dropout
    
    Parameters
    ----------
    cfg : DictConfig
        see deepethogram/configuration.py

    Returns
    -------
    str
        path to run directory
    """
    datestring = datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.run.type == 'gui':
        path = cfg.project.path if cfg.project.path is not None else os.getcwd()
        directory = os.path.join(path, 'gui_logs', datestring)
    else:
        directory = f'{datestring}_{cfg.run.model}_{cfg.run.type}'
        directory = os.path.join(cfg.project.model_path, directory)
    if cfg.notes is not None:
        directory += f'_{cfg.notes}'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    os.chdir(directory)
    return directory
    
def configure_logging(cfg: DictConfig) -> None:
    """Sets up python logging to use a specific format, and also save to disk

    Parameters
    ----------
    cfg : DictConfig
        see deepethogram.configuration
    """
    # assume current directory is run directory
    assert cfg.log.level in ['debug', 'info', 'warning', 'error', 'critical']
    
    # https://docs.python.org/3/library/logging.html#logging-levels
    log_lookup = {'critical': 50,
                  'error': 40, 
                  'warning': 30,
                  'info': 20, 
                  'debug': 10}
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logging.basicConfig(level=log_lookup[cfg.log.level], 
                        format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                        handlers=[
                            logging.FileHandler(cfg.log.level + '.log'), 
                            logging.StreamHandler()
                        ])
    
def setup_run(cfg: DictConfig) -> DictConfig:
    """Makes a run directory and configures logging.

    See projects.configure_run_directory and projects.configure_logging

    Parameters
    ----------
    cfg : DictConfig
        see deepethogram.configuration

    Returns
    -------
    DictConfig
        see deepethogram.configuration
    """
    cfg = deepethogram.projects.convert_config_paths_to_absolute(cfg)
    directory = configure_run_directory(cfg)
    cfg.run.dir = directory
    configure_logging(cfg)
    
    utils.save_dict_to_yaml(OmegaConf.to_container(cfg), 'config.yaml')
    return cfg