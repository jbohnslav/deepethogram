import glob
import logging
import multiprocessing as mp
import os
import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from vidio import VideoReader

from deepethogram import utils
# from deepethogram.dataloaders import log
from deepethogram.file_io import read_labels

log = logging.getLogger(__name__)


def purge_unlabeled_videos(video_list: list, label_list: list) -> Tuple[list, list]:
    """Get rid of any videos that contain unlabeled frames.
    Goes through all label files, loads them. If they contain any -1 values, remove both the video and the label
    from their respective lists
    """
    valid_videos = []
    valid_labels = []

    warning_string = '''Labelfile {} associated with video {} has unlabeled frames! 
        Please finish labeling or click the Finalize Labels button on the GUI.'''

    for i in range(len(label_list)):
        label = read_labels(label_list[i])
        has_unlabeled_frames = np.any(label == -1)
        if not has_unlabeled_frames:
            valid_videos.append(video_list[i])
            valid_labels.append(label_list[i])
        else:
            log.warning(warning_string.format(label_list[i], video_list[i]))
    return video_list, label_list


def purge_unlabeled_elements_from_records(records: dict) -> dict:
    valid_records = {}

    warning_message = '''labelfile {} has unlabeled frames! 
        Please finish labeling or click the Finalize Labels button on the GUI.
        Associated files: {}'''

    for animal, record in records.items():
        labelfile = record['label']

        if labelfile is None:
            log.warning('Record {} does not have a labelfile! Please start and finish labeling. '.format(animal) + \
                'Associated files: {}'.format(record))
            continue
        label = read_labels(labelfile)
        has_unlabeled_frames = np.any(label == -1)
        if has_unlabeled_frames:
            log.warning(warning_message.format(animal, record))
        else:
            valid_records[animal] = record
    return valid_records


def make_loss_weight(class_counts: np.ndarray,
                     num_pos: np.ndarray,
                     num_neg: np.ndarray,
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

    # if there are zero positive examples, we don't want the pos weight to be a large number
    # we want it to be infinity, then we will manually set it to zero
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        pos_weight = num_neg / num_pos
    # if there are zero negative examples, loss should be 1
    pos_weight[pos_weight == 0] = 1
    pos_weight_transformed = (pos_weight**weight_exp).astype(np.float32)
    # if all examples positive: will be 1
    # if zero examples positive: will be 0
    pos_weight_transformed = np.nan_to_num(pos_weight_transformed, nan=0.0, posinf=0.0, neginf=0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        softmax_weight = 1 / class_counts
    softmax_weight = np.nan_to_num(softmax_weight, nan=0.0, posinf=0.0, neginf=0)
    softmax_weight = softmax_weight / np.sum(softmax_weight)
    softmax_weight_transformed = (softmax_weight**weight_exp).astype(np.float32)

    np.set_printoptions(suppress=True)
    log.info('Class counts: {}'.format(class_counts))
    log.info('Pos weight: {}'.format(pos_weight))
    log.info('Pos weight, weighted: {}'.format(pos_weight_transformed))
    log.info('Softmax weight: {}'.format(softmax_weight))
    log.info('Softmax weight transformed: {}'.format(softmax_weight_transformed))

    return pos_weight_transformed, softmax_weight_transformed


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

    video_data = {
        'name': allnames,
        'action': allactions,
        'action_int': action_indices,
        'width': widths,
        'height': heights,
        'framecount': framenums
    }
    df = pd.DataFrame(data=video_data)
    fname = '_metadata.csv'
    if is_flow:
        fname = '_flow' + fname
    df.to_csv(os.path.join(os.path.dirname(splitdir), os.path.basename(splitdir) + fname))
    return df


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
        new_splits = np.random.choice(splits, size=(N, ), p=split_p).tolist()
        for i, k in enumerate(new_entries):
            split_dictionary[new_splits[i]].append(k)
            log.info('file {} assigned to split {}'.format(k, new_splits[i]))
    return split_dictionary


def get_split_from_records(records: dict,
                           datadir: Union[str, bytes, os.PathLike],
                           splitfile: Union[str, bytes, os.PathLike] = None,
                           supervised: bool = True,
                           reload_split: bool = True,
                           valid_splits_only: bool = True,
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
    else:
        assert os.path.isfile(splitfile), 'split file does not exist! {}'.format(splitfile)

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
