import argparse
import copy
import csv
import glob
import os
import subprocess
from urllib.request import urlretrieve
import zipfile
from typing import Union
import warnings
import shutil
import sys

import numpy as np
import pandas as pd

from deepethogram import projects
from deepethogram.file_io import VideoReader
from deepethogram.dataloaders import parse_split, train_val_test_split
from deepethogram.file_io import read_labels
from deepethogram import utils

urls = {'videos': {'val': 'https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip',
                   'test': 'https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip'},
        'labels': {'val': 'http://crcv.ucf.edu/THUMOS14/Validation_set/TH14_Temporal_annotations_validation.zip',
                   'test': 'http://crcv.ucf.edu/THUMOS14/test_set/TH14_Temporal_annotations_test.zip'}}

behaviors = ['background', 'BaseballPitch', 'BasketballDunk', 'Billiards',
             'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
             'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump',
             'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']


def download_file(url, destination_folder):
    basename = os.path.basename(url)
    assert os.path.isdir(destination_folder)
    outfile = os.path.join(destination_folder, basename)

    urlretrieve(url, outfile)


# https://stackoverflow.com/questions/3451111/unzipping-files-in-python
def unzip_file(infile, outdir):
    assert os.path.isfile(infile)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with zipfile.ZipFile(infile, 'r') as zip_obj:
        try:
            zip_obj.extractall(outdir)
        except RuntimeError as e:
            print(e)
            print('You need to put in a password to extract the Thumos14 test set.')
            print('Please use this URL: https://www.crcv.ucf.edu/THUMOS14/download.html')
            raise


def validate_thumos14(testing_directory):
    assert os.path.isdir(testing_directory)
    raise NotImplementedError


def get_zipfiles(testing_directory):
    zipfiles = {filetype: {split: os.path.join(testing_directory, os.path.basename(urls[filetype][split]))
                           for split in ['val', 'test']
                           }
                for filetype in ['videos', 'labels']
                }
    return zipfiles


def download_thumos14(testing_directory, overwrite: bool = False):
    zipfiles = get_zipfiles(testing_directory)
    for filetype in ['videos', 'labels']:
        for split in ['val', 'test']:
            if not os.path.isfile(zipfiles[filetype][split]) or overwrite:
                download_file(urls[filetype][split], testing_directory)


def read_annotation_data(annotation_file):
    assert os.path.isfile(annotation_file)
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(' ')
        line = [i for i in line if i != '']
        data[line[0]] = [float(i) for i in line[1:]]
    return data


def read_annotation_file(annotation_file):
    assert os.path.isfile(annotation_file)
    basename = os.path.basename(annotation_file)[:-4]
    behavior = basename.split('_')[0]
    data = read_annotation_data(annotation_file)
    return behavior, data


def read_annotations(annotation_path):
    annotation_files = glob.glob(annotation_path + '/*.txt')
    annotation_files.sort()

    all_data = {}
    for annotation_file in annotation_files:
        behavior, data = read_annotation_file(annotation_file)
        all_data[behavior] = data
    return all_data


def convert_time_to_frames(t: float, fps: float = 30.0):
    return int(round(t * fps))


def convert_per_behavior_to_per_video(data: dict) -> dict:
    per_video = {}
    default = {behavior: [] for behavior in behaviors}
    for behavior, behavior_dict in data.items():
        if behavior == 'Ambiguous':
            continue
        for video_name, timestamp in behavior_dict.items():
            if video_name not in per_video.keys():
                per_video[video_name] = copy.deepcopy(default)
            per_video[video_name][behavior].append(timestamp)
    return per_video


def get_val_n_labels(split_dict: dict, labelfiles: dict):
    valfiles = split_dict['val']
    sums = []
    for key in valfiles:
        labelfile = labelfiles['val'][key]
        label = read_labels(labelfile)
        sums.append(label.sum(axis=0))
    sums = np.stack(sums)
    return np.sum(sums, axis=0)


def convert_label_to_deg(videofiles: dict, video_name: str, label_dict: dict):
    videofile = videofiles[video_name]
    assert os.path.isfile(videofile)
    with VideoReader(videofile) as reader:
        nframes = len(reader)
    label = np.zeros((nframes, 21), dtype=np.uint8)
    for i, (behavior, timestamps) in enumerate(label_dict.items()):
        for timestamp in timestamps:
            start, end = convert_time_to_frames(timestamp[0]), convert_time_to_frames(timestamp[1])
            label[start:end, i] = 1
    is_bg = np.logical_not(np.any(label[:, 1:], axis=1))
    label[:, 0] = is_bg
    return label


def setup_testing_directory(datadir: Union[str, os.PathLike], overwrite: bool = False):
    testing_path_file = 'testing_directory.txt'

    should_setup = True
    if os.path.isfile(testing_path_file):
        with open(testing_path_file, 'r') as f:
            testing_directory = f.read()
            if not os.path.isfile(testing_directory):
                warnings.warn('Saved testing directory {} does not exist, downloading Thumos14...'.format(
                    testing_directory
                ))
            else:
                should_setup = False
    if not should_setup:
        return testing_directory

    testing_directory = os.path.join(datadir, 'deepethogram_testing_thumos14')
    if not os.path.isdir(testing_directory):
        os.makedirs(testing_directory)

    with open('testing_directory.txt', 'w') as f:
        f.write(testing_directory)

    return testing_directory


def clean_subdir(subdir):
    files = utils.get_subfiles(subdir)

    for file in files:
        basename = os.path.basename(file)
        if basename == 'record.yaml':
            pass
        elif basename == 'stats.yaml':
            pass
        elif basename.startswith('video') and basename.endswith('.csv'):
            pass
        elif basename.startswith('video') and basename.endswith('.mp4'):
            pass
        else:
            os.remove(file)


def clean_deg_directory(testing_directory):
    model_subfiles = utils.get_subfiles(os.path.join(testing_directory, 'models'))

    for subfile in model_subfiles:
        if os.path.isfile(subfile):
            os.remove(subfile)
        elif os.path.isdir(subfile):
            shutil.rmtree(subfile)

    datadir = os.path.join(testing_directory, 'DATA')
    subfiles = utils.get_subfiles(datadir)
    for subfile in subfiles:
        if os.path.isdir(subfile):
            clean_subdir(subfile)
        else:
            if os.path.basename(subfile) == 'split.yaml':
                pass
            else:
                os.remove(subfile)


def read_lists_for_verification() -> dict:
    def read_csv(csvfile):
        rows = []
        with open(csvfile, newline='') as csvobj:
            reader = csv.reader(csvobj)
            for row in reader:
                rows.append(row[0])
        return rows

    filelists = {}
    for split in ['val', 'test']:
        filelist = read_csv(split + '.csv')
        filelists[split] = filelist
    return filelists


def check_if_videos_exist(video_paths, n_videos):
    all_correct = True
    filelists = read_lists_for_verification()
    for split in ['val', 'test']:
        directory = video_paths[split]
        subfiles = utils.get_subfiles(directory, return_type='file')
        assert len(subfiles) >= n_videos[split]

        basenames = [os.path.splitext(os.path.basename(i))[0] for i in subfiles]

        for file_to_check in filelists[split]:
            if file_to_check not in basenames:
                all_correct = False
                break
    return all_correct


def check_if_annotations_exist(annotation_paths):
    all_correct = True
    for split, directory in annotation_paths.items():
        subfiles = utils.get_subfiles(directory, 'file')
        txts = [i for i in subfiles if i.endswith('.txt')]
        if len(txts) < 21:
            all_correct = False
            break
    return all_correct

def get_testing_directory():
    directory_file = 'testing_directory.txt'
    if os.path.isfile(directory_file):
        with open(directory_file, 'r') as f:
            testing_directory = f.read()
            return testing_directory
    else:
        raise ValueError('please run setup_data.py before attempting to run unit tests')


def prepare_thumos14_for_testing(datadir):
    testing_directory = setup_testing_directory(datadir)

    annotation_paths = {'val': os.path.join(testing_directory, 'annotation'),
                        'test': os.path.join(testing_directory, 'TH14_Temporal_Annotations_Test',
                                             'annotations', 'annotation')}
    video_paths = {'val': os.path.join(testing_directory, 'validation'),
                   'test': os.path.join(testing_directory, 'TH14_test_set_mp4')}
    n_videos = {'val': 200, 'test': 212}

    # get rid of any files that shouldn't be there
    clean_deg_directory(testing_directory)

    # CHECK IF ALREADY PREPARED
    if check_if_videos_exist(video_paths, n_videos) and check_if_annotations_exist(annotation_paths):
        pass
    else:
        # DOWNLOAD IF ZIP FILE DOESNT EXIST
        download_thumos14(testing_directory)

        # UNZIP
        zipfiles = get_zipfiles(testing_directory)
        for filetype in ['labels', 'videos']:
            for split in ['val', 'test']:
                # you might need to enter a password here
                unzip_file(zipfiles[filetype][split], testing_directory)

        for split in ['val', 'test']:
            assert os.path.isdir(annotation_paths[split])
            assert os.path.isdir(video_paths[split])

    videos = {}
    for split in ['val', 'test']:
        videos[split] = glob.glob(video_paths[split] + '/*.mp4')
        videos[split].sort()

    # WRANGLE LABEL FILES
    labelfiles = {}
    for split in ['val', 'test']:
        videofiles = {os.path.basename(i)[:-4]: i for i in videos[split]}
        label_dir = os.path.join(testing_directory, split + '_deg_labels')
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)

        annotation_path = annotation_paths[split]
        annotations = read_annotations(annotation_path)
        per_video = convert_per_behavior_to_per_video(annotations)

        labelfiles[split] = {}

        for video_name, label_dict in per_video.items():
            label = convert_label_to_deg(videofiles, video_name, label_dict)
            df = pd.DataFrame(label, columns=behaviors)
            outfile = os.path.join(label_dir, video_name + '_labels.csv')
            df.to_csv(outfile)
            labelfiles[split][video_name] = outfile

    # MAKE SPLIT
    split = [0.8, 0.2, 0.0]
    # ensure we have at least one instance of every behavior in the validation set
    np.random.seed(42)
    while True:
        split_dict = train_val_test_split(labelfiles['val'], split)
        n_labels = get_val_n_labels(split_dict, labelfiles)
        if not np.any(n_labels == 0):
            break
        else:
            print('invalid')
    split_dict['metadata']['split'][2] = None  # put this as none to remind that split came with it
    split_dict['test'] = list(labelfiles['test'].keys())

    splitfile = os.path.join(testing_directory, 'DATA', 'split.yaml')
    utils.save_dict_to_yaml(split_dict, splitfile)

    # ADD TO DEEPETHOGRAM PROJECT
    project_config = projects.initialize_project(testing_directory, 'thumos14', behaviors, make_subdirectory=False)

    for split in ['val', 'test']:
        videofiles = {os.path.basename(i)[:-4]: i for i in videos[split]}
        for video_name, labelfile in labelfiles[split].items():
            videofile = videofiles[video_name]
            labelfile = labelfiles[split][video_name]
            new_path = projects.add_video_to_project(project_config, videofile, mode='symlink')
            projects.add_label_to_project(labelfile, new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Setting up Thumos14 for deepethogram testing')
    parser.add_argument('-d', '--datadir', default=os.getcwd(),
                        help='location to store data. Needs >200GB')

    args = parser.parse_args()

    prepare_thumos14_for_testing(args.datadir)
