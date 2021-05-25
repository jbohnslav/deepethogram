import logging
import os
import sys
from typing import Union

# import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import deepethogram.file_io
from deepethogram import configuration, utils, projects

log = logging.getLogger(__name__)


class StatsRecorder:
    """ Class for computing mean and std deviation incrementally. Originally found on github here:
    https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

    I only added PyTorch compatibility.
    """
    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None, n_observations: int = None):
        self.nobservations = 0
        if mean is not None:
            assert std is not None
            assert n_observations is not None
            assert mean.shape == std.shape
            self.mean = mean
            self.std = std
            self.n_observations = n_observations

    def first_batch(self, data: Union[np.ndarray, torch.Tensor]):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        dtype = type(data)
        assert (dtype == np.ndarray or dtype == torch.Tensor)

        if dtype == np.ndarray:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        elif dtype == torch.Tensor:
            data = data.detach()  # don't accumulate gradients
            if data.ndim == 1:
                # assume it's one observation, not multiple observations with 1 dimension
                data = data.unsqueeze(0)
            # I'm mad that pytorch used dim instead of axis
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        self.nobservations = data.shape[0]
        self.ndimensions = data.shape[1]

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.first_batch(data)
            return  # only continues if we've initialized, got rid of else
        if data.shape[1] != self.ndimensions:
            raise ValueError("Data dims don't match prev observations.")
        dtype = type(data)
        assert (dtype == np.ndarray or dtype == torch.Tensor)
        if dtype == np.ndarray:
            data = np.atleast_2d(data)
            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)
        elif dtype == torch.Tensor:
            data = data.detach()  # don't accumulate gradients
            if data.ndim == 1:
                # assume it's one observation, not multiple observations with 1 dimension
                data = data.unsqueeze(0)
            newmean = data.mean(dim=0)
            newstd = data.std(dim=0)

        m = self.nobservations * 1.0
        n = data.shape[0]

        tmp = self.mean

        self.mean = m / (m + n) * tmp + n / (m + n) * newmean
        self.std = m / (m + n) * self.std ** 2 + n / (m + n) * newstd ** 2 + \
                   m * n / (m + n) ** 2 * (tmp - newmean) ** 2

        self.std = self.std**0.5

        self.nobservations += n

    def __str__(self):
        return 'mean: {} std: {} n: {}'.format(self.mean, self.std, self.nobservations)


def get_video_statistics(videofile, stride):
    image_stats = StatsRecorder()
    with deepethogram.file_io.VideoReader(videofile) as reader:
        log.debug('N frames: {}'.format(len(reader)))
        for i in tqdm(range(0, len(reader), stride)):
            image = reader[i]
            image = image.astype(float) / 255
            image = image.transpose(2, 1, 0)
            # image = image[np.newaxis,...]
            # N, C, H, W = image.shape
            image = image.reshape(3, -1).transpose(1, 0)
            # image = image.reshape(N, C, -1).squeeze().transpose(1, 0)
            # if i == 0:
            #     print(image.shape)
            image_stats.update(image)

    log.info('final stats: {}'.format(image_stats))

    imdata = {'mean': image_stats.mean, 'std': image_stats.std, 'N': image_stats.nobservations}
    for k, v in imdata.items():
        if type(v) == torch.Tensor:
            v = v.detach().cpu().numpy()
        if type(v) == np.ndarray:
            v = v.tolist()
        imdata[k] = v

    return imdata


def zscore_video(videofile: Union[str, os.PathLike], project_config: dict, stride: int = 10):
    """calculates channel-wise mean and standard deviation for input video.

    Calculates mean and std deviation independently for each input video channel. Grayscale videos are converted to RGB.
    Saves statistics to the augs/normalization dictionary in project_config. Only takes every STRIDE frames for speed.
    Calculates mean and std deviation incrementally to not load thousands of frames into memory at once:
        https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    Args:
        videofile: path to video file. Must be one of inputs to file_io/VideoReader: avi, mp4, jpg directory, or hdf5
        project_config: dictionary for your deepethogram project. Contains augs/normalization field
        stride: only every STRIDE frames will be computed. Use stride=1 for the full video

    Returns:

    """
    assert os.path.exists(videofile)
    assert projects.is_deg_file(videofile)

    # config['arch'] = 'flow-generator'
    # config['normalization'] = None
    # transforms = get_transforms_from_config(config)
    # xform = transforms['train']
    log.info('zscoring file: {}'.format(videofile))
    imdata = get_video_statistics(videofile, stride)

    fname = os.path.join(os.path.dirname(videofile), 'stats.yaml')
    dictionary = {}
    if os.path.isfile(fname):
        dictionary = utils.load_yaml(fname)

    dictionary['normalization'] = imdata
    utils.save_dict_to_yaml(dictionary, fname)
    update_project_with_normalization(imdata, project_config)


def update_project_with_normalization(norm_dict: dict, project_config: dict):
    """ Adds statistics from this video to the overall mean / std deviation for the project """
    # project_dict = utils.load_yaml(os.path.join(project_dir, 'project_config.yaml'))

    if 'normalization' not in project_config['augs'].keys():
        raise ValueError('Must have project_config/augs/normalization field: {}'.format(project_config))
    old_rgb = project_config['augs']['normalization']
    if old_rgb is not None and old_rgb['N'] is not None and old_rgb['mean'] is not None:
        old_mean_total = old_rgb['N'] * np.array(old_rgb['mean'])
        old_std_total = old_rgb['N'] * np.array(old_rgb['std'])
        old_N = old_rgb['N']
    else:
        old_mean_total = 0
        old_std_total = 0
        old_N = 0
    new_n = old_N + norm_dict['N']
    new_mean = (old_mean_total + norm_dict['N'] * np.array(norm_dict['mean'])) / new_n
    new_std = (old_std_total + norm_dict['N'] * np.array(norm_dict['std'])) / new_n
    project_config['augs']['normalization'] = {'N': new_n, 'mean': new_mean.tolist(), 'std': new_std.tolist()}
    utils.save_dict_to_yaml(project_config, os.path.join(project_config['project']['path'], 'project_config.yaml'))


# @hydra.main(config_path='../conf/zscore.yaml')
def main(cfg: DictConfig):
    assert os.path.isfile(cfg.videofile)
    project_config = utils.load_yaml(cfg.project.config_file)
    zscore_video(cfg.videofile, project_config, cfg.stride)


if __name__ == '__main__':
    config_list = ['config', 'zscore']
    run_type = 'zscore'
    model = None
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = configuration.make_config(project_path, config_list, run_type, model, use_command_line=True)
    cfg = projects.setup_run(cfg)
    main(cfg)
