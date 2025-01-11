from functools import partial
import logging
import os
import shutil
import sys
from typing import Union

import cv2
import numpy as np
from omegaconf import OmegaConf, ListConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vidio import VideoWriter

from deepethogram.configuration import make_feature_extractor_inference_cfg
from deepethogram import projects, utils
from deepethogram.data.augs import get_cpu_transforms, get_gpu_transforms
from deepethogram.data.datasets import VideoIterable
from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
from deepethogram.flow_generator.utils import flow_to_rgb_polar, flow_to_rgb
log = logging.getLogger(__name__)


def extract_movie(in_video,
                  out_video,
                  model,
                  device,
                  cpu_transform,
                  gpu_transform,
                  mean_by_channels,
                  num_workers=1,
                  num_rgb=11,
                  maxval: int = 5,
                  polar: bool = True,
                  movie_format: str = 'ffmpeg',
                  batch_size=1,
                  save_rgb_side_by_side=False) -> None:

    if polar:
        convert = partial(flow_to_rgb_polar, maxval=maxval)
    else:
        convert = partial(flow_to_rgb, maxval=maxval)
    model.eval()

    # if type(device) != torch.device:
    #     device = torch.device(device)

    dataset = VideoIterable(in_video,
                            transform=cpu_transform,
                            num_workers=num_workers,
                            sequence_length=num_rgb,
                            mean_by_channels=mean_by_channels)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    # log.debug('model training mode: {}'.format(model.training))
    with VideoWriter(out_video, movie_format) as vid:
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            if isinstance(batch, dict):
                images = batch['images']
            elif isinstance(batch, torch.Tensor):
                images = batch
            else:
                raise ValueError('unknown input type: {}'.format(type(batch)))

            if images.device != device:
                images = images.to(device)
            # images = batch['images']
            with torch.no_grad():
                images = gpu_transform['val'](images)
                flows = model(images)
                # TODO: only run optic flow calc on each frame once!
                # since we are running batches of 11 images, the batches look like
                # b0=[0,1,2,3,4,5,6,7,8,9,10]
                # b1=[1,2,3,4,5,6,7,8,9,10,11]
                # we will just hack it to take the first image. really, we should only run each batch once, then save all 11
                # images in a row
                if type(flows) == list or type(flows) == tuple:
                    flows = flows[0]
                # only support batch size of 1
                flow = flows[0, 8:10, ...]
                # squeeze batch dimension
                flow = flow.squeeze()
                flow = flow.detach().cpu().numpy().transpose(1, 2, 0)
                flow_map = convert(flow)
                if save_rgb_side_by_side:
                    images = gpu_transform['denormalize'](images)
                    im = images[:, :, 5, ...].squeeze().detach().cpu().numpy()
                    im = im.transpose(1, 2, 0) * 255
                    im = im.clip(min=0, max=255).astype(np.uint8)
                    flow_map = cv2.resize(flow_map, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
                    out = np.hstack((flow_map, im))
                else:
                    out = flow_map
                vid.write(out)


def get_run_files_from_weights(weightfile: Union[str, os.PathLike], metrics_prefix='classification') -> dict:
    """from model weights, gets the configuration for that model and its metrics file

    Parameters
    ----------
    weightfile : Union[str, os.PathLike]
        path to model weights, either .pt or .ckpt

    Returns
    -------
    dict
        config_file: path to config file
        metrics_file: path to metrics file
    """
    loaded_config_file = os.path.join(os.path.dirname(weightfile), 'config.yaml')
    if not os.path.isfile(loaded_config_file):
        # weight file should be at most one-subdirectory-down from rundir
        loaded_config_file = os.path.join(os.path.dirname(os.path.dirname(weightfile)), 'config.yaml')
        assert os.path.isfile(loaded_config_file), 'no associated config file for weights! {}'.format(weightfile)

    metrics_file = os.path.join(os.path.dirname(weightfile), f'{metrics_prefix}_metrics.h5')
    if not os.path.isfile(metrics_file):
        metrics_file = os.path.join(os.path.dirname(os.path.dirname(weightfile)), f'{metrics_prefix}_metrics.h5')
        assert os.path.isfile(metrics_file), 'no associated metrics file for weights! {}'.format(weightfile)

    return dict(config_file=loaded_config_file, metrics_file=metrics_file)


def flow_generator_inference(cfg):
    # make configuration
    cfg = projects.setup_run(cfg)
    # turn "models" in your project configuration to "full/path/to/models"
    log.info('args: {}'.format(' '.join(sys.argv)))
    log.info('configuration used in inference: ')
    log.info(OmegaConf.to_yaml(cfg))
    if 'sequence' not in cfg.keys() or 'latent_name' not in cfg.sequence.keys() or cfg.sequence.latent_name is None:
        latent_name = cfg.feature_extractor.arch
    else:
        latent_name = cfg.sequence.latent_name
    log.info('Latent name used in HDF5 file: {}'.format(latent_name))
    directory_list = cfg.inference.directory_list

    # figure out which videos to run inference on
    if directory_list is None or len(directory_list) == 0:
        raise ValueError('must pass list of directories from commmand line. '
                         'Ex: directory_list=[path_to_dir1,path_to_dir2]')
    elif type(directory_list) == str and directory_list == 'all':
        basedir = cfg.project.data_path
        directory_list = utils.get_subfiles(basedir, 'directory')
    elif isinstance(directory_list, str):
        directory_list = [directory_list]
    elif isinstance(directory_list, list):
        pass
    elif isinstance(directory_list, ListConfig):
        directory_list = OmegaConf.to_container(directory_list)
    else:
        raise ValueError('unknown value for directory list: {}'.format(directory_list))

    # video files are found in your input list of directories using the records.yaml file that should be present
    # in each directory
    records = []
    for directory in directory_list:
        assert os.path.isdir(directory), 'Not a directory: {}'.format(directory)
        record = projects.get_record_from_subdir(directory)
        assert record['rgb'] is not None
        records.append(record)
    rgb = []
    for record in records:
        rgb.append(record['rgb'])

    assert cfg.feature_extractor.n_flows + 1 == cfg.flow_generator.n_rgb, 'Flow generator inputs must be one greater ' \
                                                                          'than feature extractor num flows '
    # set up gpu augmentation
    input_images = cfg.feature_extractor.n_flows + 1
    mode = '3d' if '3d' in cfg.feature_extractor.arch.lower() else '2d'
    # get the validation transforms. should have resizing, etc
    cpu_transform = get_cpu_transforms(cfg.augs)['val']
    gpu_transform = get_gpu_transforms(cfg.augs, mode)
    log.info('gpu_transform: {}'.format(gpu_transform))

    flow_generator_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    assert os.path.isfile(flow_generator_weights)
    run_files = get_run_files_from_weights(flow_generator_weights, 'opticalflow')
    if cfg.inference.use_loaded_model_cfg:
        loaded_config_file = run_files['config_file']
        loaded_cfg = OmegaConf.load(loaded_config_file)
        loaded_model_cfg = loaded_cfg.flow_generator
        current_model_cfg = cfg.flow_generator
        model_cfg = OmegaConf.merge(current_model_cfg, loaded_model_cfg)
        cfg.flow_generator = model_cfg
        # we don't want to use the weights that the trained model was initialized with, but the weights after training
        # therefore, overwrite the loaded configuration with the current weights
        cfg.flow_generator.weights = flow_generator_weights
        # num_classes = len(loaded_cfg.project.class_names)
    log.info('model loaded')
    # log.warning('Overwriting current project classes with loaded classes! REVERT')
    model = build_flow_generator(cfg)
    model = utils.load_weights(model, flow_generator_weights, device='cpu')
    # _, _, _, _, model = model_components
    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    model = model.to(device)

    movie_format = 'ffmpeg'
    maxval = 5
    polar = True
    save_rgb_side_by_side = True
    for movie in tqdm(rgb):
        out_video = os.path.splitext(movie)[0] + '_flows'
        if movie_format == 'directory':
            pass
        elif movie_format == 'hdf5':
            out_video += '.h5'
        elif movie_format == 'ffmpeg':
            out_video += '.mp4'
        else:
            out_video += '.avi'
        if os.path.isdir(out_video):
            shutil.rmtree(out_video)
        elif os.path.isfile(out_video):
            os.remove(out_video)

        extract_movie(movie,
                      out_video,
                      model,
                      device,
                      cpu_transform,
                      gpu_transform,
                      mean_by_channels=cfg.augs.normalization.mean,
                      num_workers=1,
                      num_rgb=input_images,
                      maxval=maxval,
                      polar=polar,
                      movie_format=movie_format,
                      save_rgb_side_by_side=save_rgb_side_by_side)


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_feature_extractor_inference_cfg(project_path, use_command_line=True)
    flow_generator_inference(cfg)
