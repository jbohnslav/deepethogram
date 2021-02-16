import logging
import os
import sys
from typing import Union, Type

import h5py
# import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils import data
from tqdm import tqdm

import deepethogram.projects
from deepethogram import utils, projects
from deepethogram.data.datasets import FeatureVectorDataset
from deepethogram.sequence.train import build_model_from_cfg

log = logging.getLogger(__name__)


def infer(model: Type[nn.Module], device: Union[str, torch.device],
          activation_function: Union[str, Type[nn.Module]],
          dataloader: Union[str, os.PathLike], latent_name, sequence_length: int = 180,
          is_two_stream: bool = True):
    assert (latent_name is not None)

    gen = FeatureVectorDataset(dataloader, labelfile=None, h5_key=latent_name,
                                sequence_length=sequence_length,
                                nonoverlapping=True, store_in_ram=False, is_two_stream=is_two_stream)
    n_datapoints = gen.shape[1]
    gen = data.DataLoader(gen, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    gen = iter(gen)
    log.debug('Making sequence iterator with parameters: ')
    log.debug('file: {}'.format(dataloader))
    log.debug('seq length: {}'.format(sequence_length))

    if type(activation_function) == str:
        if activation_function == 'softmax':
            activation_function = torch.nn.Softmax(dim=1)
        elif activation_function == 'sigmoid':
            activation_function = torch.nn.Sigmoid()
        else:
            raise ValueError('unknown activation function: {}'.format(activation_function))

    if type(device) == str:
        device = torch.device(device)

    if next(model.parameters()).device != device:
        model = model.to(device)

    if next(model.parameters()).requires_grad:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if model.training:
        model = model.eval()

    all_logits = []
    all_probabilities = []
    has_printed = False
    for i in range(len(gen)):

        with torch.no_grad():
            batch = next(gen)
            batch = batch.to(device)
            logits = model(batch)

            probabilities = activation_function(logits).detach().cpu().numpy().squeeze().T
            logits = logits.detach().cpu().numpy().squeeze().T

        if not has_printed:
            log.debug('logits shape: {}'.format(logits.shape))
            has_printed = True

        end = min(i * sequence_length + sequence_length, n_datapoints)
        indices = range(i * sequence_length, end)
        # get rid of padding in final batch
        if len(indices) < logits.shape[0]:
            logits = logits[:len(indices), :]
            probabilities = probabilities[:len(indices), :]

        all_logits.append(logits)
        all_probabilities.append(probabilities)

    all_logits = np.concatenate(all_logits)
    all_probabilities = np.concatenate(all_probabilities)

    return all_logits, all_probabilities


def extract(model, outputfiles: list, thresholds: np.ndarray, final_activation: str,
            latent_name: str, output_name: str = 'tgmj', sequence_length: int = 180,
            is_two_stream: bool = True, device: str = 'cuda:1', ignore_error=True, overwrite=False,
            class_names: list = ['background']):
    torch.backends.cudnn.benchmark = True

    assert isinstance(model, torch.nn.Module)

    device = torch.device(device)
    torch.cuda.set_device(device)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    has_printed = False

    if final_activation == 'softmax':
        activation_function = torch.nn.Softmax(dim=1)
    elif final_activation == 'sigmoid':
        activation_function = torch.nn.Sigmoid()

    class_names = [n.encode("ascii", "ignore") for n in class_names]

    for i in tqdm(range(len(outputfiles))):
        outputfile = outputfiles[i]
        log.info('running inference on {}. latent name: {} output name: {}...'.format(outputfile, latent_name,
                                                                                      output_name))
        gen = FeatureVectorDataset(outputfile, labelfile=None, h5_key=latent_name,
                                    sequence_length=sequence_length,
                                    nonoverlapping=True, store_in_ram=False, is_two_stream=is_two_stream)
        n_datapoints = gen.shape[1]
        gen = data.DataLoader(gen, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        gen = iter(gen)

        log.debug('Making sequence iterator with parameters: ')
        log.debug('file: {}'.format(outputfile))
        log.debug('seq length: {}'.format(sequence_length))

        with h5py.File(outputfile, 'r+') as f:

            if output_name in list(f.keys()):
                if overwrite:
                    del (f[output_name])
                else:
                    log.info('Latent {} already found in file {}, skipping...'.format(output_name, outputfile))
                    continue
            group = f.create_group(output_name)

            for i in tqdm(range(len(gen)), leave=False):
                # for i in tqdm(range(1000)):
                with torch.no_grad():
                    try:
                        batch = next(gen)
                        # star means that if batch is a tuple of images, flows, it will pass in as sequential
                        # positional arguments
                        features = batch['features'].to(device)
                        logits = model(features)

                        if not has_printed:
                            log.debug('logits shape: {}'.format(logits.shape))
                            has_printed = True

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        # (e)
                        log.exception('Error on video {}, frame: {}'.format(outputfile, i))
                        if ignore_error:
                            log.info('continuing anyway...')
                            break
                        else:
                            raise

                    probabilities = activation_function(logits).detach().cpu().numpy().squeeze().T
                    logits = logits.detach().cpu().numpy().squeeze().T

                if i == 0:
                    group.create_dataset('thresholds', data=thresholds, dtype=np.float32)
                    group.create_dataset('logits', (n_datapoints, logits.shape[1]), dtype=np.float32)
                    group.create_dataset('P', (n_datapoints, logits.shape[1]), dtype=np.float32)
                    dt = h5py.string_dtype()
                    group.create_dataset('class_names', data=class_names, dtype=dt)
                end = min(i * sequence_length + sequence_length, n_datapoints)
                indices = range(i * sequence_length, end)
                # if we've padded the final batch
                # print('len indices: {} end: {}'.format(len(indices), end))
                if len(indices) < probabilities.shape[0]:
                    probabilities = probabilities[:len(indices), :]
                    logits = logits[:len(indices), :]
                # print('indices: {}'.format(indices))
                # print('P: {}'.format(probabilities.shape))

                group['P'][indices, :] = probabilities
                group['logits'][indices, :] = logits

                del (batch, logits, probabilities)


def main(cfg: DictConfig):
    log.info('args: {}'.format(' '.join(sys.argv)))
    # turn "models" in your project configuration to "full/path/to/models"
    log.info('configuration used: ')
    log.info(cfg.pretty())

    weights = projects.get_weightfile_from_cfg(cfg, model_type='sequence')
    assert weights is not None, 'Must either specify a weightfile or use reload.latest=True'

    if cfg.sequence.latent_name is None:
        # find the latent name used in the weight file you loaded
        rundir = os.path.dirname(weights)
        loaded_cfg = utils.load_yaml(os.path.join(rundir, 'config.yaml'))
        latent_name = loaded_cfg['sequence']['latent_name']
        # if this latent name is also None, use the arch of the feature extractor
        # this should never happen
        if latent_name is None:
            latent_name = loaded_cfg['feature_extractor']['arch']
    else:
        latent_name = cfg.sequence.latent_name
    log.info('latent name used for running sequence inference: {}'.format(latent_name))

    # the output name will be a group in the output hdf5 dataset containing probabilities, etc
    if cfg.sequence.output_name is None:
        output_name = cfg.sequence.arch
    else:
        output_name = cfg.sequence.output_name
    directory_list = cfg.inference.directory_list
    if directory_list is None or len(directory_list) == 0:
        raise ValueError('must pass list of directories from commmand line. '
                         'Ex: directory_list=[path_to_dir1,path_to_dir2] or directory_list=all')
    elif type(directory_list) == str and directory_list == 'all':
        basedir = cfg.project.data_path
        directory_list = utils.get_subfiles(basedir, 'directory')

    outputfiles = []
    for directory in directory_list:
        assert os.path.isdir(directory), 'Not a directory: {}'.format(directory)
        record = projects.get_record_from_subdir(directory)
        assert record['output'] is not None
        outputfiles.append(record['output'])


    model = build_model_from_cfg(cfg, 1024, len(cfg.project.class_names))
    log.info('model: {}'.format(model))

    model = utils.load_weights(model, weights)
    metrics_file = os.path.join(os.path.dirname(weights), 'classification_metrics.h5')
    with h5py.File(metrics_file, 'r') as f:
        try:
            thresholds = f['val']['metrics_by_threshold']['optimum'][:]
        except KeyError:
            thresholds = f['threshold_curves']['val']['optimum'][:]
        if thresholds.ndim == 2:
            thresholds = thresholds[-1, :]
        
        log.info('thresholds: {}'.format(thresholds))
    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    class_names = cfg.project.class_names
    class_names = np.array(class_names)
    extract(model, outputfiles, thresholds, cfg.feature_extractor.final_activation, latent_name, output_name,
            cfg.sequence.sequence_length, True, device, cfg.inference.ignore_error,
            cfg.inference.overwrite, class_names=class_names)


if __name__ == '__main__':
    config_list = ['config','augs','model/feature_extractor', 'model/sequence', 'inference']
    run_type = 'inference'
    model = 'sequence'
    cfg = projects.make_config_from_cli(sys.argv, config_list, run_type, model)
    cfg = projects.setup_run(cfg)
    
    main(cfg)