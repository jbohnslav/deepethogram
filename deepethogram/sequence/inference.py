import logging
import os
import sys
from typing import Union, Type

import h5py
# import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data
from tqdm import tqdm

from deepethogram import utils, projects
from deepethogram.configuration import make_sequence_inference_cfg
from deepethogram.data.datasets import FeatureVectorDataset, KeypointDataset
from deepethogram.sequence.train import build_model_from_cfg

log = logging.getLogger(__name__)


def infer(model: Type[nn.Module],
          device: Union[str, torch.device],
          activation_function: Union[str, Type[nn.Module]],
          data_file: Union[str, os.PathLike],
          latent_name: str,
          videofile: Union[str, os.PathLike],
          sequence_length: int = 180,
          is_two_stream: bool = True,
          is_keypoint: bool = False,
          expansion_method: str = 'sturman',
          stack_in_time: bool = False):
    """Runs inference of the sequence model

    Parameters
    ----------
    model : Type[nn.Module]
        sequence model
    device : Union[str, torch.device]
        Device on which to run inference. e.g. ('cpu', 'cuda:0')
    activation_function : Union[str, Type[nn.Module]]
        Either sigmoid or softmax
    data_file : Union[str, os.PathLike]
        Path to a feature vector HDF5 file, or keypoint file (currently only DeepLabCut .csvs)
    latent_name : str
        Group name in HDF5 file
    videofile: [str, os.PathLike], optional
        Path to video file. Used in normalizing keypoint features
    sequence_length : int, optional
        Number of feature vectors in each batch element, by default 180
    is_two_stream : bool, optional
        If True, load both image and flow features as input, by default True
    expansion_method: str, optional
        Method for expanding keypoints into features, by default sturman
    stack_in_time: bool, optional
        If True, stacks sequences from T x K features -> T*K features, by default False
        
    Returns
    -------
    logits: np.ndarray
        outputs before activation
    probabilities: np.ndarray
        outputs after activation

    Raises
    ------
    ValueError
        Check that activation is either sigmoid or softmax
    """
    if not is_keypoint:
        assert latent_name is not None

        gen = FeatureVectorDataset(data_file,
                                   labelfile=None,
                                   h5_key=latent_name,
                                   sequence_length=sequence_length,
                                   nonoverlapping=True,
                                   store_in_ram=False,
                                   is_two_stream=is_two_stream)
    else:
        gen = KeypointDataset(data_file,
                              labelfile=None,
                              videofile=videofile,
                              expansion_method=expansion_method,
                              sequence_length=sequence_length,
                              stack_in_time=stack_in_time,
                              nonoverlapping=not stack_in_time)

    n_datapoints = gen.shape[1]
    gen = data.DataLoader(gen, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    gen = iter(gen)
    log.debug('Making sequence iterator with parameters: ')
    log.debug('file: {}'.format(data_file))
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
            features = batch['features'].to(device)
            logits = model(features)

            probabilities = activation_function(logits).detach().cpu().numpy().squeeze().T
            logits = logits.detach().cpu().numpy().squeeze().T

        if not has_printed:
            log.debug('logits shape: {}'.format(logits.shape))
            has_printed = True

        if not stack_in_time:
            # stacking in time means that we only do one element at a time
            # therefore, we don't need this indexing logic
            end = min(i * sequence_length + sequence_length, n_datapoints)
            indices = range(i * sequence_length, end)
            # get rid of padding in final batch
            if len(indices) < logits.shape[0]:
                logits = logits[:len(indices), :]
                probabilities = probabilities[:len(indices), :]

        all_logits.append(logits)
        all_probabilities.append(probabilities)
    if stack_in_time:
        all_logits = np.stack(all_logits)
        all_probabilities = np.stack(all_probabilities)
    else:
        all_logits = np.concatenate(all_logits)
        all_probabilities = np.concatenate(all_probabilities)

    return all_logits, all_probabilities


def extract(model,
            outputfiles: list,
            thresholds: np.ndarray,
            final_activation: str,
            latent_name: str,
            output_name: str = 'tgmj',
            sequence_length: int = 180,
            is_two_stream: bool = True,
            device: str = 'cuda:0',
            ignore_error=True,
            overwrite=False,
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
    else:
        raise NotImplementedError

    class_names = [n.encode("ascii", "ignore") for n in class_names]

    for i in tqdm(range(len(outputfiles))):
        outputfile = outputfiles[i]
        log.info('running inference on {}. latent name: {} output name: {}...'.format(
            outputfile, latent_name, output_name))

        logits, probabilities = infer(model, device, activation_function, outputfile, latent_name, None,
                                      sequence_length, is_two_stream)

        # gen = FeatureVectorDataset(outputfile, labelfile=None, h5_key=latent_name,
        #                             sequence_length=sequence_length,
        #                             nonoverlapping=True, store_in_ram=False, is_two_stream=is_two_stream)
        # n_datapoints = gen.shape[1]
        # gen = data.DataLoader(gen, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        # gen = iter(gen)

        # log.debug('Making sequence iterator with parameters: ')
        # log.debug('file: {}'.format(outputfile))
        # log.debug('seq length: {}'.format(sequence_length))

        with h5py.File(outputfile, 'r+') as f:

            if output_name in list(f.keys()):
                if overwrite:
                    del (f[output_name])
                else:
                    log.info('Latent {} already found in file {}, skipping...'.format(output_name, outputfile))
                    continue
            group = f.create_group(output_name)
            group.create_dataset('thresholds', data=thresholds, dtype=np.float32)
            group.create_dataset('logits', data=logits, dtype=np.float32)
            group.create_dataset('P', data=probabilities, dtype=np.float32)
            dt = h5py.string_dtype()
            group.create_dataset('class_names', data=class_names, dtype=dt)


def sequence_inference(cfg: DictConfig):
    cfg = projects.setup_run(cfg)
    log.info('args: {}'.format(' '.join(sys.argv)))
    # turn "models" in your project configuration to "full/path/to/models"
    log.info('configuration used: ')
    log.info(OmegaConf.to_yaml(cfg))

    weights = projects.get_weightfile_from_cfg(cfg, model_type='sequence')
    assert weights is not None, 'Must either specify a weightfile or use reload.latest=True'

    run_files = utils.get_run_files_from_weights(weights)
    if cfg.sequence.latent_name is None:
        # find the latent name used in the weight file you loaded
        rundir = os.path.dirname(weights)
        loaded_cfg = utils.load_yaml(run_files['config_file'])
        latent_name = loaded_cfg['sequence']['latent_name']
        # if this latent name is also None, use the arch of the feature extractor
        # this should never happen
        if latent_name is None:
            latent_name = loaded_cfg['feature_extractor']['arch']
    else:
        latent_name = cfg.sequence.latent_name

    if cfg.inference.use_loaded_model_cfg:
        output_name = cfg.sequence.output_name
        loaded_config_file = run_files['config_file']
        loaded_model_cfg = OmegaConf.load(loaded_config_file).sequence
        current_model_cfg = cfg.sequence
        model_cfg = OmegaConf.merge(current_model_cfg, loaded_model_cfg)
        cfg.sequence = model_cfg
        # we don't want to use the weights that the trained model was initialized with, but the weights after training
        # therefore, overwrite the loaded configuration with the current weights
        cfg.sequence.weights = weights
        cfg.sequence.latent_name = latent_name
        cfg.sequence.output_name = output_name
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

    metrics_file = run_files['metrics_file']
    assert os.path.isfile(metrics_file)
    best_epoch = utils.get_best_epoch_from_weightfile(weights)
    # best_epoch = -1
    log.info('best epoch from loaded file: {}'.format(best_epoch))
    with h5py.File(metrics_file, 'r') as f:
        try:
            thresholds = f['val']['metrics_by_threshold']['optimum'][best_epoch, :]
        except KeyError:
            # backwards compatibility
            thresholds = f['threshold_curves']['val']['optimum'][best_epoch, :]
    log.info('thresholds: {}'.format(thresholds))

    class_names = list(cfg.project.class_names)
    if len(thresholds) != len(class_names):
        error_message = '''Number of classes in trained model: {}
            Number of classes in project: {}
            Did you add or remove behaviors after training this model? If so, please retrain!
        '''.format(len(thresholds), len(class_names))
        raise ValueError(error_message)

    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    class_names = cfg.project.class_names
    class_names = np.array(class_names)
    extract(model,
            outputfiles,
            thresholds,
            cfg.feature_extractor.final_activation,
            latent_name,
            output_name,
            cfg.sequence.sequence_length,
            True,
            device,
            cfg.inference.ignore_error,
            cfg.inference.overwrite,
            class_names=class_names)


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_sequence_inference_cfg(project_path, use_command_line=True)

    sequence_inference(cfg)