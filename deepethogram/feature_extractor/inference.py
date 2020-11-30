import logging
import os
import sys
from typing import Type

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from deepethogram import utils, projects
from deepethogram.data.augs import get_transforms
from deepethogram.data.datasets import SequentialIterator
from deepethogram.feature_extractor.train import build_model_from_cfg as build_feature_extractor

log = logging.getLogger(__name__)


def unpack_penultimate_layer(model: Type[nn.Module], fusion: str = 'average'):
    """ Adds the activations in the penulatimate layer of the given PyTorch module to a dictionary called 'activation'.

    Assumes the model has two subcomponents: spatial and flow models. Every time the forward pass of this network
    is run, the penultimate neural activations will be added to the activations dictionary.
    This function uses the register_forward_hook syntax in PyTorch:
    https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks

    Example:
        my_model = deg_f()
        activations = unpack_penultimate_layer(my_model)
        print(activations) # nothing in it
        outputs = my_model(some_data)
        print(activations)
        # activations = {'spatial': some 512-dimensional vector, 'flow': another 512 dimensional vector}

    Args:
        model (nn.Module): a two-stream model with subcomponents spatial and flow
        fusion (str): one of average or concatenate

    Returns:
        activations (dict): dictionary with keys ['spatial', 'flow']. After forward pass, will contain
        512-dimensional vector of neural activations (before the last fully connected layer)
    """
    activation = {}

    def get_inputs(name):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        def hook(model, inputs, output):
            if type(inputs) == tuple:
                if len(inputs) == 1:
                    inputs = inputs[0]
                else:
                    raise ValueError('unknown inputs: {}'.format(inputs))
            activation[name] = inputs.detach()

        return hook

    if fusion == 'average':
        get_penultimate_layer(model.spatial_classifier).register_forward_hook(get_inputs('spatial'))
        get_penultimate_layer(model.flow_classifier).register_forward_hook(get_inputs('flow'))
    elif fusion == 'concatenate':
        get_penultimate_layer(get_penultimate_layer(
            model.spatial_classifier)).register_forward_hook(get_inputs('spatial'))
        get_penultimate_layer(get_penultimate_layer(
            model.flow_classifier)).register_forward_hook(get_inputs('flow'))
    else:
        raise NotImplementedError
    # list(model.spatial_classifier.children())[-1].register_forward_hook(get_inputs('spatial'))
    # list(model.flow_classifier.children())[-1].register_forward_hook(get_inputs('flow'))
    return activation


def get_penultimate_layer(model: Type[nn.Module]):
    """ Function to unpack a linear layer from a nn sequential module """
    assert isinstance(model, nn.Module)
    children = list(model.children())
    return children[-1]


def extract(rgbs: list, model, final_activation: str, thresholds: np.ndarray,
            fusion: str,
            num_rgb: int, latent_name: str, class_names: list = ['background'],
            device: str = 'cuda:0',
            transform=None, ignore_error=True, overwrite=False, conv_2d:bool=False):
    """ Use the model to extract spatial and flow feature vectors, and predictions, and save them to disk.
    Assumes you have a pretrained model, and K classes. Will go through each video in rgbs, run inference, extracting
    the 512-d spatial features, 512-d flow features, K-d probabilities to disk for each video frame.
    Also stores thresholds for later reloading.

    Output file structure (outputs.h5):
        - latent_name
            - spatial_features: (T x 512) neural activations from before the last fully connected layer of the spatial
                model
            - flow_features: (T x 512) neural activations from before the last fully connected layer of the flow model
            - logits: (T x K) unnormalized logits from after the fusion layer
            - P: (T x K) values after the activation function (specified by final_activation)
            - thresholds: (K,) loaded thresholds that convert probabilities to binary predictions
            - class_names: (K,) loaded class names for your project

    Args:
        rgbs (list): list of input video files
        model (nn.Module): a hidden-two-stream deepethogram model
            see deepethogram/feature_extractor/models/hidden_two_stream.py
        final_activation (str): one of sigmoid or softmax
        thresholds (np.ndarray): array of shape (K,), thresholds between 0 and 1 that turns probabilities into
            binary predictions
        fusion (str): one of [average, concatenate]
        num_rgb (int): number of input images to your model
        latent_name (str): an HDF5 group with this name will be in your output HDF5 file.
        class_names (list): a list of class names. Will be saved so that this HDF5 file can be read without any project
            configuration files
        device (str): cuda device on which models will be run
        transform (transforms.Compose): data augmentation. Since this is inference, should only include resizing,
            cropping, and normalization
        ignore_error (bool): if True, an error on one video will not stop inference
        overwrite (bool): if an HDF5 group with the given latent_name is present in the HDF5 file:
            if True, overwrites data with current values. if False, skips that video
    """
    # make sure we're using CUDNN for speed
    torch.backends.cudnn.benchmark = True

    assert isinstance(model, torch.nn.Module)

    device = torch.device(device)
    torch.cuda.set_device(device)
    model = model.to(device)
    # freeze model and set to eval mode for batch normalization
    model.set_mode('inference')
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    has_printed = False

    if final_activation == 'softmax':
        activation_function = nn.Softmax(dim=1)
    elif final_activation == 'sigmoid':
        activation_function = nn.Sigmoid()

    class_names = [n.encode("ascii", "ignore") for n in class_names]

    log.debug('model training mode: {}'.format(model.training))
    # iterate over movie files
    for i in tqdm(range(len(rgbs))):
        rgb = rgbs[i]
        log.info('Extracting from movie {}...'.format(rgb))
        gen = SequentialIterator(rgb, num_rgb, transform=transform, device=device, stack_channels=conv_2d)


        log.debug('Making two stream iterator with parameters: ')
        log.debug('rgb: {}'.format(rgb))
        log.debug('num_images: {}'.format(num_rgb))

        basename = os.path.splitext(rgb)[0]
        # make the outputfile have the same name as the video, with _outputs appended
        h5file = basename + '_outputs.h5'
        mode = 'r+' if os.path.isfile(h5file) else 'w'
        # model.set_mode('inference')
        # our activation dictionary will automatically be updated after each forward pass
        activation = unpack_penultimate_layer(model, fusion)

        with h5py.File(h5file, mode) as f:

            if latent_name in list(f.keys()):
                if overwrite:
                    del (f[latent_name])
                else:
                    log.warning('Latent {} already found in file {}, skipping...'.format(latent_name, h5file))
                    continue
            group = f.create_group(latent_name)
            # iterate over each frame of the movie
            for i in tqdm(range(len(gen) - 1), leave=False):
                # for i in tqdm(range(1000)):
                with torch.no_grad():
                    try:
                        batch = next(gen)
                        # star means that if batch is a tuple of images, flows, it will pass in as sequential
                        # positional arguments
                        if type(batch) == tuple:
                            logits = model(*batch)
                        else:
                            logits = model(batch)
                        spatial_features = activation['spatial']
                        flow_features = activation['flow']
                        # this debug information is extremely useful. If you get strange, large values for the min
                        # or max, you should make sure that your input images are being properly normalized
                        # for each image in your input sequence. You should also make sure that weights are being
                        # reloaded properly
                        if not has_printed:
                            log.info('logits shape: {}'.format(logits.shape))
                            log.info('spatial_features shape: {}'.format(spatial_features.shape))
                            log.info('flow_features shape: {}'.format(flow_features.shape))
                            log.info('spatial: min {} mean {} max {} shape {}'.format(spatial_features.min(),
                                                                                      spatial_features.mean(),
                                                                                      spatial_features.max(),
                                                                                      spatial_features.shape))
                            log.info('flow   : min {} mean {} max {} shape {}'.format(flow_features.min(),
                                                                                      flow_features.mean(),
                                                                                      flow_features.max(),
                                                                                      flow_features.shape))
                            # a common issue I've had is not properly z-scoring input channels. this will check for that
                            if len(batch.shape) == 4:
                                N, C, H, W = batch.shape
                                log.debug('channel min:  {}'.format(batch[0].reshape(C, -1).min(dim=1).values))
                                log.debug('channel mean: {}'.format(batch[0].reshape(C, -1).mean(dim=1)))
                                log.debug('channel max : {}'.format(batch[0].reshape(C, -1).max(dim=1).values))
                                log.debug('channel std : {}'.format(batch[0].reshape(C, -1).std(dim=1)))
                            elif len(batch.shape) == 5:
                                N, C, T, H, W = batch.shape
                                log.debug('channel min:  {}'.format(batch[0].min(dim=2).values))
                                log.debug('channel mean: {}'.format(batch[0].mean(dim=2)))
                                log.debug('channel max : {}'.format(batch[0].max(dim=2).values))
                                log.debug('channel std : {}'.format(batch[0].std(dim=2)))
                            has_printed = True

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log.exception('Error on video {}, frame: {}'.format(rgb, i))
                        if ignore_error:
                            log.warning('continuing...')
                            break
                        else:
                            raise
                    probabilities = activation_function(logits).detach().cpu().numpy().squeeze()
                    logits = logits.detach().cpu().numpy().squeeze()
                    spatial_features = spatial_features.detach().cpu().numpy().squeeze()
                    flow_features = flow_features.detach().cpu().numpy().squeeze()

                if i == 0:
                    group.create_dataset('thresholds', data=thresholds, dtype=np.float32)
                    group.create_dataset('logits', (len(gen), logits.shape[0]), dtype=np.float32)
                    group.create_dataset('P', (len(gen), logits.shape[0]), dtype=np.float32)
                    group.create_dataset('spatial_features', (len(gen), spatial_features.shape[0]), dtype=np.float32)
                    group.create_dataset('flow_features', (len(gen), flow_features.shape[0]), dtype=np.float32)
                    dt = h5py.string_dtype()
                    group.create_dataset('class_names', data=class_names, dtype=dt)
                # these assignments are where it's actually saved to disk
                group['P'][i, :] = probabilities
                group['logits'][i, :] = logits
                group['spatial_features'][i, :] = spatial_features
                group['flow_features'][i, :] = flow_features
                del (batch, logits, spatial_features, flow_features, probabilities)
            gen.end()


@hydra.main(config_path='../conf/feature_extractor_inference.yaml')
def main(cfg: DictConfig):
    # turn "models" in your project configuration to "full/path/to/models"
    cfg = utils.get_absolute_paths_from_cfg(cfg)
    log.info('configuration used in inference: ')
    log.info(cfg.pretty())
    if cfg.sequence.latent_name is None:
        latent_name = cfg.feature_extractor.arch
    else:
        latent_name = cfg.sequence.latent_name
    directory_list = cfg.inference.directory_list
    if directory_list is None or len(directory_list) == 0:
        raise ValueError('must pass list of directories from commmand line. '
                         'Ex: directory_list=[path_to_dir1,path_to_dir2]')
    elif type(directory_list) == str and directory_list == 'all':
        basedir = cfg.project.data_path
        directory_list = utils.get_subfiles(basedir, 'directory')

    # video files are found in your input list of directories using the records.yaml file that should be present
    # in each directory
    records = []
    for directory in directory_list:
        assert os.path.isdir(directory), 'Not a directory: {}'.format(directory)
        record = projects.get_record_from_subdir(directory)
        assert record['rgb'] is not None
        records.append(record)
    assert cfg.feature_extractor.n_flows + 1 == cfg.flow_generator.n_rgb, 'Flow generator inputs must be one greater ' \
                                                                          'than feature extractor num flows '

    input_images = cfg.feature_extractor.n_flows + 1
    mode = '3d' if '3d' in cfg.feature_extractor.arch.lower() else '2d'
    # get the validation transforms. should have resizing, etc
    transform = get_transforms(cfg.augs, input_images, mode)['val']

    rgb = []
    for record in records:
        rgb.append(record['rgb'])

    model = build_feature_extractor(cfg)
    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    feature_extractor_weights = projects.get_weightfile_from_cfg(cfg, 'feature_extractor')
    metrics_file = os.path.join(os.path.dirname(feature_extractor_weights), 'classification_metrics.h5')
    assert os.path.isfile(metrics_file)
    with h5py.File(metrics_file, 'r') as f:
        thresholds = f['threshold_curves']['val']['optimum'][:]
        log.info('thresholds: {}'.format(thresholds))
    class_names = list(cfg.project.class_names)
    # class_names = projects.get_classes_from_project(cfg)
    class_names = np.array(class_names)
    extract(rgb,
            model,
            final_activation=cfg.feature_extractor.final_activation,
            thresholds=thresholds,
            fusion=cfg.feature_extractor.fusion,
            num_rgb=input_images,
            latent_name=latent_name,
            device=device,
            transform=transform,
            ignore_error=cfg.inference.ignore_error,
            overwrite=cfg.inference.overwrite,
            class_names=class_names,
            conv_2d=mode == '2d')

    # update each record file in the subdirectory to add our new output files
    projects.write_all_records(cfg.project.data_path)


if __name__ == '__main__':
    sys.argv = utils.process_config_file_from_cl(sys.argv)
    main()
