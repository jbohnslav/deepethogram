from collections import defaultdict
import logging
import os
import sys
import time
from typing import Type

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch import nn
from tqdm import tqdm

from deepethogram import utils, projects
from deepethogram.configuration import make_feature_extractor_inference_cfg
from deepethogram.data.augs import get_cpu_transforms, get_gpu_transforms_inference, get_gpu_transforms
from deepethogram.data.datasets import VideoIterable
from deepethogram.feature_extractor.train import build_model_from_cfg as build_feature_extractor

log = logging.getLogger(__name__)

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


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

    final_spatial_linear = get_linear_layers(model.spatial_classifier)[-1]
    final_spatial_linear.register_forward_hook(get_inputs('spatial'))
    final_flow_linear = get_linear_layers(model.flow_classifier)[-1]
    final_flow_linear.register_forward_hook(get_inputs('flow'))

    # if fusion == 'average':
    #     get_penultimate_layer(model.spatial_classifier).register_forward_hook(get_inputs('spatial'))
    #     get_penultimate_layer(model.flow_classifier).register_forward_hook(get_inputs('flow'))
    # elif fusion == 'concatenate':
    #     get_penultimate_layer(get_penultimate_layer(
    #         model.spatial_classifier)).register_forward_hook(get_inputs('spatial'))
    #     get_penultimate_layer(get_penultimate_layer(
    #         model.flow_classifier)).register_forward_hook(get_inputs('flow'))
    # else:
    #     raise NotImplementedError
    # list(model.spatial_classifier.children())[-1].register_forward_hook(get_inputs('spatial'))
    # list(model.flow_classifier.children())[-1].register_forward_hook(get_inputs('flow'))
    return activation


def get_linear_layers(model: nn.Module):
    linear_layers = []
    children = model.children()
    for child in children:
        if isinstance(child, nn.Sequential):
            linear_layers.append(get_linear_layers(child))
        elif isinstance(child, nn.Linear):
            linear_layers.append(child)
    return linear_layers


def get_penultimate_layer(model: Type[nn.Module]):
    """ Function to unpack a linear layer from a nn sequential module """
    assert isinstance(model, nn.Module)
    children = list(model.children())
    return children[-1]


def print_debug_statement(images, logits, spatial_features, flow_features, probabilities):
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
    if len(images.shape) == 4:
        N, C, H, W = images.shape
    elif images.ndim == 5:
        N, C, T, H, W = images.shape
    else:
        raise ValueError('images of unknown shape: {}'.format(images.shape))
  
    log.info('channel min:  {}'.format(images[0].reshape(C, -1).min(dim=1).values))
    log.info('channel mean: {}'.format(images[0].reshape(C, -1).mean(dim=1)))
    log.info('channel max : {}'.format(images[0].reshape(C, -1).max(dim=1).values))
    log.info('channel std : {}'.format(images[0].reshape(C, -1).std(dim=1)))


def predict_single_video(videofile, model, activation_function: nn.Module,
                         fusion: str,
                         num_rgb: int, mean_by_channels: np.ndarray,
                         device: str = 'cuda:0',
                         cpu_transform=None, gpu_transform=None,
                         should_print: bool = False, num_workers: int = 1, batch_size: int = 1):
    model.eval()
    model.set_mode('inference')

    if type(device) != torch.device:
        device = torch.device(device)

    dataset = VideoIterable(videofile, transform=cpu_transform, num_workers=num_workers, sequence_length=num_rgb,
                            mean_by_channels=mean_by_channels)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    video_frame_num = len(dataset)

    activation = unpack_penultimate_layer(model, fusion)

    buffer = {}

    has_printed = False
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
        images = gpu_transform(images)

        logits = model(images)
        spatial_features = activation['spatial']
        flow_features = activation['flow']
        # because we are using iterable datasets, each batch will be a consecutive chunk of frames from one worker
        # but they might be from totally different chunks of the video. therefore, we return the frame numbers,
        # and use this to store into our buffer in the right location
        frame_numbers = batch['framenum'].detach().cpu()

        probabilities = activation_function(logits).detach().cpu()
        logits = logits.detach().cpu()
        spatial_features = spatial_features.detach().cpu()
        flow_features = flow_features.detach().cpu()

        if not has_printed and should_print:
            print_debug_statement(images, logits, spatial_features, flow_features, probabilities)
            has_printed = True
        if i == 0:
            # print(f'~~~ N: {N} ~~~')
            buffer['probabilities'] = torch.zeros((video_frame_num, probabilities.shape[1]),
                                                  dtype=probabilities.dtype)
            buffer['logits'] = torch.zeros((video_frame_num, logits.shape[1]),
                                           dtype=logits.dtype)
            buffer['spatial_features'] = torch.zeros((video_frame_num, spatial_features.shape[1]),
                                                     dtype=spatial_features.dtype)
            buffer['flow_features'] = torch.zeros((video_frame_num, flow_features.shape[1]),
                                                  dtype=flow_features.dtype)
            buffer['debug'] = torch.zeros((video_frame_num,)).float()
        buffer['probabilities'][frame_numbers, :] = probabilities
        buffer['logits'][frame_numbers] = logits

        buffer['spatial_features'][frame_numbers] = spatial_features
        buffer['flow_features'][frame_numbers] = flow_features
        buffer['debug'][frame_numbers] += 1
    return buffer


def check_if_should_run_inference(h5file, mode, latent_name, overwrite):
    should_run = True
    with h5py.File(h5file, mode) as f:

        if latent_name in list(f.keys()):
            if overwrite:
                del f[latent_name]
            else:
                log.warning('Latent {} already found in file {}, skipping...'.format(latent_name, h5file))
                should_run = False
    return should_run


def extract(rgbs: list,
            model,
            final_activation: str,
            thresholds: np.ndarray,
            mean_by_channels,
            fusion: str,
            num_rgb: int, latent_name: str, class_names: list = ['background'],
            device: str = 'cuda:0',
            cpu_transform=None, gpu_transform=None, ignore_error=True, overwrite=False,
            num_workers: int = 1, batch_size: int = 1):
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
    # double checknig
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    if final_activation == 'softmax':
        activation_function = nn.Softmax(dim=1)
    elif final_activation == 'sigmoid':
        activation_function = nn.Sigmoid()
    else:
        raise ValueError('unknown final activation: {}'.format(final_activation))

    # 16 is a decent trade off between CPU and GPU load on datasets I've tested
    if batch_size == 'auto':
        batch_size = 16
    batch_size = min(batch_size, 16)
    log.info('inference batch size: {}'.format(batch_size))

    class_names = [n.encode("ascii", "ignore") for n in class_names]

    log.debug('model training mode: {}'.format(model.training))
    # iterate over movie files
    for i in tqdm(range(len(rgbs))):
        rgb = rgbs[i]

        basename = os.path.splitext(rgb)[0]
        # make the outputfile have the same name as the video, with _outputs appended
        h5file = basename + '_outputs.h5'
        mode = 'r+' if os.path.isfile(h5file) else 'w'

        should_run = check_if_should_run_inference(h5file, mode, latent_name, overwrite)
        if not should_run:
            continue

        # iterate over each frame of the movie
        outputs = predict_single_video(rgb, model, activation_function, fusion, num_rgb, mean_by_channels,
                                       device, cpu_transform, gpu_transform, should_print=i == 0,
                                       num_workers=num_workers, batch_size=batch_size)
        if i == 0:
            for k, v in outputs.items():
                log.info('{}: {}'.format(k, v.shape))
                if k == 'debug':
                    log.info('All should be 1.0: min: {:.4f} mean {:.4f} max {:.4f}'.format(
                        v.min(), v.mean(), v.max()
                    ))

        # if running inference from multiple processes, this will wait until the resource is available
        has_worked = False
        while not has_worked:
            try:
                f = h5py.File(h5file, 'r+')
            except OSError as e:
                log.warning('resource unavailable, waiting 30 seconds...')
                time.sleep(30)
            else:
                has_worked = True

        # these assignments are where it's actually saved to disk
        group = f.create_group(latent_name)
        group.create_dataset('thresholds', data=thresholds, dtype=np.float32)
        group.create_dataset('logits', data=outputs['logits'], dtype=np.float32)
        group.create_dataset('P', data=outputs['probabilities'], dtype=np.float32)
        group.create_dataset('spatial_features', data=outputs['spatial_features'], dtype=np.float32)
        group.create_dataset('flow_features', data=outputs['flow_features'], dtype=np.float32)
        dt = h5py.string_dtype()
        group.create_dataset('class_names', data=class_names, dtype=dt)
        del outputs
        f.close()


def feature_extractor_inference(cfg: DictConfig):    
    cfg = projects.setup_run(cfg)
    # turn "models" in your project configuration to "full/path/to/models"
    log.info('args: {}'.format(' '.join(sys.argv)))
    
    log.info('configuration used in inference: ')
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.sequence.latent_name is None:
        latent_name = cfg.feature_extractor.arch
    else:
        latent_name = cfg.sequence.latent_name
    log.info('Latent name used in HDF5 file: {}'.format(latent_name))
    directory_list = cfg.inference.directory_list
    
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
    assert cfg.feature_extractor.n_flows + 1 == cfg.flow_generator.n_rgb, 'Flow generator inputs must be one greater ' \
                                                                          'than feature extractor num flows '

    input_images = cfg.feature_extractor.n_flows + 1
    mode = '3d' if '3d' in cfg.feature_extractor.arch.lower() else '2d'
    # get the validation transforms. should have resizing, etc
    cpu_transform = get_cpu_transforms(cfg.augs)['val']
    gpu_transform = get_gpu_transforms(cfg.augs, mode)['val']
    log.info('gpu_transform: {}'.format(gpu_transform))

    rgb = []
    for record in records:
        rgb.append(record['rgb'])

    model_components = build_feature_extractor(cfg)
    _, _, _, _, model = model_components
    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    feature_extractor_weights = projects.get_weightfile_from_cfg(cfg, 'feature_extractor')
    metrics_file = os.path.join(os.path.dirname(feature_extractor_weights), 'classification_metrics.h5')

    assert os.path.isfile(metrics_file)
    with h5py.File(metrics_file, 'r') as f:
        try: 
            thresholds = f['val']['metrics_by_threshold']['optimum'][-1, :]
        except KeyError:
            # backwards compatibility
            thresholds = f['threshold_curves']['val']['optimum'][-1, :]
        log.info('thresholds: {}'.format(thresholds))
    class_names = list(cfg.project.class_names)
    # class_names = projects.get_classes_from_project(cfg)
    class_names = np.array(class_names)
    extract(rgb,
            model,
            final_activation=cfg.feature_extractor.final_activation,
            thresholds=thresholds,
            mean_by_channels=cfg.augs.normalization.mean,
            fusion=cfg.feature_extractor.fusion,
            num_rgb=input_images,
            latent_name=latent_name,
            device=device,
            cpu_transform=cpu_transform,
            gpu_transform=gpu_transform,
            ignore_error=cfg.inference.ignore_error,
            overwrite=cfg.inference.overwrite,
            class_names=class_names,
            num_workers=cfg.compute.num_workers,
            batch_size=cfg.compute.batch_size)

    # update each record file in the subdirectory to add our new output files
    # projects.write_all_records(cfg.project.data_path)


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_feature_extractor_inference_cfg(project_path, use_command_line=True)
    
    feature_extractor_inference(cfg)