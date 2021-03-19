from collections import Mapping, Container
import logging
import os
import pkgutil
from collections import OrderedDict
from inspect import isfunction
from operator import itemgetter
import sys
from types import SimpleNamespace
from typing import Union

import cv2
import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig

log = logging.getLogger(__name__)


def load_yaml(filename: Union[str, os.PathLike]) -> dict:
    """Simple wrapper around yaml.load to load yaml files as dictionaries"""
    with open(filename, 'r') as f:
        dictionary = yaml.load(f, Loader=yaml.Loader)
    return dictionary


def load_config(filename: Union[str, os.PathLike]) -> DictConfig:
    """ loads a yaml file as dictionary and converts to an omegaconf DictConfig """
    dictionary = load_yaml(filename)
    return OmegaConf.create(dictionary)


def get_minimum_learning_rate(optimizer):
    """Get the smallest learning rate from a PyTorch optimizer.
    Useful for ReduceLROnPLateau stoppers. If the minimum learning rate drops below a set value, will stop training.
    """
    min_lr = 1e9
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        if lr < min_lr:
            min_lr = lr
    return (min_lr)


def load_checkpoint(model, optimizer, checkpoint_file: Union[str, os.PathLike], config: dict,
                    overwrite_args: bool = False,
                    distributed: bool = False):
    """"Reload model and optimizer weights from a checkpoint.pt file
    Args:
        model: instance of torch.nn.Module class
        optimizer: instance of torch.optim.Optimizer class (ADAM, SGDM, etc.)
        config: dictionary of hyperparameters
        overwrite_args: if true, overwrite the input dictionary with the saved one from the checkpoint file
        distributed: if true, will prepend "module" to parameter names, which for some reason PyTorch does (or did)
    Returns:
        model: model with pretrained weights
        optimizer: optimizer with recent history of gradients
        config: depending on overwrite_args, input or reloaded hyperparameter dictionary
    """
    log.info('Reloading model from {}...'.format(checkpoint_file))
    model, optimizer_dict, _, new_args = load_state(model, checkpoint_file, distributed=distributed)
    if type(new_args) != dict:
        new_config = vars(new_args)
    else:
        new_config = new_args
    try:
        optimizer.load_state_dict(optimizer_dict)
    except Exception as e:
        log.exception('Trouble loading optimizer state dict--might have requires-grad' \
                      'for different parameters: {}'.format(e))
        log.warning('Not loading optimizer state.')
    if overwrite_args:
        config = new_config

    return model, optimizer, config


def load_weights(model, checkpoint_file: Union[str, os.PathLike], distributed: bool = False,
                 device: torch.device = None):
    """"Reload model weights from a checkpoint.pt file
    Args:
        model: instance of torch.nn.Module class
        checkpoint_file: path to checkpoint.pt
        distributed: if true, will prepend "module" to parameter names, which for some reason PyTorch does (or did)
        device (torch.device): optional device. If model trained on cuda:1, it won't be able to load to cuda:0
    Returns:
        model: model with pretrained weights

    """
    model, _, _, _ = load_state(model, checkpoint_file, distributed=distributed, device=device)

    return model


def checkpoint(model, rundir: Union[str, os.PathLike], epoch: int, args=None):
    """"
    Args:
        model: instance of torch.nn.Module class
        rundir: directory to  save checkpoint.pt to
        epoch: integer containing which epoch of training it is. Sometimes used for reloading
        args: either an argument parser or a dictionary with hyperparameters
    """
    if args is not None:
        if type(args) != dict:
            args = vars(args)
    fname = 'checkpoint.pt'
    fullfile = os.path.join(rundir, fname)
    # note: I used to save the optimizer dict as well, but this was confusing in terms of keeping track of learning
    # rates, making sure the same keys were in the optimizer dict even when you've done something like change
    # the size of the final layer of the NN (for different number of classes). I've kept the optimizer field for
    # backwards compatibility, but this should not be used
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': None,
        'hyperparameters': args
    }
    torch.save(state, fullfile)


def save_two_stream(model, rundir: Union[os.PathLike, str], config: dict = None, epoch: int = None) -> None:
    """ Saves a two-stream model to disk. Saves spatial and flow feature extractors in their own directories """
    assert os.path.isdir(rundir)
    assert isinstance(model, torch.nn.Module)
    spatialdir = os.path.join(rundir, 'spatial')
    if not os.path.isdir(spatialdir):
        os.makedirs(spatialdir)
    checkpoint(model.spatial_classifier, spatialdir, epoch, config)

    flow_classifier_dir = os.path.join(rundir, 'flow')
    if not os.path.isdir(flow_classifier_dir):
        os.makedirs(flow_classifier_dir)
    checkpoint(model.flow_classifier, flow_classifier_dir, epoch, config)

    checkpoint(model, rundir, epoch, config)


def save_hidden_two_stream(model, rundir: Union[os.PathLike, str], config: dict = None, epoch: int = None) -> None:
    """ Saves a hidden two-stream model to disk. Saves flow generator in a separate directory """
    assert os.path.isdir(rundir)
    assert isinstance(model, torch.nn.Module)
    flowdir = os.path.join(rundir, 'flow_generator')
    if not os.path.isdir(flowdir):
        os.makedirs(flowdir)
    if type(config) == DictConfig:
        config = OmegaConf.to_container(config)
    checkpoint(model.flow_generator, flowdir, epoch, config)
    save_two_stream(model, rundir, config, epoch)


def save_dict_to_yaml(dictionary: dict, filename: Union[str, bytes, os.PathLike]) -> None:
    """Simple wrapper around yaml.dump for saving a dictionary to a yaml file
    Args:
        dictionary: dict to be written
        filename: file to save dict to. Should end in .yaml
    """
    if os.path.isfile(filename):
        log.debug('File {} already exists, overwriting...'.format(filename))
    if isinstance(dictionary, DictConfig):
        dictionary = OmegaConf.to_container(dictionary)
    with open(filename, 'w') as f:
        yaml.dump(dictionary, f, default_flow_style=False)


def tensor_to_np(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Simple function for turning pytorch tensor into numpy ndarray"""
    if type(tensor) == np.ndarray:
        return tensor
    return tensor.cpu().detach().numpy()


def in_this_dir(abs_path: Union[str, os.PathLike]) -> dict:
    """ Gets list of files in a subdirectory and returns information about it.
    Designed to be a drop-in replacement for MATLAB's `dir` command :P
    Args:
        abs_path: absolute path to a directory
    Returns:
        contents: dictionary with keys 'name', 'isdir', and 'bytes', containing the name, whether or not the file is
            a directory, and the filesize in bytes of all files in the directory
    """
    backslashes = strfind(abs_path, '\\')
    if len(backslashes) > 1 and backslashes[0] != -1:
        abs_path.replace('\\', '/')
    # contents contains list of filenames or directory names
    filenames = os.listdir(abs_path)
    contents = []
    for name in filenames:
        content = {}
        content['name'] = name
        content['isdir'] = os.path.isdir(os.path.join(abs_path, name))
        content['bytes'] = os.path.getsize(os.path.join(abs_path, name))
        contents.append(content)
    # sort by name!
    contents = sorted(contents, key=itemgetter('name'))
    return contents


def get_datadir_from_paths(paths, dataset):
    """DEPRECATED.
    If you have a paths module, converts it to a dictionary and finds the data directory
    """
    dictionary = vars(paths)
    found = False
    for k, v in dictionary.items():
        if dataset in k:
            datadir = v
            found = True
    if not found:
        raise ValueError('couldn''t find dataset: {}'.format(dataset))
    return datadir


def strfind(name: str, ch: str) -> list:
    """Simple function to replicate MATLAB's strfind function"""
    inds = []
    for i in range(len(name)):
        if name.find(ch, i) == i:
            inds.append(i)
    return inds


def load_state_from_dict(model, state_dict):
    """Loads weights into a PyTorch nn.Module from a loaded state_dict.
    **Loads all weights of same name in both the model and the state dictionary, given that they are the same shape!!**
    Does not throw errors if they are not the same shape.
    Example usage: reloading ImageNet weights for a classification problem with less than 1000 classes.
    Args:
        model: instance of nn.Module
        state_dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict

    Returns:
        model with weights loaded from state_dict
    """
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k not in model_dict:
            log.warning('{} not found in model dictionary'.format(k))
        else:
            if model_dict[k].size() != v.size():
                log.warning('{} has different size: pretrained:{} model:{}'.format(k, v.size(), model_dict[k].size()))
            else:
                log.debug('Successfully loaded: {}'.format(k))
                pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    # only_in_model_dict = {k:v for k,v in state_dict.items() if k in model_dict}
    # model_dict.update(only_in_model_dict)
    # load the state dict, only for layers of same name, shape, size, etc.
    model.load_state_dict(model_dict, strict=True)
    return (model)


def load_state_dict_from_file(weights_file, distributed: bool=False):
    state = torch.load(weights_file, map_location='cpu')
        # except RuntimeError as e:
        #     log.exception(e)
        #     log.info('loading onto cpu...')
        #     state = torch.load(weights_file, map_location='cpu')

    is_pure_weights = not 'epoch' in list(state.keys())
    # load params
    if is_pure_weights:
        state_dict = state
        start_epoch = 0
    else:
        start_epoch = state['epoch']
        state_dict = state['state_dict']
        optimizer_dict = None # state['optimizer']

    first_key = next(iter(state_dict.items()))[0]
    trained_on_dataparallel = first_key[:7] == 'module.'

    if distributed and not trained_on_dataparallel:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    # if it was trained on multi-gpu, remove the 'module.' before variable names
    elif not distributed and trained_on_dataparallel:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    # sometimes I have the encoder in a structure called 'model', which means
    # all weights have 'model.' prepended
    model_prepended = first_key[:6] == 'model.'
    if model_prepended:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == 'model.':
                name = k[6:]
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict
    if not is_pure_weights:
        if 'hyperparameters' in list(state.keys()):
            args = state['hyperparameters']
        else:
            args = None
    else:
        d = {}
        args = SimpleNamespace(**d)
    return state_dict, start_epoch, args


def load_state(model, weights_file: Union[str, os.PathLike], device: torch.device = None, distributed: bool = False):
    """"Reload model and optimizer weights from a checkpoint.pt file.

    TODO: refactor this loading for pytorch 1.4+. This was written many versions ago

    Args:
        model: instance of torch.nn.Module class
        weights_file: checkpoint containing at least a state_dict, and optionally an epoch, optimizer_dict, and
            arguments
        distributed: if true, will prepend "module" to parameter names, which for some reason PyTorch does (or did)
    Returns:
        model: model with pretrained weights
        optimizer_dict: recent history of gradients from an optimizer
        start_epoch: last epoch from pretrained model
        args: SimpleNamespace containing hyperparameters
            TODO: change args to a config dictionary
    """
    # fullfile = os.path.join(model_dir,run_dir, fname)
    # state is a dictionary
    # Keys:
    #  epoch: final epoch number from training
    #  state_dict: weights
    #  args: hyperparameters
    log.info('loading from checkpoint file {}...'.format(weights_file))

    state_dict, start_epoch, args = load_state_dict_from_file(weights_file, distributed=distributed)
    # LOAD PARAMS
    model = load_state_from_dict(model, state_dict)
    optimizer_dict = None
    
    return model, optimizer_dict, start_epoch, args


def print_gpus():
    """Simple function to print available GPUs, their name, device capability, and memory.
    Note: sometimes torch.cuda.max_memory_allocated does not actually match available VRAM.
    """
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        print('GPU %d %s: Compute Capability %d.%d, Mem:%f' % (i,
                                                               torch.cuda.get_device_name(i),
                                                               int(torch.cuda.get_device_capability(i)[0]),
                                                               int(torch.cuda.get_device_capability(i)[1]),
                                                               torch.cuda.max_memory_allocated(i)))


class Normalizer:
    """Allows for easy z-scoring of tensors on the GPU.
    Example usage: You have a tensor of images of shape [N, C, H, W] or [N, T, C, H, W] in range [0,1]. You want to
        z-score this tensor.

    Methods:
        process_inputs: converts input mean, std into a torch tensor
        no_conversion: dummy method if you don't actually want to standardize the data
        handle_tensor: deals with self.mean and self.std depending on inputs. Example: your Tensor arrives on the GPU
            but your self.mean and self.std are still on the CPU. This method will move it appropriately.
        denormalize: converts standardized arrays back to their original range
        __call__: z-scores input data

    Instance variables:
        mean: mean of input data. For images, should have 2 or 3 channels
        std: standard deviation of input data
    """

    def __init__(self, mean: Union[list, np.ndarray, torch.Tensor] = None,
                 std: Union[list, np.ndarray, torch.Tensor] = None,
                 clamp: bool = True):
        """Constructor for Normalizer class.
        Args:
            mean: mean of input data. Should have 3 channels (for R,G,B) or 2 (for X,Y) in the optical flow case
            std: standard deviation of input data.
            clamp: if True, clips the output of a denormalized Tensor to between 0 and 1 (for images)
        """
        # make sure that if you have a mean, you also have a std
        # XOR
        has_mean, has_std = mean is None, std is None
        assert (not has_mean ^ has_std)

        self.mean = self.process_inputs(mean)
        self.std = self.process_inputs(std)
        # prevent divide by zero, but only change values if it's close to 0 already
        if self.std is not None:
            assert (self.std.min() > 0)
            self.std[self.std < 1e-8] += 1e-8
        log.debug('Normalizer created with mean {} and std {}'.format(self.mean, self.std))
        self.clamp = clamp

    def process_inputs(self, inputs: Union[torch.Tensor, np.ndarray]):
        """Deals with input mean and std.
        Converts to tensor if necessary. Reshapes to [length, 1, 1] for pytorch broadcasting.
        """
        if inputs is None:
            return (inputs)
        if type(inputs) == list:
            inputs = np.array(inputs).astype(np.float32)
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs)
        assert (type(inputs) == torch.Tensor)
        inputs = inputs.float()
        C = inputs.shape[0]
        inputs = inputs.reshape(C, 1, 1)
        inputs.requires_grad = False
        return inputs

    def no_conversion(self, inputs):
        """Dummy function. Allows for normalizer to be called when you don't actually want to normalize.
        That way we can leave normalize in the training loop and only optionally call it.
        """
        return inputs

    def handle_tensor(self, tensor: torch.Tensor):
        """Reshapes std and mean to deal with the dimensions of the input tensor.
        Args:
            tensor: PyTorch tensor of shapes NCHW or NCTHW, depending on if your CNN is 2D or 3D
        Moves mean and std to the tensor's device if necessary
        If you've stacked the C dimension to have multiple images, e.g. 10 optic flows stacked has dim C=20,
            repeats self.mean and self.std to match
        """
        if tensor.ndim == 4:
            N, C, H, W = tensor.shape
        elif tensor.ndim == 5:
            N, C, T, H, W = tensor.shape
        else:
            raise ValueError('Tensor input to normalizer of unknown shape: {}'.format(tensor.shape))

        t_d = tensor.device
        if t_d != self.mean.device:
            self.mean = self.mean.to(t_d)
        if t_d != self.std.device:
            self.std = self.std.to(t_d)

        c = self.mean.shape[0]
        if c < C:
            # handles the case where instead of N, C, T, H, W inputs, we have concatenated
            # multiple images along the channel dimension, so it's
            # N, C*T, H, W
            # this code simply repeats the mean T times, so it's
            # [R_mean, G_mean, B_mean, R_mean, G_mean, ... etc]
            n_repeats = C / c
            assert (int(n_repeats) == n_repeats)
            n_repeats = int(n_repeats)
            repeats = tuple([n_repeats] + [1 for i in range(self.mean.ndim - 1)])
            self.mean = self.mean.repeat((repeats))
            self.std = self.std.repeat((repeats))

        if tensor.ndim - self.mean.ndim > 1:
            # handles the case where our inputs are NCTHW
            self.mean = self.mean.unsqueeze(-1)
        if tensor.ndim - self.std.ndim > 1:
            self.std = self.std.unsqueeze(-1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converts normalized data back into its original distribution.
        If self.clamp: limits output tensor to the range (0,1). For images
        """
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)
        tensor = (tensor * self.std) + self.mean
        if self.clamp:
            tensor = tensor.clamp(min=0.0, max=1.0)
        return tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes input data"""
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)

        tensor = (tensor - self.mean) / (self.std)
        return tensor


def get_num_parameters(model) -> int:
    """Calculates the number of trainable parameters in a Pytorch model.
    Args:
        model: instance of nn.Module
    Returns:
        num_params (int): number of trainable params
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def flow_to_img_lrcn(flow: np.ndarray, max_flow: Union[float, int] = 10) -> np.ndarray:
    """Encodes an optic flow dX, dY -> a 3-channel image according to LRCN paper protocol
    https://arxiv.org/abs/1411.4389
    Args:
        flow: np.ndarray of shape [H, W, 2]
        max_flow: for conversion of float to int, you want to maximize the dynamic range of the resulting uint8.
            That means all flows above or below this value will be clipped to max_flow, -max_flow respectively.
    Returns:
        Image flow of shape [H, W, 3] (uint8)
    C=0: X normalized by the length of the optic flow vector
    C=1: Y normalized by the length of the vector
    C=2: length of vector
    """
    # input: flow, can be positive or negative
    # ranges from -20 to 20, but only 10**-5 pixels are > 10
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag[np.isinf(mag)] = 0

    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    half_range = (255 - 128) / max_flow
    img[:, :, 0] = flow[..., 0] * half_range + 128
    img[:, :, 1] = flow[..., 1] * half_range + 128
    # maximum magnitude is if x and y are both maxed
    max_magnitude = np.sqrt(max_flow ** 2 + max_flow ** 2)
    img[:, :, 2] = mag * 255 / max_magnitude
    img = img.clip(min=0, max=255).astype(np.uint8)
    return img


def flow_img_to_flow(img: np.ndarray, max_flow: Union[int, float] = 10) -> np.ndarray:
    """Converts an RGB image into an optical flow linearly, according to max_flow

    dX: the 0th (red) channel, with 255 -> max_flow, 0 -> -max_flow
    dY: the 1st (green) channel, with 255 -> max_flow, 0 -> -max_flow

    Parameters
    ----------
    img: np.ndarray
        Array of shape H, W, C (C=3) containing an encoded optic flow.
    max_flow: int, float
        Scalar. 255 in the color channels will be mapped to this value, and 0 to the negative of this value.
        Importantly, sets the dynamic range of the optic flow.

    Returns
    -------

    """
    x_channel = img[:, :, 0].astype(np.float32)
    y_channel = img[:, :, 1].astype(np.float32)
    half_range = (255 - 128) / max_flow

    dX = (x_channel - 128) / half_range
    dY = (y_channel - 128) / half_range
    return np.dstack((dX, dY))


# def encode_flow_img(flow, maxflow=10):
#     im = flow_to_img_lrcn(flow, max_flow=maxflow)
#     # print(im.shape)
#     ret, bytestring = cv2.imencode('.jpg', im)
#     return (bytestring)


# def decode_flow_img(bytestring, maxflow=10):
#     im = cv2.imdecode(bytestring, 1)
#     flow = flow_img_to_flow(im, max_flow=maxflow)
#     return (flow)


def module_to_dict(module, exclude=[], get_function=False):
    """ Converts functions in a module to a dictionary. Useful for loading model types into a dictionary """
    module_dict = {}
    for x in dir(module):
        submodule = getattr(module, x)
        # print(x, submodule)
        func = isfunction(submodule) if get_function else not isfunction(submodule)
        if (func and x not in exclude and submodule not in exclude):
            module_dict[x] = submodule
    return module_dict


def get_models_from_module(module, get_function=False):
    """ Hacky function for getting a dictionary of model: initializer from a module """
    models = {}
    for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
        # print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        total_name = module.__name__ + '.' + modname
        this_module = __import__(total_name)
        submodule = getattr(module, modname)
        # module
        this_dict = module_to_dict(submodule, get_function=get_function)
        for key, value in this_dict.items():
            if modname in key:
                models[key] = value
    return models


def load_feature_extractor_components(model, checkpoint_file: Union[str, os.PathLike], component: str, device=None):
    """ Loads individual component from a hidden two-stream model checkpoint

    Parameters
    ----------
    model: nn.Module
        pytorch model
    checkpoint_file: str, os.PathLike
        absolute path to weights on disk
    component: str
        which component to load. One of 'spatial', 'flow'
    device: str, torch.device
        optionally load model weights onto specific device.

    Returns
    -------
    model: nn.Module
        pytorch model with loaded weights
    """
    if component == 'spatial':
        key = 'spatial_classifier' + '.'
    elif component == 'flow':
        key = 'flow_classifier' + '.'
    elif component == 'fusion':
        key = 'fusion.'
    else:
        raise ValueError('component not one of spatial or flow: {}'.format(component))
    # directory = os.path.dirname(checkpoint_file)
    # subdir = os.path.join(directory, component)
    # log.info('device: {}'.format(device))
    log.info('loading component {} from file {}'.format(component, checkpoint_file))

    state_dict, _, _ = load_state_dict_from_file(checkpoint_file)
    
    # state = torch.load(checkpoint_file, map_location=device)
    # state_dict = state['state_dict']
    params = {k.replace(key, ''): v for k, v in state_dict.items() if k.startswith(key)}
    # import pdb; pdb.set_trace()
    model = load_state_from_dict(model, params)
    # import pdb; pdb.set_trace()
    # if not os.path.isdir(subdir):
    #     log.warning('{} directory not found in {}'.format(component, directory))
    #     state = torch.load(checkpoint_file, map_location=device)
    #     state_dict = state['state_dict']
    #     params = {k.replace(key, ''): v for k, v in state_dict.items() if k.startswith(key)}
    #     # import pdb; pdb.set_trace()
    #     model = load_state_from_dict(model, params)
    # else:
    #     sub_checkpoint = os.path.join(subdir, 'checkpoint.pt')
    #     model, _, _, _ = load_state(model, sub_checkpoint, device=device)
    return model


def get_subfiles(root: Union[str, bytes, os.PathLike], return_type: str = None) -> list:
    """ Helper function to get a list of files of certain type from a directory

    Parameters
    ----------
    root: str, os.PathLike
        directory
    return_type: str
        None, 'any': return all files and sub-directories
        'file': only return files, not sub-directories
        'directory': only return sub-directories, not files

    Returns
    -------
    files: list
        list of absolute paths of sub-files
    """
    assert (return_type in [None, 'any', 'file', 'directory'])
    files = os.listdir(root)
    files.sort()
    files = [os.path.join(root, i) for i in files]
    if return_type is None or return_type == 'any':
        pass
    elif return_type == 'file':
        files = [i for i in files if os.path.isfile(i)]
    elif return_type == 'directory':
        files = [i for i in files if os.path.isdir(i)]
    return files


def print_hdf5(h5py_obj, level=-1, print_full_name: bool = False, print_attrs: bool = True) -> None:
    """ Prints the name and shape of datasets in a H5py HDF5 file.
    Parameters
    ----------
    h5py_obj: [h5py.File, h5py.Group]
        the h5py.File or h5py.Group object
    level: int
        What level of the file tree you are in
    print_full_name
        If True, the full tree will be printed as the name, e.g. /group0/group1/group2/dataset: ...
        If False, only the current node will be printed, e.g. dataset:
    print_attrs
        If True: print all attributes in the file
    Returns
    -------
    None
    """

    def is_group(f):
        return type(f) == h5py._hl.group.Group

    def is_dataset(f):
        return type(f) == h5py._hl.dataset.Dataset

    def print_level(level, n_spaces=5) -> str:
        if level == -1:
            return ''
        prepend = '|' + ' ' * (n_spaces - 1)
        prepend *= level
        tree = '|' + '-' * (n_spaces - 2) + ' '
        return prepend + tree

    for key in h5py_obj.keys():
        entry = h5py_obj[key]
        name = entry.name if print_full_name else os.path.basename(entry.name)
        if is_group(entry):
            print('{}{}'.format(print_level(level), name))
            print_hdf5(entry, level + 1, print_full_name=print_full_name)
        elif is_dataset(entry):
            shape = entry.shape
            dtype = entry.dtype
            print('{}{}: {} {}'.format(print_level(level), name,
                                       shape, dtype))
    if level == -1:
        if print_attrs:
            print('attrs: ')


#
# def deep_getsizeof(o, ids):
#     """Find the memory footprint of a Python object
#
#     This is a recursive function that drills down a Python object graph
#     like a dictionary holding nested dictionaries with lists of lists
#     and tuples and sets.
#
#     The sys.getsizeof function does a shallow size of only. It counts each
#     object inside a container as pointer only regardless of how big it
#     really is.
#
#     :param o: the object
#     :param ids:
#     :return:
#     """
#     d = deep_getsizeof
#     if id(o) in ids:
#         return 0
#
#     r = sys.getsizeof(o)
#     ids.add(id(o))
#
#     if isinstance(o, str):
#         return r
#
#     if isinstance(o, Mapping):
#         return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
#
#     if isinstance(o, Container):
#         return r + sum(d(x, ids) for x in o)
#
#     return r

def print_top_largest_variables(local_call, num: int=20):
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in local_call.items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def get_hparams_from_cfg(cfg, hparams): 
    hparam_dict = {key: get_dotted_from_cfg(cfg, key) for key in hparams}
    return hparam_dict

def get_dotted_from_cfg(cfg, dotted):
    # cfg: DictConfig
    # dotted: string parameter name. can be nested. e.g. 'tune.hparams.feature_extractor.dropout_p.min'
    key_list = dotted.split('.')
    
    cfg_chunk = cfg.get(key_list[0])
    for i in range(1, len(key_list)):
        cfg_chunk = cfg_chunk.get(key_list[i])
        
    return cfg_chunk

def get_best_epoch_from_weightfile(weightfile: Union[str, os.PathLike]) -> int:
    basename = os.path.basename(weightfile)
    # in the previous version of deepethogram, load the last checkpoint
    if basename.endswith('.pt'): 
        return -1
    assert basename.endswith('.ckpt')
    basename = os.path.splitext(basename)[0]
    
    # if weightfile is the "last"
    if 'last' in basename:
        return -1

    components = basename.split('-')
    component = components[0]
    assert component.startswith('epoch')
    best_epoch = component.split('=')[1]
    return int(best_epoch)