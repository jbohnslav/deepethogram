import logging
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn

from deepethogram import utils
from .classifiers import alexnet, densenet, inception, vgg, resnet, squeezenet, resnet3d
from .utils import pop

log = logging.getLogger(__name__)


# from nvidia
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/tools.py
def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isfunction(getattr(module, x))
                 and x not in exclude
                 and getattr(module, x) not in exclude])


# model definitions can be accessed by indexing into this dictionary
# e.g. model = models['resnet50']
models = {}
for model in [alexnet, densenet, inception, vgg, resnet, squeezenet, resnet3d]:
    model_dict = module_to_dict(model)
    for key, value in model_dict.items():
        models[key] = value


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def get_cnn(model_name: str, in_channels: int = 3, reload_imagenet: bool = True, num_classes: int = 1000,
            freeze: bool = False, pos: np.ndarray = None, neg: np.ndarray = None, final_bn: bool=False,
            **kwargs):
    """ Initializes a pretrained CNN from Torchvision.

    Currently supported models:
    AlexNet, DenseNet, Inception, VGGXX, ResNets, SqueezeNets, and Resnet3Ds (not torchvision)

    Args:
        model_name (str):
        in_channels (int): number of input channels. If not 3, the per-channel weights will be averaged and replicated
            in_channels times
        reload_imagenet (bool): if True, reload imagenet weights from Torchvision
        num_classes (int): number of output classes (neurons in final FC layer)
        freeze (bool): if true, model weights will be freezed
        pos (np.ndarray): number of positive examples in training set. Used for custom bias initialization in
            final layer
        neg (np.ndarray): number of negative examples in training set. Used for custom bias initialization in
            final layer
        **kwargs (): passed to model initialization function

    Returns:
        model: a pytorch CNN

    """
    model = models[model_name](pretrained=reload_imagenet, in_channels=in_channels, **kwargs)

    if freeze:
        log.info('Before freezing: {:,}'.format(utils.get_num_parameters(model)))
        for param in model.parameters():
            param.requires_grad = False
        log.info('After freezing: {:,}'.format(utils.get_num_parameters(model)))

    # we have to use the pop function because the final layer in these models has different names
    model, num_features, final_layer = pop(model, model_name, 1)
    linear_layer = nn.Linear(num_features, num_classes, bias=not final_bn)
    modules = [model, linear_layer]
    if final_bn:
        bn_layer = nn.BatchNorm1d(num_classes)
        modules.append(bn_layer)
    # initialize bias to roughly approximate the probability of positive examples in the training set
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    if pos is not None and neg is not None:
        with torch.no_grad():
            bias = np.nan_to_num(np.log(pos / neg), neginf=0.0)
            bias = torch.nn.Parameter(torch.from_numpy(bias).float())
            if final_bn:
                bn_layer.bias = bias
            else:
                linear_layer.bias = bias
            log.info('Custom bias: {}'.format(bias))


    model = nn.Sequential(*modules)
    return model
