import logging
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from deepethogram import utils
from deepethogram.feature_extractor.models.CNN import get_cnn
from deepethogram.flow_generator import models as flow_models
from .utils import Fusion, remove_cnn_classifier_layer

flow_generators = utils.get_models_from_module(flow_models, get_function=False)

log = logging.getLogger(__name__)


class Viewer(nn.Module):
    """ PyTorch module for extracting the middle image of a concatenated stack.

    Example: you have 10 RGB images stacked in a channel of a tensor, so it has shape [N, 30, H, W].
        viewer = Viewer(10)
        middle = viewer(tensor)
        print(middle.shape) # [N, 3, H, W], taken from channels 15:18

    """

    def __init__(self, num_images, label_location):
        super().__init__()
        self.num_images = num_images
        if label_location == 'middle':
            self.start = int(num_images / 2 * 3)
        elif label_location == 'causal':
            self.start = int(num_images * 3 - 3)
        self.end = int(self.start + 3)

    def forward(self, x):
        x = x[:, self.start:self.end, :, :]
        return x


class FlowOnlyClassifier(nn.Module):
    """ Stack of flow generator module and flow classifier. Used in training Hidden two stream networks in a curriculum

    Takes a stack of images as inputs. Generates optic flow using the flow generator; computes class probabilities
    using the flow classifier.

    Example:
        flow_classifier = resnet18()
        flow_generator = TinyMotionNet()

        flow_only_classifier = FlowOnlyClassifier(flow_generator, flow_classifier)
        images = torch.rand(N, C, H, W)
        outputs = flow_only_classifier(images)
        print(outputs.shape) # [N, K]
    """

    def __init__(self, flow_generator,
                 flow_classifier,
                 freeze_flow_generator: bool = True):
        super().__init__()
        assert (isinstance(flow_generator, nn.Module) and isinstance(flow_classifier, nn.Module))

        self.flow_generator = flow_generator
        if freeze_flow_generator:
            for param in self.flow_generator.parameters():
                param.requires_grad = False
        self.flow_classifier = flow_classifier
        self.flow_generator.eval()

    def forward(self, batch):
        # never backpropagate through the flow generator. Much faster this way, and uses less VRAM
        with torch.no_grad():
            flows = self.flow_generator(batch)
        # flows will return a tuple of flows at different resolutions. Even in eval mode. 0th flow should be
        # at original image resolution
        return self.flow_classifier(flows[0])


class HiddenTwoStream(nn.Module):
    """ Hidden Two-Stream Network model
    Paper: https://arxiv.org/abs/1704.00389

    Classifies video inputs using a spatial CNN, using RGB video frames as inputs; and using a flow CNN, which
    uses optic flow as inputs. Optic flow is generated on-the-fly by a flow generator network. This has distinct
    advantages, as optic flow loaded from disk is both more discrete and has compression artifacts.
    """

    def __init__(self, flow_generator, spatial_classifier, flow_classifier,
                 classifier_name: str, num_images: int = 11,
                 num_classes: int = 1000,
                 label_location: str = 'middle',
                 fusion_style: str = 'concatenate',
                 flow_fusion_weight: float = 1.5):
        """ Hidden two-stream constructor.

        Args:
            flow_generator (nn.Module): CNN that generates optic flow from a stack of RGB frames
            spatial_classifier (nn.Module): CNN that classifies original RGB inputs
            flow_classifier (nn.Module): CNN that classifies optic flow inputs
            classifier_name (str): name of CNN (e.g. resnet18) used in both classifiers
            num_images (int): number of input images to the flow generator. Flow outputs will be num_images - 1
            num_classes (int): number of classes
            label_location (str): either middle or causal. Middle: the label will be selected from the middle of a
                stack of image frames. Causal: the label will come from the last image in the stack (no look-ahead)
            fusion_style (str): [average, concatenate] Average: logits will be averaged together (with weight on the
                flow stream, per
            flow_fusion_weight (float): how much to up-weight flow fusion. Set to 1.5 for the flow fusion, according
                to the hidden two-stream paper
        """
        super().__init__()
        assert (isinstance(flow_generator, nn.Module) and isinstance(spatial_classifier, nn.Module)
                and isinstance(flow_classifier, nn.Module))

        if fusion_style == 'average':
            # just so we can pass them to the fusion module
            num_spatial_features, num_flow_features = None, None
        elif fusion_style == 'concatenate':
            spatial_classifier, num_spatial_features = remove_cnn_classifier_layer(spatial_classifier)
            flow_classifier, num_flow_features = remove_cnn_classifier_layer(flow_classifier)
        else:
            raise ValueError('unknown fusion style: {}'.format(fusion_style))

        self.spatial_classifier = spatial_classifier
        self.flow_generator = flow_generator
        self.flow_classifier = flow_classifier
        if '3d' in classifier_name:
            self.viewer = nn.Identity()
        else:
            # self.viewer = torch.jit.script(Viewer(num_images, label_location))
            self.viewer = Viewer(num_images, label_location)
        self.fusion = Fusion(fusion_style, num_spatial_features, num_flow_features, num_classes,
                             flow_fusion_weight=flow_fusion_weight)

        self.frozen_state = {}
        self.freeze('flow_generator')

    def freeze(self, submodel_to_freeze: str):
        """ Freezes a component of the model. Useful for curriculum training

        Args:
            submodel_to_freeze (str): one of flow_generator, spatial, flow, fusion
        """
        if submodel_to_freeze == 'flow_generator':
            self.flow_generator.eval()
            for param in self.flow_generator.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'spatial':
            self.spatial_classifier.eval()
            for param in self.spatial_classifier.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'flow':
            self.flow_classifier.eval()
            for param in self.flow_classifier.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'fusion':
            self.fusion.eval()
            for param in self.fusion.parameters():
                param.requires_grad = False
        else:
            raise ValueError('submodel not found:%s' % submodel_to_freeze)
        self.frozen_state[submodel_to_freeze] = True

    def set_mode(self, mode: str):
        """ Freezes and unfreezes portions of the model, useful for curriculum training.

        Args:
            mode (str): one of spatial_only, flow_only, fusion_only, classifier, end_to_end, or inference
        """
        log.debug('setting model mode: {}'.format(mode))
        if mode == 'spatial_only':
            self.freeze('flow_generator')
            self.freeze('flow')
            self.freeze('fusion')
            self.unfreeze('spatial')
        elif mode == 'flow_only':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.unfreeze('flow')
            self.freeze('fusion')
        elif mode == 'fusion_only':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.freeze('flow')
            self.unfreeze('fusion')
        elif mode == 'classifier':
            self.freeze('flow_generator')
            self.unfreeze('spatial')
            self.unfreeze('flow')
            self.unfreeze('fusion')
        elif mode == 'end_to_end':
            self.unfreeze('flow_generator')
            self.unfreeze('spatial')
            self.unfreeze('flow')
            self.unfreeze('fusion')
        elif mode == 'inference':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.freeze('flow')
            self.freeze('fusion')
        else:
            raise ValueError('Unknown mode: %s' % mode)

    def unfreeze(self, submodel_to_unfreeze: str):
        """ Unfreezes portions of the model

        Args:
            submodel_to_unfreeze (str): one of flow_generator, spatial, flow, or fusion

        Returns:

        """
        log.debug('unfreezing model component: {}'.format(submodel_to_unfreeze))
        if submodel_to_unfreeze == 'flow_generator':
            self.flow_generator.train()
            for param in self.flow_generator.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'spatial':
            self.spatial_classifier.train()
            for param in self.spatial_classifier.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'flow':
            self.flow_classifier.train()
            for param in self.flow_classifier.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'fusion':
            self.fusion.train()
            for param in self.fusion.parameters():
                param.requires_grad = True
        else:
            raise ValueError('submodel not found:%s' % submodel_to_unfreeze)
        self.frozen_state[submodel_to_unfreeze] = False

    def get_param_groups(self):
        param_list = [{'params': self.flow_generator.parameters()},
                      {'params': self.spatial_classifier.parameters()},
                      {'params': self.flow_classifier.parameters()},
                      {'params': self.fusion.parameters()}]
        return (param_list)

    def forward(self, batch):
        with torch.no_grad():
            flows = self.flow_generator(batch)
        RGB = self.viewer(batch)
        spatial_features = self.spatial_classifier(RGB)
        # flows[0] because flow returns a pyramid of spatial resolutions, zero being the highest res
        flow_features = self.flow_classifier(flows[0])
        return self.fusion(spatial_features, flow_features)


def hidden_two_stream(classifier: str,
                      flow_gen: str,
                      num_classes: int,
                      fusion_style: str = 'average',
                      dropout_p: float = 0.9,
                      reload_imagenet: bool = True,
                      num_rgb: int = 1,
                      num_flows: int = 10,
                      pos: np.ndarray = None,
                      neg: np.ndarray = None,
                      flow_max: float = 5.0,
                      **kwargs):
    """ Wrapper for initializing hidden two stream models

    Args:
        classifier (str): a supported classifier, e.g. resnet18, vgg16
        flow_gen (str): the name of a flow generator
        num_classes (int): number of output classes
        fusion_style (str): one of average, concatenate
        dropout_p (float): amount of dropout to place on penultimate layer of each classifier
        reload_imagenet (bool): whether or not to load classifier with ImageNet weights, courtesy of PyTorch
        num_rgb (int): number of RGB frames to input to the spatial model
        num_flows (int): number of flows to output from the flow generator, input to the flow classifier
        pos (np.ndarray): shape (K,) number of positive examples in training set, used for bias initialization
        neg (np.ndarray): shape (K,) number of negative examples in training set, used for bias initialization
        flow_max (float): maximum flow. not used

    Returns:
        hidden two stream network model
    """
    assert fusion_style in ['average', 'concatenate']

    flow_generator = flow_generators[flow_gen](num_images=num_flows + 1, flow_div=flow_max)

    in_channels = num_rgb * 3 if '3d' not in classifier.lower() else 3
    spatial_classifier = get_cnn(classifier, in_channels=in_channels, dropout_p=dropout_p,
                                 num_classes=num_classes, reload_imagenet=reload_imagenet,
                                 pos=pos, neg=neg, **kwargs)

    in_channels = num_flows * 2 if '3d' not in classifier.lower() else 2
    flow_classifier = get_cnn(classifier, in_channels=in_channels, dropout_p=dropout_p,
                              num_classes=num_classes, reload_imagenet=reload_imagenet,
                              pos=pos, neg=neg, **kwargs)

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, classifier,
                            fusion_style=fusion_style,
                            num_classes=num_classes)
    return model


def deg_f(num_classes: int, dropout_p: float = 0.9, reload_imagenet: bool = True,
          pos: int = None, neg: int = None):
    """ Make the DEG-fast model. Uses ResNet18 for classification, TinyMotionNet for flow generation.
    Number of flows: 10
    Number of RGB frames for classification: 1

    Args:
        num_classes (int): number of output classes
        dropout_p (float): amount of dropout to place on penultimate layer of each classifier
        reload_imagenet (bool): whether or not to load classifier with ImageNet weights, courtesy of PyTorch
        pos (np.ndarray): shape (K,) number of positive examples in training set, used for bias initialization
        neg (np.ndarray): shape (K,) number of negative examples in training set, used for bias initialization
    Returns:
        DEG-f model
    """
    classifier = 'resnet18'
    flow_gen = 'TinyMotionNet'
    num_flows = 10
    num_rgb = 1
    fusion_style = 'average'
    model = hidden_two_stream(classifier, flow_gen, num_classes,
                              fusion_style=fusion_style,
                              dropout_p=dropout_p,
                              reload_imagenet=reload_imagenet,
                              num_rgb=num_rgb,
                              num_flows=num_flows,
                              pos=pos,
                              neg=neg)

    return model


def deg_m(num_classes: int, dropout_p: float = 0.9, reload_imagenet: bool = True,
          pos: int = None, neg: int = None):
    """ Make the DEG-medium model. Uses ResNet50 for classification, MotionNet for flow generation.
    Number of flows: 10
    Number of RGB frames for classification: 1

    Args:
        num_classes (int): number of output classes
        dropout_p (float): amount of dropout to place on penultimate layer of each classifier
        reload_imagenet (bool): whether or not to load classifier with ImageNet weights, courtesy of PyTorch
        pos (np.ndarray): shape (K,) number of positive examples in training set, used for bias initialization
        neg (np.ndarray): shape (K,) number of negative examples in training set, used for bias initialization
    Returns:
        DEG-m model
    """
    classifier = 'resnet50'
    flow_gen = 'MotionNet'
    num_flows = 10
    num_rgb = 1
    fusion_style = 'average'
    model = hidden_two_stream(classifier, flow_gen, num_classes,
                              fusion_style=fusion_style,
                              dropout_p=dropout_p,
                              reload_imagenet=reload_imagenet,
                              num_rgb=num_rgb,
                              num_flows=num_flows,
                              pos=pos,
                              neg=neg)

    return model


def deg_s(num_classes: int, dropout_p: float = 0.9, reload_imagenet: bool = True,
          pos: int = None, neg: int = None, path_to_weights: Union[str, os.PathLike] = None):
    """ Make the DEG-slow model. Uses ResNet3d-34 for classification, TinyMotionNet3D for flow generation.
    Number of flows: 10
    Number of RGB frames for classification: 11

    Args:
        num_classes (int): number of output classes
        dropout_p (float): amount of dropout to place on penultimate layer of each classifier
        reload_imagenet (bool): whether or not to load classifier with ImageNet weights, courtesy of PyTorch
        pos (np.ndarray): shape (K,) number of positive examples in training set, used for bias initialization
        neg (np.ndarray): shape (K,) number of negative examples in training set, used for bias initialization
        path_to_weights (str): since PyTorch does not have pretrained resnet3d models, must specify a path
            Can be downloaded from # https://github.com/kenshohara/3D-ResNets-PyTorch
    Returns:
        DEG-s model
    """
    classifier = 'resnet3d_34'
    flow_gen = 'TinyMotionNet3D'
    num_flows = 10
    num_rgb = 11
    fusion_style = 'average'
    model = hidden_two_stream(classifier, flow_gen, num_classes,
                              fusion_style=fusion_style,
                              dropout_p=dropout_p,
                              reload_imagenet=reload_imagenet,
                              num_rgb=num_rgb,
                              num_flows=num_flows,
                              pos=pos,
                              neg=neg,
                              path_to_weights=path_to_weights)

    return model
