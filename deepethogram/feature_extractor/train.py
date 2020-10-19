import logging
import os
import sys
import time
from typing import Union, Type

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm, trange

from deepethogram import utils, viz
from deepethogram.dataloaders import get_dataloaders_from_cfg
from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
from deepethogram.metrics import Classification
from deepethogram.projects import get_weightfile_from_cfg
from deepethogram.schedulers import initialize_scheduler
from deepethogram.stoppers import get_stopper
from .losses import BCELossCustom
from .models.CNN import get_cnn
from .models.hidden_two_stream import HiddenTwoStream, FlowOnlyClassifier

# flow_generators = utils.get_models_from_module(flow_models, get_function=False)
plt.switch_backend('agg')

# which GPUs should be available for training? I use 0,1 here manually because GPU2 is a tiny one for my displays
n_gpus = torch.cuda.device_count()
# DEVICE_IDS = [i for i in range(n_gpus)]
# DEVICE_IDS = [0, 1]

cudnn.benchmark = True
log = logging.getLogger(__name__)
# cudnn.benchmark = False
cudnn.deterministic = False


@hydra.main(config_path='../conf/feature_extractor_train.yaml')
def main(cfg: DictConfig) -> None:
    log.info('cwd: {}'.format(os.getcwd()))
    # only two custom overwrites of the configuration file
    # first, change the project paths from relative to absolute

    cfg = utils.get_absolute_paths_from_cfg(cfg)
    # second, use the model directory to find the most recent run of each model type
    # cfg = projects.overwrite_cfg_with_latest_weights(cfg, cfg.project.model_path, model_type='flow_generator')
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('configuration used ~~~~~')
    log.info(cfg.pretty())

    try:
        model = train_from_cfg(cfg)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        raise


def build_model_from_cfg(cfg: DictConfig, return_components: bool = False,
                         pos: np.ndarray = None, neg: np.ndarray = None) -> Union[Type[nn.Module], tuple]:
    """ Builds feature extractor from a configuration object.

    Parameters
    ----------
    cfg: DictConfig
        configuration, e.g. from Hydra command line
    return_components: bool
        if True, returns spatial classifier and flow classifier individually
    pos: np.ndarray
        Number of positive examples in dataset. Used for initializing biases in final layer
    neg: np.ndarray
        Number of negative examples in dataset. Used for initializing biases in final layer

    Returns
    -------
    if `return_components`:
        spatial_classifier, flow_classifier: nn.Module, nn.Module
            cnns for classifying rgb images and optic flows
    else:
        hidden two stream model: nn.Module
            hidden two stream CNN
    """
    device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
    feature_extractor_weights = get_weightfile_from_cfg(cfg, 'feature_extractor')
    num_classes = len(cfg.project.class_names)

    # if feature_extractor_weights is None:
    #     # we get the dataloaders here just for the pos and negative example fields of this dictionary. This allows us
    #     # to build our models with initialization based on the class imbalance of our dataset
    #     dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
    #                                            input_images=cfg.feature_extractor.n_flows + 1)
    # else:
    #     dataloaders = {'pos': None, 'neg': None}

    in_channels = cfg.feature_extractor.n_rgb * 3 if '3d' not in cfg.feature_extractor.arch else 3
    reload_imagenet = feature_extractor_weights is None
    if cfg.feature_extractor.arch == 'resnet3d_34':
        assert feature_extractor_weights is not None, 'Must specify path to resnet3d weights!'
    spatial_classifier = get_cnn(cfg.feature_extractor.arch, in_channels=in_channels,
                                 dropout_p=cfg.feature_extractor.dropout_p,
                                 num_classes=num_classes, reload_imagenet=reload_imagenet,
                                 pos=pos, neg=neg)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        spatial_classifier = utils.load_feature_extractor_components(spatial_classifier, feature_extractor_weights,
                                                                     'spatial', device=device)
    in_channels = cfg.feature_extractor.n_flows * 2 if '3d' not in cfg.feature_extractor.arch else 2
    flow_classifier = get_cnn(cfg.feature_extractor.arch, in_channels=in_channels,
                              dropout_p=cfg.feature_extractor.dropout_p,
                              num_classes=num_classes, reload_imagenet=reload_imagenet,
                              pos=pos, neg=neg)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        flow_classifier = utils.load_feature_extractor_components(flow_classifier, feature_extractor_weights,
                                                                  'flow', device=device)
    if return_components:
        return spatial_classifier, flow_classifier

    flow_generator = build_flow_generator(cfg)
    flow_weights = get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    flow_generator = utils.load_weights(flow_generator, flow_weights, device=device)
    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, cfg.feature_extractor.arch,
                            fusion_style=cfg.feature_extractor.fusion,
                            num_classes=num_classes)
    model.set_mode('classifier')
    return model


def train_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
    """ train DeepEthogram feature extractors from a configuration object.

    Args:
        cfg (DictConfig): configuration object generated by Hydra

    Returns:
        trained feature extractor
    """
    rundir = os.getcwd()  # done by hydra

    device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
    if device != 'cpu': torch.cuda.set_device(device)

    flow_generator = build_flow_generator(cfg)
    flow_weights = get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    log.info('loading flow generator from file {}'.format(flow_weights))

    flow_generator = utils.load_weights(flow_generator, flow_weights, device=device)
    flow_generator = flow_generator.to(device)

    dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
                                           input_images=cfg.feature_extractor.n_flows + 1)

    spatial_classifier, flow_classifier = build_model_from_cfg(cfg, return_components=True,
                                                               pos=dataloaders['pos'], neg=dataloaders['neg'])
    spatial_classifier = spatial_classifier.to(device)

    flow_classifier = flow_classifier.to(device)
    num_classes = len(cfg.project.class_names)


    utils.save_dict_to_yaml(dataloaders['split'], os.path.join(rundir, 'split.yaml'))

    criterion = get_criterion(cfg.feature_extractor.final_activation, dataloaders, device)
    steps_per_epoch = dict(cfg.train.steps_per_epoch)
    metrics = get_metrics(rundir, num_classes=num_classes,
                          num_parameters=utils.get_num_parameters(spatial_classifier))

    dali = cfg.compute.dali

    # training in a curriculum goes as follows:
    # first, we train the spatial classifier, which takes still images as input
    # second, we train the flow classifier, which generates optic flow with the flow_generator model and then classifies
    # it. Thirdly, we will train the whole thing end to end
    # Without the curriculum we just train end to end from the start
    if cfg.feature_extractor.curriculum:
        del dataloaders
        # train spatial model, then flow model, then both end-to-end
        dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
                                               input_images=cfg.feature_extractor.n_rgb)
        log.info('Num training batches {}, num val: {}'.format(len(dataloaders['train']), len(dataloaders['val'])))
        # we'll use this to visualize our data, because it is loaded z-scored. we want it to be in the range [0-1] or
        # [0-255] for visualization, and for that we need to know mean and std
        normalizer = get_normalizer(cfg, input_images=cfg.feature_extractor.n_rgb)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, spatial_classifier.parameters()), lr=cfg.train.lr,
                               weight_decay=cfg.feature_extractor.weight_decay)

        spatialdir = os.path.join(rundir, 'spatial')
        if not os.path.isdir(spatialdir):
            os.makedirs(spatialdir)
        stopper = get_stopper(cfg)
        # we're using validation loss as our key metric
        scheduler = initialize_scheduler(optimizer, cfg, mode='min', reduction_factor=cfg.train.reduction_factor)

        log.info('key metric: {}'.format(metrics.key_metric))
        spatial_classifier = train(spatial_classifier,
                                   dataloaders,
                                   criterion,
                                   optimizer,
                                   metrics,
                                   scheduler,
                                   spatialdir,
                                   stopper,
                                   device,
                                   steps_per_epoch,
                                   final_activation=cfg.feature_extractor.final_activation,
                                   sequence=False,
                                   normalizer=normalizer,
                                   dali=dali)

        log.info('Training flow stream....')
        input_images = cfg.feature_extractor.n_flows + 1
        del dataloaders
        dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
                                               input_images=input_images)

        normalizer = get_normalizer(cfg, input_images=input_images)
        log.info('Num training batches {}, num val: {}'.format(len(dataloaders['train']), len(dataloaders['val'])))
        flowdir = os.path.join(rundir, 'flow')
        if not os.path.isdir(flowdir):
            os.makedirs(flowdir)

        flow_generator_and_classifier = FlowOnlyClassifier(flow_generator, flow_classifier).to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, flow_classifier.parameters()), lr=cfg.train.lr,
                               weight_decay=cfg.feature_extractor.weight_decay)

        stopper = get_stopper(cfg)
        # we're using validation loss as our key metric
        scheduler = initialize_scheduler(optimizer, cfg, mode='min', reduction_factor=cfg.train.reduction_factor)
        flow_generator_and_classifier = train(flow_generator_and_classifier,
                                              dataloaders,
                                              criterion,
                                              optimizer,
                                              metrics,
                                              scheduler,
                                              flowdir,
                                              stopper,
                                              device,
                                              steps_per_epoch,
                                              final_activation=cfg.feature_extractor.final_activation,
                                              sequence=False,
                                              normalizer=normalizer,
                                              dali=dali)
        flow_classifier = flow_generator_and_classifier.flow_classifier
        # overwrite checkpoint
        utils.checkpoint(flow_classifier, flowdir, stopper.epoch_counter)

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, cfg.feature_extractor.arch,
                            fusion_style=cfg.feature_extractor.fusion,
                            num_classes=num_classes).to(device)
    # setting the mode to end-to-end would allow to backprop gradients into the flow generator itself
    # the paper does this, but I don't expect that users would have enough data for this to make sense
    model.set_mode('classifier')
    log.info('Training end to end...')
    input_images = cfg.feature_extractor.n_flows + 1
    dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
                                           input_images=input_images)
    normalizer = get_normalizer(cfg, input_images=input_images)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr,
                           weight_decay=cfg.feature_extractor.weight_decay)
    stopper = get_stopper(cfg)
    # we're using validation loss as our key metric
    scheduler = initialize_scheduler(optimizer, cfg, mode='min', reduction_factor=cfg.train.reduction_factor)
    log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(model)))
    model = train(model,
                  dataloaders,
                  criterion,
                  optimizer,
                  metrics,
                  scheduler,
                  rundir,
                  stopper,
                  device,
                  steps_per_epoch,
                  final_activation=cfg.feature_extractor.final_activation,
                  sequence=False,
                  normalizer=normalizer,
                  dali=dali)
    utils.save_hidden_two_stream(model, rundir, dict(cfg), stopper.epoch_counter)
    return model


def get_normalizer(cfg: DictConfig, input_images: int = 1):
    """ Returns an object for normalizing / denormalizing images.

    Example:
        # images from dataloaders should be z-scored: channel-wise mean ~=0, std ~= 1
        batch = next(dataloader)
        print(batch.mean(dim=[0,2,3])) # should be around 0
         # after denormalization, images should have min 0 and max 1, for saving, visualization, etc.
        unnormalized = normalizer.denormalize(batch)
        print(unnormalized.mean(dim=[0,2,3])) # ~0.3-0.5, depending on the data

    Args:
        cfg (DictConfig): hydra configuration
        input_images (int): number of images expected. You'll only have 3 channels in your mean, probably: for R,G,B
            but if you stack a bunch of RGB frames together, we need to subtract R,G,B,R,G,B... etc

    Returns:
        normalizer object
    """
    mode = '3d' if '3d' in cfg.feature_extractor.arch.lower() else '2d'
    if mode == '3d':
        log.info('3D convolution type detected: overriding input images to 1')
        input_images = 1
    rgb_mean = list(cfg.augs.normalization.mean) * input_images
    rgb_std = list(cfg.augs.normalization.std) * input_images
    return utils.Normalizer(mean=rgb_mean, std=rgb_std)


def get_criterion(final_activation: str, dataloaders: dict, device):
    """ Get loss function based on final activation.

    If final activation is softmax: use cross-entropy loss
    If final activation is sigmoid: use BCELoss

    Dataloaders are used to store weights based on dataset statistics, for up-weighting rare classes, etc.

    Args:
        final_activation (str): [softmax, sigmoid]
        dataloaders (dict): dictionary with keys ['train', 'val', 'test', 'loss_weight', 'pos_weight'], e.g. those
            returned from deepethogram/dataloaders.py # get_dataloaders, get_video_dataloaders, etc.
        device (str, torch.device): gpu device

    Returns:
        criterion (nn.Module): loss function
    """
    if final_activation == 'softmax':
        if 'weight' in list(dataloaders.keys()):
            weight = dataloaders['loss_weight']
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    elif final_activation == 'sigmoid':
        pos_weight = dataloaders['pos_weight']
        if type(pos_weight) == np.ndarray:
            pos_weight = torch.from_numpy(pos_weight).to(device)
        criterion = BCELossCustom(pos_weight=pos_weight).to(device)
    return criterion


def get_metrics(rundir: Union[str, bytes, os.PathLike], num_classes: int, num_parameters: Union[int, float],
                is_kinetics: bool = False, key_metric='loss'):
    """ get metrics object for classification. See deepethogram/metrics.py.

    In brief, it's a Metrics object that provides utilities for computing metrics over predictions, saving various
        metrics to disk, tracking your learning rate across epochs, etc.

    Args:
        rundir (str, os.PathLike): path to this run's directory for saving
        num_classes (int): number of classes in our dataset. Needed to compute some metrics
        num_parameters (int): number of parameters in model (will be saved)
        is_kinetics (bool): if true, don't make confusion matrices
        key_metric (str): the key metric will be used for learning rate scheduling and stopping

    Returns:
        Classification metrics object
    """
    metric_list = ['accuracy', 'mean_class_accuracy', 'f1']
    if not is_kinetics:
        metric_list.append('confusion')
    log.info('key metric: {}'.format(key_metric))
    metrics = Classification(rundir, key_metric, num_parameters,
                             num_classes=num_classes,
                             metrics=metric_list,
                             evaluate_threshold=True)
    return metrics


def train(model: Type[nn.Module],
          dataloaders: dict,
          criterion,
          optimizer,
          metrics,
          scheduler,
          rundir: Union[str, bytes, os.PathLike],
          stopper,
          device: torch.device,
          steps_per_epoch: dict,
          final_activation: str = 'sigmoid',
          sequence: bool = False,
          class_names: list = None,
          normalizer=None,
          dali: bool = False):
    """ Train feature extractor models

    Args:
        model (nn.Module): feature extractor (can also be a component, like the spatial stream or flow stream)
        dataloaders (dict): dictionary with PyTorch dataloader objects (see dataloaders.py)
        criterion (nn.Module): loss function
        optimizer (torch.optim): optimizer (SGD, SGDM, ADAM, etc)
        metrics (Metrics): metrics object for computing metrics and saving to disk (see metrics.py)
        scheduler (_LRScheduler): learning rate scheduler (see schedulers.py)
        rundir (str, os.PathLike): run directory for saving weights
        stopper (Stopper): object that stops training (see stoppers.py)
        device (str, torch.device): gpu device
        steps_per_epoch (dict): keys ['train', 'val', 'test']: number of steps in each "epoch"
        final_activation (str): either sigmoid or softmax
        sequence (bool): if True, assumes sequence inputs of shape N,K,T
        class_names (list): unused
        normalizer (Normalizer): normalizer object, used for un-zscoring images for visualization purposes

    Returns:
        model: a trained model
    """
    # check our inputs
    assert (isinstance(model, nn.Module))
    assert (isinstance(criterion, nn.Module))
    assert (isinstance(optimizer, torch.optim.Optimizer))

    # loop over number of epochs!
    for epoch in trange(0, stopper.num_epochs):
        # if our learning rate scheduler plateaus when validation metric saturates, we have to pass our "key metric" for
        # our validation set. Else, just step every epoch
        if scheduler.name == 'plateau' and epoch > 0:
            if hasattr(metrics, 'latest_key'):
                if 'val' in list(metrics.latest_key.keys()):
                    scheduler.step(metrics.latest_key['val'])
        elif epoch > 0:
            scheduler.step()
        # update the learning rate for this epoch
        min_lr = utils.get_minimum_learning_rate(optimizer)
        # store the learning rate for this epoch in our metrics file
        # print('min lr: {}'.format(min_lr))
        metrics.update_lr(min_lr)

        # loop over our training set!
        metrics, _ = loop_one_epoch(dataloaders['train'], model, criterion, optimizer, metrics, final_activation,
                                    steps_per_epoch['train'], train_mode=True, device=device, dali=dali)

        # evaluate on validation set
        with torch.no_grad():
            metrics, examples = loop_one_epoch(dataloaders['val'], model, criterion, optimizer, metrics,
                                               final_activation, steps_per_epoch['val'],
                                               train_mode=False, sequence=sequence, device=device,
                                               normalizer=normalizer, dali=dali)

            # some training protocols do not have test sets, so just reuse validation set for testing inference speed
            key = 'test' if 'test' in dataloaders.keys() else 'val'
            loader = dataloaders[key]
            # evaluate how fast inference takes, without loss calculation, which for some models can have a significant
            # speed impact
            metrics = speedtest(loader, model, metrics, steps_per_epoch['test'], device=device, dali=dali)

        # use our metrics file to output graphs for this epoch
        viz.visualize_logger(metrics.fname, examples if len(examples) > 0 else None)

        # save a checkpoint
        utils.checkpoint(model, rundir, epoch)
        # if should_update_latest_models:
        #     projects.write_latest_model(config['model'], config['classifier'], rundir, config)
        # input the latest validation loss to the early stopper
        if stopper.name == 'early':
            should_stop, _ = stopper(metrics.latest_key['val'])
        elif stopper.name == 'learning_rate':
            should_stop = stopper(min_lr)
        else:
            raise ValueError('Please select a stopping type')

        if should_stop:
            log.info('Stopping criterion reached!')
            break

    return model


def loop_one_epoch(loader, model, criterion, optimizer, metrics, final_activation, steps_per_epoch,
                   train_mode=True, device=None, sequence: bool = False, normalizer=None, supervised: bool = True,
                   dali: bool = False):
    """ Loops through one epoch of the data for training or validation.

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        pytorch dataloader. Will be either train or validation depending on value of train_mode
    model: torch.nn.Module
        CNN model
    criterion: nn.Module
        loss function
    optimizer: pytorch optimizer
        Optimizer object for updating weights given model parameters and gradients
    metrics: deepethogram.metrics.Metrics object
        instance of a Metrics class used to write metrics to disk
    final_activation: str
        either sigmoid of softmax
    steps_per_epoch: int
        How many steps to run before automatically breaking. If None, loop through entire `loader`
    train_mode: bool
        if True, update model parameters at each step. If False, don't compute gradients or update weights
    device: str, torch.Device
        GPU device to move data to
    sequence: bool
        if True, expect different shape of inputs and outputs for loss function
    normalizer: Normalizer
        Object that can z-score or un-z-score images. Used here for visualization
    supervised: bool
        if True, expect labels to be a part of dataloader

    Returns
    -------
    metrics: Metrics object
        metrics with updated values for this epoch
    examples: list
        list of images showing example figures for saving to disk
    """
    if train_mode:
        # make sure we're in train mode
        model.train()
    else:
        model.eval()
    if final_activation == 'softmax':
        activation = nn.Softmax(dim=1)
    elif final_activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        raise NotImplementedError
    # if steps per epoch is not none, make an iterable of range N (e.g. 1000 minibatches)
    num_iters = len(loader) if steps_per_epoch is None else min(steps_per_epoch, len(loader))
    t = tqdm(range(0, num_iters), leave=False)
    dataiter = iter(loader)
    mode = 'train' if train_mode else 'val'
    cnt = 0
    examples = []
    has_logged = False
    for i in t:
        t0 = time.time()
        try:
            batch = next(dataiter)
        except StopIteration:
            break

        if not dali:
            # put everything in the batch on the gpu
            batch = [i.to(device) for i in batch]
            inputs, labels = batch
        else:
            # dali should only be used
            inputs, labels = batch[0]['images'], batch[0]['labels']
            labels = labels.squeeze()

        # will print out shape and min, mean, max, std along image channels
        # we use the isEnabledFor flag so that this doesnt slow down training in the non-debug case
        if not has_logged and log.isEnabledFor(logging.DEBUG):
            if len(inputs.shape) == 4:
                N, C, H, W = inputs.shape
                log.debug('inputs shape: NCHW: {} {} {} {}'.format(N, C, H, W))
                log.debug('channel min:  {}'.format(inputs[0].reshape(C, -1).min(dim=1).values))
                log.debug('channel mean: {}'.format(inputs[0].reshape(C, -1).mean(dim=1)))
                log.debug('channel max : {}'.format(inputs[0].reshape(C, -1).max(dim=1).values))
                log.debug('channel std : {}'.format(inputs[0].reshape(C, -1).std(dim=1)))
            elif len(inputs.shape) == 5:
                N, C, T, H, W = inputs.shape
                log.debug('inputs shape: NCTHW: {} {} {} {} {}'.format(N, C, T, H, W))
                log.debug('channel min:  {}'.format(inputs[0].min(dim=2).values))
                log.debug('channel mean: {}'.format(inputs[0].mean(dim=2)))
                log.debug('channel max : {}'.format(inputs[0].max(dim=2).values))
                log.debug('channel std : {}'.format(inputs[0].std(dim=2)))
            has_logged = True

        # forward pass
        if train_mode:
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        log.debug('outputs: {}'.format(outputs))
        log.debug('labels: {}'.format(labels))
        log.debug('outputs: {}'.format(outputs.shape))
        log.debug('labels: {}'.format(labels.shape))
        log.debug('label max: {}'.format(labels.max()))
        log.debug('label min: {}'.format(labels.min()))
        # if torch.sum(labels < 0) > 0:
        #     log.warning('negative value found in labels')
        #     import pdb
        #     pdb.set_trace()
        loss = criterion(outputs, labels)
        predictions = activation(outputs)
        metrics.batch_append(predictions.detach(), labels.detach())
        metrics.loss_append(loss.item())

        if train_mode:
            # zero the parameter gradients
            optimizer.zero_grad()
            # calculate gradients
            loss.backward()
            # step in direction of gradients according to optimizer
            optimizer.step()
        else:
            if cnt < 10:
                if sequence:
                    # make sequence figures
                    fig = plt.figure(figsize=(14, 14))
                    viz.visualize_batch_sequence(inputs, predictions, labels, fig=fig)
                    img = viz.fig_to_img(fig)
                    examples.append(img)
                    plt.close(fig)
                elif hasattr(model, 'flow_generator'):
                    fig = plt.figure(figsize=(14, 14))
                    # re-compute optic flows for this batch for visualization
                    with torch.no_grad():
                        flows = model.flow_generator(inputs)
                    viz.visualize_hidden(inputs, flows, predictions, labels, fig=fig, normalizer=normalizer)
                    img = viz.fig_to_img(fig)
                    examples.append(img)
                    plt.close(fig)

        # torch.cuda.synchronize()
        time_per_image = (time.time() - t0) / inputs.shape[0]

        metrics.time_append(time_per_image)
        metrics.loss_append(loss.item())

        t.set_description('{} loss: {:.4f}'.format(mode, loss.item()))
        cnt += 1
    metrics.end_epoch(mode)
    return metrics, examples


def speedtest(loader, model, metrics, steps, supervised: bool = True, device=None, dali: bool = False):
    """ Loop through loader and compute model predictions with no loss function calculation. Approximates inference
    speed.
    """


    model.eval()

    # if steps per epoch is not none, make an iterable of range N (e.g. 1000 minibatches)
    # print(len(loader))
    num_iters = len(loader) if steps is None else min(steps, len(loader))
    t = tqdm(range(0, num_iters), leave=False)
    dataiter = iter(loader)

    cnt = 0
    for i in t:
        t0 = time.time()
        try:
            batch = next(dataiter)
        except StopIteration:
            break

        if not dali:
            # put everything in the batch on the gpu
            batch = [i.to(device) for i in batch]
            inputs, labels = batch
        else:
            inputs = batch[0]['images']

        # in case steps_per_epoch is more than number of batches in dataloader
        if inputs is None:
            break

        with torch.no_grad():
            outputs = model(inputs)

        # N,C,H,W = images.shape
        num_images = inputs.shape[0]
        time_per_image = (time.time() - t0) / (num_images + 1e-7)
        metrics.time_append(time_per_image)
        t.set_description('FPS: {:.2f}'.format(1 / (time_per_image + 1e-7)))
    metrics.end_epoch_speedtest()
    return metrics


if __name__ == '__main__':
    sys.argv = utils.process_config_file_from_cl(sys.argv)
    main()
