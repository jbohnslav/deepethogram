import gc
import logging
import os
import sys
import warnings
from typing import Union, Tuple

import cv2

cv2.setNumThreads(0)
# import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from deepethogram import utils, viz
from deepethogram.base import BaseLightningModule, get_trainer_from_cfg
from deepethogram.configuration import make_feature_extractor_train_cfg
from deepethogram.data.augs import get_gpu_transforms
from deepethogram.data.datasets import get_datasets_from_cfg
from deepethogram.feature_extractor.losses import ClassificationLoss, BinaryFocalLoss, CrossEntropyLoss
from deepethogram.feature_extractor.models.CNN import get_cnn
from deepethogram.feature_extractor.models.hidden_two_stream import HiddenTwoStream, FlowOnlyClassifier, \
    build_fusion_layer
from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
from deepethogram.losses import get_regularization_loss
from deepethogram.metrics import Classification
from deepethogram import projects
from deepethogram.stoppers import get_stopper

# hack
# https://github.com/ray-project/ray/issues/10995
os.environ["SLURM_JOB_NAME"] = "bash"

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='Your val_dataloader has `shuffle=True`, it is best practice to turn this off for validation '
    'and test dataloaders.')

# flow_generators = utils.get_models_from_module(flow_models, get_function=False)
plt.switch_backend('agg')

log = logging.getLogger(__name__)


# @profile
def feature_extractor_train(cfg: DictConfig) -> nn.Module:
    """Trains feature extractor models from a configuration. 

    Parameters
    ----------
    cfg : DictConfig
        Configuration, e.g. that returned by deepethogram.configration.make_feature_extractor_train_cfg

    Returns
    -------
    nn.Module
        Trained feature extractor
    """
    # rundir = os.getcwd()
    cfg = projects.setup_run(cfg)

    log.info('args: {}'.format(' '.join(sys.argv)))
    # change the project paths from relative to absolute
    # allow for editing
    OmegaConf.set_struct(cfg, False)
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('configuration used ~~~~~')
    log.info(OmegaConf.to_yaml(cfg))

    # we build flow generator independently because you might want to load it from a different location
    flow_generator = build_flow_generator(cfg)
    flow_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    log.info('loading flow generator from file {}'.format(flow_weights))

    flow_generator = utils.load_weights(flow_generator, flow_weights)

    _, data_info = get_datasets_from_cfg(cfg,
                                         model_type='feature_extractor',
                                         input_images=cfg.feature_extractor.n_flows + 1)

    model_parts = build_model_from_cfg(cfg, pos=data_info['pos'], neg=data_info['neg'])
    _, spatial_classifier, flow_classifier, fusion, model = model_parts
    # log.info('model: {}'.format(model))

    num_classes = len(cfg.project.class_names)

    utils.save_dict_to_yaml(data_info['split'], os.path.join(cfg.run.dir, 'split.yaml'))

    metrics = get_metrics(cfg.run.dir,
                          num_classes=num_classes,
                          num_parameters=utils.get_num_parameters(spatial_classifier),
                          key_metric='f1_class_mean_nobg',
                          num_workers=cfg.compute.metrics_workers)

    # cfg.compute.batch_size will be changed by the automatic batch size finder, possibly. store here so that
    # with each step of the curriculum, we can auto-tune it
    original_batch_size = cfg.compute.batch_size
    original_lr = cfg.train.lr

    # training in a curriculum goes as follows:
    # first, we train the spatial classifier, which takes still images as input
    # second, we train the flow classifier, which generates optic flow with the flow_generator model and then classifies
    # it. Thirdly, we will train the whole thing end to end
    # Without the curriculum we just train end to end from the start
    if cfg.feature_extractor.curriculum:
        # train spatial model, then flow model, then both end-to-end
        # dataloaders = get_dataloaders_from_cfg(cfg, model_type='feature_extractor',
        #                                        input_images=cfg.feature_extractor.n_rgb)
        datasets, data_info = get_datasets_from_cfg(cfg,
                                                    model_type='feature_extractor',
                                                    input_images=cfg.feature_extractor.n_rgb)
        stopper = get_stopper(cfg)

        criterion = get_criterion(cfg, spatial_classifier, data_info)

        lightning_module = HiddenTwoStreamLightning(spatial_classifier, cfg, datasets, metrics, criterion)
        trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
        # this horrible syntax is because we just changed our configuration's batch size and learning rate, if they are
        # set to auto. so we need to re-instantiate our lightning module
        # https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html?highlight=auto%20scale%20learning%20rate
        # I tried to do this without re-creating module, but finding the learning rate increments the epoch??
        # del lightning_module
        # log.info('epoch num: {}'.format(trainer.current_epoch))
        # lightning_module = HiddenTwoStreamLightning(spatial_classifier, cfg, datasets, metrics, criterion)
        trainer.fit(lightning_module)

        # free RAM. note: this doesn't do much
        log.info('free ram')
        del datasets, lightning_module, trainer, stopper, data_info
        torch.cuda.empty_cache()
        gc.collect()

        # return

        datasets, data_info = get_datasets_from_cfg(cfg,
                                                    model_type='feature_extractor',
                                                    input_images=cfg.feature_extractor.n_flows + 1)
        # re-initialize stopper so that it doesn't think we need to stop due to the previous model
        stopper = get_stopper(cfg)
        cfg.compute.batch_size = original_batch_size
        cfg.train.lr = original_lr

        # this class will freeze the flow generator
        flow_generator_and_classifier = FlowOnlyClassifier(flow_generator, flow_classifier)
        criterion = get_criterion(cfg, flow_generator_and_classifier, data_info)
        lightning_module = HiddenTwoStreamLightning(flow_generator_and_classifier, cfg, datasets, metrics, criterion)
        trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
        # lightning_module = HiddenTwoStreamLightning(flow_generator_and_classifier, cfg, datasets, metrics, criterion)
        trainer.fit(lightning_module)

        del datasets, lightning_module, trainer, stopper, data_info
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, fusion, cfg.feature_extractor.arch)
    model.set_mode('classifier')
    datasets, data_info = get_datasets_from_cfg(cfg,
                                                model_type='feature_extractor',
                                                input_images=cfg.feature_extractor.n_flows + 1)
    criterion = get_criterion(cfg, model, data_info)
    stopper = get_stopper(cfg)
    cfg.compute.batch_size = original_batch_size
    cfg.train.lr = original_lr

    # log.warning('SETTING ANAOMALY DETECTION TO TRUE! WILL SLOW DOWN.')
    # torch.autograd.set_detect_anomaly(True)

    lightning_module = HiddenTwoStreamLightning(model, cfg, datasets, metrics, criterion)

    trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
    # see above for horrible syntax explanation
    # lightning_module = HiddenTwoStreamLightning(model, cfg, datasets, metrics, criterion)
    trainer.fit(lightning_module)
    # trainer.test(model=lightning_module)
    return model
    # utils.save_hidden_two_stream(model, rundir, dict(cfg), stopper.epoch_counter)


def build_model_from_cfg(cfg: DictConfig,
                         pos: np.ndarray = None,
                         neg: np.ndarray = None,
                         num_classes: int = None) -> tuple:
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
    # device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    feature_extractor_weights = projects.get_weightfile_from_cfg(cfg, 'feature_extractor')
    if num_classes is None:
        num_classes = len(cfg.project.class_names)

    log.info('feature extractor weightfile: {}'.format(feature_extractor_weights))

    in_channels = cfg.feature_extractor.n_rgb * 3 if '3d' not in cfg.feature_extractor.arch else 3
    reload_imagenet = feature_extractor_weights is None
    if cfg.feature_extractor.arch == 'resnet3d_34':
        assert feature_extractor_weights is not None, 'Must specify path to resnet3d weights!'
    spatial_classifier = get_cnn(cfg.feature_extractor.arch,
                                 in_channels=in_channels,
                                 dropout_p=cfg.feature_extractor.dropout_p,
                                 num_classes=num_classes,
                                 reload_imagenet=reload_imagenet,
                                 pos=pos,
                                 neg=neg,
                                 final_bn=cfg.feature_extractor.final_bn)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        spatial_classifier = utils.load_feature_extractor_components(spatial_classifier,
                                                                     feature_extractor_weights,
                                                                     'spatial',
                                                                     device=device)
    in_channels = cfg.feature_extractor.n_flows * 2 if '3d' not in cfg.feature_extractor.arch else 2
    flow_classifier = get_cnn(cfg.feature_extractor.arch,
                              in_channels=in_channels,
                              dropout_p=cfg.feature_extractor.dropout_p,
                              num_classes=num_classes,
                              reload_imagenet=reload_imagenet,
                              pos=pos,
                              neg=neg,
                              final_bn=cfg.feature_extractor.final_bn)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        flow_classifier = utils.load_feature_extractor_components(flow_classifier,
                                                                  feature_extractor_weights,
                                                                  'flow',
                                                                  device=device)

    flow_generator = build_flow_generator(cfg)
    flow_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    flow_generator = utils.load_weights(flow_generator, flow_weights, device=device)

    spatial_classifier, flow_classifier, fusion = build_fusion_layer(spatial_classifier, flow_classifier,
                                                                     cfg.feature_extractor.fusion, num_classes)
    if feature_extractor_weights is not None:
        fusion = utils.load_feature_extractor_components(fusion, feature_extractor_weights, 'fusion', device=device)

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, fusion, cfg.feature_extractor.arch)
    # log.info(model.fusion.flow_weight)
    model.set_mode('classifier')

    return flow_generator, spatial_classifier, flow_classifier, fusion, model


class HiddenTwoStreamLightning(BaseLightningModule):
    """Lightning Module for training Feature Extractor models
    """

    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics, criterion: nn.Module):
        """constructor

        Parameters
        ----------
        model : nn.Module
            nn.Module, hidden two-stream CNNs
        cfg : DictConfig
            omegaconf configuration
        datasets : dict
            dictionary containing Dataset classes
        metrics : [type]
            metrics object for saving and computing metrics
        criterion : nn.Module
            loss function
        """
        super().__init__(model, cfg, datasets, metrics, viz.visualize_logger_multilabel_classification)

        self.has_logged_channels = False
        # for convenience
        self.final_activation = self.hparams.feature_extractor.final_activation
        if self.final_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif self.final_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.criterion = criterion

    def validate_batch_size(self, batch: dict):
        """simple check for appropriate batch sizes

        Parameters
        ----------
        batch : dict
            outputs of a DataLoader

        Returns
        -------
        batch : dict
            verified batch dictionary
        """
        if self.hparams.compute.dali:
            # no idea why they wrap this, maybe they fixed it?
            batch = batch[0]
        if 'images' in batch.keys():
            # weird case of batch size = 1 somehow getting squeezed out
            if batch['images'].ndim != 5:
                batch['images'] = batch['images'].unsqueeze(0)
        if 'labels' in batch.keys():
            if self.final_activation == 'sigmoid' and batch['labels'].ndim == 1:
                batch['labels'] = batch['labels'].unsqueeze(0)
        return batch

    def training_step(self, batch: dict, batch_idx: int):
        """Run forward pass, loss calculation, backward pass, and parameter update

        Parameters
        ----------
        batch : dict
            contains images and other information
        batch_idx : int
            index in current epoch

        Returns
        -------
        loss : torch.Tensor
            mean loss for batch for Lightning's backward + update hooks
        """
        # use the forward function
        # return the image tensor so we can visualize after gpu transforms
        images, outputs = self(batch, 'train')

        probabilities = self.activation(outputs)

        loss, loss_dict = self.criterion(outputs, batch['labels'], self.model)

        self.visualize_batch(images, probabilities, batch['labels'], 'train')

        # save the model outputs to a buffer for various metrics
        self.metrics.buffer.append('train', {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        # add the individual components of the loss to the metrics buffer
        self.metrics.buffer.append('train', loss_dict)
        # need to use the native logger for lr scheduling, etc.
        self.log('train/loss', loss.detach().cpu())
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """runs a single validation step

        Parameters
        ----------
        batch : dict
            images, labels, etc
        batch_idx : int
            index in validation epoch
        """
        images, outputs = self(batch, 'val')
        probabilities = self.activation(outputs)

        loss, loss_dict = self.criterion(outputs, batch['labels'], self.model)
        self.visualize_batch(images, probabilities, batch['labels'], 'val')
        self.metrics.buffer.append('val', {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        self.metrics.buffer.append('val', loss_dict)
        # need to use the native logger for lr scheduling, etc.
        self.log('val/loss', loss.detach().cpu())

    def test_step(self, batch: dict, batch_idx: int):
        """runs test step

        Parameters
        ----------
        batch : dict
            images, labels, etc
        batch_idx : int
            index in test epoch
        """
        images, outputs = self(batch, 'test')
        probabilities = self.activation(outputs)
        loss, loss_dict = self.criterion(outputs, batch['labels'], self.model)
        self.metrics.buffer.append('test', {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        self.metrics.buffer.append('test', loss_dict)

    def visualize_batch(self, images: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, split: str):
        """generates example images of a given batch and saves to disk as a Matplotlib figure

        Parameters
        ----------
        images : torch.Tensor
            input images
        probs : torch.Tensor
            output probabilities
        labels : torch.Tensor
            human labels
        split : str
            train, val, or test
        """
        if self.hparams.train.viz_examples == 0:
            return
        # ALWAYS VISUALIZE MODEL INPUTS JUST BEFORE FORWARD PASS
        viz_cnt = self.viz_cnt[split]
        # only save first 10 batches
        if viz_cnt > self.hparams.train.viz_examples:
            return
        # this method can be used for sequence models as well
        if hasattr(self.model, 'flow_generator'):
            with torch.no_grad():
                # re-compute optic flows for this batch for visualization
                batch_size = images.size(0)
                # don't compute flows for very large batches. only need a few random ones for
                # visualization purposes
                if batch_size > 2:
                    inds = torch.randperm(batch_size)[:2]
                    images = images[inds]
                    probs = probs[inds]
                    labels = labels[inds]

                # only output the highest res flow
                flows = self.model.flow_generator(images)[0].detach()
                inputs = self.gpu_transforms['denormalize'](images).detach()
                fig = plt.figure(figsize=(14, 14))
                viz.visualize_hidden(inputs.detach().cpu(),
                                     flows.detach().cpu(),
                                     probs.detach().cpu(),
                                     labels.detach().cpu(),
                                     fig=fig)
                # this should happen in the save figure function, but for some reason it doesn't
                viz.save_figure(fig, 'batch_with_flows', True, viz_cnt, split)
                del images, probs, labels, flows
                torch.cuda.empty_cache()
        else:
            fig = plt.figure(figsize=(14, 14))
            with torch.no_grad():
                inputs = self.gpu_transforms['denormalize'](images)
                viz.visualize_batch_spatial(inputs, probs, labels, fig=fig)
                viz.save_figure(fig, 'batch_spatial', True, viz_cnt, split)
        try:
            # should've been closed in viz.save_figure. this is double checking
            plt.close(fig)
            plt.close('all')
        except:
            pass
        torch.cuda.empty_cache()
        # self.viz_cnt[split] += 1

    def forward(self, batch: dict, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """runs forward pass, including GPU-based image augmentations

        Parameters
        ----------
        batch : dict
            images
        mode : str
            train or val, used to figure out which gpu augmenations to apply.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            [description]
        """
        batch = self.validate_batch_size(batch)
        # lightning handles transfer to device
        images = batch['images']
        # no-grad should work in the apply_gpu_transforms method; adding here just in case
        with torch.no_grad():
            # augment the input images. in training, this will perturb brightness, contrast, etc.
            # in inference, it will just convert to the right dtype and normalize
            gpu_images = self.apply_gpu_transforms(images, mode)

        if torch.sum(gpu_images != gpu_images) > 0 or torch.sum(torch.isinf(gpu_images)) > 0:
            log.error('nan in gpu augs')
            raise ValueError('nan in GPU augmentations!')
        # make sure normalization works, etc.
        self.log_image_statistics(gpu_images)

        # actually compute forward pass
        outputs = self.model(gpu_images)

        if torch.sum(outputs != outputs) > 0:
            log.error('nan in model outputs')
            raise ValueError('nan in model outputs!')

        return gpu_images, outputs

    def log_image_statistics(self, images: torch.Tensor):
        """logs the min, mean, max, and std deviation of input tensors. useful for debugging

        Parameters
        ----------
        images : torch.Tensor
            4D or 5D input images
        """
        if not self.has_logged_channels and log.isEnabledFor(logging.DEBUG):
            if len(images.shape) == 4:
                N, C, H, W = images.shape
                log.debug('inputs shape: NCHW: {} {} {} {}'.format(N, C, H, W))
                log.debug('channel min:  {}'.format(images[0].reshape(C, -1).min(dim=1).values))
                log.debug('channel mean: {}'.format(images[0].reshape(C, -1).mean(dim=1)))
                log.debug('channel max : {}'.format(images[0].reshape(C, -1).max(dim=1).values))
                log.debug('channel std : {}'.format(images[0].reshape(C, -1).std(dim=1)))
            elif len(images.shape) == 5:
                N, C, T, H, W = images.shape
                log.debug('inputs shape: NCTHW: {} {} {} {} {}'.format(N, C, T, H, W))
                log.debug('channel min:  {}'.format(images[0].min(dim=2).values))
                log.debug('channel mean: {}'.format(images[0].mean(dim=2)))
                log.debug('channel max : {}'.format(images[0].max(dim=2).values))
                log.debug('channel std : {}'.format(images[0].std(dim=2)))
            self.has_logged_channels = True

    def log_model_statistics(self, images, outputs, labels):
        # will print out shape and min, mean, max, std along image channels
        # we use the isEnabledFor flag so that this doesnt slow down training in the non-debug case
        log.debug('outputs: {}'.format(outputs))
        log.debug('labels: {}'.format(labels))
        log.debug('outputs: {}'.format(outputs.shape))
        log.debug('labels: {}'.format(labels.shape))
        log.debug('label max: {}'.format(labels.max()))
        log.debug('label min: {}'.format(labels.min()))


def get_criterion(cfg: DictConfig, model, data_info: dict, device=None):
    """Get loss function based on final activation.

    If final activation is softmax: use cross-entropy loss
    If final activation is sigmoid: use BinaryFocalLoss

    data_info are used to store weights based on dataset statistics, for up-weighting rare classes, etc.

    Parameters
    ----------
    cfg : DictConfig
        configuration
    model : nn.Module
        CNN
    data_info : dict
        dictionary with keys ['loss_weight', 'pos_weight', etc.], e.g. those
            returned from deepethogram.data.datasets.get_datasets_from_cfg
    device : str, torch.Device, optional
        cpu or GPU, by default None

    Returns
    -------
    criterion: nn.Module
        loss function

    Raises
    ------
    NotImplementedError
        if final_activation is not softmax or sigmoid
    """
    final_activation = cfg.feature_extractor.final_activation
    if final_activation == 'softmax':
        if 'weight' in list(data_info.keys()):
            weight = data_info['loss_weight']
        else:
            weight = None
        data_criterion = CrossEntropyLoss(weight=weight)

    elif final_activation == 'sigmoid':
        pos_weight = data_info['pos_weight']
        if type(pos_weight) == np.ndarray:
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device) if device is not None else pos_weight
        data_criterion = BinaryFocalLoss(pos_weight=pos_weight,
                                         gamma=cfg.train.loss_gamma,
                                         label_smoothing=cfg.train.label_smoothing)
    else:
        raise NotImplementedError

    regularization_criterion = get_regularization_loss(cfg, model)

    criterion = ClassificationLoss(data_criterion, regularization_criterion)
    criterion = criterion.to(device) if device is not None else criterion

    return criterion


def get_metrics(rundir: Union[str, bytes, os.PathLike],
                num_classes: int,
                num_parameters: Union[int, float],
                is_kinetics: bool = False,
                key_metric='loss',
                num_workers: int = 4):
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
    metrics = Classification(rundir,
                             key_metric,
                             num_parameters,
                             num_classes=num_classes,
                             evaluate_threshold=True,
                             num_workers=num_workers)
    return metrics


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_feature_extractor_train_cfg(project_path, use_command_line=True)

    feature_extractor_train(cfg)
