import gc
import logging
import os
import sys
import warnings
from typing import Union, Tuple

import cv2
cv2.setNumThreads(0)
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from deepethogram import utils, viz
from deepethogram.base import BaseLightningModule, get_trainer_from_cfg
from deepethogram.data.augs import get_gpu_transforms
from deepethogram.data.datasets import get_datasets_from_cfg
from deepethogram.feature_extractor.losses import BCELossCustom
from deepethogram.feature_extractor.models.CNN import get_cnn
from deepethogram.feature_extractor.models.hidden_two_stream import HiddenTwoStream, FlowOnlyClassifier, \
    build_fusion_layer
from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
from deepethogram.metrics import Classification
from deepethogram import projects
from deepethogram.stoppers import get_stopper

warnings.filterwarnings('ignore', category=UserWarning, message=
'Your val_dataloader has `shuffle=True`, it is best practice to turn this off for validation '
'and test dataloaders.')

# flow_generators = utils.get_models_from_module(flow_models, get_function=False)
plt.switch_backend('agg')

log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='feature_extractor_train')
def main(cfg: DictConfig) -> None:
    log.info('cwd: {}'.format(os.getcwd()))
    log.info('args: {}'.format(' '.join(sys.argv)))
    # change the project paths from relative to absolute
    cfg = projects.convert_config_paths_to_absolute(cfg)
    # allow for editing
    OmegaConf.set_struct(cfg, False)
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('configuration used ~~~~~')
    log.info(OmegaConf.to_yaml(cfg))

    try:
        train_from_cfg_lightning(cfg)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        raise


# @profile
def train_from_cfg_lightning(cfg):
    rundir = os.getcwd()  # done by hydra

    # we build flow generator independently because you might want to load it from a different location
    flow_generator = build_flow_generator(cfg)
    flow_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    log.info('loading flow generator from file {}'.format(flow_weights))

    flow_generator = utils.load_weights(flow_generator, flow_weights)

    _, data_info = get_datasets_from_cfg(cfg, model_type='feature_extractor',
                                         input_images=cfg.feature_extractor.n_flows + 1)

    criterion = get_criterion(cfg.feature_extractor.final_activation, data_info)

    model_parts = build_model_from_cfg(cfg, pos=data_info['pos'], neg=data_info['neg'])
    _, spatial_classifier, flow_classifier, fusion, model = model_parts
    log.info('model: {}'.format(model))

    num_classes = len(cfg.project.class_names)

    utils.save_dict_to_yaml(data_info['split'], os.path.join(rundir, 'split.yaml'))

    metrics = get_metrics(rundir, num_classes=num_classes,
                          num_parameters=utils.get_num_parameters(spatial_classifier))

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
        datasets, data_info = get_datasets_from_cfg(cfg, model_type='feature_extractor',
                                                    input_images=cfg.feature_extractor.n_rgb)
        stopper = get_stopper(cfg)

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

        datasets, data_info = get_datasets_from_cfg(cfg, model_type='feature_extractor',
                                                    input_images=cfg.feature_extractor.n_flows + 1)
        # re-initialize stopper so that it doesn't think we need to stop due to the previous model
        stopper = get_stopper(cfg)
        cfg.compute.batch_size = original_batch_size
        cfg.train.lr = original_lr

        # this class will freeze the flow generator
        flow_generator_and_classifier = FlowOnlyClassifier(flow_generator, flow_classifier)
        lightning_module = HiddenTwoStreamLightning(flow_generator_and_classifier, cfg, datasets, metrics, criterion)
        trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
        # lightning_module = HiddenTwoStreamLightning(flow_generator_and_classifier, cfg, datasets, metrics, criterion)
        trainer.fit(lightning_module)

        del datasets, lightning_module, trainer, stopper, data_info
        torch.cuda.empty_cache()
        gc.collect()

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, fusion, cfg.feature_extractor.arch)
    model.set_mode('classifier')
    datasets, data_info = get_datasets_from_cfg(cfg, model_type='feature_extractor',
                                                input_images=cfg.feature_extractor.n_flows + 1)
    stopper = get_stopper(cfg)
    cfg.compute.batch_size = original_batch_size
    cfg.train.lr = original_lr
    lightning_module = HiddenTwoStreamLightning(model, cfg, datasets, metrics, criterion)

    trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
    # see above for horrible syntax explanation
    # lightning_module = HiddenTwoStreamLightning(model, cfg, datasets, metrics, criterion)
    trainer.fit(lightning_module)
    utils.save_hidden_two_stream(model, rundir, dict(cfg), stopper.epoch_counter)


def build_model_from_cfg(cfg: DictConfig,
                         pos: np.ndarray = None, neg: np.ndarray = None) -> tuple:
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
    num_classes = len(cfg.project.class_names)

    in_channels = cfg.feature_extractor.n_rgb * 3 if '3d' not in cfg.feature_extractor.arch else 3
    reload_imagenet = feature_extractor_weights is None
    if cfg.feature_extractor.arch == 'resnet3d_34':
        assert feature_extractor_weights is not None, 'Must specify path to resnet3d weights!'
    spatial_classifier = get_cnn(cfg.feature_extractor.arch, in_channels=in_channels,
                                 dropout_p=cfg.feature_extractor.dropout_p,
                                 num_classes=num_classes, reload_imagenet=reload_imagenet,
                                 pos=pos, neg=neg, final_bn=cfg.feature_extractor.final_bn)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        spatial_classifier = utils.load_feature_extractor_components(spatial_classifier, feature_extractor_weights,
                                                                     'spatial', device=device)
    in_channels = cfg.feature_extractor.n_flows * 2 if '3d' not in cfg.feature_extractor.arch else 2
    flow_classifier = get_cnn(cfg.feature_extractor.arch, in_channels=in_channels,
                              dropout_p=cfg.feature_extractor.dropout_p,
                              num_classes=num_classes, reload_imagenet=reload_imagenet,
                              pos=pos, neg=neg, final_bn=cfg.feature_extractor.final_bn)
    # load this specific component from the weight file
    if feature_extractor_weights is not None:
        flow_classifier = utils.load_feature_extractor_components(flow_classifier, feature_extractor_weights,
                                                                  'flow', device=device)

    flow_generator = build_flow_generator(cfg)
    flow_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    assert flow_weights is not None, ('Must have a valid weightfile for flow generator. Use '
                                      'deepethogram.flow_generator.train or cfg.reload.latest')
    flow_generator = utils.load_weights(flow_generator, flow_weights, device=device)

    spatial_classifier, flow_classifier, fusion = build_fusion_layer(spatial_classifier, flow_classifier,
                                                                     cfg.feature_extractor.fusion,
                                                                     num_classes)
    if feature_extractor_weights is not None:
        fusion = utils.load_feature_extractor_components(fusion, feature_extractor_weights,
                                                         'fusion', device=device)

    model = HiddenTwoStream(flow_generator, spatial_classifier, flow_classifier, fusion, cfg.feature_extractor.arch)
    # log.info(model.fusion.flow_weight)
    model.set_mode('classifier')

    return flow_generator, spatial_classifier, flow_classifier, fusion, model


class HiddenTwoStreamLightning(BaseLightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics, criterion: nn.Module):
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

        # this will get overridden by the ExampleImagesCallback
        self.viz_cnt = None

    # def on_train_epoch_start(self) -> None:
    #     log.info('buffer on epoch start: {}'.format(self.metrics.buffer.data))

    def validate_batch_size(self, batch: dict):
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
        # use the forward function
        # return the image tensor so we can visualize after gpu transforms
        images, outputs = self(batch, 'train')
        probabilities = self.activation(outputs)

        loss = self.criterion(outputs, batch['labels'])

        self.visualize_batch(images, probabilities, batch['labels'], 'train')

        self.metrics.buffer.append('train', {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        # need to use the native logger for lr scheduling, etc.
        self.log('loss', loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, 'val')
        probabilities = self.activation(outputs)

        loss = self.criterion(outputs, batch['labels'])
        self.visualize_batch(images, probabilities, batch['labels'], 'val')
        self.metrics.buffer.append('val', {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        # need to use the native logger for lr scheduling, etc.
        # TESTING
        self.log('loss', loss)

    def test_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, 'test')
        probabilities = self.activation(outputs)

    def visualize_batch(self, images, probs, labels, split: str):
        if not self.hparams.train.viz:
            return
        # ALWAYS VISUALIZE MODEL INPUTS JUST BEFORE FORWARD PASS
        viz_cnt = self.viz_cnt[split]
        if viz_cnt > 10:
            return
        fig = plt.figure(figsize=(14, 14))
        if hasattr(self.model, 'flow_generator'):
            # re-compute optic flows for this batch for visualization
            with torch.no_grad():
                flows = self.model.flow_generator(images)
                inputs = self.gpu_transforms['denormalize'](images)
            viz.visualize_hidden(inputs, flows, probs, labels, fig=fig)
            viz.save_figure(fig, 'batch_with_flows', True, viz_cnt, split)
        else:
            with torch.no_grad():
                inputs = self.gpu_transforms['denormalize'](images)
            viz.visualize_batch_spatial(inputs, probs, labels, fig=fig)
            viz.save_figure(fig, 'batch_spatial', True, viz_cnt, split)
        # self.viz_cnt[split] += 1

    def forward(self, batch: dict, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.validate_batch_size(batch)
        # lightning handles transfer to device
        images = batch['images']
        images = self.apply_gpu_transforms(images, mode)

        outputs = self.model(images)
        self.log_image_statistics(images)

        return images, outputs

    def log_image_statistics(self, images):
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


def get_criterion(final_activation: str, dataloaders: dict, device=None):
    """ Get loss function based on final activation.

    If final activation is softmax: use cross-entropy loss
    If final activation is sigmoid: use BCELoss

    Dataloaders are used to store weights based on dataset statistics, for up-weighting rare classes, etc.

    Args:
        final_activation (str): [softmax, sigmoid]
        dataloaders (dict): dictionary with keys ['train', 'val', 'test', 'loss_weight', 'pos_weight'], e.g. those
            returned from deepethogram.data.dataloaders.py # get_dataloaders, get_video_dataloaders, etc.
        device (str, torch.device): gpu device

    Returns:
        criterion (nn.Module): loss function
    """
    if final_activation == 'softmax':
        if 'weight' in list(dataloaders.keys()):
            weight = dataloaders['loss_weight']
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)

    elif final_activation == 'sigmoid':
        pos_weight = dataloaders['pos_weight']
        if type(pos_weight) == np.ndarray:
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device) if device is not None else pos_weight
        criterion = BCELossCustom(pos_weight=pos_weight)
    else:
        raise NotImplementedError
    criterion = criterion.to(device) if device is not None else criterion

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


if __name__ == '__main__':
    sys.argv = projects.process_config_file_from_cl(sys.argv)
    main()
