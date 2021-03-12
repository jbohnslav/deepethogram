import logging
import os
import sys
from typing import Union, Type, Tuple
import warnings

# import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import deepethogram.projects
from deepethogram import utils, viz, projects
from deepethogram.base import BaseLightningModule, get_trainer_from_cfg
from deepethogram.configuration import make_flow_generator_train_cfg
from deepethogram.data.augs import get_gpu_transforms
from deepethogram.data.datasets import get_datasets_from_cfg
from deepethogram.flow_generator import models
from deepethogram.flow_generator.losses import MotionNetLoss
from deepethogram.flow_generator.utils import Reconstructor
from deepethogram.losses import get_regularization_loss
from deepethogram.metrics import OpticalFlow
from deepethogram.stoppers import get_stopper

warnings.filterwarnings('ignore', category=UserWarning, message=
'Your val_dataloader has `shuffle=True`, it is best practice to turn this off for validation '
'and test dataloaders.')

flow_generators = utils.get_models_from_module(models, get_function=False)

plt.switch_backend('agg')

# which GPUs should be available for training? I use 0,1 here manually because GPU2 is a tiny one for my displays
n_gpus = torch.cuda.device_count()
# DEVICE_IDS = [i for i in range(n_gpus)]
# DEVICE_IDS = [0, 1]


log = logging.getLogger(__name__)


# cudnn.benchmark = True
# cudnn.deterministic = False
# log.warning('Using nondeterministic CUDNN, may be slower')

# __all__ = ['build_model_from_cfg', 'train_from_cfg', 'train']

# @hydra.main(config_path='../conf/flow_train.yaml')


def flow_generator_train(cfg: DictConfig) -> nn.Module:
    """Trains flow generator models from a configuration. 

    Parameters
    ----------
    cfg : DictConfig
        Configuration, e.g. that returned by deepethogram.configration.make_flow_generator_train_cfg

    Returns
    -------
    nn.Module
        Trained flow generator
    """
    cfg = projects.setup_run(cfg)
    log.info('args: {}'.format(' '.join(sys.argv)))
    # only two custom overwrites of the configuration file

    # allow for editing
    OmegaConf.set_struct(cfg, False)
    # second, use the model directory to find the most recent run of each model type
    # cfg = projects.overwrite_cfg_with_latest_weights(cfg, cfg.project.model_path, model_type='flow_generator')
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('configuration used ~~~~~')
    log.info(OmegaConf.to_yaml(cfg))
    
    datasets, data_info = get_datasets_from_cfg(cfg, 'flow_generator', input_images=cfg.flow_generator.n_rgb)
    flow_generator = build_model_from_cfg(cfg)
    log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(flow_generator)))
    utils.save_dict_to_yaml(data_info['split'], os.path.join(os.getcwd(), 'split.yaml'))
    flow_weights = deepethogram.projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    if flow_weights is not None:
        print('reloading weights...')
        flow_generator = utils.load_weights(flow_generator, flow_weights, device='cpu')

    stopper = get_stopper(cfg)
    metrics = get_metrics(cfg, os.getcwd(), utils.get_num_parameters(flow_generator))
    lightning_module = OpticalFlowLightning(flow_generator, cfg, datasets, metrics, viz.visualize_logger_optical_flow)

    trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
    trainer.fit(lightning_module)
    return flow_generator
    
    
def build_model_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
    flow_generator = flow_generators[cfg.flow_generator.arch](num_images=cfg.flow_generator.n_rgb,
                                                              flow_div=cfg.flow_generator.max)
    return flow_generator


class OpticalFlowLightning(BaseLightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics, visualization_func):
        super().__init__(model, cfg, datasets, metrics, visualization_func)

        self.reconstructor = Reconstructor(self.hparams)

        self.has_logged_channels = False
        # for convenience
        
        self.criterion = get_criterion(cfg, self.model)
        # this will get overridden by the ExampleImagesCallback
        self.viz_cnt = None

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

    def common_step(self, batch: dict, batch_idx: int, split: str):
        images, outputs = self(batch, split)

        downsampled_t0, estimated_t0, flows_reshaped = self.reconstructor(images, outputs)
        loss, loss_components = self.criterion(batch, downsampled_t0, estimated_t0, flows_reshaped, self.model)
        self.visualize_batch(images, downsampled_t0, estimated_t0, flows_reshaped, split)

        to_log = loss_components
        to_log['loss'] = loss.detach()

        self.metrics.buffer.append(split, to_log)
        # need to use the native logger for lr scheduling, etc.
        key_metric = self.metrics.key_metric
        self.log(f'{split}_loss', loss)
        if split == 'val':
            self.log(f'{split}_{key_metric}', loss_components[key_metric].mean())

        return loss

    def training_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, 'test')

    def visualize_batch(self, images, downsampled_t0, estimated_t0, flows_reshaped, split: str):
        if not self.hparams.train.viz_examples:
            return
        # ALWAYS VISUALIZE MODEL INPUTS JUST BEFORE FORWARD PASS
        viz_cnt = self.viz_cnt[split]
        if viz_cnt > 10:
            return
        fig = plt.figure(figsize=(14, 14))
        batch_ind = np.random.choice(images.shape[0])
        sequence_length = int(downsampled_t0[0].shape[0] / images.shape[0])

        viz.visualize_images_and_flows(downsampled_t0, flows_reshaped, sequence_length,
                                       batch_ind=batch_ind,
                                       fig=fig, max_flow=self.hparams.flow_generator.max)
        viz.save_figure(fig, 'batch', True, viz_cnt, split)

        fig = plt.figure(figsize=(14, 14))
        sequence_ind = np.random.choice(sequence_length - 1)
        viz.visualize_multiresolution(downsampled_t0, estimated_t0, flows_reshaped, sequence_length,
                                      max_flow=self.hparams.flow_generator.max, sequence_ind=sequence_ind,
                                      batch_ind=batch_ind,
                                      fig=fig)
        viz.save_figure(fig, 'multiresolution', True, viz_cnt, split)

        fig = plt.figure(figsize=(14, 14))
        viz.visualize_batch_unsupervised(downsampled_t0, estimated_t0, flows_reshaped,
                                         batch_ind=batch_ind, sequence_ind=sequence_ind,
                                         fig=fig, sequence_length=sequence_length)
        viz.save_figure(fig, 'reconstruction', True, viz_cnt, split)

    def forward(self, batch: dict, mode: str) -> Tuple[torch.Tensor, list]:
        # try:
        #     batch = next(dataiter)
        # except StopIteration:
        #     break
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


def get_criterion(cfg, model):
    regularization_criterion = get_regularization_loss(cfg, model)
    
    if cfg.flow_generator.loss == 'MotionNet':
        criterion = MotionNetLoss(regularization_criterion, flow_sparsity=cfg.flow_generator.flow_sparsity,
                                    sparsity_weight=cfg.flow_generator.sparsity_weight,
                                    smooth_weight_multiplier=cfg.flow_generator.smooth_weight_multiplier, 
                                    )
    else:
        raise NotImplementedError
    
    return criterion


def get_metrics(cfg: DictConfig, rundir: Union[str, bytes, os.PathLike], num_parameters: Union[int, float]):
    metrics_list = ['SSIM', 'L1', 'smoothness', 'SSIM_full']
    if cfg.flow_generator.flow_sparsity:
        metrics_list.append('flow_sparsity')
    if cfg.flow_generator.loss == 'SelfSupervised':
        metrics_list.append('gradient')
        metrics_list.append('MFH')
    key_metric = 'SSIM'
    log.info('key metric is {}'.format(key_metric))
    # the metrics objects all take normal dicts instead of dict configs
    metrics = OpticalFlow(rundir, key_metric, num_parameters)
    return metrics


if __name__ == '__main__':
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_flow_generator_train_cfg(project_path, use_command_line=True)
    
    flow_generator_train(cfg)
