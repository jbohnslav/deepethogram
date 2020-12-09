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

import pytorch_lightning as pl
from tqdm import tqdm, trange


import deepethogram.projects
from deepethogram import utils, viz
from deepethogram.data.augs import get_gpu_transforms
from deepethogram.data.dataloaders import get_dataloaders_from_cfg
from deepethogram.flow_generator import models
from deepethogram.flow_generator.losses import MotionNetLoss
from deepethogram.flow_generator.utils import Reconstructor
from deepethogram.metrics import OpticalFlow
from deepethogram.schedulers import initialize_scheduler
from deepethogram.stoppers import get_stopper

flow_generators = utils.get_models_from_module(models, get_function=False)

plt.switch_backend('agg')

# which GPUs should be available for training? I use 0,1 here manually because GPU2 is a tiny one for my displays
n_gpus = torch.cuda.device_count()
# DEVICE_IDS = [i for i in range(n_gpus)]
DEVICE_IDS = [0, 1]



log = logging.getLogger(__name__)
cudnn.benchmark = False
cudnn.deterministic = False
# log.warning('Using nondeterministic CUDNN, may be slower')

# __all__ = ['build_model_from_cfg', 'train_from_cfg', 'train']

@hydra.main(config_path='../conf/flow_train.yaml')
def main(cfg: DictConfig) -> None:
    log.debug('cwd: {}'.format(os.getcwd()))
    log.info('args: {}'.format(' '.join(sys.argv)))
    # only two custom overwrites of the configuration file
    # first, change the project paths from relative to absolute

    cfg = deepethogram.projects.parse_cfg_paths(cfg)
    # second, use the model directory to find the most recent run of each model type
    # cfg = projects.overwrite_cfg_with_latest_weights(cfg, cfg.project.model_path, model_type='flow_generator')
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('Configuration used: ')
    log.info(cfg.pretty())

    try:
        model = train_from_cfg(cfg)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        raise


def build_model_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
    flow_generator = flow_generators[cfg.flow_generator.arch](num_images=cfg.flow_generator.n_rgb,
                                                              flow_div=cfg.flow_generator.max)
    return flow_generator


def train_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
    device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
    if device != 'cpu': torch.cuda.set_device(device)

    log.info('Training flow generator....')
    arch = cfg.flow_generator.arch
    gpu_transforms = get_gpu_transforms(cfg.augs, '3d' if '3d' in arch.lower() else '2d')
    dataloaders = get_dataloaders_from_cfg(cfg, model_type='flow_generator', input_images=cfg.flow_generator.n_rgb)

    # print(dataloaders)
    log.info('Num training batches {}, num val: {}'.format(len(dataloaders['train']), len(dataloaders['val'])))

    flow_generator = build_model_from_cfg(cfg)
    flow_generator = flow_generator.to(device)

    log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(flow_generator)))

    rundir = os.getcwd()  # this is configured by hydra
    # save model definition
    torch.save(flow_generator, os.path.join(rundir, cfg.flow_generator.arch + '_definition.pt'))
    utils.save_dict_to_yaml(dataloaders['split'], os.path.join(rundir, 'split.yaml'))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, flow_generator.parameters()), lr=cfg.train.lr)
    flow_weights = deepethogram.projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    if flow_weights is not None:
        print('reloading weights...')
        flow_generator = utils.load_weights(flow_generator, flow_weights, device=device)

    # stopper, early_stopping_begins = get_stopper(cfg)
    stopper = get_stopper(cfg)
    scheduler = initialize_scheduler(optimizer, cfg, mode='min')

    if cfg.flow_generator.loss == 'MotionNet':
        criterion = MotionNetLoss(flow_sparsity=cfg.flow_generator.flow_sparsity,
                                  sparsity_weight=cfg.flow_generator.sparsity_weight,
                                  smooth_weight_multiplier=cfg.flow_generator.smooth_weight_multiplier)
    else:
        raise NotImplementedError

    metrics = get_metrics(cfg, rundir, utils.get_num_parameters(flow_generator))
    reconstructor = Reconstructor(cfg)
    steps_per_epoch = cfg.train.steps_per_epoch
    if cfg.compute.fp16:
        assert torch_amp, 'must install torch 1.6 or greater to use FP16 training'
    flow_generator = train(flow_generator,
                           dataloaders,
                           criterion,
                           optimizer,
                           gpu_transforms,
                           metrics,
                           scheduler,
                           reconstructor,
                           rundir,
                           stopper,
                           device,
                           num_epochs=cfg.train.num_epochs,
                           steps_per_epoch=steps_per_epoch['train'],
                           steps_per_validation_epoch=steps_per_epoch['val'],
                           steps_per_test_epoch=steps_per_epoch['test'],
                           max_flow=cfg.flow_generator.max,
                           dali=cfg.compute.dali,
                           fp16=cfg.compute.fp16)
    return flow_generator


class HiddenTwoStreamLightning(BaseLightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics, visualization_func, data_info: dict,
                 visualize_examples: bool = True):
        super().__init__(model, cfg, datasets, metrics, visualization_func, visualize_examples)

        # self.model = model
        # self.hparams = cfg
        # self.datasets = datasets
        self.data_info = data_info
        # self.metrics = metrics
        # self.dali = dali
        # self.visualize_examples = visualize_examples

        arch = self.hparams.feature_extractor.arch
        gpu_transforms = get_gpu_transforms(self.hparams.augs, '3d' if '3d' in arch.lower() else '2d')
        self.gpu_transforms = gpu_transforms
        self.has_logged_channels = False
        # for convenience
        self.final_activation = self.hparams.feature_extractor.final_activation
        if self.final_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif self.final_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.criterion = get_criterion(self.final_activation, self.data_info)
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
        self.log('loss', self.current_epoch)

    def test_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, 'test')
        probabilities = self.activation(outputs)

    @torch.no_grad()
    def apply_gpu_transforms(self, images: torch.Tensor, mode: str) -> torch.Tensor:
        images = self.gpu_transforms[mode](images)
        return images

    def visualize_batch(self, images, probs, labels, split: str):
        if not self.visualize_examples:
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


def train(model,
          dataloaders: dict,
          criterion,
          optimizer,
          gpu_transforms: dict,
          metrics,
          scheduler,
          reconstructor,
          rundir: Union[str, bytes, os.PathLike],
          stopper,
          device: torch.device,
          num_epochs: int = 1000,
          steps_per_epoch: int = 1000,
          steps_per_validation_epoch: int = 1000,
          steps_per_test_epoch: int = 100,
          early_stopping_begins: int = 0,
          max_flow: float = 2.5,
          dali:bool=False,
          fp16: bool=False):
    # check our inputs
    assert isinstance(model, nn.Module)
    assert isinstance(criterion, nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)

    scaler = None
    if fp16:
        scaler = GradScaler()
    # loop over number of epochs!
    for epoch in trange(0, num_epochs):
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
        model, metrics = loop_one_epoch(dataloaders['train'], model, criterion, optimizer, gpu_transforms,
                                           metrics, reconstructor,
                                           steps_per_epoch, train_mode=True, device=device, dali=dali,
                                           fp16=fp16, scaler=scaler)

        # evaluate on validation set
        with torch.no_grad():
            model, metrics = loop_one_epoch(dataloaders['val'], model, criterion, optimizer, gpu_transforms,
                                                      metrics,
                                                      reconstructor,
                                                      steps_per_validation_epoch, train_mode=False,
                                                      device=device, max_flow=max_flow, dali=dali,
                                                      fp16=fp16, scaler=scaler)

            # some training protocols do not have test sets, so just reuse validation set for testing inference speed
            key = 'test' if 'test' in dataloaders.keys() else 'val'
            loader = dataloaders[key]
            # evaluate how fast inference takes, without loss calculation, which for some models can have a significant
            # speed impact
            metrics = speedtest(loader, model, gpu_transforms, metrics, steps_per_test_epoch, device=device, dali=dali,
                                fp16=fp16)

        # use our metrics file to output graphs for this epoch
        viz.visualize_logger_optical_flow(metrics.fname)

        # save a checkpoint
        utils.checkpoint(model, rundir, epoch)
        # # update latest models file
        # projects.write_latest_model(config['model'], config['flow_generator'], rundir, config)

        # input the latest validation loss to the early stopper
        if stopper.name == 'early':
            should_stop, _ = stopper(metrics.latest_key['val'])
        elif stopper.name == 'learning_rate':
            should_stop = stopper(min_lr)
        else:
            # every epoch, increment stopper
            should_stop = stopper()

        if should_stop:
            log.info('Stopping criterion reached!')
            break
    return model


def loop_one_epoch(loader, model, criterion, optimizer, gpu_transforms:dict, metrics, reconstructor, steps_per_epoch,
                   train_mode=True, device=None, max_flow: float = 2.5, dali: bool=False, fp16: bool=False,
                   scaler=None):
    if train_mode:
        # make sure we're in train mode
        model.train()
    else:
        model.eval()

    # if steps per epoch is not none, make an iterable of range N (e.g. 1000 minibatches)
    num_iters = len(loader) if steps_per_epoch is None else min(steps_per_epoch, len(loader))
    t = tqdm(range(0, num_iters), leave=False)
    dataiter = iter(loader)
    mode = 'train' if train_mode else 'val'
    cnt = 0
    has_logged = False
    for i in t:
        t0 = time.time()
        try:
            batch = next(dataiter)
        except StopIteration:
            break
        if dali:
            batch = batch[0]['images']
        else:
            batch = batch.to(device)

        with torch.no_grad():
            batch = gpu_transforms[mode](batch)

        # num_images = int(batch.shape[1] / 3) - 1
        # images = [batch[:, i * 3:i * 3 + 3, ...] for i in range(num_images)]
        # print('Training loop')
        # for j, image in enumerate(images):
        #     print('image {}: mean {:.4f} min {:.4f} max {:.4f}'.format(j, image.mean(), image.min(),
        #                                                                image.max()))
        # images = batch

        if not has_logged:
            log.debug('Batch shape: {}'.format(batch.shape))
            if len(batch.shape) == 4:
                N, C, H, W = batch.shape
                log.debug('channel min:  {}'.format(batch[0].reshape(C, -1).min(dim=1).values))
                log.debug('channel mean: {}'.format(batch[0].reshape(C, -1).mean(dim=1)))
                log.debug('channel max : {}'.format(batch[0].reshape(C, -1).max(dim=1).values))
                log.debug('channel std : {}'.format(batch[0].reshape(C, -1).std(dim=1)))
            else:
                N, C, T, H, W = batch.shape
                log.debug('channel min:  {}'.format(batch[0].min(dim=2).values))
                log.debug('channel mean: {}'.format(batch[0].mean(dim=2)))
                log.debug('channel max : {}'.format(batch[0].max(dim=2).values))
                log.debug('channel std : {}'.format(batch[0].std(dim=2)))
            has_logged = True
            # print
            # print('channel max : {}'.format(images[0].reshape(C, -1).max(dim=1).values))
            #
        if train_mode:
            if fp16:
                with autocast():
                    outputs = model(batch)
                    downsampled_t0, estimated_t0, flows_reshaped = reconstructor(batch, outputs)
                    loss, loss_components = criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)
            else:
                outputs = model(batch)
                downsampled_t0, estimated_t0, flows_reshaped = reconstructor(batch, outputs)
                loss, loss_components = criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)
        else:
            with torch.no_grad():
                if fp16:
                    with autocast():
                        outputs = model(batch)
                        downsampled_t0, estimated_t0, flows_reshaped = reconstructor(batch, outputs)
                        loss, loss_components = criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)
                else:
                    outputs = model(batch)
                    downsampled_t0, estimated_t0, flows_reshaped = reconstructor(batch, outputs)
                    loss, loss_components = criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)

        if train_mode:
            # zero the parameter gradients
            optimizer.zero_grad()
            # calculate gradients
            if fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # step in direction of gradients according to optimizer
                optimizer.step()

        # torch.cuda.synchronize()
        time_per_image = (time.time() - t0) / batch.shape[0]

        metrics.buffer.append(dict(loss=loss.detach(),
                                   time=time_per_image,
                                   **loss_components))

        # metrics.loss_components_append(loss_components)
        # metrics.time_append(time_per_image)
        # metrics.loss_append(loss.item())
        if cnt % 10 == 0:
            # for speed
            t.set_description('{} loss: {:.4f}'.format(mode, loss.item()))

        if cnt < 10:
            batch_ind = np.random.choice(batch.shape[0])
            sequence_length = int(downsampled_t0[0].shape[0] / batch.shape[0])
            fig = plt.figure(figsize=(12, 12))
            # downsampled_t0 = [i.astype(np.float32) for i in downsampled_t0]
            # estimated_t0 =   [i.astype(np.float32) for i in estimated_t0]
            # flows_reshaped = [i.astype(np.float32) for i in flows_reshaped]

            viz.visualize_images_and_flows(downsampled_t0, flows_reshaped, sequence_length,
                                           batch_ind=batch_ind,
                                           fig=fig, max_flow=max_flow)
            viz.save_figure(fig, 'batch', True, cnt, mode)


            fig = plt.figure(figsize=(12, 12))
            sequence_ind = np.random.choice(sequence_length - 1)
            viz.visualize_multiresolution(downsampled_t0, estimated_t0, flows_reshaped, sequence_length,
                                          max_flow=max_flow, sequence_ind=sequence_ind,
                                          batch_ind=batch_ind,
                                          fig=fig)
            viz.save_figure(fig, 'multiresolution', True, cnt, mode)

            fig = plt.figure(figsize=(12, 12))
            viz.visualize_batch_unsupervised(downsampled_t0, estimated_t0, flows_reshaped,
                                             batch_ind=batch_ind, sequence_ind=sequence_ind,
                                             fig=fig, sequence_length=sequence_length)
            viz.save_figure(fig, 'reconstruction', True, cnt, mode)
        cnt += 1

    metrics.buffer.append({'lr': utils.get_minimum_learning_rate(optimizer)})
    metrics.end_epoch(mode)
    return model, metrics


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
    metrics = OpticalFlow(rundir, key_metric, num_parameters,
                          metrics=metrics_list)
    return metrics


def speedtest(loader, model, gpu_transforms: dict, metrics, steps, device=None, dali:bool=False, fp16:bool=False):
    model.eval()

    # if steps per epoch is not none, make an iterable of range N (e.g. 1000 minibatches)
    # print(len(loader))
    num_iters = len(loader) if steps is None else min(steps, len(loader))
    t = tqdm(range(0, num_iters), leave=False)
    dataiter = iter(loader)
    epoch_t = time.time()
    cnt = 0
    for i in t:
        t0 = time.time()
        try:
            batch = next(dataiter)
        except StopIteration:
            break
        if dali:
            batch = batch[0]['images']
        else:
            batch = batch.to(device)

        with torch.no_grad():
            batch = gpu_transforms['val'](batch)

        with torch.no_grad():
            if fp16:
                with autocast():
                    outputs = model(batch)
            else:
                outputs = model(batch)

        # N,C,H,W = images.shape
        num_images = batch.shape[0]
        time_per_image = (time.time() - t0) / (num_images + 1e-7)
        metrics.buffer.append({'time': time_per_image})
        t.set_description('FPS: {:.2f}'.format(1 / (time_per_image + 1e-7)))
    total_t = time.time() - epoch_t
    batches_per_s = total_t / num_iters
    log.debug('batches per second in speedtest: {}'.format(batches_per_s))
    metrics.end_epoch('speedtest')
    return metrics


if __name__ == '__main__':
    sys.argv = deepethogram.projects.process_config_file_from_cl(sys.argv)
    main()
