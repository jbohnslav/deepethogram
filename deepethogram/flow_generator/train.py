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

try:
    from torch.cuda.amp import autocast, GradScaler
    torch_amp = True
except ImportError:
    torch_amp = False
from tqdm import tqdm, trange

import deepethogram.projects
from deepethogram import utils, viz
from deepethogram.dataloaders import get_dataloaders_from_cfg
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
    # only two custom overwrites of the configuration file
    # first, change the project paths from relative to absolute

    cfg = utils.get_absolute_paths_from_cfg(cfg)
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


def train(model,
          dataloaders: dict,
          criterion,
          optimizer,
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
    assert (isinstance(model, nn.Module))
    assert (isinstance(criterion, nn.Module))
    assert (isinstance(optimizer, torch.optim.Optimizer))

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
        model, metrics, _ = loop_one_epoch(dataloaders['train'], model, criterion, optimizer, metrics, reconstructor,
                                           steps_per_epoch, train_mode=True, device=device, dali=dali,
                                           fp16=fp16, scaler=scaler)

        # evaluate on validation set
        with torch.no_grad():
            model, metrics, examples = loop_one_epoch(dataloaders['val'], model, criterion, optimizer, metrics,
                                                      reconstructor,
                                                      steps_per_validation_epoch, train_mode=False,
                                                      device=device, max_flow=max_flow, dali=dali,
                                                      fp16=fp16, scaler=scaler)

            # some training protocols do not have test sets, so just reuse validation set for testing inference speed
            key = 'test' if 'test' in dataloaders.keys() else 'val'
            loader = dataloaders[key]
            # evaluate how fast inference takes, without loss calculation, which for some models can have a significant
            # speed impact
            metrics = speedtest(loader, model, metrics, steps_per_test_epoch, device=device, dali=dali,
                                fp16=fp16)

        # use our metrics file to output graphs for this epoch
        viz.visualize_logger(metrics.fname, examples)

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


def loop_one_epoch(loader, model, criterion, optimizer, metrics, reconstructor, steps_per_epoch,
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
    examples = []
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

        metrics.loss_components_append(loss_components)
        metrics.time_append(time_per_image)
        metrics.loss_append(loss.item())

        t.set_description('{} loss: {:.4f}'.format(mode, loss.item()))

        if not train_mode:
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
                img = viz.fig_to_img(fig)
                examples.append(img)
                plt.close(fig)

                fig = plt.figure(figsize=(12, 12))
                sequence_ind = np.random.choice(sequence_length - 1)
                viz.visualize_multiresolution(downsampled_t0, estimated_t0, flows_reshaped, sequence_length,
                                              max_flow=max_flow, sequence_ind=sequence_ind,
                                              batch_ind=batch_ind,
                                              fig=fig)
                img = viz.fig_to_img(fig)
                examples.append(img)
                plt.close(fig)

                fig = plt.figure(figsize=(12, 12))

                viz.visualize_batch_unsupervised(downsampled_t0, estimated_t0, flows_reshaped,
                                                 batch_ind=batch_ind, sequence_ind=sequence_ind,
                                                 fig=fig, sequence_length=sequence_length)
                img = viz.fig_to_img(fig)
                examples.append(img)
                plt.close(fig)

        cnt += 1

    metrics.end_epoch(mode)
    return model, metrics, examples


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


def speedtest(loader, model, metrics, steps, device=None, dali:bool=False, fp16:bool=False):
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
            if fp16:
                with autocast():
                    outputs = model(batch)
            else:
                outputs = model(batch)

        # N,C,H,W = images.shape
        num_images = batch.shape[0]
        time_per_image = (time.time() - t0) / (num_images + 1e-7)
        metrics.time_append(time_per_image)
        t.set_description('FPS: {:.2f}'.format(1 / (time_per_image + 1e-7)))
    total_t = time.time() - epoch_t
    batches_per_s = total_t / num_iters
    log.debug('batches per second in speedtest: {}'.format(batches_per_s))
    metrics.end_epoch_speedtest()
    return metrics


if __name__ == '__main__':
    sys.argv = utils.process_config_file_from_cl(sys.argv)
    main()
