from collections import defaultdict
from copy import deepcopy
import logging
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
try: 
    from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
        TuneReportCheckpointCallback
    from ray.tune import get_trial_dir
    from ray.tune import CLIReporter
    ray = True
except ImportError:
    ray = False
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from deepethogram.data.augs import get_gpu_transforms, get_empty_gpu_transforms
from deepethogram.callbacks import FPSCallback, DebugCallback, MetricsCallback, \
    ExampleImagesCallback, CheckpointCallback, StopperCallback
from deepethogram.metrics import Metrics, EmptyMetrics
from deepethogram.schedulers import initialize_scheduler
from deepethogram import viz, utils

log = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics: Metrics, visualization_func):
        super().__init__()

        self.model = model
        self.hparams = cfg
        self.datasets = datasets
        self.metrics = metrics
        self.visualization_func = visualization_func

        model_type = cfg.run.model
        if model_type in ['feature_extractor', 'flow_generator']:
            arch = self.hparams[model_type].arch
            gpu_transforms = get_gpu_transforms(self.hparams.augs, '3d' if '3d' in arch.lower() else '2d')
        elif model_type == 'sequence':
            gpu_transforms = get_empty_gpu_transforms()
        else:
            raise NotImplementedError
        self.model_type = model_type
        self.gpu_transforms: dict = gpu_transforms

        self.optimizer = None  # will be overridden in configure_optimizers
        self.hparams.weight_decay = None
        if 'feature_extractor' in self.hparams.keys():
            self.hparams.weight_decay = self.hparams.feature_extractor.weight_decay

        self.scheduler_mode = 'min' if self.metrics.key_metric == 'loss' else 'max'
        # need to move this to top-level for lightning's learning rate finder
        # don't set it to auto here, so that we can automatically find batch size first
        self.lr = self.hparams.train.lr if self.hparams.train.lr != 'auto' else 1e-4
        log.info('scheduler mode: {}'.format(self.scheduler_mode))
        # self.is_key_metric_loss = self.metrics.key_metric == 'loss'
        
        self.viz_cnt = defaultdict(int)
        # for hyperparameter tuning, log specific hyperparameters and metrics for tensorboard
        if 'tune' in cfg.keys():
            # print('KEYS KEYS KEYS')
            tune_keys = list(cfg.tune.hparams.keys())
            # this function goes takes a list like [`feature_extractor.dropout_p`, `train.loss_weight_exp`], and finds
            # those entries in the configuration
            self.tune_hparams = utils.get_hparams_from_cfg(cfg, tune_keys)
            self.tune_metrics = OmegaConf.to_container(cfg.tune.metrics)
        else:
            self.tune_hparams = {}
            self.tune_metrics = []
            

    def on_train_epoch_start(self) -> None:
        # self.viz_cnt['train'] = 0
        # I couldn't figure out how to make sure that this is called after BOTH train and validation ends
        if self.current_epoch > 0 and self.hparams.train.viz_metrics:
            # all models shall define a visualization function that points to the metrics file on disk
            self.visualization_func(self.metrics.fname)

    def on_test_epoch_end(self):
        self.visualization_func(self.metrics.fname)

    def get_dataloader(self, split: str):
        # for use with auto-batch-sizing. Lightning doesn't expect batch size to be nested, it expects it to be
        # top-level in self.hparams
        batch_size = self.hparams.compute.batch_size if self.hparams.compute.batch_size != 'auto' else \
            self.hparams.batch_size
        shuffles = {'train': True, 'val': True, 'test': False}
        dataloader = DataLoader(self.datasets[split], batch_size=batch_size,
                                shuffle=shuffles[split], num_workers=self.hparams.compute.num_workers,
                                pin_memory=torch.cuda.is_available(), drop_last=False)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        if 'test' in self.datasets.keys() and self.datasets['test'] is not None:
            return self.get_dataloader('test')
        else:
            raise ValueError('no test set!')

    def training_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError

    def forward(self, batch: dict, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def apply_gpu_transforms(self, images: torch.Tensor, mode: str) -> torch.Tensor:
        with torch.no_grad():
            images = self.gpu_transforms[mode](images).detach()
        return images

    def configure_optimizers(self):

        weight_decay = 0 if self.hparams.weight_decay is None else self.hparams.weight_decay

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                               weight_decay=weight_decay)
        self.optimizer = optimizer
        log.info('learning rate: {}'.format(self.lr))
        scheduler = initialize_scheduler(optimizer, self.hparams, mode=self.scheduler_mode,
                                         reduction_factor=self.hparams.train.reduction_factor)
        monitor_key = 'val_' + self.metrics.key_metric
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor_key}


# @profile
default_tune_dict = {
    'loss': 'val_loss', 
    'f1_micro': 'val_f1_class_mean', 
    'data_loss': 'val_data_loss', 
    'reg_loss': 'val_reg_loss'
}

def get_trainer_from_cfg(cfg: DictConfig, lightning_module, stopper, profiler: str = None,
                         bs_start: int=2, bs_end: int=2048) -> pl.Trainer:
    steps_per_epoch = cfg.train.steps_per_epoch
    for split in ['train', 'val', 'test']:
        steps_per_epoch[split] = steps_per_epoch[split] if steps_per_epoch[split] is not None else 1.0

    # reload_dataloaders_every_epoch = True: a bit slower, but enables validation dataloader to get the new, automatic
    # learning rate schedule.

    if cfg.compute.batch_size == 'auto' or cfg.train.lr == 'auto':
        trainer = pl.Trainer(gpus=[cfg.compute.gpu_id],
                             precision=16 if cfg.compute.fp16 else 32,
                             limit_train_batches=1.0,
                             limit_val_batches=1.0,
                             limit_test_batches=1.0,
                             num_sanity_val_steps=0)
                            # callbacks=[ExampleImagesCallback()])
        tmp_metrics = lightning_module.metrics
        tmp_workers = lightning_module.hparams.compute.num_workers
        # visualize_examples = lightning_module.visualize_examples

        if lightning_module.model_type != 'sequence':
            # there is a somewhat common error that VRAM will be maximized by the gpu-auto-tuner.
            # However, during training, we probabilistically sample colorspace transforms; in an "unlucky"
            # batch, perhaps all of the training samples are converted to HSV, hue and saturation changed, then changed 
            # back. This is rare enough to not be encountered in "auto-tuning," so we'll get a train-time error. BAD!
            # so, we crank up the colorspace augmentation probability, then pick batch size, then change it back
            original_gpu_transforms = deepcopy(lightning_module.gpu_transforms)
            
            log.debug('orig: {}'.format(lightning_module.gpu_transforms))
            
            original_augs = cfg.augs
            new_augs = deepcopy(cfg.augs)
            new_augs.color_p = 1.0
            
            arch = lightning_module.hparams[lightning_module.model_type].arch
            mode = '2d'
            gpu_transforms = get_gpu_transforms(new_augs, '3d' if '3d' in arch.lower() else '2d')
            lightning_module.gpu_transforms = gpu_transforms        
            log.debug('new: {}'.format(lightning_module.gpu_transforms))
            
        tuner = pl.tuner.tuning.Tuner(trainer)
        # hack for lightning to find the batch size
        cfg.batch_size = 2  # to start

        empty_metrics = EmptyMetrics()
        # don't store metrics when batch size finding
        lightning_module.metrics = empty_metrics
        # don't visualize our model inputs when batch size finding
        # lightning_module.visualize_examples = False
        should_viz = cfg.train.viz_examples
        lightning_module.hparams.train.viz_examples = False
        # dramatically reduces RAM usage by this process
        lightning_module.hparams.compute.num_workers = min(tmp_workers, 1)
        if cfg.compute.batch_size == 'auto':
            max_trials = int(math.log2(bs_end)) - int(math.log2(bs_start))
            log.info('max trials: {}'.format(max_trials))
            new_batch_size = trainer.tuner.scale_batch_size(lightning_module, mode='power', steps_per_trial=30,
                                                            init_val=bs_start, max_trials=max_trials)
            cfg.compute.batch_size = new_batch_size
            log.info('auto-tuned batch size: {}'.format(new_batch_size))
        if cfg.train.lr == 'auto':
            lr_finder = trainer.tuner.lr_find(lightning_module, early_stop_threshold=None,
                                              min_lr=1e-6, max_lr=10.0)
            # log.info(lr_finder.results)
            plt.style.use('seaborn')
            fig = lr_finder.plot(suggest=True, show=False)
            viz.save_figure(fig, 'auto_lr_finder', False, 0, overwrite=False)
            plt.close(fig)
            new_lr = lr_finder.suggestion()
            log.info('auto-tuned learning rate: {}'.format(new_lr))
            cfg.train.lr = new_lr
            lightning_module.lr = new_lr
            lightning_module.hparams.lr = new_lr
        del trainer, tuner
        #  restore lightning module to original state
        lightning_module.hparams.train.viz_examples = should_viz
        lightning_module.metrics = tmp_metrics
        lightning_module.hparams.compute.num_workers = tmp_workers
        if lightning_module.model_type != 'sequence':
            lightning_module.gpu_transforms = original_gpu_transforms
            log.debug('reverted: {}'.format(lightning_module.gpu_transforms))

    callback_list = [FPSCallback(),  # DebugCallback(),# SpeedtestCallback(),
                                    MetricsCallback(), ExampleImagesCallback(), CheckpointCallback(),
                                    StopperCallback(stopper)]
    if cfg.tune.use and ray: 
        callback_list.append(TuneReportCallback(OmegaConf.to_container(cfg.tune.metrics), 
                                                on='validation_end'))
        # https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
        tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=get_trial_dir(), name="", version=".", 
            default_hp_metric=False)
        refresh_rate = 100
    else:
        tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(os.getcwd())
        refresh_rate = 1
    
    # tuning fucks with the callbacks
    trainer = pl.Trainer(gpus=[cfg.compute.gpu_id],
                         precision=16 if cfg.compute.fp16 else 32,
                         limit_train_batches=steps_per_epoch['train'],
                         limit_val_batches=steps_per_epoch['val'],
                         limit_test_batches=steps_per_epoch['test'],
                         logger=tensorboard_logger,
                         max_epochs=cfg.train.num_epochs,
                         num_sanity_val_steps=0,
                         callbacks=callback_list,
                         reload_dataloaders_every_epoch=True,
                         progress_bar_refresh_rate=refresh_rate, 
                         profiler=profiler)
    torch.cuda.empty_cache()
    # gc.collect()
    
    # import signal
    # signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # log.info('trainer is_slurm_managing_tasks: {}'.format(trainer.is_slurm_managing_tasks))
    return trainer
