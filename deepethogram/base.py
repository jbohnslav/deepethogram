import logging
from typing import Tuple

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from deepethogram.data.augs import get_gpu_transforms, get_empty_gpu_transforms
from deepethogram.callbacks import FPSCallback, DebugCallback, SpeedtestCallback, MetricsCallback, \
    ExampleImagesCallback, CheckpointCallback, StopperCallback
from deepethogram.metrics import Metrics, EmptyMetrics
from deepethogram.schedulers import initialize_scheduler
from deepethogram import viz

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
        self.gpu_transforms: dict = gpu_transforms

        self.optimizer = None # will be overridden in configure_optimizers
        self.hparams.weight_decay = None
        if 'feature_extractor' in self.hparams.keys():
            self.hparams.weight_decay = self.hparams.feature_extractor.weight_decay

        self.scheduler_mode = 'min' if self.metrics.key_metric == 'loss' else 'max'
        # need to move this to top-level for lightning's learning rate finder
        # don't set it to auto here, so that we can automatically find batch size first
        self.lr = self.hparams.train.lr if self.hparams.train.lr != 'auto' else 1e-4
        log.info('scheduler mode: {}'.format(self.scheduler_mode))
        # self.is_key_metric_loss = self.metrics.key_metric == 'loss'

    def on_train_epoch_start(self) -> None:
        # self.viz_cnt['train'] = 0
        # I couldn't figure out how to make sure that this is called after BOTH train and validation ends
        if self.current_epoch > 0 and self.hparams.train.viz:
            # all models shall define a visualization function that points to the metrics file on disk
            self.visualization_func(self.metrics.fname)

    def get_dataloader(self, split: str):
        # for use with auto-batch-sizing. Lightning doesn't expect batch size to be nested, it expects it to be
        # top-level in self.hparams
        batch_size = self.hparams.compute.batch_size if self.hparams.compute.batch_size != 'auto' else \
            self.hparams.batch_size
        dataloader = DataLoader(self.datasets[split], batch_size=batch_size,
                                shuffle=True, num_workers=self.hparams.compute.num_workers,
                                pin_memory=torch.cuda.is_available(), drop_last=False)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        # We only use the test dataset to benchmark inference speed. Sometimes loss functions can be very expensive,
        # e.g. for optical flow, so we want to know how fast it is with only the forward pass and no gradients
        if 'test' in self.datasets.keys() and self.datasets['test'] is not None:
            return self.get_dataloader('test')
        else:
            return self.get_dataloader('val')

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
            images = self.gpu_transforms[mode](images)
        return images

    def configure_optimizers(self):

        weight_decay = 0 if self.hparams.weight_decay is None else self.hparams.weight_decay

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                               weight_decay=weight_decay)
        self.optimizer = optimizer
        log.info('learning rate: {}'.format(self.lr))
        scheduler = initialize_scheduler(optimizer, self.hparams, mode=self.scheduler_mode,
                                         reduction_factor=self.hparams.train.reduction_factor)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.metrics.key_metric}


# @profile
def get_trainer_from_cfg(cfg: DictConfig, lightning_module, stopper, profiler:str=None) -> pl.Trainer:
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
        tmp_metrics = lightning_module.metrics
        tmp_workers = lightning_module.hparams.compute.num_workers
        # visualize_examples = lightning_module.visualize_examples

        tuner = pl.tuner.tuning.Tuner(trainer)
        # hack for lightning to find the batch size
        cfg.batch_size = 2  # to start

        empty_metrics = EmptyMetrics()

        # don't store metrics when batch size finding
        lightning_module.metrics = empty_metrics
        # don't visualize our model inputs when batch size finding
        # lightning_module.visualize_examples = False
        should_viz = cfg.train.viz
        lightning_module.hparams.train.viz = False
        # dramatically reduces RAM usage by this process
        lightning_module.hparams.compute.num_workers = min(tmp_workers, 1)
        if cfg.compute.batch_size == 'auto':
            new_batch_size = tuner.scale_batch_size(lightning_module, mode='power', steps_per_trial=10)

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
        lightning_module.hparams.train.viz = should_viz
        lightning_module.metrics = tmp_metrics
        lightning_module.hparams.compute.num_workers = tmp_workers

    # tuning fucks with the callbacks
    trainer = pl.Trainer(gpus=[cfg.compute.gpu_id],
                         precision=16 if cfg.compute.fp16 else 32,
                         limit_train_batches=steps_per_epoch['train'],
                         limit_val_batches=steps_per_epoch['val'],
                         limit_test_batches=steps_per_epoch['test'],
                         num_sanity_val_steps=0,
                         callbacks=[FPSCallback(),  # DebugCallback(),# SpeedtestCallback(),
                                    MetricsCallback(), ExampleImagesCallback(), CheckpointCallback(),
                                    StopperCallback(stopper)],
                         reload_dataloaders_every_epoch=True,
                         profiler=profiler)

    return trainer

