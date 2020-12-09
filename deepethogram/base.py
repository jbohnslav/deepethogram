import logging
from typing import Tuple

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from deepethogram.callbacks import FPSCallback, DebugCallback, SpeedtestCallback, MetricsCallback, \
    ExampleImagesCallback, CheckpointCallback, StopperCallback
from deepethogram.metrics import Metrics, EmptyMetrics
from deepethogram.schedulers import initialize_scheduler
from deepethogram import utils

log = logging.getLogger(__name__)

class BaseLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics: Metrics, visualization_func,
                 visualize_examples: bool = True):
        super().__init__()

        self.model = model
        self.hparams = cfg
        self.datasets = datasets
        self.metrics = metrics
        self.visualization_func = visualization_func
        self.visualize_examples = visualize_examples

        self.optimizer = None # will be overridden in configure_optimizers


    def on_train_epoch_start(self) -> None:
        # self.viz_cnt['train'] = 0
        # I couldn't figure out how to make sure that this is called after BOTH train and validation ends
        if self.current_epoch > 0:
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

    def visualize_batch(self, images, probs, labels, split: str):
        raise NotImplementedError

        # self.viz_cnt[split] += 1

    def forward(self, batch: dict, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self, mode: str = 'min'):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.hparams.train.lr,
                               weight_decay=self.hparams.feature_extractor.weight_decay)
        self.optimizer = optimizer
        scheduler = initialize_scheduler(optimizer, self.hparams, mode=mode,
                                         reduction_factor=self.hparams.train.reduction_factor)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.metrics.key_metric}


# @profile
def get_trainer_from_cfg(cfg: DictConfig, lightning_module, stopper) -> pl.Trainer:
    steps_per_epoch = cfg.train.steps_per_epoch
    for split in ['train', 'val', 'test']:
        steps_per_epoch[split] = steps_per_epoch[split] if steps_per_epoch[split] is not None else 1.0

    # reload_dataloaders_every_epoch = True: a bit slower, but enables validation dataloader to get the new, automatic
    # learning rate schedule.
    trainer = pl.Trainer(gpus=[cfg.compute.gpu_id],
                         precision=16 if cfg.compute.fp16 else 32,
                         limit_train_batches=steps_per_epoch['train'],
                         limit_val_batches=steps_per_epoch['val'],
                         limit_test_batches=steps_per_epoch['test'],
                         num_sanity_val_steps=0,
                         callbacks=[FPSCallback(),  # DebugCallback(),# SpeedtestCallback(),
                                    MetricsCallback(), ExampleImagesCallback(), CheckpointCallback(),
                                    StopperCallback(stopper)],
                         reload_dataloaders_every_epoch=True)


    if cfg.compute.batch_size == 'auto':
        tmp_metrics = lightning_module.metrics
        visualize_examples = lightning_module.visualize_examples

        tuner = pl.tuner.tuning.Tuner(trainer)
        # hack for lightning to find the batch size
        cfg.batch_size = 2  # to start
        empty_metrics = EmptyMetrics()

        # don't store metrics when batch size finding
        lightning_module.metrics = empty_metrics
        # don't visualize our model inputs when batch size finding
        lightning_module.visualize_examples = False

        new_batch_size = tuner.scale_batch_size(lightning_module, mode='power')
        cfg.compute.batch_size = new_batch_size
        log.info('auto-tuned batch size: {}'.format(new_batch_size))

        # restore lightning module to original state
        lightning_module.metrics = tmp_metrics
        lightning_module.visualize_examples = visualize_examples

        del tuner

    return trainer

