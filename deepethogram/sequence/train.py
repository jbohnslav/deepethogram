import logging
import os
import sys
from typing import Type, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

import deepethogram.projects
from deepethogram.base import BaseLightningModule, get_trainer_from_cfg
from deepethogram import utils, projects, viz
from deepethogram.data.datasets import get_datasets_from_cfg
from deepethogram.feature_extractor.train import get_metrics, get_stopper, get_criterion
from deepethogram.schedulers import initialize_scheduler
from deepethogram.sequence.models.sequence import Linear, Conv_Nonlinear, RNN
from deepethogram.sequence.models.tgm import TGM, TGMJ

log = logging.getLogger(__name__)

plt.switch_backend('agg')


@hydra.main(config_path='../conf', config_name='sequence_train')
def main(cfg: DictConfig) -> None:
    log.debug('cwd: {}'.format(os.getcwd()))
    log.info('args: {}'.format(' '.join(sys.argv)))
    # only two custom overwrites of the configuration file
    # first, change the project paths from relative to absolute

    cfg = deepethogram.projects.convert_config_paths_to_absolute(cfg)
    if cfg.sequence.latent_name is None:
        cfg.sequence.latent_name = cfg.feature_extractor.arch
        # allow for editing
    OmegaConf.set_struct(cfg, False)
    # second, use the model directory to find the most recent run of each model type
    # cfg = projects.overwrite_cfg_with_latest_weights(cfg, cfg.project.model_path, model_type='flow_generator')
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('Configuration used: ')
    log.info(OmegaConf.to_yaml(cfg))

    model = train_from_cfg_lightning(cfg)


def train_from_cfg_lightning(cfg: DictConfig) -> nn.Module:
    datasets, data_info = get_datasets_from_cfg(cfg, 'sequence')
    utils.save_dict_to_yaml(data_info['split'], os.path.join(os.getcwd(), 'split.yaml'))
    model = build_model_from_cfg(cfg, data_info['num_features'], data_info['num_classes'],
                                 pos=data_info['pos'],
                                 neg=data_info['neg'])
    weights = projects.get_weightfile_from_cfg(cfg, model_type='sequence')
    if weights is not None:
        model = utils.load_weights(model, weights)
    log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(model)))
    stopper = get_stopper(cfg)

    metrics = get_metrics(os.getcwd(), data_info['num_classes'],
                          num_parameters=utils.get_num_parameters(model), key_metric='loss',
                          num_workers=cfg.compute.num_workers)
    criterion = get_criterion(cfg.feature_extractor.final_activation, data_info)
    lightning_module = SequenceLightning(model, cfg, datasets, metrics, criterion)
    trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
    trainer.fit(lightning_module)
    return model

# def train_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
#     rundir = os.getcwd()  # done by hydra
#
#     device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
#     if device != 'cpu': torch.cuda.set_device(device)
#     log.info('Training sequence model...')
#
#
#     # dataloaders = get_dataloaders_from_cfg(cfg, model_type='sequence')
#     # utils.save_dict_to_yaml(dataloaders['split'], os.path.join(rundir, 'split.yaml'))
#     log.debug('Num training batches {}, num val: {}'.format(len(dataloaders['train']), len(dataloaders['val'])))
#     model = build_model_from_cfg(cfg, dataloaders['num_features'], dataloaders['num_classes'],
#                                  pos=dataloaders['pos'],
#                                  neg=dataloaders['neg'])
#     weights = projects.get_weightfile_from_cfg(cfg, model_type='sequence')
#     if weights is not None:
#         model = utils.load_weights(model, weights)
#     model = model.to(device)
#     log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(model)))
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
#     torch.save(model, os.path.join(rundir, cfg.sequence.arch + '_definition.pt'))
#
#     stopper = get_stopper(cfg)
#     scheduler = initialize_scheduler(optimizer, cfg, mode='max', reduction_factor=cfg.train.reduction_factor)
#     metrics = get_metrics(rundir, num_classes=len(cfg.project.class_names),
#                           num_parameters=utils.get_num_parameters(model), key_metric='f1')
#     criterion = get_criterion(cfg.feature_extractor.final_activation, dataloaders, device)
#     steps_per_epoch = dict(cfg.train.steps_per_epoch)
#
#     model = train(model,
#                   dataloaders,
#                   criterion,
#                   optimizer,
#                   gpu_transforms,
#                   metrics,
#                   scheduler,
#                   rundir,
#                   stopper,
#                   device,
#                   steps_per_epoch,
#                   final_activation=cfg.feature_extractor.final_activation,
#                   sequence=True)


class SequenceLightning(BaseLightningModule):
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
        # images, outputs = self(batch, split)
        outputs = self(batch, split)
        probabilities = self.activation(outputs)

        loss = self.criterion(outputs, batch['labels'])

        # downsampled_t0, estimated_t0, flows_reshaped = self.reconstructor(images, outputs)
        # loss, loss_components = self.criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)
        self.visualize_batch(batch['features'], probabilities, batch['labels'], split)

        self.metrics.buffer.append(split, {
            'loss': loss.detach(),
            'probs': probabilities.detach(),
            'labels': batch['labels'].detach()
        })
        # need to use the native logger for lr scheduling, etc.
        self.log('loss', loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, 'test')

    def visualize_batch(self, features, predictions,labels, split: str):
        if not self.hparams.train.viz:
            return
        # ALWAYS VISUALIZE MODEL INPUTS JUST BEFORE FORWARD PASS
        viz_cnt = self.viz_cnt[split]
        # only visualize every 10 epochs for speed
        if viz_cnt > 10 or self.current_epoch % 10 != 0:
            return
        fig = plt.figure(figsize=(14, 14))
        viz.visualize_batch_sequence(features, predictions, labels, fig=fig)
        viz.save_figure(fig, 'batch', True, viz_cnt, split)


    def forward(self, batch: dict, mode: str) -> torch.Tensor:
        outputs = self.model(batch['features'])
        return outputs

    # def log_image_statistics(self, images):
    #     if not self.has_logged_channels and log.isEnabledFor(logging.DEBUG):
    #         if len(images.shape) == 4:
    #             N, C, H, W = images.shape
    #             log.debug('inputs shape: NCHW: {} {} {} {}'.format(N, C, H, W))
    #             log.debug('channel min:  {}'.format(images[0].reshape(C, -1).min(dim=1).values))
    #             log.debug('channel mean: {}'.format(images[0].reshape(C, -1).mean(dim=1)))
    #             log.debug('channel max : {}'.format(images[0].reshape(C, -1).max(dim=1).values))
    #             log.debug('channel std : {}'.format(images[0].reshape(C, -1).std(dim=1)))
    #         elif len(images.shape) == 5:
    #             N, C, T, H, W = images.shape
    #             log.debug('inputs shape: NCTHW: {} {} {} {} {}'.format(N, C, T, H, W))
    #             log.debug('channel min:  {}'.format(images[0].min(dim=2).values))
    #             log.debug('channel mean: {}'.format(images[0].mean(dim=2)))
    #             log.debug('channel max : {}'.format(images[0].max(dim=2).values))
    #             log.debug('channel std : {}'.format(images[0].std(dim=2)))
    #         self.has_logged_channels = True
    #
    # def log_model_statistics(self, images, outputs, labels):
    #     # will print out shape and min, mean, max, std along image channels
    #     # we use the isEnabledFor flag so that this doesnt slow down training in the non-debug case
    #     log.debug('outputs: {}'.format(outputs))
    #     log.debug('labels: {}'.format(labels))
    #     log.debug('outputs: {}'.format(outputs.shape))
    #     log.debug('labels: {}'.format(labels.shape))
    #     log.debug('label max: {}'.format(labels.max()))
    #     log.debug('label min: {}'.format(labels.min()))



def build_model_from_cfg(cfg: DictConfig, num_features: int, num_classes: int, neg: np.ndarray = None,
                         pos: np.ndarray = None):
    """
    Initializes a sequence model from a configuration dictionary.

    Depending on the configuration dictionary, can define a linear (logistic regression) model, a 1D-cnn, an RNN, or
    a temporal gaussian mixture model.

    Parameters
    ----------
    config: dict
        A configuration dictionary.
    num_features: int
        D in the TGM paper: the dimensionality of our inputs. Typically ~512 or 1024
    num_classes: int
        K in the TGM paper: number of output classes.

    Returns
    -------
    model: instance of torch.nn.Module
        A sequence model. Accepts as input N x D x T sequences, and returns N x K x T logits.

    Raises
    -------
    ValueError
        In case that the `arch` field of the configuration dictionary does not match linear, conv_nonlinear, rnn, tgm,
        or tgmj.

    See Also
    -------
    deepethogram.utils.load_default_configuration
    deepethogram.utils.prepare_parser_common
    deepethogram.utils.add_sequence_to_parser
    deepethogram.sequence.models
    """
    seq = cfg.sequence
    log.info('model building parameters: {}'.format(seq))
    if seq.arch == 'linear':
        model = Linear(num_features, num_classes, kernel_size=1)
    elif seq.arch == 'conv_nonlinear':
        model = Conv_Nonlinear(num_features, num_classes, hidden_size=seq.hidden_size,
                               dropout_p=seq.dropout_p)
    elif seq.arch == 'rnn':
        model = RNN(num_features, num_classes, style=seq.rnn_style, hidden_size=seq.hidden_size,
                    dropout=seq.hidden_dropout, num_layers=seq.num_layers,
                    output_dropout=seq.dropout_p, bidirectional=seq.bidirectional)
    elif seq.arch == 'tgm':
        model = TGM(num_features, classes=num_classes, n_filters=seq.n_filters,
                    filter_length=seq.filter_length, input_dropout=seq.input_dropout,
                    dropout_p=seq.dropout_p, num_layers=seq.num_layers, reduction=seq.tgm_reduction,
                    c_in=seq.c_in, c_out=seq.c_out, soft=seq.soft_attn,
                    num_features=seq.num_features)
    elif seq.arch == 'tgmj':
        model = TGMJ(num_features, classes=num_classes, n_filters=seq.n_filters,
                     filter_length=seq.filter_length, input_dropout=seq.input_dropout,
                     dropout_p=seq.dropout_p, num_layers=seq.num_layers, reduction=seq.tgm_reduction,
                     c_in=seq.c_in, c_out=seq.c_out, soft=seq.soft_attn,
                     num_features=seq.num_features, pos=pos, neg=neg, use_fe_logits=False,
                     nonlinear_classification=seq.nonlinear_classification)
    else:
        raise ValueError('arch not found: {}'.format(seq.arch))
    # print(model)
    return model


if __name__ == '__main__':
    sys.argv = deepethogram.projects.process_config_file_from_cl(sys.argv)
    main()
