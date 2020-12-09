import logging
import os
import sys
from typing import Type

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

import deepethogram.projects
from deepethogram import utils, projects
from deepethogram.data.dataloaders import get_dataloaders_from_cfg
from deepethogram.feature_extractor.train import (get_metrics, train,
                                                  get_stopper, get_criterion)
from deepethogram.schedulers import initialize_scheduler
from .models.sequence import Linear, Conv_Nonlinear, RNN
from .models.tgm import TGM, TGMJ

log = logging.getLogger(__name__)

plt.switch_backend('agg')


@hydra.main(config_path='../conf/sequence_train.yaml')
def main(cfg: DictConfig) -> None:
    log.debug('cwd: {}'.format(os.getcwd()))
    log.info('args: {}'.format(' '.join(sys.argv)))
    # only two custom overwrites of the configuration file
    # first, change the project paths from relative to absolute

    cfg = deepethogram.projects.parse_cfg_paths(cfg)
    if cfg.sequence.latent_name is None:
        cfg.sequence.latent_name = cfg.feature_extractor.arch
    # second, use the model directory to find the most recent run of each model type
    # cfg = projects.overwrite_cfg_with_latest_weights(cfg, cfg.project.model_path, model_type='flow_generator')
    # SHOULD NEVER MODIFY / MAKE ASSIGNMENTS TO THE CFG OBJECT AFTER RIGHT HERE!
    log.info('Configuration used: ')
    log.info(cfg.pretty())

    model = train_from_cfg(cfg)


def train_from_cfg(cfg: DictConfig) -> Type[nn.Module]:
    rundir = os.getcwd()  # done by hydra

    device = torch.device("cuda:" + str(cfg.compute.gpu_id) if torch.cuda.is_available() else "cpu")
    if device != 'cpu': torch.cuda.set_device(device)
    log.info('Training sequence model...')

    gpu_transforms = get_empty_gpu_transforms()

    dataloaders = get_dataloaders_from_cfg(cfg, model_type='sequence')
    utils.save_dict_to_yaml(dataloaders['split'], os.path.join(rundir, 'split.yaml'))
    log.debug('Num training batches {}, num val: {}'.format(len(dataloaders['train']), len(dataloaders['val'])))
    model = build_model_from_cfg(cfg, dataloaders['num_features'], dataloaders['num_classes'],
                                 pos=dataloaders['pos'],
                                 neg=dataloaders['neg'])
    weights = projects.get_weightfile_from_cfg(cfg, model_type='sequence')
    if weights is not None:
        model = utils.load_weights(model, weights)
    model = model.to(device)
    log.info('Total trainable params: {:,}'.format(utils.get_num_parameters(model)))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    torch.save(model, os.path.join(rundir, cfg.sequence.arch + '_definition.pt'))

    stopper = get_stopper(cfg)
    scheduler = initialize_scheduler(optimizer, cfg, mode='max', reduction_factor=cfg.train.reduction_factor)
    metrics = get_metrics(rundir, num_classes=len(cfg.project.class_names),
                          num_parameters=utils.get_num_parameters(model), key_metric='f1')
    criterion = get_criterion(cfg.feature_extractor.final_activation, dataloaders, device)
    steps_per_epoch = dict(cfg.train.steps_per_epoch)

    model = train(model,
                  dataloaders,
                  criterion,
                  optimizer,
                  gpu_transforms,
                  metrics,
                  scheduler,
                  rundir,
                  stopper,
                  device,
                  steps_per_epoch,
                  final_activation=cfg.feature_extractor.final_activation,
                  sequence=True)


def get_empty_gpu_transforms():
    gpu_transforms = dict(train=nn.Identity(),
                          val=nn.Identity(),
                          test=nn.Identity(),
                          denormalize=nn.Identity())
    return gpu_transforms



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
