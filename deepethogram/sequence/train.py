import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from deepethogram.base import BaseLightningModule, get_trainer_from_cfg
from deepethogram import utils, projects, viz
from deepethogram.configuration import make_sequence_train_cfg
from deepethogram.data.datasets import get_datasets_from_cfg
from deepethogram.feature_extractor.train import get_metrics, get_stopper, get_criterion
from deepethogram.sequence.models.mlp import MLP
from deepethogram.sequence.models.sequence import Linear, Conv_Nonlinear, RNN
from deepethogram.sequence.models.tgm import TGM, TGMJ

log = logging.getLogger(__name__)

plt.switch_backend("agg")


def sequence_train(cfg: DictConfig) -> nn.Module:
    """Trains sequence models from a configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration, e.g. that returned by deepethogram.configration.make_sequence_train_cfg

    Returns
    -------
    nn.Module
        Trained sequence model
    """
    cfg = projects.setup_run(cfg)
    log.info("args: {}".format(" ".join(sys.argv)))

    if cfg.sequence.latent_name is None:
        cfg.sequence.latent_name = cfg.feature_extractor.arch
        # allow for editing
    OmegaConf.set_struct(cfg, False)
    log.info("Configuration used: ")
    log.info(OmegaConf.to_yaml(cfg))

    datasets, data_info = get_datasets_from_cfg(cfg, "sequence")
    utils.save_dict_to_yaml(data_info["split"], os.path.join(os.getcwd(), "split.yaml"))
    model = build_model_from_cfg(
        cfg, data_info["num_features"], data_info["num_classes"], pos=data_info["pos"], neg=data_info["neg"]
    )
    weights = projects.get_weightfile_from_cfg(cfg, model_type="sequence")
    if weights is not None:
        model = utils.load_weights(model, weights)
    log.debug("model arch: {}".format(model))
    log.info("Total trainable params: {:,}".format(utils.get_num_parameters(model)))
    stopper = get_stopper(cfg)

    metrics = get_metrics(
        os.getcwd(),
        data_info["num_classes"],
        num_parameters=utils.get_num_parameters(model),
        key_metric="f1_class_mean",
        num_workers=cfg.compute.metrics_workers,
    )
    criterion = get_criterion(cfg, model, data_info)
    lightning_module = SequenceLightning(model, cfg, datasets, metrics, criterion)
    # change auto batch size parameters because large sequences can overflow RAM
    trainer = get_trainer_from_cfg(cfg, lightning_module, stopper)
    trainer.fit(lightning_module)
    return model


class SequenceLightning(BaseLightningModule):
    """Lightning Module for training sequence models"""

    def __init__(self, model: nn.Module, cfg: DictConfig, datasets: dict, metrics, criterion: nn.Module):
        super().__init__(model, cfg, datasets, metrics, viz.visualize_logger_multilabel_classification)

        self.has_logged_channels = False
        # for convenience
        self.final_activation = self.hparams.feature_extractor.final_activation
        if self.final_activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif self.final_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.criterion = criterion

    def common_step(self, batch: dict, batch_idx: int, split: str):
        """forward pass, loss computation, input visualization

        Parameters
        ----------
        batch : dict
            contains input sequences
        batch_idx : int
            index in epoch
        split : str
            train or val

        Returns
        -------
        loss: torch.Tensor
            mean loss across inputs
        """
        # images, outputs = self(batch, split)
        outputs = self(batch, split)
        probabilities = self.activation(outputs)

        loss, loss_dict = self.criterion(outputs, batch["labels"], self.model)

        # downsampled_t0, estimated_t0, flows_reshaped = self.reconstructor(images, outputs)
        # loss, loss_components = self.criterion(batch, downsampled_t0, estimated_t0, flows_reshaped)
        self.visualize_batch(batch["features"], probabilities, batch["labels"], split)

        self.metrics.buffer.append(
            split, {"loss": loss.detach(), "probs": probabilities.detach(), "labels": batch["labels"].detach()}
        )
        self.metrics.buffer.append(split, loss_dict)
        # need to use the native logger for lr scheduling, etc.
        self.log(f"{split}_loss", loss.detach())
        # if self.batch_cnt == 100:
        #     print('stop')
        # self.batch_cnt += 1
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, batch_idx, "val")

    def test_step(self, batch: dict, batch_idx: int):
        images, outputs = self(batch, "test")

    def visualize_batch(self, features, predictions, labels, split: str):
        if self.hparams.train.viz_examples == 0:
            return
        # log.info('current epoch: {}'.format(self.current_epoch))
        # ALWAYS VISUALIZE MODEL INPUTS JUST BEFORE FORWARD PASS
        viz_cnt = self.viz_cnt[split]
        # only visualize every 10 epochs for speed
        if viz_cnt > self.hparams.train.viz_examples or self.current_epoch % 10 != 0:
            return
        fig = plt.figure(figsize=(14, 14))
        # log.info('visualizing sequence batch')
        viz.visualize_batch_sequence(features, predictions, labels, fig=fig)
        viz.save_figure(fig, "batch", True, viz_cnt, split)
        # this should happen in the save figure function, but for some reason it doesn't
        plt.close(fig)
        plt.close("all")
        del fig

    def forward(self, batch: dict, mode: str) -> torch.Tensor:
        outputs = self.model(batch["features"])
        return outputs


def build_model_from_cfg(
    cfg: DictConfig, num_features: int, num_classes: int, neg: np.ndarray = None, pos: np.ndarray = None
):
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
    log.debug("model building parameters: {}".format(seq))
    if seq.arch == "linear":
        model = Linear(num_features, num_classes, kernel_size=1)
    elif seq.arch == "conv_nonlinear":
        model = Conv_Nonlinear(num_features, num_classes, hidden_size=seq.hidden_size, dropout_p=seq.dropout_p)
    elif seq.arch == "rnn":
        model = RNN(
            num_features,
            num_classes,
            style=seq.rnn_style,
            hidden_size=seq.hidden_size,
            dropout=seq.hidden_dropout,
            num_layers=seq.num_layers,
            output_dropout=seq.dropout_p,
            bidirectional=seq.bidirectional,
        )
    elif seq.arch == "tgm":
        model = TGM(
            num_features,
            classes=num_classes,
            n_filters=seq.n_filters,
            filter_length=seq.filter_length,
            input_dropout=seq.input_dropout,
            dropout_p=seq.dropout_p,
            num_layers=seq.num_layers,
            reduction=seq.tgm_reduction,
            c_in=seq.c_in,
            c_out=seq.c_out,
            soft=seq.soft_attn,
            num_features=seq.num_features,
        )
    elif seq.arch == "tgmj":
        model = TGMJ(
            num_features,
            classes=num_classes,
            n_filters=seq.n_filters,
            filter_length=seq.filter_length,
            input_dropout=seq.input_dropout,
            dropout_p=seq.dropout_p,
            num_layers=seq.num_layers,
            reduction=seq.tgm_reduction,
            c_in=seq.c_in,
            c_out=seq.c_out,
            soft=seq.soft_attn,
            num_features=seq.num_features,
            pos=pos,
            neg=neg,
            use_fe_logits=False,
            nonlinear_classification=seq.nonlinear_classification,
            final_bn=seq.final_bn,
        )
    elif seq.arch == "mlp":
        model = MLP(num_features, num_classes, dropout_p=seq.dropout_p, pos=pos, neg=neg)
    else:
        raise ValueError("arch not found: {}".format(seq.arch))
    print(model)
    return model


if __name__ == "__main__":
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_sequence_train_cfg(project_path, use_command_line=True)

    sequence_train(cfg)
