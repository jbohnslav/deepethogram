"""Behavioral regressions for direction-sensitive metric consumers.

Cover loss minimization, score maximization, and explicit custom directions.
No GPU, training data, or Ray service is required.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from deepethogram import base, stoppers, viz
from deepethogram.callbacks import StopperCallback
from deepethogram.metrics import Classification, EmptyMetrics, OpticalFlow, get_metric_mode


OBJECTIVES = [
    ("SSIM", "min"),
    ("L1", "min"),
    ("smoothness", "min"),
    ("sparsity", "min"),
    ("loss", "min"),
    ("data_loss", "min"),
    ("reg_loss", "min"),
    ("f1_class_mean", "max"),
    ("f1_class_mean_nobg", "max"),
]


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "run": {"model": "sequence", "dir": str(tmp_path)},
            "compute": {"batch_size": 2, "gpu_id": 0, "fp16": False},
            "train": {
                "lr": 0.1,
                "num_epochs": 20,
                "patience": 2,
                "scheduler": "plateau",
                "reduction_factor": 0.1,
                "min_lr": 1e-7,
                "early_stopping_begins": 0,
                "stopping_type": "early",
                "steps_per_epoch": {"train": 1, "val": 1, "test": 1},
            },
        }
    )


def make_metrics(tmp_path, key):
    cls = OpticalFlow if key in {"SSIM", "L1", "smoothness", "sparsity"} else Classification
    return cls(tmp_path, key, num_parameters=1)


@pytest.mark.parametrize("key,expected", OBJECTIVES)
def test_checkpoint_selects_better_metric(cfg, tmp_path, monkeypatch, key, expected):
    """Exercise the configured Lightning checkpoint comparator, not a copied heuristic."""
    monkeypatch.setattr(base.pl, "Trainer", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(base.pl.loggers.tensorboard, "TensorBoardLogger", Mock())
    module = SimpleNamespace(metrics=make_metrics(tmp_path, key))
    trainer = base.get_trainer_from_cfg(cfg, module, stoppers.get_stopper(cfg))
    checkpoint = next(c for c in trainer.callbacks if isinstance(c, base.pl.callbacks.ModelCheckpoint))
    better, worse = (0.1, 0.8) if expected == "min" else (0.8, 0.1)
    assert checkpoint.monitor == f"val/{key}"
    checkpoint.best_k_models = {"previous.ckpt": torch.tensor(worse)}
    checkpoint.kth_best_model_path = "previous.ckpt"
    trainer.strategy = SimpleNamespace(reduce_boolean_decision=bool)
    assert checkpoint.check_monitor_top_k(trainer, torch.tensor(better)), key
    checkpoint.best_k_models["previous.ckpt"] = torch.tensor(better)
    assert not checkpoint.check_monitor_top_k(trainer, torch.tensor(worse)), key


@pytest.mark.parametrize("key,expected", OBJECTIVES)
def test_scheduler_reduces_lr_only_after_metric_worsens(cfg, tmp_path, key, expected):
    metrics = make_metrics(tmp_path, key)
    module = base.BaseLightningModule(
        torch.nn.Linear(1, 1), cfg, {"train": SimpleNamespace(labels=None)}, metrics, None
    )
    module.hparams.train.patience = 0
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]
    scores = [0.8, 0.4, 0.6] if expected == "min" else [0.2, 0.6, 0.4]
    scheduler.step(scores[0])
    scheduler.step(scores[1])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1), "Improvement must not reduce LR"
    scheduler.step(scores[2])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "key,expected", [("SSIM", "min"), ("loss", "min"), ("f1_class_mean", "max"), ("f1_class_mean_nobg", "max")]
)
def test_early_stopping_keeps_improving_runs(cfg, tmp_path, key, expected):
    callback = StopperCallback(stoppers.get_stopper(cfg))
    metrics = make_metrics(tmp_path, key)
    module = SimpleNamespace(metrics=metrics, current_epoch=1)
    trainer = SimpleNamespace(should_stop=False)
    scores = [0.8, 0.6, 0.4, 0.2] if expected == "min" else [0.2, 0.4, 0.6, 0.8]
    for epoch, score in enumerate(scores, start=1):
        module.current_epoch = epoch
        metrics.latest_key["val"] = score
        callback.on_train_epoch_end(trainer, module)
        assert not trainer.should_stop, "An improving run must not stop"


def test_early_stopping_handles_first_non_improvement(cfg):
    stopper = stoppers.get_stopper(cfg)
    assert stopper(0.2) == (True, False)
    assert stopper(0.3) == (False, False)
    assert stopper(0.4) == (False, True)


@pytest.mark.parametrize(
    "key,scores,mode",
    [
        ("loss", [0.8, 0.1, 0.4], None),
        ("f1_class_mean", [0.1, 0.8, 0.4], None),
        ("custom_error", [0.8, 0.1, 0.4], "min"),
        ("loss", [0.1, 0.8, 0.4], "max"),
    ],
)
def test_confusion_plot_uses_best_epoch(tmp_path, monkeypatch, key, scores, mode):
    filename = tmp_path / "metrics.h5"
    matrices = np.stack([np.eye(2) * n for n in (1, 2, 3)])
    with h5py.File(filename, "w") as f:
        f.attrs["key_metric"] = key
        if mode is not None:
            f.attrs["key_metric_mode"] = mode
        f.create_dataset(f"val/{key}", data=scores)
        for split in ("train", "val"):
            f.create_dataset(f"{split}/confusion", data=matrices)
    plotted = []
    monkeypatch.setattr(viz, "plot_confusion_matrix", lambda cm, *a, **kw: plotted.append(cm))
    fig = plt.figure()
    try:
        viz.plot_confusion_from_logger(filename, fig)
        assert len(plotted) == 4
        for cm in plotted:
            np.testing.assert_array_equal(cm, matrices[1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stage", ["feature_extractor", "sequence"])
@pytest.mark.parametrize("search", ["random", "hyperopt"])
@pytest.mark.parametrize(
    "key,expected,override",
    [
        ("val/loss", "min", None),
        ("val/f1_class_mean_nobg", "max", None),
        ("val/custom", "min", "min"),
        ("val/loss", "max", "max"),
    ],
)
def test_tuning_passes_metric_direction(tmp_path, monkeypatch, stage, search, key, expected, override):
    """Call the real orchestration with a fake Ray boundary; no optional Ray install."""
    ray_modules = {}
    for name in ("ray", "ray.tune", "ray.tune.schedulers", "ray.tune.suggest", "ray.tune.suggest.hyperopt"):
        ray_modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, ray_modules[name])
    tune = ray_modules["ray.tune"]
    ray_modules["ray"].tune = tune
    tune.CLIReporter = Mock()
    tune.with_parameters = Mock()
    tune.run = Mock(return_value=SimpleNamespace(best_config={}, results_df=SimpleNamespace(to_csv=Mock())))
    ray_modules["ray.tune.schedulers"].ASHAScheduler = Mock()
    hyperopt = Mock()
    ray_modules["ray.tune.suggest.hyperopt"].HyperOptSearch = hyperopt
    # Isolate training imports too: tuning must only configure a run in this test.
    import deepethogram

    monkeypatch.setattr(deepethogram, "sequence_train", Mock(), raising=False)
    training = ModuleType("deepethogram.feature_extractor.train")
    training.feature_extractor_train = Mock()
    monkeypatch.setitem(sys.modules, training.__name__, training)
    utils_path = Path(base.__file__).parent / "tune" / "utils.py"
    spec = importlib.util.spec_from_file_location("deepethogram.tune.utils", utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, utils_module)
    spec.loader.exec_module(utils_module)
    spec = importlib.util.spec_from_file_location(f"_direction_test_{stage}", utils_path.with_name(f"{stage}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = OmegaConf.create(
        {
            "train": {"num_epochs": 3},
            "project": {"model_path": str(tmp_path)},
            "tune": {
                "grace_period": 1,
                "hparams": {},
                "search": search,
                "key_metric": key,
                "resources_per_trial": {"cpu": 1},
                "num_trials": 1,
                "name": "direction-test",
            },
        }
    )
    if override is not None:
        cfg.tune.key_metric_mode = override
    getattr(module, f"tune_{stage}")(cfg)
    directions = {
        "run": tune.run.call_args.kwargs["mode"],
        "scheduler": ray_modules["ray.tune.schedulers"].ASHAScheduler.call_args.kwargs["mode"],
    }
    if search == "hyperopt":
        directions["search"] = hyperopt.call_args.kwargs["mode"]
    assert directions == dict.fromkeys(directions, expected)


@pytest.mark.parametrize("cls", [Classification, OpticalFlow])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_custom_metric_direction_is_persisted(tmp_path, cls, mode):
    metrics = cls(tmp_path, "custom_objective", 1, key_metric_mode=mode)
    assert metrics.key_metric_mode == mode
    with h5py.File(metrics.fname, "r") as f:
        assert f.attrs["key_metric_mode"] == mode


@pytest.mark.parametrize("key,mode", [("unknown", None), ("loss", "sideways"), ("lr", None)])
def test_ambiguous_or_invalid_direction_is_rejected(key, mode):
    with pytest.raises(ValueError):
        get_metric_mode(key, mode)


def test_empty_metrics_retains_loss_direction():
    metrics = EmptyMetrics()
    assert metrics.key_metric == "loss"
    assert metrics.key_metric_mode == "min"


def test_explicit_override_reaches_training_consumers(cfg, tmp_path, monkeypatch):
    metrics = Classification(tmp_path, "f1_class_mean", 1, key_metric_mode="min")
    module = base.BaseLightningModule(
        torch.nn.Linear(1, 1), cfg, {"train": SimpleNamespace(labels=None)}, metrics, None
    )
    assert module.configure_optimizers()["lr_scheduler"].mode == "min"
    monkeypatch.setattr(base.pl, "Trainer", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(base.pl.loggers.tensorboard, "TensorBoardLogger", Mock())
    trainer = base.get_trainer_from_cfg(cfg, module, stoppers.get_stopper(cfg))
    checkpoint = next(c for c in trainer.callbacks if isinstance(c, base.pl.callbacks.ModelCheckpoint))
    assert checkpoint.mode == "min"
    stopper = next(c for c in trainer.callbacks if isinstance(c, StopperCallback))
    proxy = SimpleNamespace(metrics=metrics, current_epoch=1)
    trainer.should_stop = False
    for score in (0.8, 0.6, 0.4, 0.2):
        metrics.latest_key["val"] = score
        stopper.on_train_epoch_end(trainer, proxy)
        assert not trainer.should_stop


@pytest.mark.parametrize("mode", ["min", "max"])
def test_classification_factory_forwards_custom_direction(tmp_path, mode):
    from deepethogram.feature_extractor.train import get_metrics

    metrics = get_metrics(
        tmp_path,
        num_classes=2,
        num_parameters=1,
        key_metric="auxiliary_loss",
        key_metric_mode=mode,
    )
    assert metrics.key_metric == "auxiliary_loss"
    assert metrics.key_metric_mode == mode
    with h5py.File(metrics.fname, "r") as f:
        assert f.attrs["key_metric_mode"] == mode


@pytest.mark.parametrize("key,expected", [("loss", "min"), ("f1_class_mean_nobg", "max")])
def test_classification_factory_preserves_default_direction(tmp_path, key, expected):
    from deepethogram.feature_extractor.train import get_metrics

    metrics = get_metrics(tmp_path, num_classes=2, num_parameters=1, key_metric=key)
    assert metrics.key_metric_mode == expected
