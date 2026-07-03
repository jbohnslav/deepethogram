import torch
from omegaconf import OmegaConf

from deepethogram import losses


def test_l2_sp_missing_pretrained_architecture_falls_back_to_l2(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "project": {"pretrained_path": str(tmp_path)},
            "run": {"model": "sequence"},
            "sequence": {"arch": "missing_arch"},
            "train": {
                "regularization": {
                    "style": "l2_sp",
                    "alpha": 0.1,
                    "beta": 0.2,
                }
            },
        }
    )
    model = torch.nn.Linear(2, 1)

    def no_sequence_weights(_):
        return {"sequence": {}}

    monkeypatch.setattr(losses.projects, "get_weights_from_model_path", no_sequence_weights)

    regularization = losses.get_regularization_loss(cfg, model)

    assert isinstance(regularization, losses.L2)
    assert regularization.alpha == cfg.train.regularization.beta
