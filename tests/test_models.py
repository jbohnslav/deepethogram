import numpy as np
import torch

from deepethogram.feature_extractor.models.CNN import get_cnn


def test_get_cnn():
    model_name = "resnet18"
    num_classes = 2

    pos = np.array([0, 300])
    neg = np.array([300, 0])

    model = get_cnn(model_name=model_name, num_classes=num_classes, pos=pos, neg=neg)
    bias = list(model.children())[-1].bias

    assert torch.allclose(bias, torch.Tensor([0, 1]).float())
    print()
