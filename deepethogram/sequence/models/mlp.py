import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, D: int, classes: int, dropout_p: float = 0.4,
                 hidden_layers=(256, 128,),
                 pos=None, neg=None):
        super().__init__()

        neurons = [D]
        for hidden_layer in hidden_layers:
            neurons.append(hidden_layer)
        neurons.append(classes)

        layers = []
        for i in range(len(neurons) - 1):
            print(i, neurons[i])
            layers.append(nn.Linear(neurons[i], neurons[i + 1], bias=True))
            if i < len(neurons) - 2:
                layers.append(nn.Dropout(p=dropout_p))

        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        if pos is not None and neg is not None:
            with torch.no_grad():
                bias = np.nan_to_num(np.log(pos / neg), neginf=0.0)
                bias = torch.nn.Parameter(torch.from_numpy(bias).float())
                layers[-1].bias = bias

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        return self.model(features)
