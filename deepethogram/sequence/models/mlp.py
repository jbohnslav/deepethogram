import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron model. Baseline for sequence modeling"""

    def __init__(
        self,
        D: int,
        classes: int,
        dropout_p: float = 0.4,
        hidden_layers=(
            256,
            128,
        ),
        pos=None,
        neg=None,
    ):
        """Constructor

        Parameters
        ----------
        D : int
            Number of neurons in our input layer
        classes : int
            Number of behaviors / neurons in our output layer
        dropout_p : float, optional
            P(dropout) for layers after input, by default 0.4
        hidden_layers : tuple, optional
            Number of neurons in each hidden layer, by default (256, 128,)
        pos : np.ndarray, optional
            Number of positive examples for each class, by default None
        neg : np.ndarray, optional
            Number of negative examples for each class, by default None
        """
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
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_p))

        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        if pos is not None and neg is not None:
            with torch.no_grad():
                bias = np.nan_to_num(np.log(pos / neg), neginf=0.0, posinf=1.0)
                bias = torch.nn.Parameter(torch.from_numpy(bias).float())
                layers[-1].bias = bias

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        return self.model(features)
