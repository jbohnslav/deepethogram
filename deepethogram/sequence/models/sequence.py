from torch import nn


def conv1d_same(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    # if stride is two, output should be exactly half the size of input
    padding = kernel_size // 2 * dilation

    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class Linear(nn.Module):
    def __init__(self, num_features, num_classes, kernel_size=1):
        super().__init__()
        self.conv1 = conv1d_same(num_features, num_classes, kernel_size=kernel_size, stride=1, bias=True)

    def forward(self, x):
        return self.conv1(x)


class Conv_Nonlinear(nn.Module):
    def __init__(self, num_features, num_classes, batchnorm=True, hidden_size=64, dropout_p=0.0):
        super().__init__()

        bias = not batchnorm
        self.conv1 = conv1d_same(num_features, hidden_size, kernel_size=7, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = conv1d_same(hidden_size, num_classes, kernel_size=3, stride=1, bias=bias)

        self.activation = nn.ReLU()

        layers = []
        layers.append(self.conv1)
        if batchnorm:
            layers.append(self.bn1)
        layers.append(self.activation)
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(self.conv2)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        style="lstm",
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        output_dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        assert style in ["rnn", "lstm", "gru"]
        if style == "rnn":
            func = nn.RNN
        elif style == "lstm":
            func = nn.LSTM
        elif style == "gru":
            func = nn.GRU

        self.rnn = func(
            num_features,
            hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(output_dropout)
        size = hidden_size * 2 if bidirectional else hidden_size
        self.hidden_to_output = nn.Linear(size, num_classes)

    def forward(self, x):
        # change from N, C, L (for 1d conv) to N, L, C
        x = x.permute(0, 2, 1).contiguous()
        # hidden state is always 0 at input
        # hiddens is hidden units at each T, shape: N,L,C
        hiddens, _ = self.rnn(x)
        hiddens = self.dropout(hiddens)
        # outputs is N, L, C
        outputs = self.hidden_to_output(hiddens)
        # return outputs in shape N, C, L to be the same as conv1d
        return outputs.permute(0, 2, 1).contiguous()
