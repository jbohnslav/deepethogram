import torch
from torch import nn
import numpy as np


class NLLLossCustom(nn.Module):
    def __init__(self, alpha=0.1, weight=None, ignore_index=-1):
        super().__init__()

        self.alpha = alpha
        self.should_smooth = self.alpha != 0.0

        # self.nll = nn.NLLLoss(weight=weight, reduction='none')

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        if weight is None:
            weight = torch.from_numpy(np.array([1.0])).float()
        self.weight = weight

    def forward(self, outputs, label):
        # make sure labels are one-hot
        assert (outputs.shape == label.shape)
        N, K, T = outputs.shape
        label = label.float()

        # N = label.shape[0]
        # K = label.shape[1]
        # change from N x K x T -> N x T x K
        outputs, label = outputs.permute(0, 2, 1).contiguous(), label.permute(0, 2, 1).contiguous()
        # change from N x T x K -> N*T x K
        outputs, label = outputs.view(-1, K), label.view(-1, K)

        # figure out which index to ignore before smoothing
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)

        if self.should_smooth:
            K = label.shape[1]
            # when does label smoothing help? paper
            label = label * (1 - self.alpha) + self.alpha / K
        # take the log softmax to get log(p) from outputs
        outputs = self.log_softmax(outputs)

        if self.weight.device != label.device:
            self.weight = self.weight.to(label.device)
        # 1 if not ignore_index, else 0
        # mask = 1-(label==self.ignore_index).to(torch.float).to(outputs.device)
        # negative log likelihood
        # -y * log(p)
        loss = -((label * mask * outputs * self.weight).sum(dim=1))

        loss_over_time = loss.view(N, T)
        # sum across time, mean across batch
        loss = loss_over_time.sum(dim=1).mean()
        # import pdb
        # pdb.set_trace()

        if loss < 0:
            print('negative loss')
            import pdb
            pdb.set_trace()

        return (loss)


class BCELossCustom(nn.Module):
    def __init__(self, pos_weight=None, ignore_index=-1):
        super().__init__()
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(weight=None,
                                                      reduction='none', pos_weight=pos_weight)

        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        # if weight is None:
        #     weight = torch.from_numpy(np.array([1.0])).float()
        # self.weight = weight

    def forward(self, outputs, label):
        # make sure labels are one-hot
        assert (outputs.shape == label.shape)
        N, K, T = outputs.shape
        label = label.float()

        # N = label.shape[0]
        # K = label.shape[1]
        # change from N x K x T -> N x T x K
        outputs, label = outputs.permute(0, 2, 1).contiguous(), label.permute(0, 2, 1).contiguous()
        # change from N x T x K -> N*T x K
        outputs, label = outputs.view(-1, K), label.view(-1, K)

        # figure out which index to ignore before smoothing
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)

        bceloss = self.bcewithlogitsloss(outputs, label)
        # mask the sequences outside the range of the current movie
        # e.g. if your sequence is 30 frames long, and you start on the first frame, it contains 15 bogus frames
        # and 15 real ones
        # sum across classes
        loss = (bceloss * mask).sum(dim=1)

        # if self.weight.device != label.device:
        #     self.weight = self.weight.to(label.device)
        # # 1 if not ignore_index, else 0
        # # mask = 1-(label==self.ignore_index).to(torch.float).to(outputs.device)
        # # negative log likelihood
        # # -y * log(p)
        # loss = -( (label*mask* outputs*self.weight).sum(dim=1) )

        loss_over_time = loss.view(N, T)
        # sum across time, mean across batch
        loss = loss_over_time.sum(dim=1).mean()
        # import pdb
        # pdb.set_trace()

        if loss < 0:
            print('negative loss')
            import pdb
            pdb.set_trace()

        return loss
