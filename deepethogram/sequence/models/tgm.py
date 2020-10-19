import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_pad(stride, k, s):
    if s % stride == 0:
        return max(k - stride, 0)
    else:
        return max(k - (s % stride), 0)


class TGMLayer(nn.Module):
    # edited only slightly from https://github.com/piergiaj/tgm-icml19/
    def __init__(self, D: int = 1024, n_filters: int = 3, filter_length: int = 30, c_in: int = 1, c_out: int = 1,
                 soft: bool = False,
                 viz: bool = False):
        super().__init__()
        self.D = D
        self.n_filters = n_filters
        self.filter_length = filter_length
        self.c_in = c_in
        self.c_out = c_out
        self.soft = soft
        self.viz = viz

        # create parameteres for center and delta of this super event
        self.center = nn.Parameter(torch.FloatTensor(self.n_filters))
        self.delta = nn.Parameter(torch.FloatTensor(self.n_filters))
        self.gamma = nn.Parameter(torch.FloatTensor(self.n_filters))
        # init them around 0
        self.center.data.normal_(0, 0.5)
        self.delta.data.normal_(0, 0.01)
        self.gamma.data.normal_(0, 0.0001)

        self.soft_attn = nn.Parameter(torch.Tensor(self.c_out * self.c_in, self.n_filters))
        # edited from original code, which had no initialization
        torch.nn.init.xavier_normal_(self.soft_attn)
        # init_sparse_positive(self.soft_attn, 0.1, std=1)
        # init_sparse(self.soft_attn, 0.5, std=1)
        # torch.nn.init.orthogonal_(self.soft_attn)
        # torch.nn.init.eye_(self.soft_attn)
        # torch.nn.init.sparse_(self.soft_attn, sparsity=0.5, std=1)
        # learn c_out combinations of the c_in channels
        if self.c_in > 1 and not self.soft:
            self.convs = nn.ModuleList([nn.Conv2d(self.c_in, 1, (1, 1)) for c in range(self.c_out)])
        if self.c_in > 1 and soft:
            # self.cls_attn = nn.Parameter(torch.Tensor(1,self.c_out, self.c_in,1,1))
            cls_attn = torch.Tensor(self.c_out, self.c_in)
            torch.nn.init.xavier_normal_(cls_attn)
            # torch.nn.init.sparse_(cls_attn, sparsity=0.5, std=1)
            self.cls_attn = nn.Parameter(cls_attn.unsqueeze(2).unsqueeze(2).unsqueeze(0))
            # print(self.cls_attn.shape)

    def get_filters(self):
        device = self.center.device

        # scale to length of videos
        centers = (self.filter_length - 1) * (torch.tanh(self.center) + 1) / 2.0
        deltas = self.filter_length * (1.0 - torch.abs(torch.tanh(self.delta)))
        gammas = torch.exp(1.5 - 2.0 * torch.abs(torch.tanh(self.gamma)))

        # print(centers, deltas, gammas)
        a = torch.zeros(self.n_filters).to(device)

        # stride and center
        a = deltas[:, None] * a[None, :]
        a = centers[:, None] + a
        # print(a)
        b = torch.arange(0, self.filter_length).to(device)
        # b = b.cuda()

        f = b - a[:, :, None]
        f = f / gammas[:, None, None]

        f = f ** 2.0
        f += 1.0
        f = np.pi * gammas[:, None, None] * f
        f = 1.0 / f
        f = f / (torch.sum(f, dim=2) + 1e-6)[:, :, None]

        f = f[:, 0, :].contiguous()

        f = f.view(-1, self.n_filters, self.filter_length)
        return f

    def forward(self, x):
        # overwrite the forward pass to get the TSF as conv kernels
        t = x.size(-1)
        k = self.get_filters()
        # k = super(TGM, self).get_filters(torch.tanh(self.delta), torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)
        # k is shape 1xNxL
        k = k.squeeze()
        # is k now NxL

        kernels = k
        # make attn sum to 1 along the num_gaussians
        soft_attn = F.softmax(self.soft_attn, dim=1)
        # apply soft attention to convert (C_out*C_in x N)*(NxL) to C_out*C_in x L
        k = torch.mm(soft_attn, k)
        # k now contains a linear combination of gaussian kernels, according to soft_attn weights
        # because of softmax, the rows of k sum to one

        # make k C_out*C_in x 1x1xL
        k = k.unsqueeze(1).unsqueeze(1)
        p = compute_pad(1, self.filter_length, t)
        pad_f = p // 2
        pad_b = p - pad_f
        # x is shape CxDxT
        x = F.pad(x, (pad_f, pad_b))
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.expand(-1, self.c_in, -1, -1)

        # use groups to separate the class channels
        # but apply it in groups of C_out
        chnls = []
        for i in range(self.c_out):
            # output of C_in xDxT
            # indexing selects one row of k of shape C_in x1x1xL
            # grouped convolution applies to every C_in (of shape 1xDxT)
            r = F.conv2d(x, k[i * self.c_in:(i + 1) * self.c_in], groups=self.c_in).squeeze(1)
            # print('r: {}'.format(r.shape))
            # now, you have a stack of NxC_in x D x T
            # 1x1 conv to combine C_in to 1
            if self.c_in > 1 and not self.soft:
                r = F.relu(self.convs[i](r)).squeeze(1)
                # print('r2:{}'.format(r.shape))
                # print 'r2', r.size()
            chnls.append(r)
        # get C_out x DxT
        f = torch.stack(chnls, dim=1)
        # print('f: {}'.format(f.shape))
        f_stack = f
        if self.c_in > 1 and self.soft:
            a = F.softmax(self.cls_attn, dim=2).expand(f.size(0), -1, -1, f.size(3), f.size(4))
            # print('a:{}'.format(a.shape))
            f = torch.sum(a * f, dim=2)
        else:
            a = None
        if self.viz:
            return (kernels, k, soft_attn, f_stack, f, a)
        else:
            return f


class TGM(nn.Module):
    def __init__(self, D: int = 1024, n_filters: int = 16, filter_length: int = 30, input_dropout: float = 0.5,
                 dropout_p: float = 0.5,
                 classes: int = 8, num_layers: int = 3, reduction: str = 'max', c_in: int = 1, c_out: int = 8,
                 soft: bool = False, num_features: int = 512, viz: bool = False,
                 nonlinear_classification: bool = False, concatenate_inputs=True):
        super().__init__()

        self.D = D  # dimensionality of inputs. E.G. 1024 features from a CNN penultimate layer
        self.n_filters = n_filters  # how many gaussian filters to use (before attention)
        self.filter_length = filter_length  # shape of gaussian filters, [5,30] ~~
        self.classes = classes  # number of classes including background
        self.input_dropout = nn.Dropout(input_dropout)  # probability to dropout input channels
        self.output_dropout = nn.Dropout(dropout_p)  # probability to dropout final layer before FC
        self.num_layers = num_layers  # how many TGM layers
        assert (reduction in ['max', 'mean', 'conv1x1'])
        self.reduction = reduction  # NEW: how to go from N x C_out x D x T -> N x D x T. Paper: max
        self.c_in = c_in  # how many DxT representations there are in inputs
        self.c_out = c_out  # how many representations of the input DxT matrix in TGM layers
        self.soft = soft  # whether to use soft attention or 1d conv to reduce C_in x D x T -> 1 x D x T
        self.num_features = num_features  # conv1d N x D*2 x t -> N x num_features x T
        self.viz = viz
        self.concatenate_inputs = concatenate_inputs

        # self.add_module('d', self.dropout)
        modules = []
        for i in range(num_layers):
            c_in = self.c_in if i == 0 else self.c_out
            modules.append(TGMLayer(self.D, self.n_filters, self.filter_length, c_in, self.c_out, self.soft))

        self.tgm_layers = nn.Sequential(*modules)

        if self.reduction == 'conv1x1':
            self.reduction_layer = nn.Conv2d(self.c_out, 1, kernel_size=1, padding=0, stride=1)
        # self.sub_event1 = TGM(inp, 16, 5, c_in=1, c_out=8, soft=False)
        # self.sub_event2 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        # self.sub_event3 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        N = self.D * 2 if self.concatenate_inputs else self.D
        if nonlinear_classification:
            self.h = nn.Conv1d(N, self.num_features, 1)
            self.classify = nn.Conv1d(self.num_features, classes, 1)
        else:
            self.h = None
            self.classify = nn.Conv1d(N, classes, 1)

        # self.inp = inp
        self.viz = viz

    def forward(self, inp):

        smoothed = self.tgm_layers(inp)
        # print('smoothed before max:{}'.format(smoothed.shape))
        if self.reduction == 'max':
            smoothed = torch.max(smoothed, dim=1)[0]
        elif self.reduction == 'mean':
            smoothed = torch.mean(smoothed, dim=1)
        elif self.reduction == 'conv1x1':
            smoothed = self.reduction_layer(smoothed).squeeze()
        # sub_event = self.dropout(torch.max(sub_event, dim=1)[0])
        # print('sub_event:{}'.format(smoothed.shape))

        # concatenate original data with the learned smoothing
        if inp.shape != smoothed.shape:
            if inp.ndim == 3 and smoothed.ndim == 2:
                smoothed = smoothed.unsqueeze(0)
            else:
                print('ERROR')
                import pdb
                pdb.set_trace()

        if self.concatenate_inputs:
            tgm_module_output = torch.cat([inp, smoothed], dim=1)
        else:
            tgm_module_output = smoothed
        # dropout randomly!

        if self.h is not None:
            tgm_module_output = self.input_dropout(tgm_module_output)
            # NEW: got rid of relu on input features
            # cls = F.relu(concatenated)
            cls = F.relu(self.h(tgm_module_output))
            cls = self.output_dropout(cls)
        else:
            cls = self.output_dropout(tgm_module_output)

        if self.viz:
            return (smoothed, cls, self.classify(cls))
        else:
            return self.classify(cls)


class TGMJ(nn.Module):
    def __init__(self, D: int = 1024, n_filters: int = 16, filter_length: int = 30, input_dropout: float = 0.5,
                 dropout_p: float = 0.5,
                 classes: int = 8, num_layers: int = 3, reduction: str = 'max', c_in: int = 1, c_out: int = 8,
                 soft: bool = False, num_features: int = 512, viz: bool = False,
                 nonlinear_classification: bool = False, concatenate_inputs=True, pos=None, neg=None,
                 use_fe_logits: bool = True):
        super().__init__()

        self.D = D  # dimensionality of inputs. E.G. 1024 features from a CNN penultimate layer
        self.n_filters = n_filters  # how many gaussian filters to use (before attention)
        self.filter_length = filter_length  # shape of gaussian filters, [5,30] ~~
        self.classes = classes  # number of classes including background
        self.input_dropout = nn.Dropout(input_dropout)  # probability to dropout input channels
        self.output_dropout = nn.Dropout(dropout_p)  # probability to dropout final layer before FC
        self.num_layers = num_layers  # how many TGM layers
        assert (reduction in ['max', 'mean', 'conv1x1'])
        self.reduction = reduction  # NEW: how to go from N x C_out x D x T -> N x D x T. Paper: max
        self.c_in = c_in  # how many DxT representations there are in inputs
        self.c_out = c_out  # how many representations of the input DxT matrix in TGM layers
        self.soft = soft  # whether to use soft attention or 1d conv to reduce C_in x D x T -> 1 x D x T
        self.num_features = num_features  # conv1d N x D*2 x t -> N x num_features x T
        self.viz = viz
        self.concatenate_inputs = concatenate_inputs
        # self.add_module('d', self.dropout)
        modules = []
        for i in range(num_layers):
            c_in = self.c_in if i == 0 else self.c_out
            modules.append(TGMLayer(self.D, self.n_filters, self.filter_length, c_in, self.c_out, self.soft))

        self.tgm_layers = nn.Sequential(*modules)

        if self.reduction == 'conv1x1':
            self.reduction_layer = nn.Conv2d(self.c_out, 1, kernel_size=1, padding=0, stride=1)
        # self.sub_event1 = TGM(inp, 16, 5, c_in=1, c_out=8, soft=False)
        # self.sub_event2 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        # self.sub_event3 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        N = self.D
        if nonlinear_classification:
            self.h = nn.Conv1d(N, self.num_features, 1)
            self.h2 = nn.Conv1d(N, self.num_features, 1)
            self.classify1 = nn.Conv1d(self.num_features, classes, 1)
            self.classify2 = nn.Conv1d(self.num_features, classes, 1)
        else:
            self.h = None
            self.classify1 = nn.Conv1d(N, classes, 1)
            self.classify2 = nn.Conv1d(N, classes, 1)
            # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
            if pos is not None and neg is not None:
                with torch.no_grad():
                    bias = np.nan_to_num(np.log(pos / neg), neginf=0.0)
                    bias = torch.nn.Parameter(bias)
                    # print('Custom bias: {}'.format(bias))
                    self.classify1.bias = bias
                    self.classify2.bias = bias

        # self.inp = inp
        self.viz = viz
        self.use_fe_logits = use_fe_logits
        if self.use_fe_logits:
            self.weights = nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]).float())
        # self.weights = nn.Parameter(torch.normal(mean=0, std=torch.sqrt(torch.Tensor([2/3, 2/3, 2/3]))))
        # print('initial avg weights: {}'.format(self.weights))

    def forward(self, inp, fe_logits=None):
        smoothed = self.tgm_layers(inp)
        # print('smoothed before max:{}'.format(smoothed.shape))
        if self.reduction == 'max':
            smoothed = torch.max(smoothed, dim=1)[0]
        elif self.reduction == 'mean':
            smoothed = torch.mean(smoothed, dim=1)
        elif self.reduction == 'conv1x1':
            smoothed = self.reduction_layer(smoothed).squeeze()
        # sub_event = self.dropout(torch.max(sub_event, dim=1)[0])
        # print('sub_event:{}'.format(smoothed.shape))

        # concatenate original data with the learned smoothing
        if inp.shape != smoothed.shape:
            if inp.ndim == 3 and smoothed.ndim == 2:
                smoothed = smoothed.unsqueeze(0)
            else:
                print('ERROR')
                import pdb
                pdb.set_trace()
        outputs1 = self.input_dropout(inp)
        outputs2 = self.input_dropout(smoothed)
        if self.h is not None:
            outputs1 = F.relu(self.h(outputs1))
            outputs1 = self.classify1(self.output_dropout(outputs1))
            outputs2 = F.relu(self.h(outputs2))
            outputs2 = self.classify2(self.output_dropout(outputs2))
        else:
            outputs1 = self.classify1(outputs1)
            outputs2 = self.classify2(outputs2)

        if fe_logits is not None and self.use_fe_logits:
            # print(fe_logits.shape)
            # print('fe      : min {:.4f} mean {:.4f} max {:.4f}'.format(fe_logits.min(), fe_logits.mean(), fe_logits.max()))
            # print('outputs1: min {:.4f} mean {:.4f} max {:.4f}'.format(outputs1.min(), outputs1.mean(), outputs1.max()))
            # print('outputs2: min {:.4f} mean {:.4f} max {:.4f}'.format(outputs2.min(), outputs2.mean(), outputs2.max()))
            weights = F.softmax(self.weights, dim=0)
            # print('weights: {}'.format(weights))
            return weights[0]*outputs1 + weights[1]*outputs2 + weights[2]*fe_logits
            # return (outputs1 + outputs2 + fe_logits) / 3

        return (outputs1 + outputs2) / 2
