import logging
import math

import torch
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)

class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# UNMERGED PULL REQUEST! NOT WRITTEN BY ME BUT SUPER USEFUL!
# https://github.com/pytorch/pytorch/pull/11104
class CosineAnnealingRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts, where :math:`\eta_{max}` is set to the
    initial learning rate, :math:`T_{cur}` is the number of epochs since the
    last restart and :math:`T_i` is the number of epochs in :math:`i`-th run
    (after performing :math:`i` restarts). If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2} \eta_{mult}^i (\eta_{max}-\eta_{min})
        (1 + \cos(\frac{T_{cur}}{T_i - 1}\pi))
        T_i = T T_{mult}^i
    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that in the
    paper the :math:`i`-th run takes :math:`T_i + 1` epochs, while in this
    implementation it takes :math:`T_i` epochs only. This implementation
    also enables updating the range of learning rates by multiplicative factor
    :math:`\eta_{mult}` after each restart.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
            the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
            learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T, eta_min=0, T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = int(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                      self.T_mult))
            last_restart = int(self.T * (1 - self.T_mult ** i_restarts) / (1 - self.T_mult))

        if self.last_epoch == last_restart:
            T_i1 = self.T * self.T_mult ** (i_restarts - 1)  # T_{i-1}
            lr_update = self.eta_mult / self._decay(T_i1 - 1, T_i1)
        else:
            T_i = self.T * self.T_mult ** i_restarts
            t = self.last_epoch - last_restart
            lr_update = self._decay(t, T_i) / self._decay(t - 1, T_i)

        return [lr_update * (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    @staticmethod
    def _decay(t, T):
        """Cosine decay for step t in run of length T, where 0 <= t < T."""
        return 0.5 * (1 + math.cos(math.pi * t / T))


def initialize_scheduler(optimizer, cfg: DictConfig, mode: str = 'max', reduction_factor: float = 0.1):
    """ Makes a learning rate scheduler from an OmegaConf DictConfig

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        one of ADAM, SGDM, etc
    cfg: DictConfig
        configuration generated by Hydra
    mode: str
        min: lr will reduce when metric stops DECREASING. useful for ERRORS, e.g. loss, SSIM
        max: lr will reduce when metric stops INCREASING. useful for PERFORMANCE, e.g. accuracy, F1
    reduction_factor: float
        Factor to multiply learning rate by each time the scheduler steps
        Useful values
            0.1: reduces by a factor of 10
            0.31622: with this value, takes two decrements to reduce learning rate by 1/10
    Returns
    -------
    scheduler
        Learning rate scheduler
    """
    if cfg.train.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=0.5)
        # for convenience
        scheduler.name = 'multistep'
    elif cfg.train.scheduler == 'cosine':
        # todo: reconfigure this to use pytorch's new built-in cosine annealing
        scheduler = CosineAnnealingRestartsLR(optimizer, T=25, T_mult=1, eta_mult=0.5, eta_min=1e-7)
        scheduler.name = 'cosine'
    elif cfg.train.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=reduction_factor,
                                                               patience=cfg.train.patience, verbose=True,
                                                               min_lr=cfg.train.min_lr)
        scheduler.name = 'plateau'
    else:
        scheduler = None
    return scheduler
