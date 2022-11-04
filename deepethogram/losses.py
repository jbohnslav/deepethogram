import logging
import os

from omegaconf import DictConfig
import torch
from torch import nn

from deepethogram import projects

log = logging.getLogger(__name__)

def should_decay_parameter(name: str, param: torch.Tensor) -> bool:
    """Determines if L2 (or L2-SP) decay should be applied to parameter

    BatchNorm and bias parameters are excluded. Uses both the name of the parameter and shape of parameter to determine

    Helpful source:
    https://github.com/rwightman/pytorch-image-models/blob/198f6ea0f3dae13f041f3ea5880dd79089b60d61/timm/optim/optim_factory.py
    
    Parameters
    ----------
    name : str
        name of parameter
    param : torch.Tensor
        parameter

    Returns
    -------
    bool
        Whether or not to decay
    """
    
    if not param.requires_grad:
        return False
    elif 'batchnorm' in name.lower() or 'bn' in name.lower() or 'bias' in name.lower():
        return False
    elif param.ndim == 1:
        return False
    else:
        return True
    
def get_keys_to_decay(model: nn.Module) -> list:
    """Returns list of parameter keys in a nn.Module that should be decayed

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list
        keys in a state_dict that should be decayed
    """
    to_decay = []
    for name, param in model.named_parameters():
        if should_decay_parameter(name, param):
            to_decay.append(name)
    return to_decay


class L2(nn.Module):
    """L2 regularization
    """
    def __init__(self, model: nn.Module, alpha: float):
        super().__init__()
        
        self.alpha = alpha
        self.keys = get_keys_to_decay(model)    
    
    def forward(self, model):
        # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # note that soumith's answer is wrong because it uses W.norm, which takes the square root
        l2_loss = 0 # torch.tensor(0., requires_grad=True)
        for key, param in model.named_parameters():
            if key in self.keys:
                l2_loss += param.pow(2).sum()*0.5
                
        return l2_loss*self.alpha
    
class L2_SP(nn.Module):
    """L2_SP normalization; weight decay towards a pretrained state, instead of towards 0. 
    
    https://arxiv.org/abs/1802.01483
    @misc{li2018explicit,
        title={Explicit Inductive Bias for Transfer Learning with Convolutional Networks}, 
        author={Xuhong Li and Yves Grandvalet and Franck Davoine},
        year={2018},
        eprint={1802.01483},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """
    def __init__(self, model: nn.Module, path_to_pretrained_weights, alpha: float, beta: float):
        # 
        
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        # assert cfg.train.regularization.style == 'l2_sp'
        
        assert os.path.isfile(path_to_pretrained_weights)
        state = torch.load(path_to_pretrained_weights, map_location='cpu')
        
        pretrained_state = state['state_dict']
        
        self.pretrained_keys, self.new_keys = self.get_keys(model, pretrained_state)
        
        log.debug('pretrained keys for L2SP: {}'.format(self.pretrained_keys))
        log.debug('Novel keys for L2SP: {}'.format(self.new_keys))
        
        for key in self.pretrained_keys:
            # can't register a buffer with dots in the keys
            self.register_buffer(self.dots_to_underscores(key), pretrained_state[key])
    
    @staticmethod
    def dots_to_underscores(key):
        return key.replace('.', '_')
        
    def get_keys(self, model: nn.Module, pretrained_state):
        """Gets parameter names that are in both current model and pretrained weights, and unique keys to our model

        Parameters
        ----------
        model : nn.Module
            Model to train
        pretrained_state : state_dict
            State dict from pretraining

        Returns
        -------
        is_in_pretrained: list
            Keys in both model and pretrained weights
        not_in_pretrained: list
            Keys uniquely in current model
        """

        to_decay = get_keys_to_decay(model)
        model_state = model.state_dict()
        is_in_pretrained, not_in_pretrained = [], []
        for key in to_decay:
            match = False

            if key in pretrained_state.keys():
                if model_state[key].shape == pretrained_state[key].shape:
                    match = True
            if match:
                is_in_pretrained.append(key)
            else:
                not_in_pretrained.append(key)
        return is_in_pretrained, not_in_pretrained
    
    def forward(self, model):
        towards_pretrained, towards_0 = 0, 0
        
        # not passing keep_vars will detach the tensor from the computation graph, resulting in no effect on the 
        # training but also no error messages
        model_state = model.state_dict(keep_vars=True)
        pretrained_state = self.state_dict(keep_vars=True)

        for key in self.pretrained_keys:
            model_param = model_state[key]
            pretrained_param = pretrained_state[self.dots_to_underscores(key)]
            towards_pretrained += (model_param - pretrained_param).pow(2).sum()*0.5

        for key in self.new_keys:
            model_param = model_state[key]
            towards_0 += model_param.pow(2).sum()*0.5
            
        if towards_pretrained != towards_pretrained or towards_0 != towards_0:
            msg = 'invalid loss in L2-SP: towards pretrained: {} towards 0: {}'.format(towards_pretrained, towards_0)
            raise ValueError(msg)
        # alternate method. same result, ~50% slower
        #         towards_pretrained, towards_0 = 0, 0

        #         for key, param in model.named_parameters():
        #             if key in self.pretrained_keys:
        #                 pretrained_param = getattr(self, self.dots_to_underscores(key))
        #                 towards_pretrained += (param - pretrained_param).pow(2).sum()*0.5
        #             elif key in self.new_keys:
        #                 towards_0 += param.pow(2).sum()*0.5
        
        return towards_pretrained*self.alpha + towards_0*self.beta

def get_regularization_loss(cfg: DictConfig, model):
    if cfg.train.regularization.style == 'l2':
        log.info('Regularization: L2. alpha: {} '.format(cfg.train.regularization.alpha))
        regularization_criterion = L2(model, cfg.train.regularization.alpha)
    elif cfg.train.regularization.style == 'l2_sp':
        pretrained_dir = cfg.project.pretrained_path
        assert os.path.isdir(pretrained_dir)
        weights = projects.get_weights_from_model_path(pretrained_dir)
        pretrained_file = weights[cfg.run.model].get(cfg[cfg.run.model].arch, [])
        
        if len(pretrained_file) == 0:
            log.warning('No pretrained file found. Regularization: L2. alpha={}'.format(
                cfg.train.regularization.beta
            ))
            regularization_criterion = L2(model, cfg.train.regularization.beta)
        elif len(pretrained_file) == 1:
            
            pretrained_file = pretrained_file[0]
            log.info('Regularization: L2_SP. Pretrained file: {} alpha: {} beta: {}'.format(
                pretrained_file, cfg.train.regularization.alpha, cfg.train.regularization.beta
            ))
            regularization_criterion = L2_SP(model, pretrained_file, cfg.train.regularization.alpha, 
                                            cfg.train.regularization.beta)
        else:
            raise ValueError('unsure what weights to use: {}'.format(pretrained_file))
    else:
        raise NotImplementedError
    
    return regularization_criterion