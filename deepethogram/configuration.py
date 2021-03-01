import os
from typing import Union

from omegaconf import OmegaConf, DictConfig

import deepethogram
from deepethogram import projects

def config_string_to_path(config_path, string): 
    fullpath = os.path.join(config_path, *string.split('/'))  + '.yaml'
    assert os.path.isfile(fullpath)
    return fullpath


def load_config_by_name(string, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(deepethogram.__file__), 'conf')
        
    path = config_string_to_path(config_path, string)
    return OmegaConf.load(path)


def make_config(project_path: Union[str, os.PathLike], config_list: list, run_type: str, model: str, 
               use_command_line: bool=False, preset: str=None, debug: bool=False) -> DictConfig:
    
    # config_path = os.path.join(os.path.dirname(deepethogram.__file__), 'conf')
    
    user_cfg = projects.get_config_file_from_project_path(project_path)

    # order of operations: first, defaults specified in config_list
    # then, if preset is specified in user config or at the command line, load those preset values
    # then, append the user config
    # then, the command line args
    # so if we specify a preset and manually change, say, the feature extractor architecture, we can do that
    if 'preset' in user_cfg:
        config_list.append('preset/' + user_cfg.preset)
    
    if use_command_line:
        command_line_cfg = OmegaConf.from_cli()
        if 'preset' in command_line_cfg:
            config_list.append('preset/' + command_line_cfg.preset)
    
    # add this option so we can add a preset programmatically
    if preset is not None:
        assert preset in ['deg_f', 'deg_m', 'deg_s']
        config_list.append('preset/' + preset)
        
    if debug:
        config_list.append('debug')
    # config_files = [config_string_to_path(config_path, i) for i in config_list]
    
    cfgs = [load_config_by_name(i) for i in config_list]
    
    # cfgs = [OmegaConf.load(i) for i in config_files]    
    
        # first defaults; then user cfg; then cli
    if use_command_line:
        cfg = OmegaConf.merge(*cfgs, user_cfg, command_line_cfg)
    else:
        cfg = OmegaConf.merge(*cfgs, user_cfg)

    cfg.run = {'type': run_type, 'model': model}
    return cfg

def make_flow_generator_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig: 
    config_list = ['config','augs','model/flow_generator','train']
    run_type = 'train'
    model = 'flow_generator'
    
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                     **kwargs)
    
    return cfg

def make_feature_extractor_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig: 
    config_list = ['config','augs','model/flow_generator','train', 'model/feature_extractor']
    run_type = 'train'
    model = 'feature_extractor'
    
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                     **kwargs)
    
    return cfg

def make_feature_extractor_inference_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig: 
    config_list = ['config','augs','model/feature_extractor', 'model/flow_generator', 'inference']
    run_type = 'inference'
    model = 'feature_extractor'
    
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                     **kwargs)
    
    return cfg
    
def make_sequence_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig: 
    config_list = ['config','model/feature_extractor', 'train', 'model/sequence']
    run_type = 'train'
    model = 'sequence'
    
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                     **kwargs)
    
    return cfg

def make_sequence_inference_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig: 
    config_list = ['config','augs','model/feature_extractor', 'model/sequence', 'inference']
    run_type = 'inference'
    model = 'sequence'
    
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                     **kwargs)
    
    return cfg