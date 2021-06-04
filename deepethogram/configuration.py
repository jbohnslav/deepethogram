import os
from typing import Union

from omegaconf import OmegaConf, DictConfig

import deepethogram
from deepethogram import projects


def config_string_to_path(config_path: Union[str, os.PathLike], string: str) -> str:
    """Converts a string name to an absolute path

    Parameters
    ----------
    config_path : Union[str, os.PathLike]
        absolute path to deepethogram/deepethogram/conf directory
    string : str
        name of configuration. 

    Returns
    -------
    str
        Absolute path to configuration default file
        
    Examples
    --------
    >>> config_string_to_path('path/to/deepethogram/deepethogram/conf', 'tune/feature_extractor')
    'path/to/deepethogram/deepethogram/conf/tune/feature_extractor.yaml'
        
    """
    fullpath = os.path.join(config_path, *string.split('/')) + '.yaml'
    assert os.path.isfile(fullpath), f'{fullpath} not found'
    return fullpath


def load_config_by_name(string: str, config_path: Union[str, os.PathLike] = None) -> DictConfig:
    """Loads a default configuration by name

    Parameters
    ----------
    string : str
        Name of configuration. examples: 'train', 'zscore', 'tune/tune', 'model/feature_extractor'
    config_path : Union[str, os.PathLike], optional
        Path to deepethogram conf directory. If None, finds automatically, by default None

    Returns
    -------
    DictConfig
        Configuration loaded from YAML file
        
    Examples
    --------
    >>> load_config_by_name('model/feature_extractor')
    feature_extractor:
        arch: resnet18
        fusion: average
        sampler: null
        final_bn: false
        sampling_ratio: null
        final_activation: sigmoid
        dropout_p: 0.25
        n_flows: 10
        n_rgb: 1
        curriculum: false
        inputs: both
        weights: null
        train:
        steps_per_epoch:
            train: 1000
            val: 1000
            test: null
        num_epochs: 20
    """

    if config_path is None:
        config_path = os.path.join(os.path.dirname(deepethogram.__file__), 'conf')

    path = config_string_to_path(config_path, string)
    return OmegaConf.load(path)


def make_config(project_path: Union[str, os.PathLike],
                config_list: list,
                run_type: str,
                model: str,
                use_command_line: bool = False,
                preset: str = None,
                debug: bool = False) -> DictConfig:
    """Makes a configuration for model training or inference. 
    
    A list of default configurations are composed into one single cfg. From the project path, the project configuration
    is found and loaded. If a preset is specified either in the config_list or in the project config, load "preset" 
    parameters. 

    Order of composition: 
    1. Defaults
    2. Preset
    3. Project configuration
    4. Command line
    
    This means if you specify the value of a parameter (say, dropout probability) in multiple places, the last one 
    (highest number in above list) will be chosen. This means we can specify a default dropout (0.25); for your project,
    you can specify a new default in your project_config (e.g. 0.5). For an experiment, you can use the commmand line 
    to set `feature_extractor.dropout_p=0.75`. If its in all 3 places, the command line "wins" and the actual dropout is 
    0.75. 

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project. Should contain: project_config.yaml, models directory, DATA directory
    config_list : list
        List of string names of default configurations. Each of them is the name of a file or sub-file in the 
        deepethogram/conf directory. 
    run_type : str
        Train, inference, or gui
    model : str
        feature_extractor, flow_generator, or sequence
    use_command_line : bool, optional
        If True, command line arguments are parsed and composed into the 
    preset : str, optional
        One of deg_f, deg_m, deg_s, by default None
    debug : bool, optional
        If True, reduce number of steps and number of epochs. Useful for quick debugging, by default False

    Returns
    -------
    DictConfig
        [description]
    """
    # config_path = os.path.join(os.path.dirname(deepethogram.__file__), 'conf')

    user_cfg = projects.get_config_from_path(project_path)

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

    # first defaults; then user cfg; then cli
    if use_command_line:
        cfg = OmegaConf.merge(*cfgs, user_cfg, command_line_cfg)
    else:
        cfg = OmegaConf.merge(*cfgs, user_cfg)

    cfg.run = {'type': run_type, 'model': model}
    return cfg


def make_flow_generator_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for training flow generators

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        flow generator config
    """
    config_list = ['config', 'augs', 'train', 'model/flow_generator']
    run_type = 'train'
    model = 'flow_generator'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg


def make_feature_extractor_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for training feature extractor models

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        feature extractor train config
    """
    config_list = ['config', 'augs', 'train', 'model/flow_generator', 'model/feature_extractor']
    run_type = 'train'
    model = 'feature_extractor'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg


def make_feature_extractor_inference_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for running inference with feature extractor models

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        feature extractor inference config
    """
    config_list = ['config', 'augs', 'model/feature_extractor', 'model/flow_generator', 'inference', 'postprocessor']
    run_type = 'inference'
    model = 'feature_extractor'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg


def make_sequence_train_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for training sequence models

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        sequence train config
    """
    config_list = ['config', 'model/feature_extractor', 'train', 'model/sequence']
    run_type = 'train'
    model = 'sequence'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg


def make_sequence_inference_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for running inference with sequence models

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        sequence inference config
    """
    config_list = ['config', 'augs', 'model/feature_extractor', 'model/sequence', 'inference']
    run_type = 'inference'
    model = 'sequence'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg


def make_postprocessing_cfg(project_path: Union[str, os.PathLike], **kwargs) -> DictConfig:
    """Makes configuration for postprocessing predictions

    Parameters
    ----------
    project_path : Union[str, os.PathLike]
        Path to deepethogram project

    Returns
    -------
    DictConfig
        postprocessing config
    """
    config_list = ['config', 'model/sequence', 'postprocessor']
    run_type = 'inference'
    model = 'sequence'

    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, **kwargs)

    return cfg