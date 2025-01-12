from omegaconf import OmegaConf

try:
    import ray  # noqa: F401
    from ray import tune  # noqa: F401
except ImportError:
    print("To use the deepethogram.tune module, you must `pip install 'ray[tune]`")
    raise


# code modified from official ray docs:
# https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
def dict_to_dotlist(cfg_dict):
    dotlist = [f"{key}={value}" for key, value in cfg_dict.items()]
    return dotlist


def generate_tune_cfg(cfg):
    """from a configuration, e.g. conf/tune/feature_extractor.yaml, generate a search space for specific hyperparameters"""

    def get_space(hparam_dict):
        if hparam_dict.space == "uniform":
            return tune.uniform(hparam_dict.min, hparam_dict.max)
        elif hparam_dict.space == "log":
            return tune.loguniform(hparam_dict.min, hparam_dict.max)
        elif hparam_dict.space == "choice":
            return tune.choice(OmegaConf.to_container(hparam_dict.choices))
        else:
            raise NotImplementedError

    tune_cfg = {}
    for key, value in cfg.tune.hparams.items():
        tune_cfg[key] = get_space(value)

    return tune_cfg
