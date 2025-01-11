import os

from deepethogram import projects, utils, viz
from deepethogram.configuration import make_flow_generator_train_cfg
from deepethogram.flow_generator.train import (get_datasets_from_cfg, build_model_from_cfg, get_metrics,
                                               OpticalFlowLightning)
from setup_data import (make_project_from_archive, project_path)


def test_metrics():
    make_project_from_archive()
    cfg = make_flow_generator_train_cfg(project_path=project_path)
    cfg = projects.setup_run(cfg)

    datasets, data_info = get_datasets_from_cfg(cfg, 'flow_generator', input_images=cfg.flow_generator.n_rgb)
    flow_generator = build_model_from_cfg(cfg)
    utils.save_dict_to_yaml(data_info['split'], os.path.join(os.getcwd(), 'split.yaml'))
    flow_weights = projects.get_weightfile_from_cfg(cfg, 'flow_generator')
    if flow_weights is not None:
        print('reloading weights...')
        flow_generator = utils.load_weights(flow_generator, flow_weights, device='cpu')

    # stopper = get_stopper(cfg)
    metrics = get_metrics(cfg, os.getcwd(), utils.get_num_parameters(flow_generator))
    lightning_module = OpticalFlowLightning(flow_generator, cfg, datasets, metrics, viz.visualize_logger_optical_flow)
    assert lightning_module.scheduler_mode == 'min'
    assert lightning_module.metrics.key_metric == 'SSIM'