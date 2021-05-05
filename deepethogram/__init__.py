# from deepethogram import feature_extractor, flow_generator, gui, sequence, dataloaders, metrics, utils, viz, zscore
# from deepethogram import feature_extractor, flow_generator, gui, sequence, dataloaders,
# from deepethogram.flow_generator.train import flow_generator_train
# from deepethogram.feature_extractor.train import feature_extractor_train
# from deepethogram.feature_extractor.inference import feature_extractor_inference
# from deepethogram.sequence.train import sequence_train
# from deepethogram.sequence.inference import sequence_inference
import importlib.util

spec = importlib.util.find_spec('hydra')
if spec is not None:
    raise ValueError('Hydra installation found. Please run pip uninstall hydra-core')
# try:
#     import hydra
# except Exception as e:
#     print(e)
#     # hydra is not found
#     pass
# else:
#
