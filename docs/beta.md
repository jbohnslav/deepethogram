# DeepEthogram Beta

DeepEthogram is now in Beta, version 0.1! There are major changes to the codebase and to model training and inference.
Model performance, measured by F1, accuracy, etc. should be higher in version 0.1. Model training times and inference
times should be dramatically reduced.

**Important note: your old project files, models, and (most importantly) human labels will all still work!** However,
I do recommend training new feature extractor and sequence models, as performance should improve somewhat. This will
be the last major refactor of DeepEthogram (model improvements and new features will still come out), however I will
not be majorly changing dependencies after this. Future upgrades will be easier in a uv-managed environment.

## Summary of changes
* Basic training pipeline re-implemented with PyTorch Lightning. This gives us some great features, such as tensorboard
logging, automatic batch sizing, and Ray Tune integration.
* Image augmentations moved to GPU with Kornia. [see Performance guide for details](performance.md)
* New, parallelized inference
* Hyperparameter tuning
* New defaults for all models to improve performance
* improved unit tests
* new `configuration` module to make generation of configurations (e.g. `cfg`) more understandable and easy
* Refactor of the whole data module
* (alpha): support for importing DeepLabCut keypoints to train sequence models
* new performance documentation, among others

## Migration guide

There are some new dependency changes; making sure that install works correctly is the hardest part about migration.
For current releases, use the uv-first workflow from [installation.md](installation.md). If you are maintaining a version
prior to 0.4.0, keep the legacy installation notes from that page in mind before trying to reuse an older environment.

The cleanest path is to migrate into a fresh uv-managed environment instead of upgrading an older pip or conda environment in place:

```bash
git clone https://github.com/jbohnslav/deepethogram.git
cd deepethogram
uv sync
```

If you need to clean up an old environment first, use the uv equivalents for the legacy pip commands:

* uninstall hydra: `uv pip uninstall hydra-core`
* uninstall DeepEthogram: `uv pip uninstall deepethogram`
* install the current release for regular use: `uv pip install deepethogram`

### upgrade issues
* `AttributeError: type object 'OmegaConf' has no attribute 'to_yaml'`
  * this indicates that OmegaConf did not successfully upgrade to version 2.0+, and also likely that there was a problem
  with your upgrade. please follow the above steps. If you're sure that everything else installed correctly, you can run
  `uv add omegaconf`
* `error: torch 1.5.1 is installed but torch>=1.6.0 is required by {'kornia'}`
  * this indicates that your PyTorch version is too low. Please uninstall and reinstall PyTorch.
* `ValueError: Hydra installation found. Please run uv pip uninstall hydra-core`
  * do as the error message says: run `uv pip uninstall hydra-core`
  * if you've already done this, you might have to manually delete hydra files. Mine were at
  `'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\hydra_core-0.11.3-py3.7.egg\\hydra'`. Please delete the `hydra_core` folder.
