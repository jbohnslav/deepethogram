# DeepEthogram Beta

DeepEthogram is now in Beta, version 0.1! There are major changes to the codebase and to model training and inference. 
Model performance, measured by F1, accuracy, etc. should be higher in version 0.1. Model training times and inference 
times should be dramatically reduced. 

**Important note: your old project files, models, and (most importantly) human labels will all still work!** However, 
I do recommend training new feature extractor and sequence models, as performance should improve somewhat. 

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

* activate your conda environment, e.g. `conda activate deg`
* uninstall hydra: `pip uninstall hydra-core`
* uninstall pytorch to upgrade: `conda uninstall pytorch`
* uninstall pyside2 via conda: `conda uninstall pyside2`
* upgrade pytorch. note that cudatoolkit version is not important `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
* Uninstall DeepEthogram: `pip uninstall deepethogram`
* Install the new version: `pip install deepethogram`

### upgrade issues
* `AttributeError: type object 'OmegaConf' has no attribute 'to_yaml'`
  * this indicates that OmegaConf did not successfully upgrade to version 2.0+, and also likely that there was a problem
  with your upgrade. please follow the above steps. If you're sure that everything else installed correctly, you can run
  `pip install --upgrade omegaconf`
* `error: torch 1.5.1 is installed but torch>=1.6.0 is required by {'kornia'}`
  * this indicates that your PyTorch version is too low. Please uninstall and reinstall PyTorch. 
* `ValueError: Hydra installation found. Please run pip uninstall hydra-core`
  * do as the error message says: run `pip uninstall hydra-core`
  * if you've already done this, you might have to manually delete hydra files. Mine were at 
  `'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\hydra_core-0.11.3-py3.7.egg\\hydra'`. Please delete the `hydra_core` folder.
