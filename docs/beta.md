# DeepEthogram Beta

DeepEthogram is now in Beta, version 0.1! There are major changes to the codebase and to model training and inference. 
Model performance, measured by F1, accuracy, etc. should be higher in version 0.1. Model training times and inference 
times should be dramatically reduced. 

## Summary of changes
* Basic training pipeline re-implemented with PyTorch Lightning. This gives us some great features, such as tensorboard 
logging, automatic batch sizing, and Ray Tune integration. 
* Image augmentations moved to GPU with Kornia. [see Performance guide for details](performance.md)

## Migration guide
* activate your conda environment, e.g. `deg`
* pip uninstall hydra-core