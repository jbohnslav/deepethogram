# DeepEthogram
- Written by Jim Bohnslav, except where as noted
- JBohnslav@gmail.com

DeepEthogram is an open-source package for automatically classifying each frame of a video into a set of pre-defined 
behaviors. Designed for neuroscience research, it could be used in any scenario where you need to detect actions from 
each frame of a video.

Example use cases:
* Measuring itching or scratching behaviors to assess the differences between wild-type and mutant animals
* Measuring the amount of time animals spend courting, and comparing between experimental conditions

DeepEthogram uses state-of-the-art algorithms for *temporal action detection*. We build on the following previous machine 
learning research into action detection:
* [Hidden Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/abs/1704.00389)
* [Temporal Gaussian Mixture Layer for Videos](https://arxiv.org/abs/1803.06316)

![deepethogram schematic](docs/images/deepethogram_schematic.png)

## Installation
For full installation instructions, see [this readme file](docs/installation.md). 

In brief: 
* [Install PyTorch](https://pytorch.org/) 
* Install PySide2: `conda install -c conda-forge pyside2==5.13.2` 
* `pip install deepethogram`

## News
Due to paper revisions, DeepEthogram is going under major re-development. I might take longer than average to respond to your emails / issues on GitHub. However, I'm confident that the ongoing update to DeepEthogram will significantly improve model performance and train / inference time. Thanks for your patience!

## Pretrained models
Rather than start from scratch, we will start with model weights pretrained on the Kinetics700 dataset. Go to 
To download the pretrained weights, please use [this Google Drive link](https://drive.google.com/file/d/1ntIZVbOG1UAiFVlsAAuKEBEVCVevyets/view?usp=sharing).
Unzip the files in your `project/models` directory. The path should be 
`your_project/models/pretrained/{models 1:6}`. 

## Licensing
Copyright (c) 2020 - President and Fellows of Harvard College. All rights reserved.

This software is free for academic use. For commercial use, please contact the Harvard Office of Technology 
Development (hms_otd@harvard.edu) with cc to Dr. Chris Harvey. For details, see [license.txt](license.txt). 

## Usage
### [To use the GUI, click](docs/using_gui.md)
#### [To use the command line interface, click](docs/using_CLI.md)

## Dependencies
The major dependencies for DeepEthogram are as follows: 
* PyTorch, torchvision: all the neural networks, training, and inference pipelines were written in PyTorch
* pyside2: for the GUI
* opencv: for video and image reading and writing
* opencv_transforms: for fast image augmentation
* scikit-learn, scipy: for binary classification metrics
* matplotlib: plotting metrics and neural network outputs
* pandas: reading and writing CSVs
* h5py: saving inference outputs as HDF5 files
* hydra: for smoothly integrating configuration files and command line inputs
* tifffile: for writing neural network outputs as tiff stacks
* tqdm: for nice progress bars

## Hardware requirements
For GUI usage, we expect that the users will be working on a local workstation with a good NVIDIA graphics card. For 
training via a cluster, you can use the CLI yourself. 

* CPU: 8 cores or more for parallel data loading
* Hard Drive: SSD at minimum, NVMe drive is better.
* GPU: DeepEthogram speed is directly related to GPU performance. An NVIDIA GPU is absolutely required, as PyTorch uses 
CUDA, while AMD does not. 
The more VRAM you have, the more data you can fit in one batch, which generally increases performance. a
I'd recommend 6GB VRAM at absolute minimum. 8GB is better, with 10+ GB preferred.
Recommended GPUs: `RTX 3090`, `RTX 3080`, `Titan RTX`, `2080 Ti`, `2080 super`, `2080`, `1080 Ti`, `2070 super`, `2070` 
Some older ones might also be fine, like a `1080` or even `1070 Ti`/ `1070`. 

## Changelog
#### 0.0.2
* fixed divide by zero bug
* fixed bias initialization
* fixed issue where HDF5 files could have byte keys instead of strings
* fixed various capitalization issues
* fixed video conversion bugs
* added better support for videos encoded as PNG directories
#### 0.0.1.post1
* bug fixes and video conversion scripts added
#### 0.0.1
* initial commit
