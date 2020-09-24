# DeepEthogram
- Written by Jim Bohnslav, except where as noted
- JBohnslav@gmail.com

DeepEthogram is an open-source package for automatically classifying each frame of a video into a set of pre-defined 
behaviors. Designed for neuroscience research, it could be used in any scenario where you need to detect actions from 
each frame of a video.

Example use cases:
* Measuring itching or scratching behaviors to assess the differences between wild-type and mutant animals
* Measuring the amount of time animals spend courting, and comparing between experimental conditions

DeepEthogram uses state-of-the-art algorithms for *action detection*. We build on the following previous machine 
learning research into action detection:
* [Hidden Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/abs/1704.00389)
* [Temporal Gaussian Mixture Layer for Videos](https://arxiv.org/abs/1803.06316)

![deepethogram schematic](docs/images/deepethogram_schematic.png)

## Installation
For full installation instructions, see [this readme file](docs/installation.md). In brief, [install PyTorch](pytorch.org). 
Then, `pip install deepethogram`. # currently doesn't work because I haven't put it on PyPi

## Hardware requirements
For GUI usage, we expect that the users will be working on a local workstation with a good NVIDIA graphics card. For 
training via a cluster, you can use the CLI yourself. 

* CPU: 8 cores or more for parallel data loading
* Hard Drive: SSD at minimum, NVMe drive is better.
* GPU: DeepEthogram speed is directly related to GPU performance. An NVIDIA GPU is absolutely required, as PyTorch uses 
CUDA, while AMD does not. AMD GPUs are not supported. 
The more VRAM you have, the more data you can fit in one batch, which increases performance (due to batch normalization). 
I'd recommend 6GB VRAM at absolute minimum. 8GB is better, with 10+ GB preferred. Price is directly proportional to performance. 
Recommended GPUs: `Titan RTX`, `2080 Ti`, `2080 super`, `2080`, `1080 Ti`, `2070 super`, `2070`, `2060 Super`, `2060`. Some
older ones might also be fine, like a `1080` or even `1070 Ti`/ `1070`. 

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


