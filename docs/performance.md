# Performance

This document will describe how to minimize model training and inference time. It does not have to do with model accuracy; for that, please see [the model performance docs](model_performance.md)

## Hardware requirements
For GUI usage, we expect that the users will be working on a local workstation with a good NVIDIA graphics card. For training via a cluster, you can use the command line interface.
As of DeepEthogram version 0.1, we use the GPU more heavily for all tasks. If you have a limited budget, spending the
majority of it on a GPU is the highest priority.

* CPU: 4 cores or more for parallel data loading
* Hard Drive: SSD at minimum, NVMe drive is better.
* GPU: DeepEthogram speed is directly related to GPU performance. An NVIDIA GPU is absolutely required, as PyTorch uses
CUDA, while AMD does not.
The more VRAM you have, the more data you can fit in one batch, which generally increases performance. a
I'd recommend 6GB VRAM at absolute minimum. 8GB is better, with 10+ GB preferred.
Recommended GPUs: `RTX 3090`, `RTX 3080`, `Titan RTX`, `2080 Ti`, `2080 super`, `2080`, `1080 Ti`, `2070 super`, `2070`
Some older ones might also be fine, like a `1080` or even `1070 Ti`/ `1070`.

# Bottlenecks
In diagnosing performance issues, it is helpful to know the basics of how data is loaded and run through the network.

## Training pipeline

1. Grab random frames from random videos. This randomness is extremely important; through gradient
descent, we will be training our model parameters sequentially. Without randomizing the order of video snippets,
our model would be biased towards the first frames of the first videos. The input to hidden-two-stream models are clips of 11 images. So, we need to randomly select a video from our training set, open it, and seek to a random starting point, and read 11 images into CPU memory (RAM). This happens in parallel, with each CPU core reading one clip at a time. The number of workers is controlled by `cfg.compute.num_workers`.
2. Not all our videos are necessarily the same size. Therefore, crop or resize our clip into a consistent shape. This is
done on the CPU.
3. Here is ordinarily where image augmentation (randomly changing brightness, contrast, rotating, flipping, etc.) would
occur. However, I use Kornia to do this on the GPU for speed.
4. Stack our clips, currently of shape (3, 11, Height, Width), into a single batch of shape (N, 3, 11, Height, Width).
N is controlled by `cfg.compute.batch_size`.
5. Move our batch from CPU to GPU memory. All subsequent operations are now done on the GPU.
6. Perform image augmentations with Kornia.
7. Perform the forward pass through our neural network.
8. Compute the loss
9. Compute the gradients of the loss w.r.t. our parameters
10. Optimize the parameters of our network using the gradients (we use ADAM).

Now that we know how the pipeline works at a high level, we can start to see where our bottlenecks might occur.

* `1.` Reading frames from disk. If we use a batch size of 32, we are loading 11 frames from 32 videos, or 352 images
for one batch. If we have a fast GPU, at ~256x256 resolution, DEG_f feature extractors can train at ~3.5 batches per
second. This is ~1200+ images that need to be read per second, from `32*3.5=112` random video locations. This means
two things:
  * We need to make sure we have a solid-state hard drive, or an NVMe hard drive. NOT a mechanical hard drive.
  * We need to optimize our video format to make random reads faster. Normal video encodings, like libx264 or MJPG,
  typically have very fast sequential reads, and very slow random reads. If your videos have endings like .avi or .mp4,
  it is likely that it will be extremely slow to randomly read frames from them, and this will slow down your entire
  training.
  * SOLUTION: the fastest way to randomly read videos is to store them as folders full of images. If we use .jpg
  compression, your videos will be re-compressed, and have new artifacts. If we use .PNG compression, the videos will
  be lossless-ly encoded, which means there will be no new artifacts. However, PNG images take up far more space on disk
  than something like a .mp4 file, or .jpg images. However, for datasets numbering in the 10s of videos, I think this is
  a worthwhile tradeoff; we are trading more space on the disk for less time in training.
    * Solution 1: Use the function `projects.convert_all_videos`. Using `movie_format='directory'` means that we will
    convert our video into a big directory full of .png files.
    Example:
      ```python
      from deepethogram.projects import convert_all_videos
      convert_all_videos('PATH/TO/MY/project_config.yaml', movie_format='directory')
      ```
    * Solution 2: Use the function `projects.convert_all_videos`. Using `movie_format='hdf5'` means that we will
    convert our video into PNGs; however, instead of a big directory, we will save the PNG bytestrings inside an
    HDF5 file. This means it will be faster to move around (copying or cutting / pasting directories full of images
    takes forever). The only downside of using `hdf5` as our movie format is that it is a less common filetype; you cant
    just open it up in your file browser. However, this is the format I use for all DeepEthogram training.
    * Solution 3: Use the function `projects.convert_all_videos` using `movie_format='directory'` or
    `movie_format='hdf5'`, along with `codec='.jpg'`. This will re-compress your images as .jpgs, saving filespace at
    the expense of image quality.
    Example:
        ```python
        from deepethogram.projects import convert_all_videos
        convert_all_videos('PATH/TO/MY/project_config.yaml', movie_format='directory', codec='.jpg')
        ```
    * Solution 4: Use `file_io.convert_video` along with your custom code to resize your images while you save them as
    .PNGs. The input to the network is usually not larger than ~256x256 in resolution; if we resize them when
    converting, we could potentially save both time and space. If you want me to code this for you, please raise an
    issue on GitHub.
* `3`: Image augmentations. If our network is capable of running at 1200 images per second, randomly changing the
brightness, color, contrast, rotating, etc for 1200 images on the CPU could bottleneck our training. For this reason,
with DeepEthogram v0.1 and above, I've converted the entire augmentation pipeline to Kornia, which implemented
augmentations on the GPU. Thanks to the people at Kornia for implementing Video Transforms as of version 0.5.

In my experience, either reading from disk or performing augmentations are the most likely places to slow down training.

## Inference
As of DeepEthogram 0.1, I've implemented a much faster parallel processing pipeline for video inference. We use only
sequential reads from disk (see training section), while also loading images in parallel and running our network in
batches. The code for this can be found in `deepethogram.data.datasets.VideoIterable`.

# Model type
In the DeepEthogram paper, we describe 3 models; `DEG_f`,` DEG_m`, and `DEG_s`. These use ResNet18, ResNet50, and
3D-ResNet34 as feature extractors, respectively. We recommend only using `DEG_m` or `DEG_f` by default; if you have
access to high-quality GPUs (e.g. RTX3090s, Titan RTX, etc), or need especially accurate results for your project, you
can try `DEG_s`.

# Image size
For both training and inference, speed will be extremely proportional to image resolution. You should never use raw
acquired video, such as HD (1920 x 1080). The size of the input images are determined by this section of your
`project_config.yaml`:
```yaml
augs:
  resize:
  - 224
  - 224
```
Note that we default to resizing to 224 x 224. In the paper, we use the following image sizes:
* 224 x 224
* 256 x 256
* 352 x 224 # Homecage videos

For the flow generators that I wrote, we can use images of any resolution by default. For ResNets, however, our image
sizes must be multiples of 32:
```python
[32*i for i in range(1, 15)]
# [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448]
```
Note that I chose `352 x 224` for the Homecage dataset for this reason.
