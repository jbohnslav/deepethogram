# Using DeepEthogram in Docker

DeepEthogram's Docker images are built with uv, so you do not need conda inside the container.

Install Docker: https://docs.docker.com/get-docker/

Install the NVIDIA Container Toolkit if you want GPU access:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## Running the GUI on Linux with training support

In a terminal, run `xhost +local:docker`. You'll need to do this each time you restart.

```bash
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --shm-size 16G -v /media:/media -it jbohnslav/deepethogram:full
```

Explanation:

* `--gpus all`: required to expose NVIDIA GPUs inside the container
* `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw`: forwards your X11 display
* `--shm-size 16G`: helps PyTorch multiprocessing workloads
* `-v /media:/media`: mounts your data drive; replace `/media` with the path that makes sense on your machine

The `full` image launches `deepethogram` by default.

## Running the GUI without training support

Again, replace `/media` with the path to your data.

```bash
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it jbohnslav/deepethogram:gui
```

## Running tests in the headless image

```bash
docker run --gpus all -v /media:/media -it jbohnslav/deepethogram:headless pytest -v
```

## Verifying the images

Use the repo-local images if you built them yourself:

```bash
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it deepethogram:full
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it deepethogram:gui
docker run --gpus all -it deepethogram:headless pytest -v
```

## Building it yourself

To build the `full` image from this repository:

```bash
docker buildx build --load --target full -t deepethogram:full -f docker/Dockerfile .
```
