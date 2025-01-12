# Using deepethogram in Docker
Install Docker: https://docs.docker.com/get-docker/

Install nvidia-docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## running the gui on Linux with training support
In a terminal, run `xhost +local:docker`. You'll need to do this every time you restart.

To run, type this command: `docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --shm-size 16G -v /media:/media -it jbohnslav/deepethogram:full python -m deepethogram`

Explanation
* `--gpus all`: required to have GPUs accessible in the container
* `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw`: so that the container has access to your screen
* `--shm-size 16G`: required for pytorch to be able to use multiprocessing. we might be able to lower this amount
* `-v /media:/media`: use this to mount your data hard drive inside the container. Replace with whatever works for your system. For example, if your data lives on a drive called `/mnt/data/DATA`, replace this with `-v /mnt:/mnt`
* `it deepethogram:dev python -m deepethogram`: run the deepethogram GUI in interactive mode

## Running the GUI without training support (no pytorch, etc.)
Again, change `/media` to your hard drive with your training data

`docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it jbohnslav/deepethogram:gui python -m deepethogram`

## Running the CLI without GUI support
`docker run --gpus all -v /media:/media -it jbohnslav/deepethogram:headless pytest tests/`

## making sure all 3 images work
#### full
* GUI: `docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it deepethogram:full python -m deepethogram`
* tests: `docker run --gpus all -it deepethogram:full pytest tests/`

#### gui only
* GUI: `docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media:/media -it deepethogram:gui python -m deepethogram`

#### CLI only
* tests: `docker run --gpus all -it deepethogram:full pytest tests/`

# building it yourself
To build the container with both GUI and model training support:
* `cd` to your `deepethogram` directory
* `docker build -t deepethogram:full -f docker/Dockerfile-full .`
