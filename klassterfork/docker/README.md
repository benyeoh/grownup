# Docker for Klassterfork

## Installing
Ensure that GPU driver version is >=460.32.03.

Install Docker 19.03 and the `nvidia-container-toolkit`. See: https://github.com/NVIDIA/nvidia-docker

Otherwise if you're using Docker 18.09, please install the `nvidia-docker` wrapper.

## Building
To build a klassterfork docker image, run `build_docker.sh`

## Usage
Example, to train a model using docker version < 19.03:

`sudo nvidia-docker run --rm -it klassterfork ./python/ktf/train/trainer.py -c path/to/json/config`

To specify using GPUs 0 and 1 for training

`sudo nvidia-docker run --rm -it -e CUDA_VISIBLE_DEVICES=0,1 klassterfork ./python/ktf/train/trainer.py -c path/to/json/config`

Example, for docker version >= 19.03:

`sudo docker run --rm --gpus all -it klassterfork ./python/ktf/train/trainer.py -c path/to/json/config`

Example, for < 8GB RAM GPUs:

`sudo nvidia-docker run --rm -it -e TF_FORCE_GPU_ALLOW_GROWTH=true klassterfork ./python/ktf/train/trainer.py -c path/to/json/config`
