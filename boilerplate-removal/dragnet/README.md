# Dragnet benchmarking

## How to install
The easiest way to make sure you have all pre-requisites is to build a Docker image with the following command:

```bash
./build_docker.sh
```

Note: Your Docker version should be >= 19.04.

## How to run

### Run a bash shell within a container
Once you've built the `dragnet` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] dragnet /bin/bash
```

### Prepare datasets

CleanEval:
```
./prepare_data_cleaneval.py -i /hpc-datasets/web/cleaneval/raw -o <prepared data for training / extraction>
```

Dragnet:
```
./prepare_data_dragnet.py -i /hpc-datasets/web/dragnet/raw -o <prepared data for training / extraction>
```

### Train

```
./train.py -i <prepared dataset dev dir> -e <prepared dataset test dir> -o <output saved model dir>
```

Example:

```
./train.py -i /hpc-datasets/web/cleaneval/dragnet/dev -e /hpc-datasets/web/cleaneval/dragnet/test -o <output saved model dir>
```