# Web2Text
See: https://github.com/dalab/web2text

## How to install
Build a docker by running `./build_docker.sh`

## How to run

### Running bash in the docker container
To run the environment, you can use `./run_bash.sh` with optional docker parameters. Example:

```bash
./run_bash.sh -v /hpc-datasets:/hpc-datasets -v $(pwd):/pwd
```

### 1. Preparing data

```bash
./prepare_data_dragnet.sh -r /hpc-datasets/web/dragnet/raw -t /hpc-datasets/web/dragnet/web2text/train -e /hpc-datasets/web/dragnet/web2text/extract
```

### 2. Training

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true ./train.py /hpc-datasets/web/dragnet/web2text/train/dev/block_features.npy /hpc-datasets/ben/web2text/dragnet/saved_model2/ /hpc-datasets/web/dragnet/web2text/train/test/block_features.npy 1.0
```

### 3. Extracting content
The input path to the `-h` parameter is the folder containing all HTML files corresponding to the pre-processed HTMLs in the folder specified by the `-d` parameter.


```bash
CUDA_VISIBLE_DEVICES= ./extract_content.sh -h /hpc-datasets/web/dragnet/raw/test_html -d /hpc-datasets/web/dragnet/web2text/extract -l /hpc-datasets/ben/web2text/dragnet/all/labels/ -m /hpc-datasets/ben/web2text/dragnet/saved_model -e /hpc-datasets/ben/web2text/dragnet/all/extracted
```