# Web2Text
See: https://github.com/dalab/web2text

## How to install
Build a docker by running `./build_docker.sh`

## How to run

### Run a bash shell within a container
Once you've built the `web2text` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] web2text /bin/bash
```

### 1. Preparing data

```bash
./prepare_data_dragnet.sh -r <INPUT_RAW_DIR> -t <TRAIN_DIR> -e <PREPRO_DIR>
```

### 2. Training

```bash
./train.py <OUTPUT_TRAIN_DIR>/dev/block_features.npy <SAVED_MODEL_DIR> <TRAIN_DIR>/test/block_features.npy 1.0
```

### 3. Extracting content
The input path to the `-h` parameter is the folder containing all HTML files corresponding to the pre-processed HTMLs in the folder specified by the `-d` parameter.

```bash
./extract_content.sh -h <INPUT_RAW_DIR>/test_html -d <PREPRO_DIR> -l <LABELS_DIR> -m <SAVED_MODEL_DIR> -e <EXTRACTED_DIR>
```