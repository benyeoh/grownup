# Goose
Based on: https://github.com/grangier/python-goose

## How to install
The easiest way to make sure you have all pre-requisites is to build a Docker image with the following command:

```bash
./build_docker.sh
```

Note: Your Docker version should be >= 19.04.

## How to run

### Run a bash shell within a container
Once you've built the `goose` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] goose /bin/bash
```

### Extract content

```bash
./extract_content.py -i <INPUT_HTML_DIR> -c <INPUT_LABEL_DIR> -o <EXTRACTED_DIR>
```