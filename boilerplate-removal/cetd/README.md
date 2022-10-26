# CETD
Based on: https://github.com/benprofessionaledition/content-extraction which is itself based on https://github.com/FeiSun/ContentExtraction

Some modifications and fixes are added in `patch` folder. 

### How to build docker

Run `./build_docker.sh`

## How to run

### Run a bash shell within a container
Once you've built the `cetd` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] cetd /bin/bash
```

### Extract content
Since no training is required, you can just proceed to extract content given the HTML files and ground truth files. The ground truth files are simply pre-processed to remove some unnecessary markups and written out again.

```bash
./extract_content.py -i <INPUT_HTML_DIR> -c <INPUT_LABEL_DIR> -o <EXTRACTED_DIR>
```
