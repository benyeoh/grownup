# CETD
Based on: https://github.com/benprofessionaledition/content-extraction which is itself based on https://github.com/FeiSun/ContentExtraction

Some modifications and fixes are added in `patch` folder. 

### How to build docker

Run `./build_docker.sh`

## How to run

### Running docker image

Run `./run_bash.sh -v /hpc-datasets:/hpc-datasets [<other optional docker run args>]`

### Extract content
Since no training is required, you can just proceed to extract content given the HTML files and ground truth files. The ground truth files are simply pre-processed to remove some unnecessary markups and written out again.

Dragnet: 

```bash
./extract_content.py -i /hpc-datasets/web/dragnet/raw/test_html/ -c /hpc-datasets/web/dragnet/raw/test_cleaned/ -o <output folder>
```

Cleaneval:

```bash
./extract_content.py -i /hpc-datasets/web/cleaneval/raw/test/input -c /hpc-datasets/web/cleaneval/raw/test/gold_std -o <output folder>
```
