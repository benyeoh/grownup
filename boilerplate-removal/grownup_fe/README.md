# GROWN+UP feature extractor benchmarking

## Pre-requisites
### Building Docker image
The easiest way to make sure you have all pre-requisites is to build a Docker image with the following command:

```bash
./build_docker.sh
```

Note: Your Docker version should be >= 19.04.

### Run a bash shell within a container
Once you've built the `grownup_boil_remove` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] grownup_boil_remove /bin/bash
```

## Fine-tuning and extracting content from webpage 
### 1. Prepare train/test datasets for CleanEval / Dragnet from raw
Prepare both training and test splits for datasets. Warning: This can take >24 hrs.

```bash
./prepare_data_[cleaneval|dragnet].sh -i <dataset raw input dir> -o <dataset processed output dir>

```

### 2. Fine-tune on CleanEval / Dragnet and save fine-tuned weights
Fine tune on relevant datasets. Take note that using the default fine tuning settings requires at least 16GB of VRAM.

```bash
./train_[cleaneval|dragnet].sh -i <dataset processed output dir>/train -o <trained weights .h5 path> 
```



### 3. Extract CleanEval / Dragnet textual content from test dataset
Now that we've fine-tuned the model, we can extract the textual content from the dataset with:

```bash
./extract_[cleaneval|dragnet].py -i <dataset processed output dir>/test -o <output extracted text dir> -w <trained weights .h5 path>
```

## FAQ

### Getting `Out Of Memory` errors when training or extracting data?
Most likely, you have insufficient VRAM to run the models on the GPU. You may wish to use a smaller batch size (and learning rate) to reduce VRAM requirements. You can adjust these parameters in the training config .hjson file. Otherwise, you may disable GPU usage by setting the environment var `CUDA_VISIBLE_DEVICES=`.