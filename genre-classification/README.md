# GROWN+UP feat. extractor genre classification benchmark

## Pre-requisites
### Building Docker image
The easiest way to make sure you have all pre-requisites is to build a Docker image with the following command:

```bash
./build_docker.sh
```

Note: Your Docker version should be >= 19.04.

### Run a bash shell within a container
Once you've built the `grownup_genre_class` Docker image, run a bash shell using the image. Example:

```bash
docker run -it --net=host --rm --gpus all [-v <SRC MOUNT>:<DST MOUNT>] grownup_genre_class /bin/bash
```

## Fine-tuning and classifying webpages 
### 1. Prepare train/test datasets for KI-04 / 7-Web-Genre
Prepare datasets and generate 10-fold CV splits. Warning: This can take >24 hrs.

```bash
./prepare_data_[ki04|web7].sh -i <dataset raw input dir> -o <dataset processed output dir> -s <seed int>

```

### 2. Fine-tune on KI-04 / 7-Web-Genre
Fine tune on relevant datasets with 10-fold splits and save trained weights for each split. Take note that using the default fine tuning settings requires at least 16GB of VRAM.

```bash
./train_[ki04|web7].sh -i <dataset processed output dir>/split_<seed int> -o <trained weights dir> 
```

### 3. Evaluate scores
Now that we've fine-tuned the model, we can evaluate the genre classification accuracy with:

```bash
./evaluate_[ki04|web7].py -i <dataset processed output dir>/split_<seed int> -w <trained weights dir>
```

## FAQ

### Getting `Out Of Memory` errors when training or extracting data?
Most likely, you have insufficient VRAM to run the models on the GPU. You may wish to use a smaller batch size (and learning rate) to reduce VRAM requirements. You can adjust these parameters and many others in the training config .hjson file. Otherwise, you may disable GPU usage by setting the environment var `CUDA_VISIBLE_DEVICES=` but would be horrendously slow.