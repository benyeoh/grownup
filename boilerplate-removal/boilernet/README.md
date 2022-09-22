# Boilernet benchmarking
See: https://github.com/mrjleo/boilernet

## How to install
Build a docker by running `./build_docker.sh`

## How to run

### Running bash in the docker container
To run the environment, you can use `./run_bash.sh` with optional docker parameters. Example:

```bash
./run_bash.sh -v /hpc-datasets:/hpc-datasets -v $(pwd):/pwd
```

### Dataset preparation
To prepare for training and standalone extraction for the CleanEval dataset, run:

```bash
./boilernet/prepare_data_cleaneval.sh -r <input/path/cleaneval/raw/> -h <output/path/cleaneval/html> -p <output/path/cleaneval/train_data> -e <output/path/cleaneval/for_extract>
```

Example:
```bash
./boilernet/prepare_data_cleaneval.sh -r /hpc-datasets/web/cleaneval/raw/ -h /hpc-datasets/web/cleaneval/boilernet/raw/ -p /hpc-datasets/web/cleaneval/boilernet/prepro/ -e /hpc-datasets/web/cleaneval/boilernet/extract_data_all
```

Run the correspoding command `prepare_data_dragnet.sh` to prepare data for Dragnet dataset.


### Training
Train the model with this command:

```bash
./boilernet/train.sh -i <input/path/to/prepro/data> -o <output/path/to/model/results> [-d <dropout>] [-e <number of epochs>]
```

Example:

```bash
./boilernet/train.sh -i /hpc-datasets/web/dragnet/boilernet/prepro -o /hpc-datasets/ben/boilernet/dragnet/model_training_d0.1/ -d 0.1
```


### Extraction
After training with data, you will want to start extracting text from some dataset. Run:

```bash
./boilernet/extract_content.py -i <input/path/for_extract> -o <output/path/results> --model-path <path/to/model/ckpt.h5>
```

Example:
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true ./boilernet/extract_content.py -i /hpc-datasets/web/dragnet/boilernet/extract_data_all/ \
                                                              -o /hpc-datasets/ben/boilernet/dragnet/all_d0.1 \
                                                              --model-path /hpc-datasets/ben/boilernet/dragnet/model_training_d0.1/ckpt/model.028.h5
```
