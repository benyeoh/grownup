# Dragnet benchmarking

## How to install
Build a docker by running `./build_docker.sh`

## How to run

### Running bash in the docker container
To run the environment, you can use `./run_bash.sh` with optional docker parameters. Example:

```
./run_bash.sh -v /hpc-datasets:/hpc-datasets -v $(pwd):/pwd
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