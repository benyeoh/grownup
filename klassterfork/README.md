# Klassterfork Python
This repo contains utilities and scripts to facilitate development and prototyping of deep-learning models using Tensorflow.

## 1. Pre-requisites
The hardware / software requirements for ktf are:

* Ubuntu 18.04 or newer
* Docker 19.03 or newer with GPU support (Optional but recommended)
* NVIDIA GPU with CUDA 11.2.1 support (GPU driver version: >=460.32.03)
Typically, you don't need to care on Cuda library installation if you use Docker
* [Git LFS](https://git-lfs.github.com/)
**This is super important and is required when you clone / pull from this repo, since some large data (ie, pretrained weights) is stored in LFS.**

## 2. How to install (assuming you don't want to use a Docker image)
1. Ensure that you have the required GPU drivers and libraries installed:
    * We need to install CUDA 11.2.1, if it's not already installed. There are many guides for this. For myself, I first added the NVIDIA PPAs with:
        ```sh
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
        sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
        sudo apt-get update
        ```
    * Then I simply installed `cuda-11-2` with:
        ```sh
        sudo apt-get install cuda-11-2
        ```
    * If it complains of missing dependencies of `cuda-drivers`, you might have to manually install the right version of `cuda-drivers` that corresponds to your GPU driver version first. For example, `sudo apt-get install cuda-drivers=470.141.03-1`. Then try installing `cuda-11-2` again.
    * You might also need to install CuDNN v8.1:
        ```sh
        # Install development and runtime libraries (~4GB)
        apt-get install -y --no-install-recommends \
        libcudnn8=8.1.1.33-1+cuda11.2 \
        libcudnn8-dev=8.1.1.33-1+cuda11.2
        ```
    * Finally, make sure your cudart .so's are cached with:
        ```sh
        sudo ldconfig
        ```        
2. Install Python (>=3.7) (recommended to use a PPA as detailed here: https://websiteforstudents.com/installing-the-latest-python-3-7-on-ubuntu-16-04-18-04/)
3. Install Python (>=3.7) developer libraries: `sudo apt-get install python3.7-dev` 
4. (Optional) Recommended that you create a `virtualenv` for ktf and related projects:
    * Run `pip3 install virtualenv`
    * Run `which python3.7` to get the path to your python3.7 interpreter
    * Run `virtualenv -p <python3.7_path> <venv_path_to_create>` to create a venv for python3.7
    * Run `source <venv_path>/bin/activate` to activate your venv (and `deactivate` to deactivate it later)
5. At this point, you should be in your virtualenv that you just created. See step 4.
6. In the `<ktf_project_root>/python` folder, run `pip install -r requirements.txt` to install required packages
7. In the `<ktf_project_root>/python` folder, run `pip install -r requirements_ext.txt` to install required packages
8. Test that everything is installed correctly with `python -c "import tensorflow as tf;tf.test.is_gpu_available()"`
    * You should see an output similar to:
      ```
      Created TensorFlow device (/device:GPU:0 with 5033 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5)
      ```

ktf currently uses **Tensorflow 2.5.0**.

For development within the project itself, including running functional tests and notebooks and other ktf package scripts, it is not necessary to install the ktf package.
All scripts and notebooks defined within the `klassterfork/python` subdirectory should be able to run as-is.

For external development using the ktf package, you must first install the package with `pip install --force-reinstall .`.

## 3. Architecture
The architecture of ktf encourages breaking up an implementation into reusable "components" rather than a single monolithic feature.
This is to facilitate sharing of useful features.

We briefly describe the folder structure below:

```
klassterfork # Project root
├── ktf # Python package where all the good stuff goes
│   ├── datasets # All generic dataset readers and parsers should go here
│   ├── models # Common end-to-end generic models should be defined in the root
│   │   ├── components # Reusable network components are defined here
│   │   ├── networks # Networks are defined here
│   │   ├── losses # Loss functions
│   │   └── ...
│   └── train # Boilerplate code for training arbitrary models
├── scripts # Some utility scripts to print and save/load weights from models
├── docker # Docker image building scripts
└── runs # Training config parameters and some data to reproduce past results
```

### Methodology to use ktf
To facilitate prototyping of models, ktf allows you to mix and match models and datasets and train models defined in the ktf package without ever writing python boilerplate code.

#### How to train with no python code
ktf supports training with deep customizability of the training parameters as-is by running the `klassterfork/python/ktf/train/trainer.py` script with a configuration JSON file.

This is a good example [training config](runs/web/cleaneval/cleaneval.hjson) with some comments interspersed.

An example command line to train would be: `TF_FORCE_GPU_ALLOW_GROWTH=true ./klassterfork/python/ktf/train/trainer.py -c <path_to_json_config>`

Note the `TF_FORCE_GPU_ALLOW_GROWTH=true` bash variable before the actual command. This tells Tensorflow not to pre-allocate a large chunk of GPU memory at once.
If you're getting `CUDNN_STATUS_INTERNAL_ERROR` and your program crashes because of OOM, this is a possible culprit.

#### How to build a docker environment to train models
If you find it a hassle to setup your environment, and/or want to train models on different hosts, klassterfork supports building docker images with ktf installed.

To build docker images, run: `./docker/build_docker.sh`

To train models with a docker image, a sample command would be:

```
docker run -it --gpus all --rm -v /hpc-datasets:/hpc-datasets <DOCKER_IMAGE> ./python/ktf/train/trainer.py -c ./python/runs/stanford_cars/baseline_improved/config.json
```

#### How to train with python code
Sometimes you would like to quickly train with custom user-side code. In this case we will need to write minimal python boiler plate:

```python
import ktf.models
import ktf.train


if __name__ == "__main__":
    train_env = ktf.train.Environment()
    # Start training with the config
    train_env.run([
        {
            "datasets": ...
            "model": ...
            "loss": ...
        }
    ])
```

#### How to programatically use a model defined in ktf
```python
import ktf.models
import ktf.train

if __name__ == "__main__":
    # Either just pass a JSON filepath to DynamicConfig  ...
    model = ktf.train.DynamicConfig("path/to/config.json")

    # ... or pass a dict to DynamicConfig 
    same_model = ktf.train.DynamicConfig({
        "ktf.models.Sequential": {
            "name": "gcn_trans",
            "layers": [
                {
                    "ktf.models.networks.RecurrentCGN": {
                        "name": "gcn",
                        "layers": [
                            2,
                            2,
                            2,
                            2,
                            2,
                        ],
                        "debug_self_loops": null,
                        "num_prelim_blocks": [768, 256],
                        "hidden_base_feature_size": 256,
                        "output_feature_size": null,
                        "use_residuals": true,
                        "aggregator": "gat_gated",
                        "normalization": "layer",
                        "recurrent_cell": "lstm",
                        "aggregator_activation": "relu",
                        "recurrent_activation": "tanh",
                        "dropout": 0.2
                    }
                },

                {
                    "ktf.models.networks.TransGraphNet": {
                        "name": "transgraph",
                        "num_layers": 5,
                        "embedding_size": 256,
                        "num_heads": 4,
                        "dropout": 0.0
                    }
                }
            ]
        }
    }).get_model()
    
    # ... or just define your model directly
    same_model2 = ktf.models.Sequential(
        name="gcn_trans"
        layers=[
            ktf.models.networks.RecurrentCGN(
                ...
            ),
            ktf.models.networks.TransGraphNet(
                ...
            )
        ]
    )

```


## 4. FAQ

### Help! When I try to run training on my command line or jupyter notebook, it crashes with `CUDNN_STATUS_INTERNAL_ERROR`.

Run your TF application or jupyter notebook with `TF_FORCE_GPU_ALLOW_GROWTH=true [<tf_application>|<jupyter notebook>]`

### I'm running training or unit tests using a GPU from a docker container and it keeps crashing! I tried using TF_FORCE_GPU_ALLOW_GROWTH=true too!

When running a docker container on some GPUs, you most likely need to include `--shm-size=256m` or `--shm-size=512m` to set enough shared memory for Tensorflow's GPU workings.
