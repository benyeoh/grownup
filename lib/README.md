# Klassterfork Python
This repo contains utilities and scripts to facilitate development and prototyping of deep-learning models using Tensorflow.

## TOC
- [1. Pre-requisites](#1-pre-requisites)
- [2. How to install](#2-how-to-install)
- [3. Architecture](#3-architecture)
  * [Methodology to use ktf](#methodology-to-use-ktf)
    + [How to train with no python code](#how-to-train-with-no-python-code)
    + [How to train with python code](#how-to-train-with-python-code)
  * [Methodology when adding new features](#methodology-when-adding-new-features)
    + [Adding a new dataset](#adding-a-new-dataset)
    + [Adding a new network](#adding-a-new-network)
    + [Adding new programmatic features or algorithms (new optimizers, gradient descent algorithms, special train steps, dataset processors for triplet loss etc)](#adding-new-programmatic-features-or-algorithms--new-optimizers--gradient-descent-algorithms--special-train-steps--dataset-processors-for-triplet-loss-etc-)
    + [Adding a new functional test](#adding-a-new-functional-test)
- [4. FAQ](#4-faq)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## 1. Pre-requisites
The hardware / software requirements for ktf are:

* Ubuntu 16.04 or newer
* NVIDIA GPU with CUDA 11.2.1 support (GPU driver version: >=460.32.03)
* Git LFS

## 2. How to install
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

Instead of implementing an end-to-end ResNet model for classification for example, contributors should break up ResNet into its
different layers/blocks and should not conflate ResNet used in a classification use-case with its core purpose as
a feature encoder. (ResNet can also be used for regression).

The structure of the project should nudge people towards particular workflows. We briefly describe the folder structure below:

```
klassterfork # Project root
└── python # Python framework
    ├── ktf # Python package where all new features go
    │   ├── datasets # All generic dataset readers and parsers should go here
    │   ├── models # Common end-to-end generic models should be defined in the root
    │   │   ├── components # Reusable network components are defined here
    │   │   ├── networks # Networks are defined here
    │   │   ├── losses # Loss functions
    │   │   └── ...
    │   ├── train # Boilerplate code for training arbitrary models flexibly
    ├── notebooks # Tutorials (which are also tests) and model viewers
    └── tests # Functional tests
```

### Methodology to use ktf
To facilitate prototyping of models, ktf allows you to mix and match models and datasets and train models defined in the ktf package without ever writing python boilerplate code.

#### How to train with no python code
ktf supports training with deep customizability of the training parameters as-is by running the `klassterfork/python/ktf/train/trainer.py` script with a configuration JSON file.
The limitation here is that all training parameters must be available in either the ktf packages or other packages imported by ktf (like tensorflow)

A simple JSON training config is shown below that trains 1 stage with default Adam optimizer and keras training loop:

```yaml
[
    {
        "datasets": { "ktf.datasets.digits.affnist.from_tfrecord": { "affnist_tfrecord_dir": "data/affnist" } },
        "model": { "ktf.models.OneHeadNet": { "num_outputs": 10 } },
        "loss": { "tf.keras.losses.SparseCategoricalCrossentropy": { "from_logits": true } },        
    }
]
```

And by contrast, here's a fairly complex JSON config:

_✨New✨: ktf now supports [hjson](https://hjson.github.io/), a superset of json that supports comments (as well as some other features). Comments can be indicated with `#`, `//` or in a block like `/* block comment */`!_


```yaml
# Top level list to hold 1 or more train cycle parameters.
# If the list contains 2 dictionaries of parameters for example, the trainer will execute 2 training cycles
[
    # 1st train cycle parameters
    {
        # Define the train/validation datasets
        "datasets": {
            # Read from affnist tfrecord dataset
            "ktf.datasets.digits.affnist.from_tfrecord": {
                # Define parameters for the function (same as the python function definition)
            
                "affnist_tfrecord_dir": "data/affnist", # Path of directory to tfrecords
                "batch_size": 64,                       # Batch size
                "force_rgb": true                       # Force 3 output channels instead of 1 channel
            }
        },

        # Define the model
        "model": {
            # Use OneHeadNet (ie, 1 base model with 1 dense head)
            "ktf.models.OneHeadNet": {
                # Define parameters for OneHeadNet (same as the __init__ constructor)
                
                "base_model": {
                    # Use keras pre-trained MobileNetV2 for the base model for OneHeadNet
                    "tf.keras.applications.MobileNetV2": {
                        "include_top": false,           # No top since OneHeadNet gives us that
                        "input_shape": [40, 40, 3],     # Same as input for dataset
                        "pooling": "avg"                # Average pooling
                    }
                },

                "num_outputs": 10,      # Number of outputs for head - same as number of classes
                "dropout": 0.4          # Dropout rate since we will severely overfit for small data
            }
        },

        # Define the model loss
        "loss": {
            "tf.keras.losses.SparseCategoricalCrossentropy": {
                "from_logits": true
            }
        },
        
        # Define the metrics. Note that it takes a list
        "metrics": [ {
                "tf.keras.metrics.SparseCategoricalAccuracy": {}
            }
        ],
        
        # Define the optimizer 
        "optimizer": {
            "tf.keras.optimizers.Adam": {
                "learning_rate": 0.0005
            }
        },

        # Define the training loop
        "train_loop": {
            # We will in most cases use KerasTrainLoop. For custom training loops use ktf.train.CustomTrainLoop
            # By default, the training will save and load model weights from `/tmp` before and after every train cycle.
            # It is also possible to save and load a list of **submodel** weights 

            "ktf.train.KerasTrainLoop": {
                "num_epochs": 10,
                "callbacks":[
                    {
                        "tf.keras.callbacks.TensorBoard":{
                            "log_dir": "/tmp/ktf/tensorboard", 
                            "histogram_freq": 1, 
                            "write_graph": true
                        }
                    }
                ]                                               # Output tensorboard logs
                "freeze_submodels": ["base_model"]              # Freeze the submodel with the specifed name.
                                                                # Nested submodels can be scoped with "@" (ex, "base_model@layer1@resample" etc)
            }
        }
    },

    # 2nd train cycle parameters.
    # Note that it is okay to define only some parameters, in which case undefined parameters 
    # in subsequent cycles will re-use earlier definitions
    {
        "optimizer" : {
            "tf.keras.optimizers.Adam": {
                "learning_rate": 0.00025
            }
        },
        
        "train_loop": {
            "ktf.train.KerasTrainLoop": {
                "num_epochs": 5,
                "num_valid_steps": null,
                "callbacks":[
                    {
                        "tf.keras.callbacks.TensorBoard":{
                            "log_dir": "/tmp/ktf/tensorboard", 
                            "histogram_freq": 1, 
                            "write_graph": true
                        }
                    }
                ]
            }
        }
    }
]
```

An example command line to train would be: `TF_FORCE_GPU_ALLOW_GROWTH=true ./klassterfork/python/ktf/train/trainer.py -c <path_to_json_config>`

Note the `TF_FORCE_GPU_ALLOW_GROWTH=true` bash variable before the actual command. This tells Tensorflow not to pre-allocate a large chunk of GPU memory at once.
If you're getting `CUDNN_STATUS_INTERNAL_ERROR` and your program crashes, this is a likely culprit.

#### How to build a docker environment to train models
If you find it a hassle to setup your environment, and/or want to train models on different hosts, klassterfork supports building docker images with ktf installed.

To build docker images, run: `./docker/build_docker.sh`

To push docker images to `dr1.klass.dev:5000`, run : `./docker/push_docker.sh`

To clean dangling images, run: `./docker/clean_docker.sh`

To train models with a docker image, a sample command would be:

```
docker run -it --gpus all --rm -v /hpc-datasets:/hpc-datasets <DOCKER_IMAGE> ./python/ktf/train/trainer.py -c ./python/runs/stanford_cars/baseline_improved/config.json
```

#### How to train with python code
Sometimes you would like to quickly train with custom user-side code that is not defined in ktf or other imported packages. In this case we will need to write minimal python boiler plate:

```python
#!/usr/bin/env python
import ktf.models
import ktf.train

# Important: Use this decorator so ktf can see it from the config
@ktf.train.export_config
class MyCustomModel(ktf.models.OneHeadNet)
    def __init__(self, **kwargs):
        super(MyCustomModel, self).__init__(num_outputs=10, **kwargs)


if __name__ == "__main__":
    train_env = ktf.train.Environment()
    # Start training with the config
    train_env.run([
        {
            "datasets": { "ktf.datasets.digits.affnist.from_tfrecord": { "affnist_tfrecord_dir": "data/affnist" } },
            "model": { "MyCustomModel": { "name": "my_hello_world_model" } },
            "loss": { "tf.keras.losses.SparseCategoricalCrossentropy": { "from_logits": True } },
        }
    ])
```

### Methodology when adding new features
The following section details how to add new features to the framework.

#### Adding a new dataset
1. Create a new folder in `klassterfork/python/ktf/datasets` with the name of the dataset (ie, mnist, imagenet, stamford_cars etc)
2. Create a raw data parser in `klassterfork/python/ktf/datasets/<dataset_name>/parser.py`. This parser should have 1 public method called `parse` that returns a tuple of values. It should take no arguments.
    * At the declaration of the of your parser class, write a docstring (ie, enclosing `"""`) describing briefly the dataset and add website links if applicable.
3. Create a raw-to-tfrecord converter in `klassterfork/python/ktf/datasets/<dataset_name>/raw_to_tfrecord.py`. This converts from the raw dataset using the parser to multiple tfrecord files.
4. Create a tf.data.Dataset pipeline in `klassterfork/python/ktf/datasets/<dataset_name>/dataset.py` that reads tfrecord files, splits the data, and returns a training and validation tf.data.Dataset tuple (called `from_tfrecord`). Can **optionally** also include a raw to tf.data.Dataset function in the same format.
5. Write a jupyter notebook in `klassterfork/python/notebooks/viewers/` with filename `<dataset_name>.ipynb` that reads the raw data and displays it
6. Import the functions and classes that we want to expose in the `__init__.py` script in the containing package folder.
7. Use the raw-to-tfrecord converter to convert the raw data. (You should probably store it somewhere appropriate on `/hpc-datasets/`)
8. In the same jupyter notebook as step 6, implement in another cell reading from the tfrecord files using the dataset reader and display the data. (It should be the same layout except for some shuffling/reshaping maybe).

#### Adding a new network
1. Understand the core functionality of the network. Is it an encoding network, can it be used in composition with other networks, regression, classification, sequence encoding?
2. Create a new file in `klassterfork/python/ktf/models/network`. Implement the network using model subclassing (see https://www.tensorflow.org/guide/keras/custom_layers_and_models#building_models)
3. As you design and implement, refactor network submodules **that can be reused** into `klassterfork/python/ktf/models/components`.
    * In general, all submodules of a network that are a **composition** of `tf.keras.layers.Layer` objects should inherit from `tf.keras.Model`.
    * Otherwise, if a submodule requires weights and are **not a composition** of `tf.keras.layers.Layer`, inherit from `tf.keras.layers.Layer` instead.
    * If a submodule both requires weights and is a composition of `tf.keras.layers.Layer`, you should probably breakdown the design more.
    * If submodules are very specific to the network and cannot be potentially re-used, it is okay to define those submodules within the same file as the network definition.
4. At the declaration of your model class(es), write a docstring (ie, enclosing `"""`) describing in some detail the model, model parameters, and add links to appropriate papers if applicable.
5. Implement a top level container if required in `klassterfork/python/ktf/models` to compose similar networks for a specific use case. Example, multi-head classification/regression container, sequence encoder/decoder container.
6. Import the functions and classes that we want to expose in the `__init__.py` script in the containing package folder.
7. Write a jupyter notebook in `klassterfork/python/notebooks/tutorials/` with filename `<model_name>.ipynb` that demonstrates how to use the model. This is also a test to verify it works.
8. Finally, weights for the network shouldn't be stored as part of the repository, but instead stored in `/hpc-datasets/ktf/weights/<network_type>/<network_name>/`.

#### Adding new programmatic features or algorithms (new optimizers, gradient descent algorithms, special train steps, dataset processors for triplet loss etc)
1. Add python file(s) in appropriate folders.
    * Optimizers, gradient descent algorithms, train steps should go to subfolders in `klassterfork/python/ktf/train` for example.
2. Import the functions and classes that we want to expose in the `__init__.py` script in the containing package folder.
3. Add functional tests in `klassterfork/python/tests`.
    * Your functional tests must check and **raise an assert** if issues are encountered. If the program finishes without crashing, it is deemed to have passed
4. Add a tutorial or viewer notebook in `klassterfork/python/notebooks` if appropriate.

#### Adding a new functional test
1. Implement the test in 1 python or bash script and add it to the `klassterfork/python/tests` folder.
2. Make sure the script is executable with `chmod +x <script_file>`
3. Always assume that the user can run your scripts from any directory. If you need to `os.chdir()`, make sure you restore when you finish the script
4. Your python script should `assert <_expr_>` assumed behavior. Bash scripts should immediately fail when any command it runs crashes.
5. When your script finishes, it should print `*********** Passed ***********`.
6. The assumption is that when your script finishes without crashing, it has passed the functional test.

## 4. FAQ

### Help! When I try to run training on my command line or jupyter notebook, it crashes with `CUDNN_STATUS_INTERNAL_ERROR`.

Run your TF application or jupyter notebook with `TF_FORCE_GPU_ALLOW_GROWTH=true [<tf_application>|<jupyter notebook>]`

### What is `hpc-datasets`?

`hpc-datasets` is a Ceph DFS (distributed file system) share for storing datasets and model weights. It's somewhat messy as it was copied from `/dev-datasets` but moving forward the following conventions should be followed when adding new content:
* Datasets should be named properly. Use the common name - ie, stamford_cars, vmmrdb_cars, affnist, etc etc
* Use lowercase only
* Use logical folder groupings
> As explained by Ben:
> * Example, I put affNIST into /hpc_datasets/digits/affnist/raw/ for the raw data and /hpc-datasets/digits/affnist/tfrecord for the converted tfrecord files. If I wanted to add a MNIST or SHVN dataset, I'll put them in the same digits folder (because they are all dealing with character digit recognition)
> * If I were working on stamford cars and coco bags for example, I'll put them in cars and bags top-level folders respectively

### How do I mount `hpc-datasets`?

Run:
```
sudo mount -t ceph -o name=klass,secret=AQBDWVBgL2I3LhAACmySqXgrl67tcG7737yf7A== kldfsdev01.klass.dev:6789,kldfsdev02.klass.dev:6789,kldfsdev03.klass.dev:6789:/ /hpc-datasets
```

### My training runs very slowly. What can I do to find out why?

Excellent question! First of all, you should arm yourself with the basics that is already documented, namely this Tensorflow [article](https://www.tensorflow.org/guide/data_performance).

Usually, the obvious performance issues should already be handled by KTF since it tries to "railroad" users towards best practices. However this safety net is not always sufficient.

To look deeper into performance issues, we analyze training performance statistics with [Tensorboard](https://www.tensorflow.org/guide/profiler#profiler_tools). The steps to getting Tensorboard running with training profiler statistics are:

1. Add a Tensorboard callback to your Keras callback list:
```python
{
    "tf.keras.callbacks.TensorBoard": {
        "log_dir": "/hpc-datasets/<your name>/ktf/runs/tensorboard",
        "profile_batch": [10, 20] # This tells the profiler to collect stats from train step 10 to 20
    }
}
```

2. Run your training as per normal. Note that during the specified train steps, you should see output like this:

```s
021-07-30 14:21:12.200702: I tensorflow/core/profiler/lib/profiler_session.cc:66] Profiler session collecting data.
2021-07-30 14:21:12.338683: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1743] CUPTI activity buffer flushed
2021-07-30 14:21:17.076833: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:673]  GpuTracer has collected 909656 callback api events and 909256 activity events.
2021-07-30 14:21:37.605262: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-30 14:21:57.171055: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: /hpc-datasets/ben/ktf/runs/tensorboard/train/plugins/profile/2021_07_30_14_21_37
2021-07-30 14:22:06.464225: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to /hpc-datasets/ben/ktf/runs/tensorboard/train/plugins/profile/2021_07_30_14_21_37/97e1e7b26d82.trace.json.gz
```

3. On your host PC, make sure you have Tensorboard and the profile plugin package installed (if you have installed `requirements_ext.tx` you should be covered, otherwise just `pip install tensorboard tensorboard-plugin-profile`). Then run Tensorboard with:

`tensorboard --logdir /hpc-datasets/<your name>/ktf/runs/tensorboard/ --load_fast=false`

The `--logdir` should point to the same folder in the Keras callback.

You should see an output like:
```s
2021-07-30 23:24:20.728807: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then point your browser at the given URL.

4. Open the `Profile` tab if one exists. If not go to the dropdown menu on the upper right and scroll down to `Profile`. If this is still missing, you either did not install the profile plugin or your command line is missing `--load_fast=false`. Maybe check for error messages.

5. Read [this workflow](https://www.tensorflow.org/guide/data_performance_analysis) and familiarize yourself with profiling using Tensorboard and hopefully find your bottleneck. I would start first by checking if you're CPU/IO or GPU bound, then look at the most expensive operations on both GPU and CPU, and then finally try to account for the "blank" spaces in between operations on the GPU using the `trace viewer`.

6. Lastly, if your models are small-ish (check if GPU is mostly idle) and you are struggling to feed the GPU enough work, you are most likely CPU or IO bound. One thing to try is setting this environment variable `TF_GPU_THREAD_MODE=gpu_private` when you train. This will [dedicate some CPU threads](https://www.tensorflow.org/guide/gpu_performance_analysis) exclusively to launch CUDA kernels on the GPU and could improve performance.

### I'm running training or unit tests using a GPU from a docker container and it keeps crashing! I tried using TF_FORCE_GPU_ALLOW_GROWTH=true too!

When running a docker container on some GPUs, you most likely need to include `--shm-size=256m` or `--shm-size=512m` to set enough shared memory for Tensorflow's GPU workings.

### How to run training jobs using Argo workflows ?

See the `argo_train.sh` script and accompanying README in [this](scripts/) folder.
