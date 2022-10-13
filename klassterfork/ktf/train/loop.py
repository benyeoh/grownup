import os
import time
import glob

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class TrainLoop:
    """Base class for derived training loops.

    Implements common utilities like:

    1. Saving/loading (sub)model weights
    2. Setting up and compiling models within a distribution strategy context
    3. Freezing submodel layers.

    For saving and loading weights, the train loop automatically selects the latest saved weights with the same
    name as the model. If there additional metrics used during validation, the last metric score is also used
    as part of the filename, example "one_head_net-0.97-124566135577.tf" where `0.97` was the last metric score.
    Otherwise will be formatted as None.

    Note: While this class supports saving/loading weights in either h5 or tf format, the latter is preferred
        as there is a bug with saving/loading weights if layers are frozen

    The class also uses a distribution strategy for training. (Currently fixed as MirroredStrategy).
    """

    def __init__(self,
                 save_dir="/tmp/ktf/train_loop",
                 save_submodels=[],
                 freeze_submodels=[],
                 pretrained_weights=None):
        """Initialize train loop.

        Args:
            save_dir: (Optional) The directory where model weights are saved (and loaded) after every training cycle.
                If None, model weights are not saved/loaded
            save_submodels: (Optional) A list of additional submodels to save. Referenced by name.
                Nested submodels can be referenced with '@'. Example, `base_model@layer1`
            freeze_submodels: (Optional) A list of submodels to disable trainable weights
            pretrained_weights: (Optional) Either a (submodel name, weights file path) list of tuples
                for loading pre-trained weights, or a weights file path string. If None, pretrained weights are
                not loaded
        """
        self._save_dir = save_dir
        self._save_submodels = save_submodels
        self._freeze_submodels = freeze_submodels
        self._save_ext = "tf"
        self._pretrained_weights = pretrained_weights

    def _find_submodel(self, model, name):
        cur_submodel = model
        for cur_submodel_str in name.split("@"):
            cur_submodel = cur_submodel.get_layer(cur_submodel_str)
        return cur_submodel

    def _load_model_weights(self, model, path):
        return model.load_weights(path)

    def _load_pretrained_weights(self, model, load_specs):
        if isinstance(load_specs, list):
            for name, path in load_specs:
                submodel = self._find_submodel(model, name)
                if path is not None:
                    print("\nLoading pre-trained weights model %s using %s ...\n" % (submodel.name, path))
                    self._load_model_weights(submodel, path)
                else:
                    self._load_weights(submodel)
        elif isinstance(load_specs, str):
            print("\nLoading pre-trained weights model %s using %s ...\n" % (model.name, load_specs))
            self._load_model_weights(model, load_specs)
        else:
            raise ValueError("Either a model_name->file_path dict or file_path string is expected.")

    def _save_weights(self, model, score):
        os.makedirs(self._save_dir, exist_ok=True)

        cur_time = int(time.time() * 1000.0)
        main_model_filename = "{}-{:.3f}-{}.{}".format(model.name, score, cur_time, self._save_ext)
        save_filepath = os.path.join(self._save_dir, main_model_filename)
        print("\nSaving weights %s ..." % save_filepath)
        model.save_weights(save_filepath)

        save_paths = []
        save_paths.append(save_filepath)

        for submodel_str in self._save_submodels:
            cur_submodel = self._find_submodel(model, submodel_str)
            submodel_filename = "{}-{:.3f}-{}.{}".format(submodel_str, score, cur_time, self._save_ext)
            save_filepath = os.path.join(self._save_dir, submodel_filename)
            print("Saving submodel weights %s ..." % save_filepath)
            cur_submodel.save_weights(save_filepath)
            save_paths.append(save_filepath)
        print()
        return save_paths

    def _load_weights(self, model):
        glob_pattern = "{}-*.{}".format(model.name, self._save_ext)
        if self._save_ext == "tf":
            glob_pattern += ".index"

        filepaths = glob.glob(os.path.join(self._save_dir, glob_pattern))
        if len(filepaths) > 0:
            save_times = [int(filename.split("-")[-1].split(".")[0]) for filename in filepaths]
            latest_filepath = filepaths[np.argmax(save_times)].split(
                ".%s" % self._save_ext)[0] + (".%s" % self._save_ext)
            print("\nLoading weights %s ...\n" % latest_filepath)
            try:
                model.load_weights(latest_filepath)
            except:
                print("\nError loading weights: %s. You should:\n"
                      "    (1) Check if your models have the same shape as the saved weights\n"
                      "    (2) Or delete the saved weights\n"
                      "    (3) Or use a different save directory\n" % latest_filepath)
                raise
        else:
            print("\nWeights for %s not found in %s ...\n" % (model.name, self._save_dir))

    def call(self, *args, **kwargs):
        raise NotImplementedError("Training loop must be implemented!")

    def __call__(self,
                 train_dataset,
                 valid_dataset,
                 create_model_fn,
                 create_loss_fn,
                 loss_weights,
                 create_metrics_fn,
                 create_optimizer_fn,
                 on_setup_done_fn=None):
        """Functor to setup the context and train with various parameters.

        Args:
            train_dataset: The training tf.data.Dataset
            valid_dataset: The validation tf.data.Dataset
            create_model_fn: A functor that creates a tf.keras.Model object
            create_loss_fn: A functor that creates a tf.keras.losses.Loss object
            loss_weights: A list of values specifying the weight of each loss
            create_metrics_fn: A functor that creates a tf.keras.metrics.Metric object
            create_optimizer_fn: A functor that creates a tf.keras.optimizers.Optimizer object
            on_setup_done_fn: A functor callback after model setup is done

        Returns:
            res: A dictionary containing entries "results" for the training results
            and "save_paths" for the paths to the saved weights for each (sub)model
        """

        keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            print("\nRunning in distributed scope, %d GPUs...\n" % (strategy.num_replicas_in_sync))
            print("Creating model ...")
            model = create_model_fn()
            print("Creating loss ...")
            loss = create_loss_fn()
            print("Creating metrics ...")
            metrics = create_metrics_fn()
            print("Creating optimizer ...")
            optimizer = create_optimizer_fn()

            if not model.built:
                print("Build model ...")
                model(next(iter(train_dataset))[0], training=True)

            for submodel_str in self._freeze_submodels:
                cur_submodel = self._find_submodel(model, submodel_str)
                print("Freezing submodel layer: %s (%s) ..." % (cur_submodel.name, submodel_str))
                cur_submodel.trainable = False

            print("Compiling model ...")
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
            model.summary()

            if self._save_dir:
                self._load_weights(model)
            if self._pretrained_weights:
                self._load_pretrained_weights(model, self._pretrained_weights)

            if on_setup_done_fn:
                on_setup_done_fn(model, loss, metrics, optimizer)

        res = self.call(model,
                        train_dataset,
                        valid_dataset,
                        strategy=strategy,
                        loss=loss,
                        metrics=metrics,
                        optimizer=optimizer)

        print()
        score = None
        final_res = res["final"]
        if final_res is not None:
            if type(final_res) is list or type(final_res) is tuple:
                score = final_res[-1]
            else:
                score = final_res

        save_path = None
        if self._save_dir:
            save_path = self._save_weights(model, score)
        return {"results": res, "save_path": save_path}
