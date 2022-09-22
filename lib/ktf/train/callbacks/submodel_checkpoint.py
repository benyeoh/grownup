import tensorflow as tf
import numpy as np


class SubModelCheckpoint(tf.keras.callbacks.Callback):
    """A custom Keras callback to save submodels of a model

    It is assumed that verbose = 1 and submodel checkpointing only runs at the end of epoch
    """

    def __init__(self,
                 filepath,
                 submodel_name,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 **kwargs):
        """Initialize custom callback with parameters

        Args:
            filepath: Filepath to save submodel
            submodel_name: Name of submodel to be saved. Nested submodels can be referenced with '@'.
                Example, `base_model@layer1`
            monitor: (Optional) Metric to monitor for best values
            save_best_only: (Optional) True if want to save for best validation values. False to save at the end of every epoch.
            save_weights_only: (Optional) True if want to save weights only. False to save entire submodel.
            mode: (Optional) Mode for comparison of current value and historical best value.
                Either 'min' or 'max' or 'auto'. Default is 'auto'
        """

        super(SubModelCheckpoint, self).__init__(**kwargs)
        self._filepath = filepath
        self._submodel_name = submodel_name
        self._monitor = monitor
        self._save_best_only = save_best_only
        self._save_weights_only = save_weights_only
        if mode == "min":
            self._compare = np.less
            self._best = np.Inf
        elif mode == "max":
            self._compare = np.greater
            self._best = np.NINF
        elif mode == "auto":
            if ("accuracy" in monitor or
                "acc" in monitor or
                "score" in monitor or
                monitor.startswith("fmeasure")):  # noqa
                self._compare = np.greater
                self._best = np.NINF
            else:
                self._compare = np.less
                self._best = np.Inf
        else:
            raise Exception("mode argument has to be 'min' or 'max' or 'auto'")

    def on_train_begin(self, logs=None):
        submodel = self._find_submodel(self._submodel_name)
        if self._save_weights_only:
            self._save = submodel.save_weights
        else:
            self._save = submodel.save

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self._monitor)
        if not current:
            print("\nCan save best model only with %s available, skipping" % self._monitor)
            return

        epoch = epoch + 1
        final_file_path = self._get_file_path(epoch, logs)
        if self._save_best_only:
            if self._compare(current, self._best):
                print("\nEpoch {:05d}: {} improved from {:.5f} to {:.5f}, saving {} to {}".format(
                    epoch, self._monitor, self._best, current, self._submodel_name, final_file_path))
                self._best = current
                self._save(final_file_path)
            else:
                print("\nEpoch {:05d}: {} did not improve from {:.5f}".format(epoch, self._monitor, self._best))
        else:
            print("\nSaving {} to {}".format(self._submodel_name, final_file_path))
            self._save(final_file_path)

    def _get_file_path(self, epoch, logs):
        return self._filepath.format(epoch=epoch, **logs)

    def _find_submodel(self, name):
        cur_submodel = self.model
        for cur_submodel_str in name.split("@"):
            cur_submodel = cur_submodel.get_layer(cur_submodel_str)
        return cur_submodel
