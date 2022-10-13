import gc

import tensorflow as tf

from ..loop import TrainLoop


class KerasTrainLoop(TrainLoop):
    """Implements a typical keras model training procedure.
    """

    def __init__(self,
                 num_epochs=10,
                 num_steps=None,
                 num_valid_steps=None,
                 valid_freq=1,
                 callbacks=None,
                 verbose=1,
                 run_eagerly=False,
                 force_gc=False,
                 **kwargs):
        """Initialization function.

        Args:
            num_epochs: (Optional) Number of epochs to train
            num_steps: (Optional) Number of train steps per epoch.
                If None, trains until end of dataset (which could be infinite)
            num_valid_steps: (Optional) Number of validation steps per epoch.
                If None, trains until end of dataset (which could be infinite)
            callbacks: (Optional) List of Keras callbacks to be run during model.fit()
            verbose: (Optional) Set Keras verbose level during model.fit()
            run_eagerly: (Optional) If True, force model.fit() to run with eager execution. Default is False.
            force_gc: (Optional) If True, force python garbage collection and clear keras session variables after
                every epoch. Default is False.
            **kwargs: (Optional) Base class parameters
        """
        super(KerasTrainLoop, self).__init__(**kwargs)

        class _ClearMemory(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                # TODO: This triggers some bug in Tensorflow with distribution strategy :/
                # tf.keras.backend.clear_session()

        self._initial_epoch = 0
        self._num_epochs = num_epochs
        self._num_steps = num_steps
        self._num_valid_steps = num_valid_steps
        self._valid_freq = valid_freq
        if callbacks is not None and not isinstance(callbacks, list):
            callbacks = [callbacks]
        self._train_callbacks = callbacks
        if force_gc:
            if self._train_callbacks is None:
                self._train_callbacks = [_ClearMemory()]
            else:
                self._train_callbacks.append(_ClearMemory())

        self._verbose = verbose
        self._run_eagerly = run_eagerly

    def call(self, model, train_dataset, valid_dataset, **kwargs):
        old_run_eagerly = model.run_eagerly
        model.run_eagerly = self._run_eagerly
        hist = model.fit(x=train_dataset,
                         epochs=self._num_epochs,
                         verbose=self._verbose,
                         callbacks=self._train_callbacks,
                         validation_data=valid_dataset,
                         validation_steps=self._num_valid_steps,
                         class_weight=None,
                         sample_weight=None,
                         initial_epoch=self._initial_epoch,
                         steps_per_epoch=self._num_steps,
                         validation_freq=self._valid_freq)

        if valid_dataset:
            print("Performing final validation ...")
            eval_res = model.evaluate(x=valid_dataset, steps=self._num_valid_steps, verbose=self._verbose)
        else:
            eval_res = None

        model.run_eagerly = old_run_eagerly
        return {"final": eval_res, "history": hist}
