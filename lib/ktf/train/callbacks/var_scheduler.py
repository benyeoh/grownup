import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class VarScheduler(keras.callbacks.Callback):
    """Scheduler for any keras.optimizers.Optimizer variable

    Example usage:
    ```python
        # This function keeps the learning rate at 0.001 for the first ten epochs
        # and decreases it exponentially after that.
        def scheduler(epoch):
            if epoch < 10:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (10 - epoch))
        callback = tf.keras.callbacks.VarScheduler(scheduler, "learning_rate")
        model.fit(data, labels, epochs=100, callbacks=[callback],
                  validation_data=(val_data, val_labels))
    ```
    
    Example usage using dynamic config dict:
    ```python
        ...
        
        "optimizer": {
            "tfa.optimizers.AdamW": {
                "weight_decay": 1.0,
                "learning_rate": 0.0
            }
        },
            
        "train_loop": {
                "ktf.train.KerasTrainLoop": {
                    "num_epochs": 10,
                    "save_dir": None,
                    "callbacks": [
                        {
                            "ktf.train.callbacks.VarScheduler": {
                                "schedule": {
                                    "tf.keras.experimental.CosineDecayRestarts": {
                                        "initial_learning_rate": 0.005,
                                        "first_decay_steps": 10,
                                        "t_mul": 2
                                    }
                                },

                                "var_name": "learning_rate"
                            }
                        },
                        
                        {
                            "ktf.train.callbacks.VarScheduler": {
                                "schedule": {
                                    "tf.keras.experimental.CosineDecayRestarts": {
                                        "initial_learning_rate": 0.000025,
                                        "first_decay_steps": 10,
                                        "t_mul": 2
                                    }
                                },

                                "var_name": "weight_decay"
                            }
                        }
                    ]
                }
            }
    ```
    """

    def __init__(self, schedule, var_name, verbose=0):
        """Initializer

        Args:
            schedule: a function that takes an epoch index as input (integer, indexed from 0)
                and returns a new learning rate as output (float).
            var_name: A string name for the Optimizer variable to schedule
            verbose: int. 0: quiet, 1: update messages.
        """
        super(VarScheduler, self).__init__()
        self._schedule = schedule
        self._verbose = verbose
        self._var_name = var_name

    def on_epoch_begin(self, epoch, logs=None):
        # The optimizer should support get/setattr on it's hyper parameters with overloading
        # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L665-L692
        if not hasattr(self.model.optimizer, self._var_name):
            raise ValueError('Optimizer must have a %s attribute.' % self._var_name)

        try:
            # New API
            val = float(K.get_value(getattr(self.model.optimizer, self._var_name)))
            val = self._schedule(epoch, val)
        except TypeError:
            # Support for old API for backward compatibility
            val = self._schedule(epoch)

        if not isinstance(val, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(val, tf.Tensor) and not val.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')

        var = getattr(self.model.optimizer, self._var_name)
        if isinstance(var, tf.Tensor):
            K.set_value(var, K.get_value(val))
            if self._verbose > 0:
                print('\nEpoch %05d: VarScheduler reducing %s Tensor to %s.' % (epoch + 1, self._var_name, val))
        else:
            setattr(self.model.optimizer, self._var_name, K.get_value(val))
            if self._verbose > 0:
                print('\nEpoch %05d: VarScheduler reducing %s attribute to %s.' % (epoch + 1, self._var_name, val))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = K.get_value(getattr(self.model.optimizer, self._var_name))
        logs[self._var_name] = val
