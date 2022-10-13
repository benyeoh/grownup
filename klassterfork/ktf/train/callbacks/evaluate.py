import tensorflow as tf
import numpy as np


class Evaluate(tf.keras.callbacks.Callback):
    """Perform evaluation of a dataset at the end of an arbitrary epoch(s). This is useful
    when we want to evaluate against a test dataset with the best model so far during
    a training/validation cycle.
    """

    def __init__(self,
                 dataset,
                 name="cb_eval",
                 monitor="val_loss",
                 mode='auto',
                 default_metric_vals={},
                 custom_eval_fn=None,
                 custom_eval_metrics=None,
                 **kwargs):
        """Initialize custom callback with parameters

        Args:
            dataset: A tf.Dataset to evaluate or tuple of tf.Dataset where the 1st element is the dataset to evaluate
            name: (Optional) A unique name required if multiple `Evaluate` callbacks are used. Default is `cb_eval`.
            monitor: (Optional) Either:
                1. An int - The frequency to evaluate at
                2. A list or tuple of int - The epoch(s) to evaluate at
                3. A str - The best-so-far metric to determine whether to evaluate at.
                Default is 'val_loss'.
            mode: (Optional) Only used if `monitor` is a str where `mode` is the comparison of current value and historical best value.
                Either 'min' or 'max' or 'auto'. Default is 'auto'.
            default_metric_vals: (Optional) Default dictionary of values per metric to log when only sparsely evaluating outputs.
                Unfortunately, keras History does not record the actual epoch of the logged result so if you were to only run `model.evaluate
                once every 2 frames, the output log of the evaluation callback results would contain half the entries of the actual number
                of epochs in the array. Also, keras_tuner stuff needs dense log outputs. If an empty dictionary is passed in, we try to
                derive a sensible default value to log instead. 
            custom_eval_fn: (Optional) A custom evaluation function to use instead. A custom evaluation function
                            accepts three arguments, the model, datasets, and metrics.
            custom_eval_metrics: (Optional) Metric(s) to be used in custom_eval_fn, the number of metrics should match the custom_eval_fn.
                            We need custom_eval_metrics because we are unable to access the metrics used by the model to perform evaluation.
            kwargs: (Optional) Args for base keras.callbacks.Callback class.
        """

        super(Evaluate, self).__init__(**kwargs)
        self._dataset = dataset
        if isinstance(self._dataset, (list, tuple)):
            self._dataset = self._dataset[0]
        self._monitor = monitor
        self._custom_eval_fn = custom_eval_fn
        self._custom_eval_metrics = custom_eval_metrics
        self._name = name
        self._default_metric_vals = default_metric_vals

        # NOTE: This part here is more or less lifted from keras.callbacks.ModelCheckpoint
        # with some slight changes
        if mode == "min":
            self._compare = np.less
            self._best = np.Inf
        elif mode == "max":
            self._compare = np.greater
            self._best = np.NINF
        elif mode == "auto":
            if (isinstance(monitor, str) and
                ("accuracy" in monitor or
                "acc" in monitor or
                "score" in monitor or
                monitor.startswith("fmeasure"))):  # noqa
                self._compare = np.greater
                self._best = np.NINF
            else:
                self._compare = np.less
                self._best = np.Inf
        else:
            raise Exception("mode argument has to be 'min' or 'max' or 'auto'")

    def _evaluate(self, logs=None):
        print("\nEvaluating (Callback)...")
        if self._custom_eval_fn is None:
            res = self.model.evaluate(x=self._dataset, verbose=2)
            if logs:
                for val, metric_name in zip(res, self.model.metrics_names):
                    logs["%s_%s" % (self._name, metric_name)] = val
            print()
        else:
            self._custom_eval_fn(self.model, self._dataset, self._custom_eval_metrics)
            self._print_metrics()

    def _log_defaults(self, logs=None):
        if logs:
            for metric_name in self.model.metrics_names:
                if metric_name in self._default_metric_vals:
                    val = self._default_metric_vals[metric_name]
                elif ("accuracy" in metric_name or
                    "acc" in metric_name or
                    "score" in metric_name or
                    "recall" in metric_name or
                    "precision" in metric_name or
                    metric_name.startswith("fmeasure")):  # noqa
                    val = -np.inf
                elif ("loss" in metric_name or
                    "error" in metric_name or
                    "err" in metric_name):  # noqa
                    val = np.inf
                else:
                    raise Exception(
                        "If not evaluating at every epoch, then `default_metric_vals` need to be defined with `%s`" % metric_name)
                logs["%s_%s" % (self._name, metric_name)] = val

    def _print_metrics(self):
        # TODO: Add support for updating logs
        res = [met.result().numpy() for met in self._custom_eval_metrics]
        message = ("Evaluation on Test Set - Metrics: ["
                   + "{:.5f}, " * (len(self._custom_eval_metrics) - 1)
                   + "{:.5f}" * (len(self._custom_eval_metrics) >= 1) + "]\n")
        message = message.format(*res)
        print(message)

        for met in self.model.metrics:
            met.reset_states()

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self._monitor, str):
            current = logs.get(self._monitor)
            if not current:
                print("Can only evaluate with %s available, skipping\n" % self._monitor)
                self._log_defaults(logs=logs)
                return
            if self._compare(current, self._best):
                self._evaluate(logs=logs)
                self._best = current
            else:
                self._log_defaults(logs=logs)
        elif isinstance(self._monitor, (list, tuple)):
            epoch = epoch + 1
            if epoch in self._monitor:
                self._evaluate(logs=logs)
            else:
                self._log_defaults(logs=logs)
        else:
            # Assume it's a scalar integer
            epoch = epoch + 1
            if (epoch % self._monitor) == 0:
                self._evaluate(logs=logs)
            else:
                self._log_defaults(logs=logs)

    def on_train_end(self, logs=None):
        # We cannot run a final evaluation here
        # because that apparently erases the logs :(
        pass


@tf.function
def mean_teacher_evaluate_fn(model, dataset, metrics):
    """ Custom Evaluate Function for MeanTeacherTrainLoop """
    for input in dataset:
        x, y_true = input
        student_output, teacher_output = model(x, training=False)

        # Student model classification on Labelled Data
        metrics[0].update_state(y_true, student_output)

        # Teacher model classification on Labelled Data
        metrics[1].update_state(y_true, teacher_output)

        # Consistency Metrics
        metrics[2].update_state(student_output, teacher_output)
