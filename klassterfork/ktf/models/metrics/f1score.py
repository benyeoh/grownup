import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
import numpy as np


class F1Score(tf.keras.metrics.Metric):
    """A proper (micro) F1 score keras compatible metric as compared
    to tfa.metrics.F1Score which computes the score per batch (!), as
    opposed to computing the score per epoch

    Can be used as drop-in replacement if required in the model.fit() function
    or training scripts or can be used standalone.
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        """Initializer for the metric.

        Args:
            thresholds: (Optional) A scalar or list of threshold values for determining
                a positive prediction. If multiple thresholds are provided, the computation
                will be repeated for every threshold. Default is None (ie, 0.5)
            top_k: (Optional) Number of top matches for positive label. Default is None (ie, 1)
            class_id: (Optional) The specific class ID to consider for matches, while ignoring others.
                Default is None
            name: (Optional) Name of the metric. Default is None
            dtype: (Optional) Type of the metric. Default is None
        """

        super(F1Score, self).__init__(name=name, dtype=dtype)
        self._init_thresholds = thresholds
        self._top_k = top_k
        self._class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self._thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self._true_positives = self.add_weight(
            'true_positives',
            shape=(len(self._thresholds),),
            initializer=tf.keras.initializers.Zeros())
        self._false_positives = self.add_weight(
            'false_positives',
            shape=(len(self._thresholds),),
            initializer=tf.keras.initializers.Zeros())
        self._false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self._thresholds),),
            initializer=tf.keras.initializers.Zeros())

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false positive statistics.
        Args:
            y_true: The ground truth values, with the same dimensions as `y_pred`.
                Will be cast to `bool`.
            y_pred: The predicted values. Each element must be in the range `[0, 1]`.
                sample_weight: Optional weighting of each example. Defaults to 1. Can be a
                `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
                be broadcastable to `y_true`.
        Returns:
            Update op.
        """
        if (y_true.shape[-1] != y_pred.shape[-1]) or (len(y_true.shape) != len(y_pred.shape)):
            # Assume it's sparse, so convert to one hot
            y_true = tf.squeeze(tf.one_hot(y_true, y_pred.shape[-1]), axis=[-2])

        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self._true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self._false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self._false_negatives,
            },
            y_true,
            y_pred,
            thresholds=self._thresholds,
            top_k=self._top_k,
            class_id=self._class_id,
            sample_weight=sample_weight)

    def result(self):
        result = math_ops.div_no_nan(self._true_positives,
                                     self._true_positives + self._false_positives)
        precision = result[0] if len(self._thresholds) == 1 else result
        result = math_ops.div_no_nan(self._true_positives,
                                     self._true_positives + self._false_negatives)
        recall = result[0] if len(self._thresholds) == 1 else result
        return 2.0 * math_ops.div_no_nan(precision * recall, precision + recall)

    def reset_state(self):
        num_thresholds = len(self._thresholds) if isinstance(self._thresholds, list) else 1
        tf.keras.backend.batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self._init_thresholds,
            'top_k': self._top_k,
            'class_id': self._class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
