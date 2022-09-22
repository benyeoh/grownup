import math
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class SparseCategoricalClassWeights:
    """Functor to compute the per-sample weights for different categorical classes with sparse labels
    """

    def __init__(self, class_weights, normalize_weights=True):
        """Initializes with class weights.

        Args:
            class_weights: A list or numpy array of weights for each class ordered by the (zero-based) class labels.
                Must be as large as the number of classes. Example: `[1.0, 1.0, 2.0, 0.5]`.
            normalize_weights: (Optional) If True, class weights are normalized so that `[1.0, 1.0, 2.0]` and
                `[500, 500, 1000]` give the same weights. Default is True.
        """

        self._class_weights = np.array(class_weights, dtype=np.float32)
        if normalize_weights:
            self._class_weights = (self._class_weights / np.sum(self._class_weights)) * len(self._class_weights)

    def __call__(self, y_true, y_pred, loss):
        """Computes the loss weights per sample

        Args:
            y_true: A Tensor of ground truth labels
            y_pred: A Tensor of predicted labels
            loss: A Tensor of losses per sample

        Returns:
            A Tensor of weights per sample
        """
        class_weights = tf.convert_to_tensor(self._class_weights)
        y_true = tf.convert_to_tensor(y_true)

        # We need this because somewhere along the way in the dataset API and keras.model.fit(),
        # y_true with shape [batch_size] gets reshaped into [batch_size, 1] ...
        # In order to make sure computations are correct, we need to remove that last
        # dimension before multiplication with the loss samples
        if len(y_true.shape) > len(loss.shape):
            # We expect that the y_true and loss has the same shape
            # if not, we attempt to remove the last dimension of y_true
            y_true = tf.squeeze(y_true, axis=-1)
            assert len(y_true.shape) == len(loss.shape)

        loss_weights = tf.gather(class_weights, y_true)
        return loss_weights


class CategoricalClassWeights:
    """Functor to compute the per-sample weights for different categorical classes
    """

    def __init__(self, class_weights, normalize_weights=True):
        """Initializes with class weights.

        Args:
            class_weights: A list or numpy array of weights for each class ordered by the (zero-based) class labels.
                Must be as large as the number of classes. Example: `[1.0, 1.0, 2.0, 0.5]`.
            normalize_weights: (Optional) If True, class weights are normalized so that `[1.0, 1.0, 2.0]` and
                `[500, 500, 1000]` give the same weights. Default is True.
        """

        self._class_weights = np.array(class_weights, dtype=np.float32)
        if normalize_weights:
            self._class_weights = (self._class_weights / np.sum(self._class_weights)) * len(self._class_weights)

    def __call__(self, y_true, y_pred, loss):
        """Computes the loss weights per sample

        Args:
            y_true: A Tensor of ground truth labels
            y_pred: A Tensor of predicted labels
            loss: A Tensor of losses per sample

        Returns:
            A Tensor of weights per sample
        """
        class_weights = tf.convert_to_tensor(self._class_weights)
        y_true = tf.cast(y_true, dtype=tf.float32)
        for _ in range(1, len(y_true.shape)):
            class_weights = tf.expand_dims(class_weights, axis=0)
        loss_weights = tf.reduce_sum(y_true * class_weights, axis=-1)
        return loss_weights


class BinaryClassWeights:
    """Functor to compute per-sample weights for positive and negative (binary) class
    """

    def __init__(self, class_weights, normalize_weights=True):
        """Initializes with class weights.

        Args:
            class_weights: A length 2 list or array of weights. Index 0 is the positive class weight, and
                index 1 is the negative class weight.
            normalize_weights: (Optional) If True, class weights are normalized so that `[1.0, 2.0]` and
                `[500, 1000]` give the same weights. Default is True.
        """
        assert len(class_weights) == 2
        self._class_weights = np.array(class_weights, dtype=np.float32)
        if normalize_weights:
            self._class_weights = (self._class_weights / np.sum(self._class_weights)) * len(self._class_weights)

    def __call__(self, y_true, y_pred, loss):
        """Computes the loss weights per sample

        Args:
            y_true: A Tensor of ground truth labels
            y_pred: A Tensor of predicted labels
            loss: A Tensor of losses per sample

        Returns:
            A Tensor of weights per sample
        """
        class_weights = tf.convert_to_tensor(self._class_weights)
        # We need the "squeeze" because Keras binary crossentropy always
        # averages the last dimension and removes it
        loss_weights = (tf.cast(y_true, tf.float32) * class_weights[0] +
                        (1.0 - tf.cast(y_true, tf.float32)) * class_weights[1])
        if len(loss_weights.shape) > len(loss.shape):
            loss_weights = tf.squeeze(loss_weights, axis=-1)
        return loss_weights


class ValueMatchWeights:
    """Functor to compute per-sample weights for labels matching a predefined set of values-to-weights
    """

    def __init__(self, value_spec, batch_dims=-1, normalize_weights=True):
        """Initializes with the value/weight specifications to match against when computing the loss weights.

        Args:
            value_spec: A str or dict. If str, the argument is interpreted as a path to a json file containing
                the specifications. Otherwise, the specification is a dict that looks like this:

                ```
                {
                    # Values to match with
                    "match": [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [-12345.0, -0.1112, 0.431]
                    ],

                    # Loss weights to use when matched
                    "weights": [
                        0.0,
                        1.0,
                        1.5
                    ]
                }
                ```
            For labels that do not match any of the specified values, the resulting weight is set to 1.0

            normalize_weights: (Optional) If True, weights are normalized. The default weight of 1.0
                for unmatched labels is also included in the normalization. Default is True.
        """
        if isinstance(value_spec, str):
            # Interpret as a path
            with open(value_spec, "r") as fd:
                value_spec = json.load(fd)
        self._match_values = value_spec["match"]
        self._weights = np.array(value_spec["weights"], dtype=np.float32)
        self._def_weight = 1.0
        if normalize_weights:
            self._weights = (self._weights /
                             (np.sum(self._weights) + self._def_weight) * (len(self._weights) + 1.0))
            self._def_weight = (self._def_weight /
                                (np.sum(self._weights) + self._def_weight) * (len(self._weights) + 1.0))

        self._batch_dims = batch_dims

    def __call__(self, y_true, y_pred, loss):
        """Computes the loss weights per sample.
        For floating point labels, a default epsilon of 1e-5 is used for comparisons.
        For integer labels, direct equality is used.

        Args:
            y_true: A Tensor of ground truth labels
            y_pred: A Tensor of predicted labels
            loss: A Tensor of losses per sample

        Returns:
            A Tensor of weights per sample
        """
        y_true = tf.convert_to_tensor(y_true)
        weights_so_far = tf.ones(tf.shape(y_true)[:self._batch_dims], dtype=tf.float32) * self._def_weight
        for match_value, match_weights in zip(self._match_values, self._weights):
            if y_true.dtype.is_floating:
                match_res = tf.math.abs(tf.convert_to_tensor(match_value) - y_true) < 1e-5
            else:
                match_res = tf.convert_to_tensor(match_value) == y_true
            match_res = tf.cast(tf.reduce_all(match_res, axis=-1), tf.float32)
            weights_so_far = match_res * match_weights + (1.0 - match_res) * weights_so_far
        return weights_so_far


def weighted_loss_wrap(keras_loss_class,
                       sample_weight_fn=lambda y_t, y_p, loss: tf.ones_like(loss),
                       **kwargs):
    """Top level wrapper to inject per-sample weighted losses for any keras Loss classes.

    Actual per-sample weights (and therefore also per-label/per-class weights) are computed
    from a user-defined functor.

    This is preferable to using the `sample_weights` and `class_weights` parameters in `keras.Model.fit()`
    for a few reasons.

    Firstly, this allows per-sample weights to be computed differently (or ignored) per Loss object
    in a multi-loss setting. Morever, this does not require modifications of the input dataset and
    flexibly supports per-sample/label/class or any other user-defined loss weights computation.

    Lastly, it is just cleaner to define loss weights in the context of the Loss object itself.
    For example, it does not make sense to define a `class_weights` parameter in `keras.Model.fit()`
    when the loss used is mean-squared error.

    Example usage:
        bce_w = ktf.models.losses.weighted_loss_wrap(tf.keras.losses.BinaryCrossentropy,
                                                     sample_weight_fn=ktf.models.losses.BinaryClassWeights([1.0, 1.0]))

        ...

        mse_w2 = ktf.models.losses.weighted_loss_wrap(tf.keras.losses.MeanSquaredError,
                                                      sample_weight_fn=ktf.models.losses.ValueMatchWeights(
                                                          value_spec="data/value_match.json"))

    Args:
        keras_loss_class: The Keras-compatible Loss class to wrap
        sample_weight_fn: (Optional) A functor that returns a Tensor of loss weights per sample.
            Default is a lambda that just returns ones.
        **kwargs: (Optional) Additional intialization arguments for keras_loss_class

    Returns:
        A Keras compatible Loss object with weighted loss functionality
    """

    class _LoserWrapper(keras_loss_class):
        def __init__(self,
                     name=keras_loss_class.__name__ + "_weight_wrap",
                     **kwargs):
            super(_LoserWrapper, self).__init__(name=name, **kwargs)

        def call(self, y_true, y_pred):
            loss = super(_LoserWrapper, self).call(y_true, y_pred)
            return loss * sample_weight_fn(y_true, y_pred, loss)

    return _LoserWrapper(**kwargs)


class MultiClassBinaryBalancedWeights:
    def __init__(self, class_weights):
        """Initializes with a set of positive and negative class weights balancing it using some formula
        from DeepMAR http://ir.ia.ac.cn/bitstream/173211/12681/1/multiattribute_acpr15.pdf

        Args:
            class_weights: A list corresponding to the positive ratio of each class
                eg [0.5,0.7], 50% of the dataset has class 0 marked as exist,
                70% of the dataset has class 1 marked as exist
        """
        self._pos_weights = []
        self._neg_weights = []

        class_weights = np.array(class_weights, dtype=np.float32)
        class_weights = np.array(class_weights, dtype=np.float32)

        self._pos_weights = np.exp(1.0 - class_weights)  # Lesser weightage where attribute exist
        self._neg_weights = np.exp(class_weights)  # Higher weightage where attribute exist

    def __call__(self, y_true, y_pred, loss):
        pos_weights = tf.convert_to_tensor(self._pos_weights)
        neg_weights = tf.convert_to_tensor(self._neg_weights)

        loss_weights = tf.zeros_like(y_true, dtype=tf.float32)  # Init weights as zero, same as y_pred shape
        loss_weights = tf.where(tf.equal(y_true, 1), pos_weights, loss_weights)
        loss_weights = tf.where(tf.equal(y_true, 0), neg_weights, loss_weights)
        num_classes = len(self._pos_weights)
        return loss_weights * num_classes
