import math

import numpy as np
import tensorflow as tf

from .weighted_loss import weighted_loss_wrap, BinaryClassWeights, MultiClassBinaryBalancedWeights


class BinaryCrossentropy(tf.keras.losses.Loss):
    """This BinaryCrossentropy class differs from the one in TF Keras in only one place.
    See the `call` fn below for details.    
    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 name='bce',
                 **kwargs):
        """Initializes with class weights.

        Args:
            from_logits: (Optional) Whether to interpret y_pred as a tensor of logit values.
                By default, we assume that y_pred contains probabilities (i.e., values in [0, 1]).
                **Note - Using from_logits=True may be more numerically stable.
            label_smoothing: (Optional) Float in [0, 1]. When 0, no smoothing occurs. When > 0, we compute the loss
                between the predicted labels and a smoothed version of the true labels, where the smoothing
                squeezes the labels towards 0.5. Larger values of label_smoothing correspond to heavier smoothing.
            name: (Optional) Name of the Loss object
            **kwargs: (Optional) Additional arguments for the Loss object
        """
        super(BinaryCrossentropy, self).__init__(name=name, **kwargs)
        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """Computes the binary crossentropy loss. (Stolen from Keras)

        The only difference between this and tf.keras.losses.BinaryCrossentropy
        is that this class does NOT do a mean reduction in the last dimension when it returns
        the result.

        So you would want to use this class if you're using ktf.models.losses.weighted_loss_wrap
        or whenever you require raw unreduced loss values per sample.

        Args:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
            from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
              we assume that `y_pred` encodes a probability distribution.
            label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
        Returns:
            Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self._label_smoothing, dtype=tf.keras.backend.floatx())

        def _smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = tf.cond(label_smoothing > 0, _smooth_labels, lambda: y_true)
        losses = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self._from_logits)
        return losses


# Convenience alias for weighted BinaryCrossentropy (KTF version)
BinaryCrossentropyW = lambda class_weights, **kwargs: weighted_loss_wrap(BinaryCrossentropy,
                                                                         sample_weight_fn=BinaryClassWeights(
                                                                             class_weights),
                                                                         **kwargs)
MultiClassBinaryCrossentropyBW = (lambda class_weights, **kwargs:
                                  weighted_loss_wrap(BinaryCrossentropy,
                                                     sample_weight_fn=MultiClassBinaryBalancedWeights(class_weights),
                                                     **kwargs))
