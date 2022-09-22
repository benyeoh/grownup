import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class ArcLoss(keras.losses.Loss):
    """ArcLoss loss function based on this paper: https://arxiv.org/abs/1801.07698

    It defines an arcloss margin that encourages greater separation of inter-class feature variations
    and tighter groupings of intra-class variations. This margin is applied in angular space of the features.

    This context is only appropriate if we interpret the head Dense layer weights as a matrix of 
    mean identity feature vectors, therefore if we L2 normalize each row of the matrix and the input feature vector,
    the output of the Dense layer would be the angle between each identity feature vector with the input.

    In this implementation, we also add a cosloss based on this paper: https://arxiv.org/abs/1801.09414

    Important note: 
        If training with ArcLoss, it is recommended to use a multi-stage training process where:
            1. Model looks something like: Conv layers->[Optional Pooling]->BatchNorm->Dense->BatchNorm
                - Note the "BatchNorm->Dense->BatchNorm" part. The output is the high-level features
            2. We use a model with a DenseL2Norm head where the weights and input feature vectors are L2 normalized.
            3. Training is first done with `arcloss_margin` and `cosloss_margin` set to 0
            4. Once model has converged, we then set the `arcloss_margin` and `cosloss_margin` to a higher number and retrain
    """

    def __init__(self,
                 scale=30.0,
                 arcloss_margin=0.5,
                 cosloss_margin=0.0,
                 **kwargs):
        """Initialization for arcloss.

        If using custom training steps, it is important to set the kwarg `reduction` to `tf.keras.losses.Reduction.NONE`

        Args:
            scale: (Optional) The scale to apply to the feature vector logits after applying the margin
            arcloss_margin: (Optional) Arcloss margin
            cosloss_margin: (Optional) Cosine loss margin
            **kwargs: (Optional) tf.keras.losses.Loss kwargs
        """
        super(ArcLoss, self).__init__(**kwargs)
        self._scale = scale
        self._arcloss_margin = arcloss_margin
        self._cosloss_margin = cosloss_margin

    def _margin_loss(self, y_true, y_pred):
        # Explicitly set shape of tensor to provide additional static information
        y_true.set_shape([None, 1])

        # Get element of y_pred that correspond to the target class for each batch
        # We assume y_pred is a normalized matrix/vector product
        cos_theta = tf.gather_nd(y_pred, tf.cast(y_true, tf.int32), batch_dims=1)
        sin_theta = tf.sqrt(1.0 - (cos_theta * cos_theta))

        cos_margin = math.cos(self._arcloss_margin)
        sin_margin = math.sin(self._arcloss_margin)

        # This cosine(threshold_angle) threshold is the point where we will overflow if we apply the arcloss margin
        threshold_cos = math.cos(math.pi - self._arcloss_margin)

        # This is just to check if we have crossed the threshold
        threshold_cond = cos_theta - threshold_cos

        # cos(theta + margin)
        cos_theta_arcloss_margin = cos_theta * cos_margin - sin_theta * sin_margin

        # In the case of an overflow the gradients might swing in the opposite direction because the loss actually
        # decreases in that case (since angles wrap around). We prevent that by applying a fixed overflow factor if
        # it crosses the threshold. There is a discontinuity in the resulting loss function but the wraparound
        # of angles is prevented and the gradient still points in the correct direction
        arcface_overflow_factor = math.sin(math.pi - self._arcloss_margin) * self._arcloss_margin
        cos_theta_arcloss_margin_overflow = cos_theta - arcface_overflow_factor

        # Use the workaround in the event where we detect an overflow and also apply cosine loss
        cos_theta_margin = tf.compat.v2.where(threshold_cond > 0.0,
                                              cos_theta_arcloss_margin,
                                              cos_theta_arcloss_margin_overflow) - self._cosloss_margin

        y_true_one_hot = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32),
                                    y_pred.shape[-1],
                                    on_value=1.0,
                                    off_value=0.0)

        # Finally modify the logit for the actual class with the margin loss and scale it (for softmax)
        diff_margin_one_hot = tf.expand_dims(cos_theta_margin - cos_theta, axis=1) * y_true_one_hot
        logits = (y_pred + diff_margin_one_hot) * self._scale

        return keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True, axis=-1)

    def call(self, y_true, y_pred):
        return self._margin_loss(y_true, y_pred)


class LinearArcLoss(keras.losses.Loss):
    """Li-ArcFace loss based on this paper: https://arxiv.org/abs/1907.12256

    This is a "linear" version of ArcFace loss as described above. Instead of adding the margins to the logits and
    interpreting it in cos(theta) space, the logits are remapped to linear space.

    Important note: 
        If training with ArcLoss, it is recommended to use a multi-stage training process where:
            1. Model looks something like: Conv layers->[Optional Pooling]->BatchNorm->Dense->BatchNorm
                - Note the "BatchNorm->Dense->BatchNorm" part. The output is the high-level features
            2. We use a model with a DenseL2Norm head where the weights and input feature vectors are L2 normalized.
            3. Training is first done with `arcloss_margin` and `cosloss_margin` set to 0
            4. Once model has converged, we then set the `arcloss_margin` and `cosloss_margin` to a higher number and retrain
    """

    def __init__(self,
                 scale=30.0,
                 arcloss_margin=0.5,
                 **kwargs):
        """Initialization for linear arcloss.

        If using custom training steps, it is important to set the kwarg `reduction` to `tf.keras.losses.Reduction.NONE`

        Args:
            scale: (Optional) The scale to apply to the feature vector logits after applying the margin
            arcloss_margin: (Optional) Arcloss margin
            cosloss_margin: (Optional) Cosine loss margin
            **kwargs: (Optional) tf.keras.losses.Loss kwargs
        """
        super(LinearArcLoss, self).__init__(**kwargs)
        self._scale = scale
        self._arcloss_margin = arcloss_margin

    def _arcloss_loss_linear(self, y_true, y_pred):
        y_true.set_shape([None, 1])

        # Get the angles and add margin and remap it to [-1, 1] linearly
        linear_pred = (math.pi - 2.0 * tf.acos(y_pred)) / math.pi
        linear_margin = -2.0 * self._arcloss_margin / math.pi

        y_true_one_hot = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32),
                                    y_pred.shape[-1],
                                    on_value=tf.cast(linear_margin, tf.float32),
                                    off_value=tf.cast(0.0, tf.float32))

        # Finally modify the logit for the actual class with the margin loss and scale it (for softmax)
        diff_margin_one_hot = y_true_one_hot
        logits = (linear_pred + diff_margin_one_hot) * self._scale

        return keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True, axis=-1)

    def call(self, y_true, y_pred):
        return self._arcloss_loss_linear(y_true, y_pred)
