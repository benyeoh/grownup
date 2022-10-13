import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class DenseL2Norm(keras.layers.Dense):
    """DenseL2Norm layer

    Basically just a typical Dense layer with operations to l2 normalize
    both the kernel and inputs before multiplication

    This is typically useful for classification training where
    we assume that each input vector in feature space are pointing in the same
    directions if they belong to the same class, and pointing away if they
    are different.

    You would typically have 'use_bias=False' for this particular use case.

    Other constructor parameters are the same as the base Dense layer.    
    """

    def __init__(self, *args, **kwargs):
        super(DenseL2Norm, self).__init__(*args, **kwargs)

    def _l2_norm(self, x, axis):
        # Rather than calling tf.nn.l2_normalize,
        # this is needed because of a Keras bug regarding namespaces when saving .h5
        # https://github.com/keras-team/keras/issues/3974
        math_ops = tf.math
        square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
        x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, 1e-12))
        return math_ops.multiply(x, x_inv_norm)

    def call(self, inputs):
        inputs = tf.cast(inputs, self._compute_dtype)
        kernel = self._l2_norm(self.kernel, axis=0)
        features = self._l2_norm(inputs, axis=-1)
        outputs = tf.matmul(features, kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
