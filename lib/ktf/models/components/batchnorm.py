import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import nn, array_ops, math_ops
import warnings


class BatchNormZeroBias(keras.layers.BatchNormalization):
    """BatchNormZeroBias layer

    A Batch Normalization layer that fixes the bias/beta parameter to zero.

    This is because the softmax loss is angular and the hypersphere is nearly symmetric
    about the origin of the coordinate axis. By removing the bias parameter,
    the features are constrained around the origin of the coordinate axis.

    See: https://arxiv.org/pdf/1903.07071.pdf
    """

    def __init__(self, *args, **kwargs):
        super(BatchNormZeroBias, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = array_ops.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = array_ops.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]     # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale = _broadcast(self.gamma)
        # Evan: Offset refers to the beta variable, or the bias of the batch norm layer. We set this to None.
        # This is the only line that's changed from keras.layers.BatchNormalization.call()
        offset = None

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = control_flow_util.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = control_flow_util.smart_cond(training,
                                                         lambda: adj_scale,
                                                         lambda: array_ops.ones_like(adj_scale))
                adj_bias = control_flow_util.smart_cond(training,
                                                        lambda: adj_bias,
                                                        lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = control_flow_util.smart_cond(training, lambda: mean,
                                                lambda: ops.convert_to_tensor_v2(moving_mean))
            variance = control_flow_util.smart_cond(
                training, lambda: variance,
                lambda: ops.convert_to_tensor_v2(moving_variance))

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                # Keras assumes that batch dimension is the first dimension for Batch
                # Normalization.
                input_batch_size = array_ops.shape(inputs)[0]
            else:
                input_batch_size = None

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training, input_batch_size)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   input_batch_size)

            def mean_update():
                def true_branch(): return _do_update(self.moving_mean, new_mean)
                def false_branch(): return self.moving_mean
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(self.moving_stddev,
                                               math_ops.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon))

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    def true_branch(): return _do_update(self.moving_variance, new_variance)

                def false_branch(): return self.moving_variance
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        # math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs
