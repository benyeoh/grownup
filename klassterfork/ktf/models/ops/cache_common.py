import math

import tensorflow as tf

from .compute_cache import ComputeCache


@ComputeCache.register()
def get_range_tensor_shape(tensor, axis):
    """Gets a 1D Tensor from an axis of the input tensor shape whose values are
    [0, .... , len(tensor.shape[axis]) - 1].

    Args:
        tensor: A Tensor whose shape to use
        axis: An int axis specifying the dimension of the shape to generate range values

    Returns:
        A 1D Tensor
    """
    return tf.range(0, tf.shape(tensor)[axis], 1)


@ComputeCache.register()
def get_slice(tensor, *args):
    """Returns a slice of the input tensor given a variable length number of arguments
    whose values correspond to the parameters for a slice, per axis of the tensor.

    Example:
        get_slice(tensor, None, [5, None]) == tensor[:, 5:]
        get_slice(tensor, [None, -1], [1, 5], 7) == tensor[:-1, 1:5, 7]

    Args:
        tensor: A Tensor to slice
        *args: A variable number of arguments for slicing. Each argument is either a scalar or
            a list/tuple corresponding to slicing parameters. Each argument also corresponds
            to the appropriate axis of the input tensor to slice. 

    Returns:
        A Tensor
    """
    slice_tup = tuple([slice(*ll) if isinstance(ll, (list, tuple)) else slice(ll) for ll in args])
    return tensor[slice_tup]


@ComputeCache.register()
def gather_nd(params, indices):
    """Perform a gather operation given parameters and indices.
    See: https://www.tensorflow.org/api_docs/python/tf/gather_nd

    Args:
        params: A Tensor. The tensor from which to gather values.
        indices: A Tensor. Must be one of the following types: int32, int64. Index tensor.

    Returns:
        A Tensor. Has the same type as params.
    """
    return tf.gather_nd(params, indices)
