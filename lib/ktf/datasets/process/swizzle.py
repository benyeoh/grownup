import tensorflow as tf


class Swizzle:
    """ A utility function to swizzle input->output ordering for dataset.
        Overloaded class for dataset processing from Swizzle Layer.

    Example:

        inputs = [x, y, z, w], ordering = [3, 2, 1, 0] -> outputs = [w, z, y, x]
        inputs = [x, y, z, w], ordering = [0, 0] -> outputs = [x, x]
        inputs = [x, y, z, w], ordering = [[0, 1], [2, 3]] -> outputs = [[x, y], [z, w]]
        inputs = x, ordering = [0, 0] -> outputs = [x, x]
        inputs = [x, y], ordering = 1 -> outputs = y
    """

    def __init__(self, ordering, is_deterministic=False):
        """Init function.

        Args:
            ordering: A nested list where each element is an integer index into the inputs,
                or an integer scalar. See above example.
            is_deterministic: bool, whether should preserver input order.
            **kwargs: (Optional) Base layer arguments.
        """
        super(Swizzle, self).__init__()
        self._ordering = ordering
        self._is_deterministic = is_deterministic

    def __call__(self, ds):
        def _map_fn(*inputs):
            if not isinstance(inputs, (list, tuple)):
                # Make it a list for indexing later
                inputs = [inputs]

            if isinstance(self._ordering, (tuple, list)):
                def _make_list(order):
                    res = []
                    for index in order:
                        if isinstance(index, list):
                            res.append(_make_list(index))
                        else:
                            res.append(inputs[index])
                    return tuple(res)
                return _make_list(self._ordering)
            else:
                return inputs[self._ordering]
        return ds.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=self._is_deterministic)
