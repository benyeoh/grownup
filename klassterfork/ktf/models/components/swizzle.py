import math

import tensorflow as tf
import tensorflow.keras as keras


class Swizzle(keras.layers.Layer):
    """A utility layer to swizzle input->output ordering.

    Example:
        inputs = [x, y, z, w], ordering = [3, 2, 1, 0] -> outputs = [w, z, y, x]
        inputs = [x, y, z, w], ordering = [0, 0] -> outputs = [x, x]
        inputs = [x, y, z, w], ordering = [[0, 1], [2, 3]] -> outputs = [[x, y], [z, w]]
        inputs = x, ordering = [0, 0] -> outputs = [x, x]
        inputs = [x, y], ordering = 1 -> outputs = y
        inputs = {"bboxes": w, "scores": x, "classes": y, "counts": z},
            ordering = ["bboxes", "scores"] -> outputs = [w, x]
        inputs = {"bboxes": w, "scores": x, "classes": y, "counts": z},
            ordering = [["bboxes", "scores"], ["classes", "counts"]] -> outputs = [[w, x], [y, z]]
    """

    def __init__(self, ordering, **kwargs):
        """Init function.

        Args:
            ordering: A nested list where each element is an integer index (or key) into the inputs,
                an integer scalar, or string. See above examples.
            **kwargs: (Optional) Base layer arguments.
        """
        super(Swizzle, self).__init__(**kwargs)
        self._ordering = ordering

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list, dict)):
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
                return res
            return _make_list(self._ordering)
        else:
            return inputs[self._ordering]
