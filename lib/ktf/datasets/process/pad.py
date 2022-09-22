import tensorflow as tf


class ImgPad:
    """Pads images in a TF Dataset using tf.pad

    Currently this class only allows for constant padding
    TODO: Allow for more complicated paddings
    """

    def __init__(self, paddings, mode="CONSTANT", pad_value=0):
        """Accepts similar arguments as to tf.pad

        Args:
            paddings: An integer or list. If an integer is of provided, this will be the amount of padding on all edges.
                If a list is provided, it should be a list of 4 elements whose values correspond to
                [pad_top, pad_bottom, pad_left, pad_right].
            constant_values: A scalar pad value to use.
        """
        if isinstance(paddings, list):
            pad_top, pad_bottom, pad_left, pad_right = paddings
            # Assumes NHWC
            self._paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

        elif isinstance(paddings, int):
            pad_top = pad_bottom = pad_left = pad_right = paddings
            # Assumes NHWC
            self._paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        self._mode = mode.upper()
        self._pad_value = pad_value

    def __call__(self, dataset):
        """Runs padding on the dataset in parallel and returns a dataset containing padded images

        Args:
            dataset: A TF Dataset object with shape: ((None, None, None, 3), (None,))
                with each element representing (batched_images, batched_labels)

        """
        return dataset.map(self._pad_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _pad_batch(self, img_batch, label_batch):
        padded_img_batch = tf.pad(img_batch, self._paddings, mode=self._mode, constant_values=self._pad_value)
        return (padded_img_batch, label_batch)
