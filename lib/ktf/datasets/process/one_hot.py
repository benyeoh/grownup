import tensorflow as tf


class OneHot:
    """Converts sparse labels to one hot"""

    def __init__(self, num_classes):
        self._num_classes = num_classes
        pass

    def __call__(self, dataset):
        """Runs augmentation on dataset in parallel and returns augmented dataset

        Args:
            dataset: A TF Dataset object with shape: ((None, None, None, 3), (None,)) and type: (tf.float32, tf.int32)
                     with each element representing (batched_images, batched_labels)

        """
        return dataset.map(self._apply_one_hot, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _apply_one_hot(self, img_batch, label_batch):
        one_hot_labels = tf.one_hot(label_batch, self._num_classes,
                                    on_value=1, off_value=0)
        return (img_batch, one_hot_labels)
