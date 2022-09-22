import tensorflow as tf


class Batch:
    """Batches the input dataset
    """

    def __init__(self, batch_size):
        """Initialization for this functor

        Args:
            batch_size: Size of each batch in the dataset
        """

        self._batch_size = batch_size

    def __call__(self, ds):
        return ds.batch(self._batch_size)


class Unbatch:
    """Unbatches the input dataset
    """

    def __init__(self):
        pass

    def __call__(self, ds):
        return ds.unbatch()
