import tensorflow as tf


class MapInput:
    """Preprocess image
    """

    def __init__(self, map_fn, input_indices=[0], unwrap_batch=False, is_deterministic=False, **kwargs):
        """Initialization of the functor.

        Args:
            map_fn: String of function used to preprocess image
            kwargs: Variable arguments for the user-defined map_fn
            input_indices: (Optional) A list of indices to elements in the dataset. Defaults to the 1st element
            unwrap_batch: (Optional) Whether to unbatch input datasets. In the case where the user defined map_fn is
                unable to process batched inputs, set this to True. Defaults to False.
            is_deterministic: (Optional) If True, the output elements in the dataset is in the same ordering as the
                input. If False, the ordering is arbitrary but has better performance. Defaults to False.
        """
        self._map_fn = map_fn
        self._kwargs = kwargs
        self._indices = input_indices
        self._unwrap_batch = unwrap_batch
        self._is_deterministic = is_deterministic

    def _convert_single_input(self, input):
        return eval(self._map_fn)(input, **self._kwargs)

    def _convert_no_batch(self, ds):

        def _per_elem_convert(*args):
            res_args = list(args)
            for idx in self._indices:
                # Convert each element in the dataset
                res_args[idx] = self._convert_single_input(args[idx])
            return tuple(res_args)

        return ds.map(_per_elem_convert,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=self._is_deterministic)

    def _convert(self, ds):
        def _per_elem_convert(args):
            return [self._convert_single_input(arg) for arg in args]

        def _ds_map_fn(*args):
            res_args = list(args)

            # Extract relevant elements in the dataset
            imgs = [args[idx] for idx in self._indices]

            # Convert per batch
            result = tf.map_fn(_per_elem_convert, imgs, parallel_iterations=64)

            # Get resulting elements
            for i, idx in enumerate(self._indices):
                res_args[idx] = result[i]
            return tuple(res_args)

        return ds.map(_ds_map_fn,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=self._is_deterministic)

    def __call__(self, ds):
        return self._convert(ds) if self._unwrap_batch else self._convert_no_batch(ds)
