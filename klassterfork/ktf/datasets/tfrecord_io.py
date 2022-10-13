import numpy as np
import tensorflow as tf

"""Mappings to convert from native python
and numpy datatypes to tf types
"""
NP_TO_TF_TYPE = {
    float: tf.float32,
    np.float32: tf.float32,
    np.float64: tf.float64,

    int: tf.int64,
    np.int: tf.int64,
    np.int32: tf.int64,
    np.int64: tf.int64,

    str: tf.string,
    bytes: tf.string,
    np.int8: tf.string,
}


"""Mappings to convert from tf types
to protocol buffer example feature functions
"""
TF_TO_PB_TYPE = {
    tf.string: lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=x)),

    tf.float32: lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x)),
    tf.float64: lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x)),

    tf.int32: lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
    tf.int64: lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
}


def get_serialize_example(feature_desc_list):
    def _shape_wrapper(fn):
        return lambda x: fn([x])

    def _str_to_bytes_wrapper(fn):
        return lambda x: fn(tf.compat.as_bytes(x))

    def _serialize_feature(x):
        x = tf.convert_to_tensor(x)
        # TFRecords only support tf.int64
        if x.dtype == tf.int32:
            x = tf.cast(x, tf.int64)
        x = tf.io.serialize_tensor(x)
        # Check if x is an EagerTensor
        if isinstance(x, type(tf.constant(0))):
            x = x.numpy()
        return TF_TO_PB_TYPE[tf.string]([x])

    feature_type = []
    for desc in feature_desc_list:
        feature_func = TF_TO_PB_TYPE[NP_TO_TF_TYPE[desc[1]]]
        if len(desc[2]) == 0:
            feature_func = _shape_wrapper(feature_func)
        if desc[1] == str:
            feature_func = _str_to_bytes_wrapper(feature_func)
        if None in desc[2]:
            feature_func = _serialize_feature
        feature_type.append(feature_func)

    def _serialize_example(*argv):
        features = {}
        for i, feat in enumerate(argv):
            features[feature_desc_list[i][0]] = feature_type[i](feat)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    return _serialize_example


class RecordWriter:
    """Simple wrapper around tf.io.TFRecordWriter to serialize
    a list of features into protobuf examples and writes into tfrecord files
    """

    def __init__(self, path, feature_desc_list, use_compression=False):
        """Accepts a destination path for the tfrecord file and
        a list of feature descriptions that specify the format of
        each element in the tfrecord

        Args:
            path: tfrecord file path
            feature_desc_list: A list of 3-tuples describing the features to write to a tfrecord
                The format of the 3-tuple (name, type, shape) is as follows:
                    name: a string describing the name of the feature
                    type: can be one of str, int, float or numpy dtypes
                    shape: a list describing the shape of the feature. Can be empty list [] if feature is a scalar.
                        Allows for the use of None to describe dimensions of varying size
            use_compression: If True, uses gzip compression on elements of tfrecord files
        """
        options = tf.io.TFRecordOptions(compression_type="GZIP") if use_compression else None
        self._writer = tf.io.TFRecordWriter(path, options=options)
        self._serialize_example_fn = get_serialize_example(feature_desc_list[:])

    def __entry__(self):
        return self

    def __exit__(self):
        self.close()

    def write(self, features):
        """Writes one row of features into a tfrecord

        Args:
            features: A list of features conforming to the
                feature_desc_list passed in the constructor
        """
        serialized = self._serialize_example_fn(*features)
        self._writer.write(serialized)

    def write_from_gen(self, generator):
        """Writes one row of features into a tfrecord

        Args:
            generator: A generator that returns a list of features conforming to the
                feature_desc_list passed in the constructor
        """
        for features in generator():
            self.write(features)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


class RecordDatasetIO:
    """Reads tfrecords and transforms elements into tf.data.Dataset objects in an efficient manner.

    Prefer to use `read_batch` as opposed to `read`, as the former can be much faster when there
    is heavy data transformation further down the pipeline

    Also supports experimental writing to a tfrecord file from tf.data.Dataset objects
    """

    def __init__(self, feature_desc_list):
        """Accepts a list of feature descriptions that specify the format of
        each element in the tfrecord

        Args:
            feature_desc_list: A list of 3-tuples describing the features to write to a tfrecord
                The format of the 3-tuple (name, type, shape) is as follows:
                    name: a string describing the name of the feature
                    type: can be one of str, int, float or numpy dtypes
                    shape: a list describing the shape of the feature. Can be empty list [] if feature is a scalar.
                        Allows for the use of None to describe dimensions of varying size. These features will be
                        returned as a RaggedTensor which can then be iterated over to retrieve the per-element tensors
        """
        self._feature_desc_list = feature_desc_list[:]

    def _write(self, path, ds_slices):
        def _tf_serialize_example(*argv):
            res = tf.py_func(get_serialize_example(self._feature_desc_list), argv, tf.string)
            return tf.reshape(res, ())

        serialized_res = ds_slices.map(_tf_serialize_example)
        writer = tf.data.experimental.TFRecordWriter(path)
        return writer.write(serialized_res)

    def _read(self, paths, num_parallel_reads=tf.data.AUTOTUNE, use_compression=False, cache=False):
        compression_type = "GZIP" if use_compression else ""
        ds = tf.data.TFRecordDataset(paths,
                                     compression_type=compression_type,
                                     num_parallel_reads=num_parallel_reads)
        if cache:
            print("\nNOTE: Using cache: %s\n" % cache)
            ds = ds.cache() if cache == True else ds.cache(cache)
        return ds

    def write_from_gen(self, path, generator):
        """Writes into tfrecord specified by `path` from a generator function

        Args:
            path: Path to the tfrecord destination file
            generator: Generator function that returns a list of features in the specified
                `feature_desc_list` passed in the constructor
        """

        gen_out_type = []
        gen_out_shape = []
        for desc in self._feature_desc_list:
            gen_out_type.append(NP_TO_TF_TYPE[desc[1]])
            gen_out_shape.append(tf.TensorShape(desc[2]))
        ds = tf.data.Dataset.from_generator(generator, tuple(gen_out_type), output_shapes=tuple(gen_out_shape))
        return self._write(path, ds)

    def write_from_tensors(self, path, tensors):
        """Writes `tensors` into tfrecord specified by `path`

        Args:
            path: Path to the tfrecord destination file
            tensors: Tensors to write to tfrecord. Tensors are sliced in the batch dimension
        """
        return self._write(path, tf.data.Dataset.from_tensor_slices(tensors))

    def read(self, paths, num_parallel_reads=tf.data.AUTOTUNE, use_compression=False, cache=False):
        """Reads elements from a list of tfrecord `paths` into a tf.data.Dataset object and parses it
        into the format specified in `feature_desc_list` passed in the constructor

        The tfrecord files are read in parallel and data may be interleaved if num_parallel_reads >= 1.

        Warning: This method is suboptimal. Prefer to use `read_batch`.

        Args:
            paths: A list of file paths to tfrecord files
            num_parallel_reads: (Optional) If >= 1, files are read in parallel and output may be interleaved
            use_compression: (Optional) If True, specifies that compression was used for the tfrecord files

        Returns:
            tf.data.Dataset
        """
        ds = self._read(paths, num_parallel_reads, use_compression, cache)

        # While generating the feature maps, we check which tensors need to be deserialized
        features = {}
        serialized_tensors = []
        for (name, dtype, shape) in self._feature_desc_list:
            dtype = NP_TO_TF_TYPE[dtype]
            if None in shape:
                serialized_tensors.append((name, dtype, shape))
                shape, dtype = tuple(), tf.string
            features[name] = tf.io.FixedLenFeature(shape, dtype)

        def _parse_example(proto):
            example = tf.io.parse_single_example(proto, features)
            # Deserialize tensors and provide shape info from self._feature_desc_list
            for (name, dtype, shape) in serialized_tensors:
                example[name] = tf.map_fn(lambda x: tf.ensure_shape(tf.io.parse_tensor(x, dtype), shape),
                                          example[name],
                                          swap_memory=False,
                                          fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0, dtype=dtype))
            return example

        return ds.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def read_batch(self,
                   paths,
                   batch_size,
                   shuffle_size=None,
                   repeat=False,
                   parse_fn=None,
                   num_parallel_reads=tf.data.AUTOTUNE,
                   use_compression=False,
                   cache=False,
                   drop_remainder=False,
                   deterministic=None):
        """Reads elements from a list of tfrecord `paths` into a tf.data.Dataset object and parses it
        into the format specified in `feature_desc_list` passed in the constructor. It does this in batch_size
        batches which increases performance

        The tfrecord files are read in parallel and data may be interleaved if num_parallel_reads >= 1.

        Args:
            paths: A list of file paths to tfrecord files
            batch_size: Size of each batch of elements
            shuffle_size: (Optional) The size of the shuffle buffer. If None, shuffle is not used
            parse_fn: (Optional) User-specified python callback to process each batch of elements
            num_parallel_reads: (Optional) If >= 1, files are read in parallel and output may be interleaved
            use_compression: (Optional) If True, specifies that compression was used for the tfrecord files
            cache: (Optional) If True, specifies that the dataset will be cached in local RAM when read. This results
                in significant performance gains if reading from storage is slow, but will require a lot of RAM
                if the dataset is large.
                If False, specifies that the dataset is not cached.
                If `cache`  is a path string, specifies that the dataset will be cached in the specified path
                when read.
                Default is False.
            drop_remainder: (Optional) If True, the final batch of the dataset will be dropped if it has fewer than
                `batch_size` elements.
            deterministic: (Optional) Controls whether the output of `map` (ie, data transform) has a 
                deterministic ordering or not. Default is None, which assumes that the ordering is deterministic
                if `shuttle_size` is also None, otherwise will interpreted as False.
        Returns:
            tf.data.Dataset
        """
        ds = self._read(paths, num_parallel_reads, use_compression, cache)
        if shuffle_size is not None:
            ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        # While generating the feature maps, we check which tensors need to be deserialized
        features = {}
        serialized_tensors = []
        for (name, dtype, shape) in self._feature_desc_list:
            dtype = NP_TO_TF_TYPE[dtype]
            if None in shape:
                serialized_tensors.append((name, dtype, shape))
                shape, dtype = tuple(), tf.string
            features[name] = tf.io.FixedLenFeature(shape, dtype)

        def _parse_examples(proto):
            examples = tf.io.parse_example(proto, features)
            # Deserialize tensors and provide shape info from self._feature_desc_list
            for (name, dtype, shape) in serialized_tensors:
                examples[name] = tf.map_fn(lambda x: tf.ensure_shape(tf.io.parse_tensor(x, dtype), shape),
                                           examples[name],
                                           swap_memory=False,
                                           fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0, dtype=dtype))

            if parse_fn is not None:
                return parse_fn(examples)
            else:
                return examples

        if deterministic is None:
            if shuffle_size is not None:
                # We assume that if the user wants shuffled data,
                # he must not care very much about deterministic ordering
                # In fact, the more random the better
                deterministic = False
            else:
                # Otherwise, I suppose we default to True
                deterministic = True

        ds = ds.map(_parse_examples, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)
        return ds.prefetch(tf.data.AUTOTUNE)
