import os
import glob
import json
import random

import numpy as np
import tensorflow as tf

import ktf.datasets
import ktf.datasets.web


# Define the format for reading/writing tfrecords
FEATURE_DESC_GRAPH = [('adj', int, [-1]),         # Placeholder
                      ('feat', float, [-1]),      # Placeholder
                      ('label_id', np.int32, [])]


def parse_config(config_path):
    if config_path:
        with open(config_path, "r") as fd:
            config_json = json.loads(fd.read())
            return ktf.datasets.web.parse_graph_config_json(config_json)
    return ktf.datasets.web.parse_graph_config_json(None)


def _setup_config(tfrecord_dir,
                  validation_split,
                  val_rand_seed,
                  config_path,
                  log_train_valid_files_path=None):

    if config_path is None:
        config_path = os.path.join(tfrecord_dir[0] if isinstance(tfrecord_dir, list) else tfrecord_dir, "config.json")

    config = parse_config(config_path)

    if not isinstance(tfrecord_dir, list):
        tfrecord_dir = [tfrecord_dir]

    all_files_by_category = {}
    for dir in tfrecord_dir:
        paths = glob.glob(os.path.join(dir, "*.tfrecord"), recursive=True)
        for path in paths:
            cat = path.split("/")[-2]
            if cat not in all_files_by_category:
                all_files_by_category[cat] = []
            all_files_by_category[cat].append(path)

    if val_rand_seed:
        random.seed(val_rand_seed)

    trained_files = []
    valid_files = []
    for _, all_files in all_files_by_category.items():
        random.shuffle(all_files)
        if validation_split > 0.0:
            num_valid = max(int(len(all_files) * validation_split), 1)
        else:
            num_valid = 0
        trained_files.extend(all_files[num_valid:])
        valid_files.extend(all_files[:num_valid])

    trained_categories = {}
    valid_categories = {}
    for path in trained_files:
        cat = path.split("/")[-2]
        if cat not in trained_categories:
            trained_categories[cat] = 0
        if cat not in valid_categories:
            valid_categories[cat] = 0
        trained_categories[cat] += 1
    print("Train categories:")
    print(trained_categories)

    valid_categories = {}
    for path in valid_files:
        cat = path.split("/")[-2]
        if cat not in trained_categories:
            trained_categories[cat] = 0
        if cat not in valid_categories:
            valid_categories[cat] = 0
        valid_categories[cat] += 1
    print("Valid categories:")
    print(valid_categories)

    print("Num train files: %d" % len(trained_files))
    print("Num valid files: %d" % len(valid_files))

    if log_train_valid_files_path:
        with open(log_train_valid_files_path, "w") as fd:
            train_valid_files_log = {"train": trained_files, "valid": valid_files}
            json.dump(train_valid_files_log, fd, indent=4)

    return config, trained_files, valid_files


def from_tfrecord_graph(tfrecord_dir,
                        config_path=None,
                        batch_size=64,
                        validation_split=0.2,
                        val_rand_seed=None,
                        log_train_valid_files_path=None,
                        expand_labels_depth=None,
                        shuffle_size=None,
                        repeat=False,
                        num_parallel_reads=tf.data.AUTOTUNE):
    """Reads from tfrecord generated with `html_to_tfrecord.py`

    Args:
        tfrecord_dir: A `str` with the directory path for tfrecord files, or a list of `str` directory paths
        config_path: (Optional) Path to html->graph configuration file exported from html_to_graph.py.
            If None, assumes it is in the same folder as `tfrecord_dir`
        batch_size: (Optional) Batch size. Default is 64
        validation_split: (Optional) Validation set split (from the number of files). Default is 0.2
        shuffle_size: (Optional) Size of the shuffle buffer. Default is None.
        repeat: (Optional) Repeat the dataset
        num_parallel_reads: (Optional) Number of parallel reads when reading from TFRecords

    Returns:
        A tuple of train_dataset, validation_dataset
    """

    config, trained_files, valid_files = _setup_config(
        tfrecord_dir, validation_split, val_rand_seed, config_path, log_train_valid_files_path)
    max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]

    def _get_decode_example(is_train):
        def _decode_example(examples):
            adj = tf.cast(examples["adj"], tf.int32)
            feat = tf.cast(examples["feat"], tf.float32)
            label_id = examples["label_id"]
            if expand_labels_depth:
                # Expand to one hot labels
                label_id = tf.one_hot(label_id, expand_labels_depth)
            if is_train and config["eigvec_offset"]:
                feat = ktf.datasets.web.tensor_rand_flip_eigvec(
                    feat, config["eigvec_offset"][0], config["eigvec_offset"][1], batch_dims=1)
            return ((adj, feat), label_id)
        return _decode_example

    feature_desc = [
        ('adj', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('label_id', np.int32, [])]

    record_io = ktf.datasets.RecordDatasetIO(feature_desc)
    train_ds = record_io.read_batch(trained_files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_get_decode_example(True),
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True)
    valid_ds = record_io.read_batch(valid_files,
                                    batch_size,
                                    shuffle_size=None,
                                    repeat=repeat,
                                    parse_fn=_get_decode_example(False),
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True) if len(valid_files) > 0 else None

    return (train_ds, valid_ds)
