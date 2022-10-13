import os
import glob
import multiprocessing
import random
import json
import gzip
import time

import tensorflow as tf
from networkx.readwrite import json_graph
from faster_fifo import Queue

import ktf.datasets
import ktf.datasets.web

from .parser import ParserHTML

from ktf.datasets.web import parse_graph_config_json


# This describes the features in the tfrecord file for this dataset.
# We use this for reading and writing tfrecord files
FEATURE_DESC = [('adj', int, [-1]),     # Placeholder
                ('feat', float, [-1]),  # Placeholder
                ('node', int, []),
                ('label', int, [])]


def parse_config(config_path):
    if config_path:
        with open(config_path, "r") as fd:
            config_json = json.loads(fd.read())
            return parse_graph_config_json(config_json)
    return parse_graph_config_json(None)


def _from_tfrecord(files, batch_size, shuffle_size, repeat, max_nodes_neighbours_feat_size,
                   expand_binary_class=False, config=None, cache=False):
    def _decode_example(examples):
        label = (tf.cast(examples["label"], tf.int32))
        label = (tf.one_hot(label, 2)) if expand_binary_class else label
        feat = tf.cast(examples["feat"], tf.float32)
        if config["eigvec_offset"]:
            feat = ktf.datasets.web.tensor_rand_flip_eigvec(
                feat, config["eigvec_offset"][0], config["eigvec_offset"][1], batch_dims=1)
        return ((tf.cast(examples["adj"], tf.int32),
                 feat,
                 tf.cast(examples["node"], tf.int32)),
                label)

    if len(files) > 0:
        feature_desc = [
            ('adj', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
            ('feat', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
            ('node', int, []),
            ('label', int, [])]

        record_io = ktf.datasets.RecordDatasetIO(feature_desc)
        return record_io.read_batch(files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=8,
                                    use_compression=True,
                                    cache=cache)
    return None


def from_tfrecord(tfrecord_dir,
                  expand_binary_class=False,
                  max_nodes_neighbours_feat_size=None,
                  validation_split=0.2,
                  batch_size=64,
                  shuffle_size=None,
                  repeat=False,
                  config_path=None,
                  cache=False):
    """Reads from tfrecord files into training and validation tf.data.Dataset objects.

    Args:
        tfrecord_dir: Path to the directory containing converted tfrecord files
        max_nodes_neighbours_feat_size: (Optional) A tuple specifying [max # nodes, max # neighbours, feature size] of the graph.
            If None, will try to infer from the name of the tfrecord directory. Default is None.
        validation_split: (Optional) Ratio of tfrecord files in directory to use for validation
        batch_size: (Optional) Batch size of dataset
        shuffle_size: (Optional) Size of shuffle buffer. If None, shuffle is not used
        repeat: (Optional) If True, dataset is repeated when end is reached
        cache: (Optional) If True, specifies that the dataset will be cached in local RAM when read. This results
                in significant performance gains if reading from storage is slow, but will require a lot of RAM
                if the dataset is large.
                If False, specifies that the dataset is not cached.
                If `cache`  is a path string, specifies that the dataset will be cached in the specified path
                when read.
                Default is False.
    """
    if config_path is None:
        config_path = os.path.join(tfrecord_dir, "config.json")

    config = parse_config(config_path)

    all_files = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    num_valid = int(len(all_files) * validation_split)
    num_train = len(all_files) - num_valid

    if max_nodes_neighbours_feat_size is None:
        if config["max_nodes_neighbours_feat_size"]:
            max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]
        else:
            # Try to decipher from tfrecord dir
            try:
                max_nodes_neighbours_feat_size = [int(v) for v in
                                                  os.path.basename(os.path.dirname(os.path.join(tfrecord_dir, ""))).split("_")[-3:]]
            except ValueError as e:
                print("Unable to deciper shape from tfrecord dir. "
                      "Please either rename tfrecord directory with `name_<# nodes>_<# neighbours>_<feature size>` "
                      "or manually input these when calling `from_tfrecord(...)`.")
                raise

    train_dataset = _from_tfrecord(all_files[num_valid:], batch_size, shuffle_size, repeat,
                                   max_nodes_neighbours_feat_size=max_nodes_neighbours_feat_size,
                                   expand_binary_class=expand_binary_class,
                                   config=config,
                                   cache=cache)
    valid_dataset = _from_tfrecord(all_files[:num_valid], batch_size,
                                   shuffle_size, repeat,
                                   max_nodes_neighbours_feat_size=max_nodes_neighbours_feat_size,
                                   expand_binary_class=expand_binary_class,
                                   config=config,
                                   cache=cache) if num_valid > 0 else None
    return (train_dataset, valid_dataset)
