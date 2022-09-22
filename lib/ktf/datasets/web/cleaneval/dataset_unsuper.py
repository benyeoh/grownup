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
from .dataset import parse_config


def _from_tfrecord_unsupervised(files,
                                batch_size,
                                shuffle_size,
                                repeat,
                                prob_mask,
                                max_nodes_neighbours_feat_size,
                                raw_config,
                                tag_feature_list,
                                parsed_config=None):

    tag_feature_offset = []
    tag_feature_size = []
    for full_key in tag_feature_list:
        feature_desc = raw_config["feature_desc"]["feature_offsets"]
        for key in full_key.split("."):
            feature_desc = feature_desc[key]
        tag_feature_offset.append(feature_desc["idx"])
        tag_feature_size.append(feature_desc["len"])

    def _decode_example(examples):
        #label = (tf.cast(examples["label"], tf.int32))
        #label = (tf.one_hot(label, 2)) if expand_binary_class else label
        feat = tf.cast(examples["feat"], tf.float32)
        node = tf.cast(examples["node"], tf.int32)
        if parsed_config["eigvec_offset"]:
            feat = ktf.datasets.web.tensor_rand_flip_eigvec(
                feat, parsed_config["eigvec_offset"][0], parsed_config["eigvec_offset"][1], batch_dims=1)

        src_feats = ktf.datasets.web.tensor_extract_feats(feat, node, tag_feature_offset, tag_feature_size)
        zeroed_feat = ktf.datasets.web.tensor_rand_mask_feat(feat, node, prob_mask=prob_mask)
        return ((tf.cast(examples["adj"], tf.int32),
                 zeroed_feat,
                 node),
                tuple(src_feats))

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
                                    use_compression=True)
    return None


def from_tfrecord_unsupervised(tfrecord_dir,
                               tag_feature_list,
                               prob_mask=0.85,
                               config_path=None,
                               validation_split=0.2,
                               batch_size=64,
                               shuffle_size=None,
                               repeat=False):
    """Reads from tfrecord files into training and validation tf.data.Dataset objects.

    This transforms the raw input (graph) tensors for basic unsupervised learning by
    extracting features from target nodes to use as labels.

    Args:
        tfrecord_dir: Path to the directory containing converted tfrecord files
        tag_feature_list: A list of feature names (corresponding to the config file) to extract for unsupervised
            learning.
        config_path: (Optional) Path to the configuration file generated during tfrecord creation.
            If None, will default to the config.json in the tfrecord directory.
        prob_mask: (Optional) The probability that a target node will be zero'ed (vs left as-is).
        validation_split: (Optional) Ratio of tfrecord files in directory to use for validation
        batch_size: (Optional) Batch size of dataset
        shuffle_size: (Optional) Size of shuffle buffer. If None, shuffle is not used
        repeat: (Optional) If True, dataset is repeated when end is reached
    """
    config_json = None
    if config_path is None:
        config_path = os.path.join(tfrecord_dir, "config.json")

    with open(config_path, "r") as fd:
        config_json = json.loads(fd.read())

    config = parse_config(config_path)

    all_files = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    num_valid = int(len(all_files) * validation_split)
    num_train = len(all_files) - num_valid

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

    train_dataset = _from_tfrecord_unsupervised(all_files[num_valid:], batch_size, shuffle_size, repeat,
                                                prob_mask=prob_mask,
                                                max_nodes_neighbours_feat_size=max_nodes_neighbours_feat_size,
                                                raw_config=config_json,
                                                tag_feature_list=tag_feature_list,
                                                parsed_config=config)

    valid_dataset = _from_tfrecord_unsupervised(all_files[:num_valid], batch_size, shuffle_size, repeat,
                                                prob_mask=prob_mask,
                                                max_nodes_neighbours_feat_size=max_nodes_neighbours_feat_size,
                                                raw_config=config_json,
                                                tag_feature_list=tag_feature_list,
                                                parsed_config=config)
    return (train_dataset, valid_dataset)
