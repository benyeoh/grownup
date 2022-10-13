import os
import glob
import multiprocessing
import random
import pickle
import gzip
import time
import json
import traceback
import sys

import tensorflow as tf
from networkx.readwrite import json_graph
from faster_fifo import Queue

import ktf.datasets
import ktf.datasets.web


# We use this for reading and writing tfrecord files
FEATURE_DESC_GRAPH = [('adj', int, [-1]),         # Placeholder
                      ('feat', float, [-1]),      # Placeholder
                      ('node_list', int, [-1])]


def parse_graph_config_json(config_json):
    """Parse the exported graph config file (from `html_to_graph.py`) and transform it to a format
    used in unsupervised learning.

    Args:
        config_json: The dictionary derived from the config json file exported by `html_to_graph.py`

    Returns:
        A dictionary with parameters used for unsupervised learning
    """
    config = {
        "eigvec_offset": None,
        "max_nodes_neighbours_feat_size": None
    }
    try:
        eigvec_offset = (config_json["feature_desc"]["feature_offsets"]["tag_graph_eigen"]["idx"],
                         config_json["feature_desc"]["feature_offsets"]["tag_graph_eigen"]["len"])
        print("Eigenvectors found. Position: %d, Len: %d" % (eigvec_offset[0], eigvec_offset[1]))
        config["eigvec_offset"] = eigvec_offset
    except (TypeError, KeyError):
        print("No eigenvectors found.")

    try:
        graph_params = config_json["graph_params"]
        feature_size = config_json["feature_desc"]["feature_size"]
        print("Max nodes, neighbours, feat_size found. %d, %d, %d" %
              (graph_params[1], graph_params[2], feature_size))
        config["max_nodes_neighbours_feat_size"] = (graph_params[1], graph_params[2], feature_size)
    except (TypeError, KeyError):
        print("No max nodes, neighbours, feat_size found.")
    print()
    return config


def from_tfrecord_graph_unsupervised(tfrecord_dir,
                                     tag_feature_list,
                                     config_path=None,
                                     prob_mask=0.85,
                                     max_num_nodes=16,
                                     use_node_list=False,
                                     batch_size=64,
                                     validation_split=0.2,
                                     shuffle_size=None,
                                     repeat=False):
    """Reads from tfrecord generated with `html_to_graph.py` and applies
    transformations applicable to unsupervised learning

    Args:
        tfrecord_dir: A `str` with the directory path for tfrecord files, or a list of `str` directory paths
        tag_feature_list: A list of tag feature names to mask and extract labels for unsupervised learning. The
            tag feature names can be referenced from `TagFeatures.get_feature_size_and_offsets()["feature_offsets"]`
            Example, `["class_attr", "text.embedding"]`
        config_path: (Optional) Path to html->graph configuration file exported from html_to_graph.py.
            If None, assumes it is in the same folder as `tfrecord_dir`
        prob_mask: (Optional) Probability that nodes to predict will be masked out. Default is 0.85
        max_num_nodes: (Optional) The maximum number of nodes to predict per sample. Default is 16. Higher is more
            efficient but has a similar effect of increasing batch size and possibly affecting results
        use_node_list: (Optional) Do node prediction from a pre-defined list of nodes from the tfrecord. If False,
            will predict from any node in the graph.
        batch_size: (Optional) Batch size. Default is 64
        validation_split: (Optional) Validation set split (from the number of files). Default is 0.2
        shuffle_size: (Optional) Size of the shuffle buffer. Default is None.
        repeat: (Optional) Repeat the dataset

    Returns:
        A tuple of `(train_dataset, validation_dataset)`
    """
    if config_path is None:
        config_path = os.path.join(tfrecord_dir[0] if isinstance(tfrecord_dir, list) else tfrecord_dir, "config.json")

    if not isinstance(tfrecord_dir, list):
        tfrecord_dir = [tfrecord_dir]

    with open(config_path, "r") as fd:
        config_json = json.loads(fd.read())
        config = parse_graph_config_json(config_json)

    all_files = []
    for dir in tfrecord_dir:
        all_files.extend(glob.glob(os.path.join(dir, "*.tfrecord")))

    num_valid = int(len(all_files) * validation_split)
    num_train = len(all_files) - num_valid

    max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]

    tag_feature_offset = []
    tag_feature_size = []
    for full_key in tag_feature_list:
        feature_desc = config_json["feature_desc"]["feature_offsets"]
        for key in full_key.split("."):
            feature_desc = feature_desc[key]
        tag_feature_offset.append(feature_desc["idx"])
        tag_feature_size.append(feature_desc["len"])

    def _decode_example(examples):
        adj = tf.cast(examples["adj"], tf.int32)
        feat = tf.cast(examples["feat"], tf.float32)
        node_list = tf.cast(examples["node_list"], tf.int32)
        if config["eigvec_offset"]:
            feat = ktf.datasets.web.tensor_rand_flip_eigvec(feat,
                                                            config["eigvec_offset"][0],
                                                            config["eigvec_offset"][1],
                                                            batch_dims=1)
        if use_node_list:
            _, nodes_per_batch = ktf.datasets.web.tensor_rand_sel_nodes_from_list(
                node_list, max_num_nodes=max_num_nodes)
        else:
            nodes_per_batch = ktf.datasets.web.tensor_rand_sel_nodes(adj, max_num_nodes=max_num_nodes)
        src_feats = ktf.datasets.web.tensor_extract_feats(feat, nodes_per_batch, tag_feature_offset, tag_feature_size)
        zeroed_feat = ktf.datasets.web.tensor_rand_mask_feat(feat, nodes_per_batch, prob_mask=prob_mask)
        return ((adj, zeroed_feat, nodes_per_batch), tuple(src_feats))

    feature_desc = [
        ('adj', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('node_list', int, [192])]

    record_io = ktf.datasets.RecordDatasetIO(feature_desc)
    train_ds = record_io.read_batch(all_files[num_valid:],
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=16,
                                    use_compression=True)
    valid_ds = record_io.read_batch(all_files[:num_valid],
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=16,
                                    use_compression=True)
    return train_ds, valid_ds


def from_pkl_graph_unsupervised(pkl_dir,
                                tag_feature_list,
                                config_path=None,
                                num_iters=None,
                                prob_mask=0.85,
                                max_num_nodes=16,
                                use_node_list=False,
                                batch_size=64,
                                validation_split=0.2,
                                shuffle_size=None,
                                max_cache_pkls=64,
                                num_parallel_calls=8):
    """Reads from tfrecord generated with `html_to_graph.py` and applies
    transformations applicable to unsupervised learning

    Args:
        pkl_dir: A `str` with the directory path for tfrecord files, or a list of `str` directory paths
        tag_feature_list: A list of tag feature names to mask and extract labels for unsupervised learning. The
            tag feature names can be referenced from `TagFeatures.get_feature_size_and_offsets()["feature_offsets"]`
            Example, `["class_attr", "text.embedding"]`
        config_path: (Optional) Path to html->graph configuration file exported from html_to_graph.py.
            If None, assumes it is in the same folder as `tfrecord_dir`
        num_iters: (Optional) Number of iteration samples to generate
        prob_mask: (Optional) Probability that nodes to predict will be masked out. Default is 0.85
        max_num_nodes: (Optional) The maximum number of nodes to predict per sample. Default is 16. Higher is more
            efficient but has a similar effect of increasing batch size and possibly affecting results
        use_node_list: (Optional) Do node prediction from a pre-defined list of nodes from the tfrecord. If False,
            will predict from any node in the graph.
        batch_size: (Optional) Batch size. Default is 64
        validation_split: (Optional) Validation set split (from the number of files). Default is 0.2
        shuffle_size: (Optional) Size of the shuffle buffer. Default is None.
        max_cache_pkls: (Optional) Number of pkls in the cache at any time. Higher helps to buffer against spikes
            but uses more memory. Default is 64.
        num_parallel_calls: (Optional) Number of pkls read and processed in parallel. Higher is faster (slightly),
            but uses significantly more memory. Default is 8.

    Returns:
        A tuple of `(train_dataset, validation_dataset)`
    """
    if config_path is None:
        config_path = os.path.join(pkl_dir[0] if isinstance(pkl_dir, list) else pkl_dir, "config.json")

    with open(config_path, "r") as fd:
        config_json = json.loads(fd.read())
        config = parse_graph_config_json(config_json)

    def _load_pkl(file):
        with gzip.open(file, "rb") as fd:
            adj, feats, nodes = pickle.load(fd)
            return adj, feats, nodes

    def _producer_pkl(files, queue, seed):
        rand = random.Random(seed)
        max_batch = 32
        max_num_pkls = max_cache_pkls
        random.shuffle(files)

        pkls = []
        for _ in range(max_num_pkls):
            file = files.pop(0)
            files.append(file)
            pkls.append(_load_pkl(file))

        while True:
            data = []
            for i in range(max_batch):
                if rand.random() <= 1.0 / max_batch:
                    file = files.pop(0)
                    files.append(file)
                    pkls.pop(0)
                    pkls.append(_load_pkl(file))

                adj, feats, nodes = rand.choice(pkls)
                data.append((adj, feats, nodes))
            try:
                queue.put(data, block=True, timeout=3600)
            except Exception as e:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)

    all_files = []
    if isinstance(pkl_dir, list):
        for dir in pkl_dir:
            all_files.extend(glob.glob(os.path.join(dir, "*.pkl.gz")))
    else:
        all_files.extend(glob.glob(os.path.join(pkl_dir, "*.pkl.gz")))

    num_valid_files = int(len(all_files) * validation_split)
    valid_files = all_files[:num_valid_files]
    train_files = all_files[num_valid_files:]

    samp_adj, samp_feats, samp_nodes = _load_pkl(all_files[0])
    train_queue = Queue(1024 * 1024 * 1024 * 4)
    valid_queue = None

    if num_valid_files > 0:
        valid_queue = Queue(1024 * 1024 * 1024)

    for i in range(num_parallel_calls):
        multiprocessing.Process(target=_producer_pkl,
                                args=(train_files,
                                      train_queue,
                                      int(time.time() * 100000 + i * 2)),
                                daemon=True).start()

        if valid_queue:
            multiprocessing.Process(target=_producer_pkl,
                                    args=(valid_files,
                                          valid_queue,
                                          int(time.time() * 100000 + i * 2 + 1)),
                                    daemon=True).start()

    tag_feature_offset = []
    tag_feature_size = []
    for full_key in tag_feature_list:
        feature_desc = config_json["feature_desc"]["feature_offsets"]
        for key in full_key.split("."):
            feature_desc = feature_desc[key]
        tag_feature_offset.append(feature_desc["idx"])
        tag_feature_size.append(feature_desc["len"])

    max_num_consumers = 2

    def _wrapper_consumer(queue):
        def _consumer():
            count = 0
            while True:
                try:
                    batch_data_many = queue.get_many(block=True, timeout=3600)
                    for batch_data in batch_data_many:
                        for data in batch_data:
                            yield data
                            count += 1
                            if num_iters is not None and count >= (num_iters * batch_size) / max_num_consumers:
                                return
                except Exception as e:
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)  # Fixed format
                    raise

        def _ds_gen(id):
            return tf.data.Dataset.from_generator(_consumer,
                                                  output_types=(tf.int32, tf.float32, tf.int32),
                                                  output_shapes=(samp_adj.shape, samp_feats.shape, samp_nodes.shape))
        return _ds_gen

    train_ds = tf.data.Dataset.range(max_num_consumers)
    train_ds = train_ds.interleave(_wrapper_consumer(train_queue),
                                   cycle_length=tf.data.experimental.AUTOTUNE,
                                   deterministic=False)

    if shuffle_size is not None:
        train_ds = train_ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)

    def _transform_data(adj, feat, nodes):
        if config["eigvec_offset"]:
            feat = ktf.datasets.web.tensor_rand_flip_eigvec(feat,
                                                            config["eigvec_offset"][0],
                                                            config["eigvec_offset"][1],
                                                            batch_dims=1)
        # Do stuff here to mask and so on
        if use_node_list:
            nodes_list = nodes
            _, nodes_per_batch = ktf.datasets.web.tensor_rand_sel_nodes_from_list(
                nodes_list, max_num_nodes=max_num_nodes)
        else:
            nodes_per_batch = ktf.datasets.web.tensor_rand_sel_nodes(adj, max_num_nodes=max_num_nodes)
        src_feats = ktf.datasets.web.tensor_extract_feats(feat, nodes_per_batch, tag_feature_offset, tag_feature_size)
        zeroed_feat = ktf.datasets.web.tensor_rand_mask_feat(feat, nodes_per_batch, prob_mask=prob_mask)
        return ((adj, zeroed_feat, nodes_per_batch), tuple(src_feats))

    train_ds = train_ds.map(_transform_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    if valid_queue:
        valid_ds = tf.data.Dataset.range(max_num_consumers)
        valid_ds = valid_ds.interleave(_wrapper_consumer(valid_queue),
                                       cycle_length=tf.data.experimental.AUTOTUNE,
                                       deterministic=False)
        valid_ds = valid_ds.batch(batch_size)
        valid_ds = valid_ds.map(_transform_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        valid_ds = None

    return (train_ds, valid_ds)
