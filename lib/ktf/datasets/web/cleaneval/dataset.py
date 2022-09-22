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


def from_json_graph(json_graph_dir,
                    max_depth=7,
                    max_neighbours=30,
                    max_nodes=500,
                    validation_split=0.2,
                    batch_size=64,
                    shuffle_size=None,
                    num_parallel_calls=8,
                    config_path=None):

    config = parse_config(config_path)

    def _load_samplers(file):
        with gzip.open(file, "rt", encoding="ascii") as fd:
            graph_data = json.loads(fd.read())
            new_graph = json_graph.node_link_graph(graph_data["graph"])
            content_sampler = (ktf.datasets.web.GraphNodeSampler(new_graph, **graph_data["content_config"])
                               if len(graph_data["content_config"]["node_list"]) > 0
                               else None)
            non_content_sampler = (ktf.datasets.web.GraphNodeSampler(new_graph, **graph_data["non_content_config"])
                                   if len(graph_data["non_content_config"]["node_list"]) > 0
                                   else None)
            if content_sampler is None:
                return non_content_sampler, non_content_sampler
            elif non_content_sampler is None:
                return content_sampler, content_sampler
            else:
                return content_sampler, non_content_sampler

    def _producer_json_graph(files, queue, seed):
        rand = random.Random(seed)
        max_batch = 32
        max_num_samplers = 8
        random.shuffle(files)

        samplers = []
        for _ in range(max_num_samplers):
            file = files.pop(0)
            files.append(file)
            samplers.append(_load_samplers(file))

        while True:
            data = []
            for _ in range(max_batch):
                if rand.random() <= 1.0 / max_batch:
                    file = files.pop(0)
                    files.append(file)
                    samplers.pop(0)
                    samplers.append(_load_samplers(file))

                samplers_pair = rand.choice(samplers)
                if rand.random() >= 0.5:
                    node_adj, node_features, node, _ = samplers_pair[0].sample(max_depth,
                                                                               max_neighbours,
                                                                               max_nodes)
                    is_content = 1
                else:
                    node_adj, node_features, node, _ = samplers_pair[1].sample(max_depth,
                                                                               max_neighbours,
                                                                               max_nodes)
                    is_content = 0
                data.append(((node_adj, node_features, node), is_content))
            queue.put(data, block=True, timeout=3600.0)

    def _get_consumer_gen(queue):
        def _gen():
            while True:
                batch_data = queue.get(block=True, timeout=3600.0)
                for data in batch_data:
                    yield data
        return _gen

    all_files = glob.glob(os.path.join(json_graph_dir, "*.json.gz"))
    num_valid_files = int(len(all_files) * validation_split)
    valid_files = all_files[:num_valid_files]
    train_files = all_files[num_valid_files:]

    sampler, _ = _load_samplers(all_files[0])
    samp_adj, samp_feats, _, _ = sampler.sample(max_depth,
                                                max_neighbours,
                                                max_nodes)

    train_queue = Queue(1024 * 1024 * 1024 * 2)
    valid_queue = None

    if num_valid_files > 0:
        valid_queue = Queue(1024 * 1024 * 1024)

    for i in range(num_parallel_calls):
        multiprocessing.Process(target=_producer_json_graph,
                                args=(train_files,
                                      train_queue,
                                      int(time.time() * 100000 + i * 2)),
                                daemon=True).start()

        if valid_queue:
            multiprocessing.Process(target=_producer_json_graph,
                                    args=(valid_files,
                                          valid_queue,
                                          int(time.time() * 100000 + i * 2 + 1)),
                                    daemon=True).start()

    def _transform_data(x, y):
        adj = x[0]
        feat = ktf.datasets.web.tensor_rand_flip_eigvec(
            x[1], config["eigvec_offset"][0], config["eigvec_offset"][1], batch_dims=1)
        node = x[2]
        return (adj, feat, node), y

    train_ds = tf.data.Dataset.from_generator(_get_consumer_gen(train_queue),
                                              output_types=((tf.int32, tf.float32, tf.int32), tf.int32),
                                              output_shapes=((samp_adj.shape, samp_feats.shape, ()), ()))

    if shuffle_size is not None:
        train_ds = train_ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)
    if config["eigvec_offset"]:
        train_ds = train_ds.map(_transform_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    if valid_queue:
        valid_ds = tf.data.Dataset.from_generator(_get_consumer_gen(valid_queue),
                                                  output_types=((tf.int32, tf.float32, tf.int32), tf.int32),
                                                  output_shapes=((samp_adj.shape, samp_feats.shape, ()), ()))
        valid_ds = valid_ds.batch(batch_size)
        if config["eigvec_offset"]:
            valid_ds = valid_ds.map(_transform_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        valid_ds = None

    return (train_ds, valid_ds)


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
