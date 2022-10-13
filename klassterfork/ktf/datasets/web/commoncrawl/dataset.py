import os
import glob
import json

import numpy as np
import tensorflow as tf

import ktf.datasets
import ktf.datasets.web


# We use this for reading and writing tfrecord files
FEATURE_DESC_GRAPH = [('adj1', int, [-1]),         # Placeholder
                      ('feat1', float, [-1]),      # Placeholder
                      ('node_list1', int, [-1]),
                      ('group1', np.int64, []),
                      ('adj2', int, [-1]),         # Placeholder
                      ('feat2', float, [-1]),      # Placeholder
                      ('node_list2', int, [-1]),
                      ('group2', np.int64, [])]


def _setup_config(tfrecord_dir,
                  tag_feature_list,
                  validation_split,
                  config_path=None):
    if config_path is None:
        config_path = os.path.join(tfrecord_dir[0] if isinstance(tfrecord_dir, list) else tfrecord_dir, "config.json")

    if not isinstance(tfrecord_dir, list):
        tfrecord_dir = [tfrecord_dir]

    with open(config_path, "r") as fd:
        config_json = json.loads(fd.read())
        config = ktf.datasets.web.parse_graph_config_json(config_json)

    all_files = []
    for dir in tfrecord_dir:
        all_files.extend(glob.glob(os.path.join(dir, "*.tfrecord")))

    pruned_files = {}
    for file in all_files:
        if (os.path.exists(os.path.join(os.path.dirname(file), "complete.txt")) or
            os.path.exists(os.path.join(os.path.dirname(file), "processed.txt"))):  # noqa
            dirname = os.path.dirname(file)
            if dirname not in pruned_files:
                pruned_files[dirname] = [file]
            else:
                pruned_files[dirname].append(file)
        else:
            print("Skipping %s. complete.txt or processed.txt missing." % file)

    num_valid = int(len(pruned_files) * validation_split)
    pruned_files_list = list(pruned_files.items())
    trained_files = [file for _, files in pruned_files_list[num_valid:] for file in files]
    valid_files = [file for _, files in pruned_files_list[:num_valid] for file in files]

    print("Num trained files: %d" % len(trained_files))
    print("Num valid files: %d" % len(valid_files))

    tag_feature_offset = []
    tag_feature_size = []
    if tag_feature_list:
        for full_key in tag_feature_list:
            feature_desc = config_json["feature_desc"]["feature_offsets"]
            for key in full_key.split("."):
                feature_desc = feature_desc[key]
            tag_feature_offset.append(feature_desc["idx"])
            tag_feature_size.append(feature_desc["len"])

    return config, trained_files, valid_files, tag_feature_offset, tag_feature_size


def from_tfrecord_graph_unsupervised(tfrecord_dir,
                                     tag_feature_list,
                                     config_path=None,
                                     prob_mask=0.85,
                                     max_num_nodes=16,
                                     use_node_list=False,
                                     batch_size=64,
                                     validation_split=0.2,
                                     shuffle_size=None,
                                     repeat=False,
                                     num_parallel_reads=tf.data.AUTOTUNE):
    """Reads from tfrecord generated with `html_to_graph_pair.py` and applies 
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
    config, trained_files, valid_files, tag_feature_offset, tag_feature_size = _setup_config(tfrecord_dir,
                                                                                             tag_feature_list,
                                                                                             validation_split,
                                                                                             config_path=config_path)

    max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]

    def _decode_example(examples):
        def _process_autoencode(adj, feat, node_list):
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
            src_feats = ktf.datasets.web.tensor_extract_feats(
                feat, nodes_per_batch, tag_feature_offset, tag_feature_size)
            zeroed_feat = ktf.datasets.web.tensor_rand_mask_feat(feat, nodes_per_batch, prob_mask=prob_mask)
            return ((adj, zeroed_feat, nodes_per_batch), tuple(src_feats))

        adj1 = tf.cast(examples["adj1"], tf.int32)
        feat1 = tf.cast(examples["feat1"], tf.float32)
        node_list1 = tf.cast(examples["node_list1"], tf.int32)
        group1 = examples["group1"]
        adj2 = tf.cast(examples["adj2"], tf.int32)
        feat2 = tf.cast(examples["feat2"], tf.float32)
        node_list2 = tf.cast(examples["node_list2"], tf.int32)
        group2 = examples["group2"]

        adj = tf.concat([adj1, adj2], axis=0)
        feat = tf.concat([feat1, feat2], axis=0)
        node_list = tf.concat([node_list1, node_list2], axis=0)

        return _process_autoencode(adj, feat, node_list)

    feature_desc = [
        ('adj1', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat1', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('node_list1', int, [192]),
        ('group1', np.int64, []),
        ('adj2', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat2', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('node_list2', int, [192]),
        ('group2', np.int64, [])
    ]

    record_io = ktf.datasets.RecordDatasetIO(feature_desc)
    train_ds = record_io.read_batch(trained_files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True,
                                    drop_remainder=True)
    valid_ds = record_io.read_batch(valid_files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True,
                                    drop_remainder=True)
    return train_ds, valid_ds


def from_tfrecord_graph_unsupervised_pairs(tfrecord_dir,
                                           tag_feature_list,
                                           config_path=None,
                                           prob_mask=0.85,
                                           max_num_nodes=16,
                                           use_node_list=False,
                                           permute_ratio=0.5,
                                           batch_size=64,
                                           validation_split=0.2,
                                           shuffle_size=None,
                                           repeat=False,
                                           num_parallel_reads=tf.data.AUTOTUNE,
                                           debug_force_cache=False):
    """Reads from tfrecord generated with `html_to_graph_pair.py` and applies
    transformations applicable to joint unsupervised learning with "same website" prediction

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
        permute_ratio: (Optional) The (ratio) number of elements in the batch where the ordering of the elements
            within the batch is shuffled. This basically controls the number of matching pairs vs not-matching pairs
            for classification.
        batch_size: (Optional) Batch size. Default is 64
        validation_split: (Optional) Validation set split (from the number of files). Default is 0.2
        shuffle_size: (Optional) Size of the shuffle buffer. Default is None.
        repeat: (Optional) Repeat the dataset
        debug_force_cache: (Optional) Debug option mainly for testing convergence.

    Returns:
        A tuple of `(train_dataset, validation_dataset)`
    """
    config, trained_files, valid_files, tag_feature_offset, tag_feature_size = _setup_config(tfrecord_dir,
                                                                                             tag_feature_list,
                                                                                             validation_split,
                                                                                             config_path=config_path)

    max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]

    def _decode_example(examples):
        def _process_autoencode(adj, feat, node_list):
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
            src_feats = ktf.datasets.web.tensor_extract_feats(
                feat, nodes_per_batch, tag_feature_offset, tag_feature_size)
            zeroed_feat = ktf.datasets.web.tensor_rand_mask_feat(feat, nodes_per_batch, prob_mask=prob_mask)
            return ((adj, zeroed_feat, nodes_per_batch), tuple(src_feats))

        adj1 = tf.cast(examples["adj1"], tf.int32)
        feat1 = tf.cast(examples["feat1"], tf.float32)
        node_list1 = tf.cast(examples["node_list1"], tf.int32)
        group1 = examples["group1"]
        adj2 = tf.cast(examples["adj2"], tf.int32)
        feat2 = tf.cast(examples["feat2"], tf.float32)
        node_list2 = tf.cast(examples["node_list2"], tf.int32)
        group2 = examples["group2"]

        permute_len = tf.cast((tf.cast(tf.shape(feat2)[0], tf.float32) * permute_ratio), tf.int32)
        rand_indices = tf.range(permute_len - 1, -1, -1)
        adj2 = tf.concat([tf.gather(adj2, rand_indices), adj2[permute_len:]], axis=0)
        feat2 = tf.concat([tf.gather(feat2, rand_indices), feat2[permute_len:]], axis=0)
        node_list2 = tf.concat([tf.gather(node_list2, rand_indices), node_list2[permute_len:]], axis=0)
        group2 = tf.concat([tf.gather(group2, rand_indices), group2[permute_len:]], axis=0)

        # NOTE: Alternative implementation below with fewer but more expensive opts
        # rand_indices = tf.expand_dims((tf.range(permute_len - 1, -1, -1)), axis=-1)
        # adj2 = tf.tensor_scatter_nd_update(adj2, rand_indices, adj2[:permute_len])
        # feat2 = tf.tensor_scatter_nd_update(feat2, rand_indices, feat2[:permute_len])
        # node_list2 = tf.tensor_scatter_nd_update(node_list2, rand_indices, node_list2[:permute_len])
        # group2 = tf.tensor_scatter_nd_update(group2, rand_indices, group2[:permute_len])

        pair_match = tf.cast(group1 == group2, tf.int32)
        if len(tag_feature_size) == 0 or max_num_nodes == 0:
            return (((adj1, feat1, group1), (adj2, feat2, group2)), pair_match)
        else:
            input1, gt1 = _process_autoencode(adj1, feat1, node_list1)
            input2, gt2 = _process_autoencode(adj2, feat2, node_list2)
            return ((input1 + (group1,), input2 + (group2,)), (pair_match, ) + gt1 + gt2)

    feature_desc = [
        ('adj1', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat1', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('node_list1', int, [192]),
        ('group1', np.int64, []),
        ('adj2', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat2', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('node_list2', int, [192]),
        ('group2', np.int64, [])
    ]

    record_io = ktf.datasets.RecordDatasetIO(feature_desc)
    train_ds = record_io.read_batch(trained_files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True,
                                    drop_remainder=True)
    valid_ds = record_io.read_batch(valid_files,
                                    batch_size,
                                    shuffle_size=shuffle_size,
                                    repeat=repeat,
                                    parse_fn=_decode_example,
                                    num_parallel_reads=num_parallel_reads,
                                    use_compression=True,
                                    drop_remainder=True)
    if debug_force_cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()

    return train_ds, valid_ds
