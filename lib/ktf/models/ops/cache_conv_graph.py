import math

import tensorflow as tf

from .compute_cache import ComputeCache


@ComputeCache.register()
def get_adjacency_exists(adjacency):
    """Checks if neighbours in the graph adjacency Tensor exists.

    Args:
        adjacency: A Tensor with shape (batch, # vertices, # edge types, # neighbours)
            with graph adjacency information

    Returns:
        A Tensor
    """
    return adjacency >= 0


@ComputeCache.register()
def get_valid_indices(adjacency_exists):
    """Extracts indices to valid feature vertices.

    Args:
        adjacency_exists: An int32 or int64 or bool Tensor with shape
            (batch, # vertices, # edge types, # neighbours) describing valid entries in
            an adjacency tensor.

    Returns:
        A (?, 2) shaped Tensor of valid indices to a feature Tensor
    """
    return tf.where(tf.reduce_any(tf.reduce_any(adjacency_exists, axis=-1), axis=-1))


@ComputeCache.register()
def get_valid_indices_from_adj(adjacency):
    """Gets valid indices from adjacency Tensor.

    Args:
        adjacency: A Tensor with shape (batch, # vertices, # edge types, # neighbours)
            with graph adjacency information

    Returns:
        A (?, 2) shaped Tensor of valid indices to a feature Tensor
    """
    adjacency_exists = get_adjacency_exists(adjacency)
    return get_valid_indices(adjacency_exists)


@ComputeCache.register()
def get_nb_indices_at_edge(i, adjacency_exists):
    """Extracts valid indices to an adjacency Tensor for an edge type

    Args:
        i: An int index of the edge type
        adjacency_exists: An int32 or int64 or bool Tensor with shape
            (batch, # vertices, # edge types, # neighbours) describing valid entries in
            an adjacency tensor.

    Returns:
        A (?, 3) shaped Tensor of valid indices to an adjacency Tensor
    """
    return tf.cast(tf.where(adjacency_exists[:, :, i, :]), dtype=tf.int32)


@ComputeCache.register()
def get_vtx_nb_indices_at_edge(i, neighbours_indices, adjacency):
    """Gets indices to neighbours features for an edge type.

    Args:
        i: An int index of the edge type
        neighbours_indices: A (?, 3) shaped Tensor of valid indices to an adjacency Tensor
        adjacency: A Tensor with shape (batch, # vertices, # edge types, # neighbours)
            with graph adjacency information

    Returns:
        A (?, 2) shaped Tensor of indices to neighbour features
    """
    return tf.concat([neighbours_indices[:, 0:1], tf.expand_dims(
        tf.gather_nd(adjacency[:, :, i, :], neighbours_indices), axis=-1)], axis=-1)


@ComputeCache.register()
def scatter_valid_features(valid_features, adjacency):
    """Scatters valid (flattened) features to a Tensor with shape corresponding to
    an adjacency Tensor. Basically the reverse operation of `get_valid_features`.

    Args:
        valid_features: A 2D Tensor of features retrieved using `get_valid_features` or similar
        adjacency: A Tensor with shape (batch, # vertices, # edge types, # neighbours)
            with graph adjacency information

    Returns:
        A Tensor of shape (batch, # vertices, # features) with empty entries zero-filled
    """
    valid_indices = get_valid_indices_from_adj(adjacency)
    if len(valid_features.shape) > 1:
        vertex_features_shape = tf.cast(tf.concat([tf.shape(adjacency)[:2],     # Batch, Vertices
                                                   tf.shape(valid_features)[-1:]],   # Features
                                                  axis=-1), tf.int64)
    else:
        vertex_features_shape = tf.cast(tf.concat([tf.shape(adjacency)[:2]],   # Batch, Vertices
                                                  axis=-1), tf.int64)
    return tf.scatter_nd(valid_indices, valid_features, vertex_features_shape)
