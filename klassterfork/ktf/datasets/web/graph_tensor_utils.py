import tensorflow as tf


def tensor_rand_flip_eigvec(feats, offset, size, batch_dims=1):
    # Randomly flip the sign of eigenvectors (they are equivalent to the original) for use in training
    feats_t = tf.transpose(feats, list(range(batch_dims)) + [len(feats.shape) - 1, len(feats.shape) - 2])
    rand = tf.random.uniform(tf.concat([tf.shape(feats_t)[:batch_dims], [size]], axis=-1))
    sel = tf.cast(tf.where(rand > 0.5), tf.int32) + tf.constant(([0] * batch_dims) + [offset])
    neg_eigvec = -tf.gather_nd(feats_t, sel)
    feats_t_neg = tf.tensor_scatter_nd_update(feats_t, sel, neg_eigvec)
    feats_neg = tf.transpose(feats_t_neg, list(range(batch_dims)) + [len(feats.shape) - 1, len(feats.shape) - 2])
    return feats_neg


def tensor_rand_sel_nodes(adj, max_num_nodes=16):
    # Each batch element contains the number of valid vertices
    valid_per_batch = tf.reduce_sum(tf.cast(tf.reduce_any(adj >= 0, axis=[-2, -1]), tf.float32),
                                    axis=-1,
                                    keepdims=True)
    # Generate an array of integers up to max_num_nodes with replacement (for now)
    rand_idx = tf.cast(
        tf.random.uniform(tf.concat([tf.shape(valid_per_batch)[0:1], [max_num_nodes]], axis=-1)) * valid_per_batch,
        tf.int32)
    return rand_idx


def tensor_rand_sel_nodes_from_list(nodes_list, max_num_nodes=16):
    # Each batch element contains the number of valid vertices
    valid_indices_per_batch = tf.reduce_sum(tf.cast(nodes_list >= 0, tf.float32),
                                            axis=-1,
                                            keepdims=True)
    rand_shape = tf.concat([tf.shape(valid_indices_per_batch)[0:1], [max_num_nodes]], axis=-1)
    rand_idx_to_node_idx = tf.cast(tf.random.uniform(rand_shape) * valid_indices_per_batch, tf.int32)
    rand_idx = tf.gather(nodes_list, rand_idx_to_node_idx, batch_dims=1)

    # Return selected nodes (rand_idx) and their indices in feature tensor (rand_idx_to_node_idx)
    return rand_idx_to_node_idx, rand_idx


def tensor_extract_feats(feats, src_node, offsets, sizes):
    src_feats = tf.gather_nd(feats, tf.expand_dims(src_node, axis=-1), batch_dims=1)

    extracted_feats = []
    for (offset, size) in zip(offsets, sizes):
        cur_feats = src_feats[..., offset:offset + size]
        extracted_feats.append(cur_feats)
    return extracted_feats


def tensor_rand_mask_feat(feats, src_node, prob_mask=0.85):
    src_node_indices = tf.where(src_node >= 0)
    src_node_feat_indices = tf.stack([src_node_indices[:, 0],   # Batch
                                      tf.cast(tf.gather_nd(src_node, src_node_indices), tf.int64)],  # Indices
                                     axis=-1)
    rand = tf.random.uniform(tf.shape(src_node_indices)[0:1])
    sel = tf.where(rand <= prob_mask)
    to_zero_indices = tf.gather_nd(src_node_feat_indices, sel)
    src_zeros = tf.zeros(tf.concat([tf.shape(to_zero_indices)[0:1], tf.shape(feats)[-1:]], axis=-1), dtype=feats.dtype)
    src_zeroed_feats = tf.tensor_scatter_nd_update(feats, to_zero_indices, src_zeros)
    return src_zeroed_feats
