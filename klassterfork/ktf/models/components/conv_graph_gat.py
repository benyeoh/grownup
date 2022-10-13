import tensorflow as tf
import tensorflow.keras as keras

from .super_model import SuperModel
import ktf.models.ops.cache_conv_graph as cache_conv_graph
import ktf.models.ops.cache_common as cache_common


class AggrGATMean(keras.layers.Layer):
    """Graph ATtention network mean/sum aggregation.
    """

    def __init__(self,
                 units,
                 use_sum,
                 group_edges=False,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initializes graph attention aggregation.

        Args:
            units: Number of output units.
            use_sum: Multiply aggregated attention-weighted features by the degree of the vertex
            group_edges: (Optional) Compute attention for each edge type separately. Default is False
            use_bias: (Optional) Use bias weights for linear transformation.
            kernel_initialize: (Optional) Keras initializer for kernel weights.
            bias_initializer: (Optional) Keras initializer for bias weights
            kernel_regularizer: (Optional) Keras regularizer for the kernel weights.
            bias_regularizer: (Optional) Keras regularizer for the bias weights.
            kernel_constraint: (Optional) Keras constraints for the kernel weights.
            bias_constraint: (Optional) Keras constraints for the bias weights.
            **kwargs: (Optional) Keras base layer parameters.
        """
        super(AggrGATMean, self).__init__(**kwargs)

        self._dense_factory = lambda: keras.layers.Dense(units,
                                                         activation=None,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint,
                                                         use_bias=use_bias)
        # Create MLP for attention weights
        self._att_dense = keras.layers.Dense(1, activation=keras.layers.LeakyReLU(name="leaky_relu"))
        self._units = units
        self._use_sum = use_sum
        self._group_edges = group_edges

    def build(self, input_shapes):
        adj_input_shape = tf.TensorShape(input_shapes[0])

        self._num_vertices = adj_input_shape[1]
        self._num_edge_types = adj_input_shape[2]
        self._num_neighbours = adj_input_shape[-1]

        self._flat_adj_shape = (-1, self._num_vertices, self._num_edge_types * self._num_neighbours)

        # We create a kernel per edge type in a batch
        self._dense_layers = []
        for i in range(self._num_edge_types):
            self._dense_layers.append(self._dense_factory())
        self.built = True

    def call(self, inputs):
        adjacency = inputs[0]
        features = inputs[1]
        flattened_indices = inputs[2]
        flattened_features = inputs[3]

        vertex_neighbours_shape = tf.shape(adjacency)

        neighbour_atts = tf.fill(vertex_neighbours_shape, -200.0)

        all_xform_feats = []
        all_full_edge_idxs = []
        for i, (edge_idxs, edge_feats) in enumerate(zip(flattened_indices, flattened_features)):
            # For each edge type transform the features for each vertex neighbours
            edge_type = tf.expand_dims(tf.fill(tf.shape(edge_idxs)[:1], i), axis=-1)
            full_edge_idxs = tf.concat([edge_idxs[:, :2], edge_type, edge_idxs[:, 2:]], axis=-1)
            edge_xform_features = self._dense_layers[i](edge_feats)

            # Then calculate the attention logits
            edge_src_node_feats = tf.gather_nd(features, edge_idxs[:, :2])
            edge_att_feats = tf.concat([edge_src_node_feats, edge_xform_features], axis=-1)
            edge_att_logits = tf.squeeze(self._att_dense(edge_att_feats), axis=-1)

            # And scatter them back into the same tensor layout as the adjacency tensor
            neighbour_atts = tf.tensor_scatter_nd_update(neighbour_atts, full_edge_idxs, edge_att_logits)

            all_xform_feats.append(edge_xform_features)
            all_full_edge_idxs.append(full_edge_idxs)

        if not self._group_edges:
            # If we calculate edges globally, we calculate softmax for all vertices and edge types
            neighbour_atts = tf.reshape(tf.nn.softmax(tf.reshape(neighbour_atts,
                                                                 self._flat_adj_shape),
                                                      axis=-1),
                                        vertex_neighbours_shape)
        else:
            # Otherwise we calculate for vertices per edge type
            neighbour_atts = tf.nn.softmax(neighbour_atts, axis=-1)

        vertex_features_shape = tf.concat([tf.shape(adjacency)[:2],   # Batch, Vertices
                                           tf.shape(adjacency)[3:4],  # Neighbours
                                           [self._units]],            # Features
                                          axis=-1)
        degree = None
        if self._use_sum:
            degree = tf.reduce_sum(tf.cast(adjacency >= 0, tf.int32), axis=-1)
            if not self._group_edges:
                degree = tf.reduce_sum(degree, axis=-1)
            degree = tf.expand_dims(tf.cast(degree, flattened_features[0].dtype), axis=-1)

        summed_feats = None
        for i, (edge_idxs, edge_xform_feats) in enumerate(zip(flattened_indices, all_xform_feats)):
            # Calculate the attention weighted contribution per neighbour
            edge_atts = tf.expand_dims(tf.gather_nd(neighbour_atts, all_full_edge_idxs[i]), axis=-1)
            edge_xform_att_feats = all_xform_feats[i] * edge_atts
            nb_xform_att_feats = tf.scatter_nd(edge_idxs, edge_xform_att_feats, vertex_features_shape)
            edge_summed_feats = tf.reduce_sum(nb_xform_att_feats, axis=-2)
            if self._use_sum and self._group_edges:
                edge_summed_feats = edge_summed_feats * degree[:, :, i, :]
            if summed_feats is None:
                summed_feats = edge_summed_feats
            else:
                summed_feats = summed_feats + edge_summed_feats

        if self._use_sum and not self._group_edges:
            summed_feats = summed_feats * degree

        return summed_feats


class AggrGATGated(keras.layers.Layer):
    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initializes gated aggregation. See https://arxiv.org/abs/1711.07553.

        Args:
            units: Number of output units.
            use_bias: (Optional) Use bias weights for linear transformation.
            kernel_initialize: (Optional) Keras initializer for kernel weights.
            bias_initializer: (Optional) Keras initializer for bias weights
            kernel_regularizer: (Optional) Keras regularizer for the kernel weights.
            bias_regularizer: (Optional) Keras regularizer for the bias weights.
            kernel_constraint: (Optional) Keras constraints for the kernel weights.
            bias_constraint: (Optional) Keras constraints for the bias weights.
            **kwargs: (Optional) Keras base layer parameters.
        """
        super(AggrGATGated, self).__init__(**kwargs)

        self._dense_factory = lambda: keras.layers.Dense(units,
                                                         activation=None,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint,
                                                         use_bias=use_bias)

        self._gate_factory = lambda: keras.layers.Dense(units,
                                                        activation=None,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=bias_initializer,
                                                        kernel_regularizer=kernel_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        use_bias=False)

        self._units = units

    def build(self, input_shapes):
        adj_input_shape = tf.TensorShape(input_shapes[0])

        self._num_edge_types = adj_input_shape[2]

        # We create a kernel per edge type in a batch
        self._dense_per_edge_type = []
        for i in range(self._num_edge_types):
            self._dense_per_edge_type.append(self._dense_factory())

        self._gate = self._gate_factory()
        self._gate_per_edge_type = []
        for i in range(self._num_edge_types):
            self._gate_per_edge_type.append(self._gate_factory())

        self.built = True

    def call(self, inputs):
        adjacency = inputs[0]
        features = inputs[1]
        flattened_indices = inputs[2]
        flattened_features = inputs[3]

        valid_indices = cache_conv_graph.get_valid_indices_from_adj(adjacency)
        valid_features = cache_common.gather_nd(features, valid_indices)

        id_flattened = cache_common.get_range_tensor_shape(valid_indices, 0)
        ids = cache_conv_graph.scatter_valid_features(id_flattened, adjacency)

        src_gated = self._gate(valid_features)

        summed_feats = tf.zeros(tf.concat([tf.shape(features)[:2], [self._units]], axis=-1), features.dtype)
        for i, (edge_idxs, edge_feats) in enumerate(zip(flattened_indices, flattened_features)):
            # Calculate the gate values
            sliced_edge_idxs = cache_common.get_slice(edge_idxs, None, [None, 2])
            edge_ids_flattened = cache_common.gather_nd(ids, sliced_edge_idxs)
            edge_src_gated = tf.gather_nd(src_gated, tf.expand_dims(edge_ids_flattened, axis=-1))
            gate = tf.sigmoid(edge_src_gated + self._gate_per_edge_type[i](edge_feats))

            # Then modulate the output for each neighbour per edge
            edge_xform_features = self._dense_per_edge_type[i](edge_feats) * gate

            # And scatter them back into the same tensor layout as the adjacency tensor
            summed_feats = tf.tensor_scatter_nd_add(summed_feats, sliced_edge_idxs, edge_xform_features)

        return summed_feats


class ConvGraphGAT(SuperModel):
    """Graph ATtention network Keras-compatible layer, as described in: https://arxiv.org/abs/1710.10903
    """

    def __init__(self,
                 units,
                 num_heads=4,
                 aggregator="gat_mean",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initializes graph convolution layer.

        Args:
            units: Number of output units.
            aggregator: (Optional) Aggregator for neighbouring nodes. Values are
                "gat_mean", "gat_mean_grp, "gat_sum", gat_sum_grp or "gat_gated". Default is "gat_mean"
            activation: (Optional) Keras activation function. Default is None.
            use_bias: (Optional) Use bias weights for linear transformation.
            kernel_initialize: (Optional) Keras initializer for kernel weights.
            bias_initializer: (Optional) Keras initializer for bias weights
            kernel_regularizer: (Optional) Keras regularizer for the kernel weights.
            bias_regularizer: (Optional) Keras regularizer for the bias weights.
            kernel_constraint: (Optional) Keras constraints for the kernel weights.
            bias_constraint: (Optional) Keras constraints for the bias weights.
            **kwargs: (Optional) Keras base layer parameters.
        """
        super(ConvGraphGAT, self).__init__(**kwargs)

        if units % num_heads != 0:
            raise ValueError("Number of output units (%d) must be divisible by number of attention heads (%d)."
                             % (units, num_heads))

        num_units_per_head = units // num_heads

        def _gat_mean_factory(use_sum, group_edges):
            def _factory(name):
                return AggrGATMean(units=num_units_per_head,
                                   use_sum=use_sum,
                                   group_edges=group_edges,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint,
                                   name=name)
            return _factory

        def _gat_gated_factory():
            def _factory(name):
                return AggrGATGated(units=num_units_per_head,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    name=name)
            return _factory

        aggr_factory = {
            "gat_mean": _gat_mean_factory(use_sum=False, group_edges=False),
            "gat_mean_grp": _gat_mean_factory(use_sum=False, group_edges=True),
            "gat_sum": _gat_mean_factory(use_sum=True, group_edges=False),
            "gat_sum_grp": _gat_mean_factory(use_sum=True, group_edges=True),
            "gat_gated": _gat_gated_factory(),
        }.get(aggregator, aggregator)

        self._aggregators = [aggr_factory("aggr_%s_h%d" % (aggregator, i)) for i in range(num_heads)]

        self._activation = activation
        if isinstance(self._activation, str):
            self._activation = keras.activations.get(self._activation)

    def build(self, input_shapes):
        if (len(input_shapes) != 2 or
            len(tf.TensorShape(input_shapes[0])) != 4 or
                len(tf.TensorShape(input_shapes[1])) != 3):
            raise ValueError("The number of input tensors should be 2. "
                             "input_shapes[0] should be the adjacency matrix equivalent with "
                             "(batch, vertex, relationship, vertices_index) shape. "
                             "input_shapes[1] should be the node features with "
                             "(batch, vertex, features) shape.")

        adj_input_shape = tf.TensorShape(input_shapes[0])
        # features_input_shape = tf.TensorShape(input_shapes[1])

        self._num_edge_types = adj_input_shape[2]
        self.built = True

    def call(self, inputs, training=False):
        """Call function implementation for the Keras layer.

        The function operates on every vertex with at least 1 neighbour in the adjacency Tensor in inputs[0],
        and outputs a Tensor of shape (num_batches, max_num_vertices, output_feature_size).

        Note that for vertices that are not defined in the adjacency Tensor,
        the corresponding vertex in the output Tensor will either be:
            1. zero'ed, if the input feature size is different from the output feature size. Or
            2. remain the same as the input, if the input feature size is the same as the output feature size.

        Args:
            inputs: A length 2 list of Tensors with the following format:
                Index 0 should be of shape (num_batches, max_num_vertices, max_num_edge_types, max_num_neighbours).
                    num_batches - The number of batches
                    max_num_vertices - The maximum number of vertices for all graphs in this batch
                    max_num_edge_types - The maximum number of edge types for all graphs in this batch
                    max_num_neighbours - A max_num_neighbours sized array of integers representing the node indices
                        that the current vertex is adjacent to. For non-existent neighbours, use -1 to pad the array.

                Index 1 should be of shape (num_batches, max_num_vertices, input_feature_size).
                    num_batches - As above
                    max_num_vertices - As above
                    input_feature_size - Input feature vector for the vertex

            training: (Optional) True if run during training, False otherwise.

        Returns:
            A Tensor of shape (num_batches, max_num_vertices, output_feature_size). The output feature
            is self of length ._units.
        """
        adjacency = inputs[0]
        features = inputs[1]

        adjacency_exists = cache_conv_graph.get_adjacency_exists(adjacency)

        flattened_indices = []
        flattened_features = []

        for i in range(self._num_edge_types):
            # Gather the (valid) neighbour ID for every vertex
            neighbours_indices = cache_conv_graph.get_nb_indices_at_edge(i, adjacency_exists)

            # Replace the proxy indices with the actual neighbour ID
            vertices_neighbours_flattened = cache_conv_graph.get_vtx_nb_indices_at_edge(
                i, neighbours_indices, adjacency)

            # Then gather the features for each neighbour and transform them
            neighbours_features_flattened = cache_common.gather_nd(features, vertices_neighbours_flattened)

            flattened_features.append(neighbours_features_flattened)
            flattened_indices.append(neighbours_indices)

        head_feats = []
        for aggr in self._aggregators:
            # Aggregate the neighbour features
            aggr_feats = aggr((adjacency, features, flattened_indices, flattened_features))
            if self._activation:
                aggr_feats = self._activation(aggr_feats)
            head_feats.append(aggr_feats)

        if len(head_feats) > 1:
            final_feats = tf.concat(head_feats, axis=-1)
        else:
            final_feats = head_feats[0]

        # TODO: Optionally add reprojection here
        return final_feats
