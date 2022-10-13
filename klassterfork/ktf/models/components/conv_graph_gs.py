import tensorflow as tf
import tensorflow.keras as keras

from ..components import SuperModel


class AggrGSMean(keras.layers.Layer):
    """GraphSAGE mean/sum layer
    """

    def __init__(self, use_sum, **kwargs):
        super(AggrGSMean, self).__init__(**kwargs)
        self._use_sum = use_sum

    def call(self, inputs):
        adjacency = inputs[0]
        flattened_indices = inputs[1]
        flattened_features = inputs[2]

        # We group all transformed neighbour features of each vertex together per edge type
        vertex_features_shape = tf.concat([tf.shape(adjacency)[:2],
                                           tf.shape(adjacency)[3:4],
                                           tf.shape(flattened_features[0])[1:2]], axis=-1)
        aggr_feats = []

        # We compute the degree of each vertex per edge type
        degree = tf.expand_dims(tf.reduce_sum(tf.cast(adjacency >= 0, tf.int32), axis=-1), axis=-1)
        degree = tf.cast(tf.maximum(degree, 1), flattened_features[0].dtype)

        for i, (indices, features) in enumerate(zip(flattened_indices, flattened_features)):
            neighbour_features = tf.scatter_nd(indices, features, vertex_features_shape)

            # Calculate mean of neighbour features
            mean_features = tf.reduce_sum(neighbour_features, axis=-2)
            if not self._use_sum:
                mean_features = (mean_features / degree[:, :, i, :])

            aggr_feats.append(mean_features)
        return aggr_feats


class AggrGSMaxPool(keras.layers.Layer):
    """GraphSAGE max-pool layer
    """

    def __init__(self, activation="relu", **kwargs):
        super(AggrGSMaxPool, self).__init__(**kwargs)

        self._activation = activation

        if isinstance(self._activation, str):
            self._activation = keras.activations.get(self._activation)

    def build(self, input_shapes):
        adj_input_shape = tf.TensorShape(input_shapes[0])
        features_input_shape = tf.TensorShape(input_shapes[2][0])

        # We create a kernel per edge type in a batch
        self._dense_layers = []
        for i in range(adj_input_shape[2]):
            self._dense_layers.append(keras.layers.Dense(features_input_shape[-1]))

        self.built = True

    def call(self, inputs):
        adjacency = inputs[0]
        flattened_indices = inputs[1]
        flattened_features = inputs[2]

        min_val = -1e9
        if self._activation is not None:
            min_val = self._activation(min_val)

        # We group all transformed neighbour features of each vertex together per edge type
        vertex_features_shape = tf.concat([tf.shape(adjacency)[:2],
                                           tf.shape(adjacency)[3:4],
                                           [self._dense_layers[0].units]], axis=-1)
        aggr_feats = []
        for i, (indices, features) in enumerate(zip(flattened_indices, flattened_features)):
            xform_features = self._dense_layers[i](features)
            if self._activation is not None:
                xform_features = self._activation(xform_features)

            # We need to initialize a Tensor of a large negative value to scatter into because
            # we are doing a max pool
            neighbour_xform_features = tf.tensor_scatter_nd_update(
                tf.cast(tf.fill(vertex_features_shape, min_val), xform_features.dtype),
                indices,
                xform_features)

            # Finally max pool all features and return the result
            pooled_features = tf.reduce_max(neighbour_xform_features, axis=-2)
            aggr_feats.append(pooled_features)
        return aggr_feats


class ConvGraphGS(SuperModel):
    """GraphSAGE Keras-compatible layer, as described in: http://snap.stanford.edu/graphsage/
    """

    def __init__(self,
                 units,
                 aggregator="gs_mean",
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
                "gs_mean", "gs_sum" or "gs_maxpool". Default is "gs_mean"
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

        super(ConvGraphGS, self).__init__(**kwargs)

        self._units = units
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)

        self._aggregator = {
            "gs_sum": AggrGSMean(name="aggr_mean", use_sum=True),
            "gs_mean": AggrGSMean(name="aggr_mean", use_sum=False),
            "gs_maxpool": AggrGSMaxPool(name="aggr_pool", activation=activation),
        }.get(aggregator, aggregator)

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
        features_input_shape = tf.TensorShape(input_shapes[1])

        self._num_edge_types = adj_input_shape[2]
        self._kernel = self.add_weight("kernel",
                                       shape=[features_input_shape[2] * self._num_edge_types, self._units],
                                       initializer=self._kernel_initializer,
                                       regularizer=self._kernel_regularizer,
                                       constraint=self._kernel_constraint,
                                       trainable=True)
        if self._use_bias:
            self._bias = self.add_weight("bias",
                                         shape=[self._units, ],
                                         initializer=self._bias_initializer,
                                         regularizer=self._bias_regularizer,
                                         constraint=self._bias_constraint,
                                         trainable=True)
        else:
            self._bias = None
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

        # Filter invalid entries in the adjacency matrices
        adjacency_exists = adjacency >= 0

        flattened_indices = []
        flattened_features = []

        # Transform all neighbour features per edge type
        for i in range(self._num_edge_types):
            # Gather the (valid) neighbour ID for every vertex
            neighbours_indices = tf.cast(tf.where(adjacency_exists[:, :, i, :]), dtype=tf.int32)
            neighbours_flattened = tf.expand_dims(tf.gather_nd(adjacency[:, :, i, :], neighbours_indices), axis=-1)

            # Replace the proxy indices with the actual neighbour ID
            vertices_neighbours_flattened = tf.concat([neighbours_indices[:, 0:1], neighbours_flattened], axis=-1)

            # Then gather the features for each neighbour and transform them
            neighbours_features_flattened = tf.gather_nd(features, vertices_neighbours_flattened)
            flattened_features.append(neighbours_features_flattened)
            flattened_indices.append(neighbours_indices)

        # Aggregate the neighbour features
        aggregated_feats = self._aggregator((adjacency, flattened_indices, flattened_features))

        # Concat GS style
        indices = tf.where(tf.reduce_any(tf.reduce_any(adjacency_exists, axis=-1), axis=-1))
        final_feats = tf.concat(aggregated_feats, axis=-1)
        flattened_xform_features = tf.matmul(tf.gather_nd(final_feats, indices), self._kernel)
        if self._bias is not None:
            flattened_xform_features = tf.nn.bias_add(flattened_xform_features, self._bias)
        if self._activation is not None:
            flattened_xform_features = self._activation(flattened_xform_features)

        final_xform_feats = tf.scatter_nd(indices, flattened_xform_features,
                                          tf.cast(tf.concat([tf.shape(features)[:-1], [self._units]], axis=-1), tf.int64))
        return final_xform_feats
