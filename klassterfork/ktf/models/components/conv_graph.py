import tensorflow as tf
import tensorflow.keras as keras


import ktf.models.ops.cache_conv_graph as cache_conv_graph
import ktf.models.ops.cache_common as cache_common
from ktf.models.ops.compute_cache import ComputeCache


class AggrConvGraphMean:
    """Mean aggregator implementation for ConvGraph.
    """

    def __init__(self, use_sum, activation='relu'):
        self._activation = activation
        if isinstance(self._activation, str):
            if self._activation == "leaky_relu":
                self._activation = keras.layers.LeakyReLU(name="leaky_relu")
            else:
                self._activation = keras.activations.get(self._activation)
        self._use_sum = use_sum

    def __call__(self, adjacency, flattened_indices, flattened_features):
        """Mean aggregation function.

        Args:
            adjacency: The adjacency Tensor of shape
                (num_batches, max_num_vertices, max_num_edge_types, max_num_neighbours) describing
                each vertex of the graph and its neighbours.
            flattened_indices: A list of int32 Tensors per edge type, of shape (num_vertices, 3) that describe
                the indices into the adjacency Tensor for each valid neighbour of every vertex, per edge type.
            flattened_features: A list of Tensors per edge type, of shape (num_vertices, feature_size) that describe
                the features of every neighbour defined in flattened_indices.

        Returns:
            A Tensor of shape (num_batches, max_num_vertices, feature_size) representing the aggregated
            feature per vertex.
        """

        # We group all transformed neighbour features of each vertex together per edge type
        vertex_features_shape = tf.concat([tf.shape(adjacency)[:2],
                                           tf.shape(adjacency)[3:4],
                                           tf.shape(flattened_features[0])[1:2]], axis=-1)

        summed_features = None

        # We compute the degree of each vertex per edge type
        degree = tf.expand_dims(tf.reduce_sum(tf.cast(adjacency >= 0, tf.int32), axis=-1), axis=-1)
        degree = tf.cast(tf.maximum(degree, 1), flattened_features[0].dtype)

        for i, (indices, features) in enumerate(zip(flattened_indices, flattened_features)):
            neighbour_features = tf.scatter_nd(indices, features, vertex_features_shape)

            # Calculate mean of neighbour features
            mean_features = (tf.reduce_sum(neighbour_features, axis=-2))
            if not self._use_sum:
                mean_features = mean_features / degree[:, :, i, :]

            if summed_features is not None:
                summed_features += mean_features
            else:
                summed_features = mean_features

        # Finally pass through activation and the summed featuers
        return self._activation(summed_features) if self._activation else summed_features


class AggrConvGraphMaxPool:
    def __init__(self, activation='relu'):
        self._activation = activation
        if isinstance(self._activation, str):
            if self._activation == "leaky_relu":
                self._activation = keras.layers.LeakyReLU(name="leaky_relu")
            else:
                self._activation = keras.activations.get(self._activation)

    def __call__(self, adjacency, flattened_indices, flattened_features):
        """Max pooling aggregation function.

        Args:
            adjacency: The adjacency Tensor of shape
                (num_batches, max_num_vertices, max_num_edge_types, max_num_neighbours) describing
                each vertex of the graph and its neighbours.
            flattened_indices: A list of int32 Tensors per edge type, of shape (num_vertices, 3) that describe
                the indices into the adjacency Tensor for each valid neighbour of every vertex, per edge type.
            flattened_features: A list of Tensors per edge type, of shape (num_vertices, feature_size) that describe
                the features of every neighbour defined in flattened_indices.

        Returns:
            A Tensor of shape (num_batches, max_num_vertices, feature_size) representing the aggregated
            feature per vertex.
        """

        # We group all transformed neighbour features of each vertex together per edge type
        vertex_features_shape = tf.concat([tf.shape(adjacency)[:2],
                                           tf.shape(adjacency)[3:4],
                                           tf.shape(flattened_features[0])[1:2]], axis=-1)
        neighbour_features_list = []
        for i, (indices, features) in enumerate(zip(flattened_indices, flattened_features)):
            activated_features = self._activation(features) if self._activation else features

            # We need to initialize a Tensor of a large negative value to scatter into because
            # we are doing a max pool
            neighbour_features = tf.tensor_scatter_nd_update(
                tf.cast(tf.fill(vertex_features_shape, -1e9), features.dtype),
                indices,
                activated_features)
            neighbour_features_list.append(neighbour_features)

        # Finally max pool all features and return the result
        adjacency_features = tf.stack(neighbour_features_list, axis=2)
        pooled_features = tf.reduce_max(
            tf.reduce_max(adjacency_features, axis=-2),
            axis=-2)
        return pooled_features


class ConvGraph(keras.layers.Layer):
    """Graph neural network Keras-compatible layer, as described in:

    1. https://arxiv.org/pdf/1812.08434.pdf
    2. https://tkipf.github.io/graph-convolutional-networks/

    and combining ideas from:

    3. https://arxiv.org/abs/1703.06103
    """

    def __init__(self,
                 units,
                 aggregator="mean",
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
            aggregator: (Optional) Aggregator to use. Values are "mean" or "maxpool". Default is "mean"
            activation: (Optional) Keras activation to use. Default is "relu"
            use_bias: (Optional) Use bias weights for linear transformation.
            kernel_initialize: (Optional) Keras initializer for kernel weights.
            bias_initializer: (Optional) Keras initializer for bias weights
            kernel_regularizer: (Optional) Keras regularizer for the kernel weights.
            bias_regularizer: (Optional) Keras regularizer for the bias weights.
            kernel_constraint: (Optional) Keras constraints for the kernel weights.
            bias_constraint: (Optional) Keras constraints for the bias weights.
            **kwargs: (Optional) Keras base layer parameters.
        """

        super(ConvGraph, self).__init__(**kwargs)

        self._units = units
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)

        self._aggregator = {
            "sum": AggrConvGraphMean(activation=activation, use_sum=True),
            "mean": AggrConvGraphMean(activation=activation, use_sum=False),
            "maxpool": AggrConvGraphMaxPool(activation=activation),
        }.get(aggregator, aggregator)

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

        self._kernels = []
        self._biases = []

        # We create a kernel per edge type in a batch
        for i in range(adj_input_shape[2]):
            kernel = self.add_weight(
                'kernel_%d' % i,
                shape=[features_input_shape[2], self._units],
                initializer=self._kernel_initializer,
                regularizer=self._kernel_regularizer,
                constraint=self._kernel_constraint,
                trainable=True)
            if self._use_bias:
                bias = self.add_weight(
                    'bias_%d' % i,
                    shape=[self._units, ],
                    initializer=self._bias_initializer,
                    regularizer=self._bias_regularizer,
                    constraint=self._bias_constraint,
                    trainable=True)
            else:
                bias = None

            self._kernels.append(kernel)
            self._biases.append(bias)
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
        for i, (kernel, bias) in enumerate(zip(self._kernels, self._biases)):
            # Gather the (valid) neighbour ID for every vertex
            neighbours_indices = tf.cast(tf.where(adjacency_exists[:, :, i, :]), dtype=tf.int32)
            neighbours_flattened = tf.expand_dims(tf.gather_nd(adjacency[:, :, i, :], neighbours_indices), axis=-1)

            # Replace the proxy indices with the actual neighbour ID
            vertices_neighbours_flattened = tf.concat([neighbours_indices[:, 0:1], neighbours_flattened], axis=-1)

            # Then gather the features for each neighbour and transform them
            neighbours_features_flattened = tf.gather_nd(features, vertices_neighbours_flattened)
            flattened_xform_features = tf.matmul(neighbours_features_flattened, kernel)
            if bias is not None:
                flattened_xform_features = tf.nn.bias_add(flattened_xform_features, bias)
            flattened_features.append(flattened_xform_features)
            flattened_indices.append(neighbours_indices)

        # Aggregate the neighbour features
        aggregated_feats = self._aggregator(adjacency, flattened_indices, flattened_features)

        # Flag vertices as valid/invalid depending on whether they have proper adjacency matrices
        vertices = tf.expand_dims(tf.reduce_any(tf.reduce_any(adjacency_exists, axis=-1), axis=-1), axis=-1)

        # Then finally return the feats for updated vertices while leaving the rest untouched IF
        # the output feature dimension is the same as the input feature dimension
        # If the output feature dimension is different from the input dimension, then for un-updated
        # vertices we resize the feature dimension to the output size and initialize them to zero
        def_features = features
        if self._kernels[0].shape[0] != self._kernels[0].shape[1]:
            def_features = tf.zeros(tf.concat([tf.shape(features)[:-1], [self._units]], axis=-1), dtype=features.dtype)
        return tf.where(vertices, aggregated_feats, def_features)


class ConvGraphSelfLoop(keras.layers.Layer):
    """ Self-loop graph layer. This is graph "convolution" analogous to a 1x1 2D convolution
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 activation='relu',
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initializes graph convolution self-loop layer.

        Args:
            units: Number of output units.
            use_bias: (Optional) Use bias weights for linear transformation.
            activation: (Optional) A keras activation object
            kernel_initialize: (Optional) Keras initializer for kernel weights.
            bias_initializer: (Optional) Keras initializer for bias weights
            kernel_regularizer: (Optional) Keras regularizer for the kernel weights.
            bias_regularizer: (Optional) Keras regularizer for the bias weights.
            kernel_constraint: (Optional) Keras constraints for the kernel weights.
            bias_constraint: (Optional) Keras constraints for the bias weights.
            **kwargs: (Optional) Keras base layer parameters.
        """

        super(ConvGraphSelfLoop, self).__init__(**kwargs)

        self._units = units
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)
        self._activation = activation
        if isinstance(self._activation, str):
            if self._activation == "leaky_relu":
                self._activation = keras.layers.LeakyReLU(name="leaky_relu")
            else:
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

        self._kernel = self.add_weight("kernel",
                                       shape=[features_input_shape[2], self._units],
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
        valid_indices = cache_conv_graph.get_valid_indices_from_adj(adjacency)
        valid_features = cache_common.gather_nd(features, valid_indices)

        valid_features = tf.matmul(valid_features, self._kernel)
        if self._bias is not None:
            valid_features = tf.nn.bias_add(valid_features, self._bias)
        if self._activation:
            valid_features = self._activation(valid_features)

        def_features = features
        if self._kernel.shape[0] != self._kernel.shape[1]:
            def_features = tf.zeros(tf.concat([tf.shape(features)[:-1], [self._units]], axis=-1), dtype=features.dtype)

        features = tf.tensor_scatter_nd_update(def_features, valid_indices, valid_features)
        ComputeCache.assign(cache_common.gather_nd, (features, valid_indices), valid_features)
        return features
