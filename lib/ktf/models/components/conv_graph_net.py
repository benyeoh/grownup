import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

from .conv_graph import ConvGraph, ConvGraphSelfLoop
from .conv_graph_gs import ConvGraphGS
from .conv_graph_gat import ConvGraphGAT
from .super_model import SuperModel
from .dense_l2norm import DenseL2Norm

import ktf.models.ops.cache_conv_graph as cache_conv_graph
import ktf.models.ops.cache_common as cache_common
from ktf.models.ops.compute_cache import ComputeCache


class ConvGraphNetBlock(SuperModel):
    """ ConvGraphNet layer/block. Defined in the same spirit as the ResNet block.
    """

    def __init__(self,
                 num_outputs,
                 self_loops_only=False,
                 use_residuals=False,
                 is_shallow=False,
                 normalization="layer",
                 dropout=0.0,
                 aggregator="mean",
                 activation="relu",
                 **kwargs):
        """ Initialize block.

        Args:
            num_outputs: The size of outputs of this feature block
            self_loops_only: (Optional) Apply only self loop convolutions. Default is False
            use_residuals: (Optional) Apply residual computations similar to ResNet. Default is False
            dropout: (Optional) Dropout rate. Default is 0 (no dropout)
            aggregator: (Optional) The aggregator to use for graph convolutions. Possible entries are
                "mean", "sum", "maxpool", "gs_mean", "gs_sum", "gs_maxpool", "gat_mean[1|4|8]", "gat_sum[1|4|8]",
                "gat_mean_grp[1|4|8]", "gat_sum_grp[1|4|8]", "gat_gated". Default is "mean"
            normalization: (Optional) Use "layer", "batch" or "group[2|4|8]" normalization. Default is "layer".
            is_shallow: (Optional) Only use 1 graph convolutional layer per block. Default is False
            activation: (Optional) Keras activation function for the network. Default is "relu"
            **kwargs: (Optional) Other layer arguments.
        """

        super(ConvGraphNetBlock, self).__init__(**kwargs)

        self._use_residuals = use_residuals
        self._num_outputs = num_outputs

        self._normalization = {
            "layer": lambda: keras.layers.LayerNormalization(epsilon=1e-5),
            "layer_simple": lambda: keras.layers.LayerNormalization(epsilon=1e-5, center=False, scale=False),
            "batch": lambda: keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9),
            "group": lambda: tfa.layers.GroupNormalization(axis=-1),
            "group8": lambda: tfa.layers.GroupNormalization(groups=8, axis=-1),
            "group4": lambda: tfa.layers.GroupNormalization(groups=4, axis=-1),
            None: lambda: None
        }.get(normalization)

        self._aggregator = aggregator
        self._is_shallow = is_shallow
        self._self_loops_only = self_loops_only

        self._activation = activation
        if isinstance(self._activation, str):
            if self._activation == "leaky_relu":
                self._activation = keras.layers.LeakyReLU(name="leaky_relu")
            else:
                self._activation = keras.activations.get(self._activation)

        if dropout > 0.0:
            self._dropout = keras.layers.Dropout(rate=dropout)
        else:
            self._dropout = None

    def build(self, input_shapes):
        if (len(input_shapes) < 2 or
                len(tf.TensorShape(input_shapes[0])) != 4 or len(tf.TensorShape(input_shapes[1])) != 3):
            raise ValueError("The number of input tensors should be >= 2. "
                             "input_shapes[0] should be the adjacency matrix equivalent with "
                             "(batch, vertex, relationship, vertices_index) shape. "
                             "input_shapes[1] should be the node features with "
                             "(batch, vertex, features) shape.")

        adj_input_shape = tf.TensorShape(input_shapes[0])
        features_input_shape = tf.TensorShape(input_shapes[1])

        def _gat_factory(*args, **kwargs):
            num_heads = {
                "gat_mean": 4,
                "gat_mean1": 1,
                "gat_mean4": 4,
                "gat_mean8": 8,
                "gat_mean_grp": 4,
                "gat_mean_grp1": 1,
                "gat_mean_grp4": 4,
                "gat_mean_grp8": 8,
                "gat_sum": 4,
                "gat_sum1": 1,
                "gat_sum4": 4,
                "gat_sum8": 8,
                "gat_sum_grp": 4,
                "gat_sum_grp1": 1,
                "gat_sum_grp4": 4,
                "gat_sum_grp8": 8,
                "gat_gated": 1
            }.get(kwargs["aggregator"])

            kwargs["aggregator"] = {
                "gat_mean1": "gat_mean",
                "gat_mean4": "gat_mean",
                "gat_mean8": "gat_mean",
                "gat_mean_grp1": "gat_mean_grp",
                "gat_mean_grp4": "gat_mean_grp",
                "gat_mean_grp8": "gat_mean_grp",
                "gat_sum1": "gat_sum",
                "gat_sum4": "gat_sum",
                "gat_sum8": "gat_sum",
                "gat_sum_grp1": "gat_sum_grp",
                "gat_sum_grp4": "gat_sum_grp",
                "gat_sum_grp8": "gat_sum_grp"
            }.get(kwargs["aggregator"], kwargs["aggregator"])

            return ConvGraphGAT(*args, num_heads=num_heads, **kwargs)

        conv_graph = {
            "sum": ConvGraph,
            "mean": ConvGraph,
            "maxpool": ConvGraph,
            "gs_sum": ConvGraphGS,
            "gs_mean": ConvGraphGS,
            "gs_maxpool": ConvGraphGS,
            "gat_mean": _gat_factory,
            "gat_mean1": _gat_factory,
            "gat_mean4": _gat_factory,
            "gat_mean8": _gat_factory,
            "gat_mean_grp": _gat_factory,
            "gat_mean_grp1": _gat_factory,
            "gat_mean_grp4": _gat_factory,
            "gat_mean_grp8": _gat_factory,
            "gat_sum": _gat_factory,
            "gat_sum1": _gat_factory,
            "gat_sum4": _gat_factory,
            "gat_sum8": _gat_factory,
            "gat_sum_grp": _gat_factory,
            "gat_sum_grp1": _gat_factory,
            "gat_sum_grp4": _gat_factory,
            "gat_sum_grp8": _gat_factory,
            "gat_gated": _gat_factory
        }.get(self._aggregator)

        self._cg1 = (conv_graph(self._num_outputs, activation=None, aggregator=self._aggregator, name="cg1")
                     if not self._self_loops_only else ConvGraphSelfLoop(self._num_outputs, activation=None, name="cg1"))
        self._ln1 = self._normalization()

        if not self._is_shallow:
            self._cg2 = ConvGraphSelfLoop(self._num_outputs, activation=None, name="cg2")
            self._ln2 = self._normalization()
        else:
            self._cg2 = None
            self._ln2 = None

        if self._use_residuals and features_input_shape[2] != self._num_outputs:
            self._cgr = ConvGraphSelfLoop(self._num_outputs, activation=None, name="cgr")
            self._lnr = self._normalization()
        else:
            self._cgr = None
            self._lnr = None

    def call(self, inputs, training=False):
        adjacency = inputs[0]
        features = inputs[1]

        valid_indices = cache_conv_graph.get_valid_indices_from_adj(adjacency)

        features = self._cg1((adjacency, features), training=training)
        valid_features = cache_common.gather_nd(features, valid_indices)
        valid_features = self._ln1(valid_features) if self._ln1 is not None else valid_features

        if self._cg2:
            valid_features = self._activation(valid_features) if self._activation is not None else valid_features
            features = tf.tensor_scatter_nd_update(features, valid_indices, valid_features)
            ComputeCache.assign(cache_common.gather_nd, (features, valid_indices), valid_features)

            features = self._cg2((adjacency, features), training=training)
            valid_features = cache_common.gather_nd(features, valid_indices)
            valid_features = self._ln2(valid_features) if self._ln2 is not None else valid_features

        if self._use_residuals:
            residual = inputs[1]
            if self._cgr:
                # Resample the inputs to have the same feature vector size
                residual = self._cgr((adjacency, residual), training=training)
                valid_residual = cache_common.gather_nd(residual, valid_indices)
                valid_residual = self._lnr(valid_residual) if self._lnr is not None else valid_residual
            else:
                valid_residual = cache_common.gather_nd(residual, valid_indices)
            valid_features = valid_features + valid_residual
            features = residual

        valid_features = self._activation(valid_features) if self._activation is not None else valid_features

        if self._dropout and training:
            valid_features = self._dropout(valid_features, training=training)

        features = tf.tensor_scatter_nd_update(features, valid_indices, valid_features)
        ComputeCache.assign(cache_common.gather_nd, (features, valid_indices), valid_features)
        return features


class GraphLinkPredictHead(SuperModel):
    """ This layer is used to extract and compare the embeddings of 2 node features
    to get an output for graph binary link prediction.
    """

    def __init__(self, activation=None, scale=1.0, **kwargs):
        super(GraphLinkPredictHead, self).__init__(**kwargs)
        self._scale = scale
        self._activation = activation
        if isinstance(self._activation, str):
            self._activation = keras.activations.get(activation)

    def call(self, inputs, training=False):
        features = inputs[1]
        node1 = inputs[2]
        if len(node1.shape) != 1:
            node1 = tf.squeeze(node1, axis=-1)
        node2 = inputs[3]
        if len(node2.shape) != 1:
            node2 = tf.squeeze(node2, axis=-1)

        node1_features = tf.nn.l2_normalize(tf.gather(features, node1, batch_dims=1), axis=-1)
        node2_features = tf.nn.l2_normalize(tf.gather(features, node2, batch_dims=1), axis=-1)

        proj = tf.reduce_sum(node1_features * node2_features, axis=-1)
        if training:
            proj = proj * self._scale
        if self._activation:
            proj = self._activation(proj)
        return proj


class GraphNodeClassifyHead(SuperModel):
    """ This layer is used to extract a node embedding and perform multi-label classification
    """

    def __init__(self, num_classes, norm_features=False, activation=None, **kwargs):
        super(GraphNodeClassifyHead, self).__init__(**kwargs)
        if num_classes is not None:
            if norm_features:
                self._head = DenseL2Norm(num_classes, use_bias=False, activation=activation, name="head")
            else:
                self._head = keras.layers.Dense(num_classes, activation=activation, name="head")
        else:
            self._head = None
            if isinstance(activation, str):
                activation = keras.activations.get(activation)
            self._activation = activation
            if self._activation is None:
                self._activation = lambda x: x

    def call(self, inputs, training=False):
        features = inputs[1]
        node1 = inputs[2]
        if len(node1.shape) != 1 and node1.shape[-1] == 1:
            node1 = tf.squeeze(node1, axis=-1)

        node1_features = tf.gather(features, node1, batch_dims=1)
        return self._head(node1_features) if self._head else self._activation(node1_features)


class GraphNodeBinaryClassifyHead(SuperModel):
    """ Convenience function for binary classification with option to output one-hot with 2 separate classes
    Warning: Uses sigmoid / softmax activation as output
    """

    def __init__(self, expand_class=False, **kwargs):
        super(GraphNodeBinaryClassifyHead, self).__init__(**kwargs)
        self._head = keras.layers.Dense(1, activation="sigmoid", name="head")
        self._expand_class = expand_class

    def call(self, inputs, training=False):
        features = inputs[1]
        node1 = inputs[2]
        if len(node1.shape) != 1 and node1.shape[-1] == 1:
            node1 = tf.squeeze(node1, axis=-1)

        node1_features = tf.gather(features, node1, batch_dims=1)
        res = self._head(node1_features)
        if self._expand_class:
            anti_res = 1.0 - res
            categorical_res = tf.concat([anti_res, res], axis=-1)
            return categorical_res
        else:
            return res


class GraphReadoutSum(SuperModel):
    """Simple graph readout using sum / avg of all vertices
    """

    def __init__(self, use_mean=False, **kwargs):
        """Init fn.

        Args:
            use_mean: (Optional) Average after summation if True. Default is False
            **kwargs: (Optional) SuperModel base parameters
        """
        super(GraphReadoutSum, self).__init__(**kwargs)
        self._use_mean = use_mean

    def call(self, inputs, training=False):
        adjacency = inputs[0]
        features = inputs[1]

        # Here we sum the valid vertices
        # Actually, invalid vertices should already be 0s in the Tensor, but just being double sure...
        flattened_indices = tf.where(tf.reduce_all(tf.reduce_all(adjacency < 0, axis=-1), axis=-1))
        zero_feats = tf.zeros(tf.concat([tf.shape(flattened_indices)[0:1], tf.shape(features)[-1:]], axis=-1),
                              dtype=features.dtype)
        res_features = tf.tensor_scatter_nd_update(features, flattened_indices, zero_feats)
        res = tf.reduce_sum(res_features, axis=1, keepdims=False)

        if self._use_mean:
            # Compute the total number of valid vertices by the adjacency tensor for averaging
            degree = tf.reduce_sum(tf.reduce_sum(tf.cast(adjacency >= 0, tf.int32), axis=-1), axis=-1)
            vertex_valid = tf.cast(tf.minimum(degree, 1), features.dtype)
            total_valid = tf.reduce_sum(vertex_valid, axis=-1, keepdims=True)
            res = res / tf.maximum(total_valid, 1)
        return res


class GraphReadoutMax(SuperModel):
    """Simple graph readout using max of all vertices
    """

    def __init__(self, **kwargs):
        """Init fn.

        Args:
            use_mean: (Optional) Average after summation if True. Default is False
            **kwargs: (Optional) SuperModel base parameters
        """
        super(GraphReadoutMax, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        adjacency = inputs[0]
        features = inputs[1]

        flattened_indices = tf.where(tf.reduce_all(tf.reduce_all(adjacency < 0, axis=-1), axis=-1))
        zero_feats = tf.zeros(tf.concat([tf.shape(flattened_indices)[0:1], tf.shape(features)[-1:]], axis=-1),
                              dtype=features.dtype)
        res_features = tf.tensor_scatter_nd_update(features, flattened_indices, zero_feats)
        return tf.reduce_max(res_features, axis=1, keepdims=False)
