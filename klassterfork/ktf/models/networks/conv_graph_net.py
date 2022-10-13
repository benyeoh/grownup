import tensorflow as tf
import tensorflow.keras as keras


from ktf.models.components import (SuperModel, ConvGraphNetBlock, ConvGraphSelfLoop)
import ktf.models.ops.cache_conv_graph as cache_conv_graph
import ktf.models.ops.cache_common as cache_common
from ktf.models.ops.compute_cache import ComputeCache


class RecurrentCGN(SuperModel):
    """A graph convolution / GraphSAGE-based network that generates node embeddings for a graph, with
    optional residual computations similar to ResNet. Also uses layer normalizations instead of batch normalizations.

    This model also adds a recurrent layer to facilitate processing of larger contexts.
    """

    def __init__(self,
                 layers,
                 output_feature_size=None,
                 hidden_base_feature_size=128,
                 use_residuals=True,
                 normalization="layer",
                 dropout=0.0,
                 aggregator="mean",
                 recurrent_cell="lstm",
                 recurrent_activation="tanh",
                 aggregator_activation="relu",
                 output_activation="relu",
                 num_prelim_blocks=0,
                 debug_self_loops=None,
                 use_compute_cache=True,
                 **kwargs):
        """Initialize the network.

        Args:
            layers: A list of integers representing the number of blocks per layer. Ex, a list [3, 1] would
                create a network that has 2 convolution layers with 3 blocks in the 1st layer and 1 block in
                the second layer. The number of convolution layers would correspond to the maximum number of hops
                from a node's neighborhood that the node can aggregate from.
            output_feature_size: (Optional) The size of the output feature. If None, the output will be
                the final convolution layer
            hidden_base_feature_size: (Optional) The size of the hidden layer features. Default is 128.
            use_residuals: (Optional) Use residual computations. Default is True.
            normalization: (Optional) Use "layer", "batch" or "group[2|4|8]" normalization. Default is "layer".
            aggregator: (Optional) The aggregator to use for graph convolutions. Possible entries are
                "sum", "mean", "maxpool", "gs_sum", "gs_mean", "gs_maxpool",
                "gat_mean", "gat_sum", "gat_mean_grp[2|4|8]", "gat_sum_grp[2|4|8]" and "gat_gated". Default is "mean".
            recurrent_cell: (Optional) Recurrent cell type. Either "lstm" or "gru". Default is "lstm"
            recurrent_activation: (Optional) Keras activation function for the recurrent cell network. Default is "tanh"
            aggregator_activation: (Optional) Keras activation function for the network. Default is "relu"
            num_prelim_blocks: (Optional) The number of self-convolution blocks to apply to process inputs before
                passing downstream to the rest of the network. Default is 0.
            dropout: (Optional) Dropout rate. Default is 0.
            use_compute_cache: (Optional) A bool or dictionary of parameters for the ComputeCache. Default is True.
            **kwargs: (Optional) Optional arguments
        """
        super(RecurrentCGN, self).__init__(**kwargs)

        self._use_compute_cache = use_compute_cache

        if isinstance(recurrent_activation, str):
            if recurrent_activation == "leaky_relu":
                recurrent_activation = keras.layers.LeakyReLU(name="leaky_relu")

        if recurrent_cell == "lstm":
            self._rcell = keras.layers.LSTMCell(hidden_base_feature_size,
                                                activation=recurrent_activation)
        elif recurrent_cell == "gru":
            self._rcell = keras.layers.GRUCell(hidden_base_feature_size,
                                               activation=recurrent_activation)
        else:
            raise ValueError("Valid types are 'lstm' and 'gru'.")

        self._prelim_blocks = []
        if not isinstance(num_prelim_blocks, (list, tuple)):
            num_prelim_blocks = [hidden_base_feature_size] * int(num_prelim_blocks)
        for i, feat_size in enumerate(num_prelim_blocks):
            self._prelim_blocks.append(ConvGraphNetBlock(feat_size,
                                                         self_loops_only=True,
                                                         normalization=normalization,
                                                         use_residuals=use_residuals,
                                                         dropout=dropout,
                                                         activation=aggregator_activation,
                                                         name="prelim_block%d" % i))
        self._blocks = []
        for i, l in enumerate(layers):
            block = [ConvGraphNetBlock(hidden_base_feature_size,
                                       self_loops_only=False if (debug_self_loops is None
                                                                 or i < debug_self_loops) else True,
                                       normalization=normalization,
                                       use_residuals=use_residuals,
                                       dropout=dropout,
                                       aggregator=aggregator,
                                       activation=aggregator_activation,
                                       name="layer%d_block0" % i)]
            for j in range(1, l):
                block.append(ConvGraphNetBlock(hidden_base_feature_size,
                                               self_loops_only=True,
                                               normalization=normalization,
                                               use_residuals=use_residuals,
                                               dropout=dropout,
                                               activation=aggregator_activation,
                                               name="layer%d_block%d" % (i, j)))
            self._blocks.append(block)

        if output_feature_size is not None:
            self._feature_block = ConvGraphSelfLoop(output_feature_size,
                                                    use_bias=True,
                                                    activation=output_activation)
        else:
            self._feature_block = None

    def _compute(self, inputs, training=False):
        adjacency = inputs[0]
        features = inputs[1]
        valid_indices = cache_conv_graph.get_valid_indices_from_adj(adjacency)

        for block in self._prelim_blocks:
            features = block((adjacency, features), training=training)

        rstate = self._rcell.get_initial_state(batch_size=tf.shape(valid_indices)[0], dtype=features.dtype)

        for block in self._blocks:
            for conv_graph_blk in block:
                features = conv_graph_blk((adjacency, features), training=training)

            valid_features = cache_common.gather_nd(features, valid_indices)

            flattened_features, rstate = self._rcell(valid_features, rstate, training=training)
            valid_features = flattened_features
            features = tf.scatter_nd(valid_indices, valid_features, tf.cast(tf.shape(features), tf.int64))
            ComputeCache.assign(cache_common.gather_nd, (features, valid_indices), valid_features)

        if self._feature_block:
            features = self._feature_block((adjacency, features), training=training)

        return (adjacency, features) + tuple(inputs[2:])

    def call(self, inputs, training=False):
        if self._use_compute_cache:
            kwargs = self._use_compute_cache if isinstance(self._use_compute_cache, dict) else {}
            with ComputeCache.push_context(**kwargs):
                return self._compute(inputs, training=training)
        else:
            return self._compute(inputs, training=training)
