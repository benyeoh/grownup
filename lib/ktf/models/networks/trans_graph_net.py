import tensorflow as tf
import tensorflow.keras as keras

from ktf.models.components import (SuperModel, ConvGraphSelfLoop, ResidualAttentionBlock)


class TransGraphNet(SuperModel):

    def __init__(self,
                 num_layers,
                 embedding_size=512,
                 num_heads=8,
                 dropout=0.0,
                 **kwargs):
        """Initialize the network.

        Args:
            num_layers: An integer representing the number of transformer layers.
            embedding_size: (Optional) The size of the transformer embeddings. Default is 512
            num_heads: (Optional) Number of attention heads for the transformer embeddings. Default is 8.
            dropout: (Optional) Dropout rate. Default is 0.
            **kwargs: (Optional) Optional arguments
        """
        super(TransGraphNet, self).__init__(**kwargs)

        self._prelim = ConvGraphSelfLoop(embedding_size, use_bias=True, activation=None)

        self._blocks = []
        for i in range(num_layers):
            block = ResidualAttentionBlock(embedding_size, num_heads, dropout=dropout,
                                           batch_first=True, name="attn_block%d" % i)
            self._blocks.append(block)

        self._num_heads = num_heads
        self._cls_embeddings = self.add_weight(
            'cls_embeddings',
            shape=[embedding_size],
            initializer=keras.initializers.get("glorot_uniform"),
            trainable=True)

    def _compute(self, inputs, training=False):
        adjacency = inputs[0]
        features = inputs[1]

        # Resample to embedding size
        features = self._prelim((adjacency, features), training=training)

        shape = tf.shape(features)
        batch_size = shape[0]
        max_vertices = shape[1]

        # attn_mask = [B, N], attn_mask_w_cls = [B, N + 1]
        attn_mask = tf.reduce_any(tf.reduce_any(adjacency >= 0, axis=-1), axis=-1)
        attn_mask_w_cls = tf.concat([attn_mask, tf.ones([tf.shape(attn_mask)[0], 1], dtype=tf.bool)], axis=-1)

        # attn_mask_heads = [B * H, 1, N + 1]
        attn_mask_w_cls = tf.reshape(tf.tile(tf.expand_dims(attn_mask_w_cls, axis=1), [1, self._num_heads, 1]),
                                     [batch_size * self._num_heads, 1, max_vertices + 1])

        # Shape = [B, N + 1, F]
        tiled_cls = tf.tile(tf.reshape(self._cls_embeddings, [
                            1, 1, self._cls_embeddings.shape[-1]]), [batch_size, 1, 1])
        features_w_cls = tf.concat([features, tiled_cls], axis=1)

        for block in self._blocks:
            features_w_cls = block(features_w_cls, attn_mask=attn_mask_w_cls, training=training)

        features = features_w_cls[:, :-1]
        cls_feats = features_w_cls[:, -1]
        return (adjacency, features) + tuple(inputs[2:]) + (cls_feats,)

    def call(self, inputs, training=False):
        return self._compute(inputs, training=training)
