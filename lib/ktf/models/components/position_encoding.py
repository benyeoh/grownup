import tensorflow as tf
import math
from .super_model import SuperModel


class PositionEmbeddingSine(SuperModel):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        Args:
            num_pos_feats: The feature dimension for each position along x-axis or y-axis. Note the final returned 
                dimension for each position is 2 times of this value
            temperature: The temperature used for scaling the position embedding
            normalize: Whether to normalize the position embedding. Defaults to False.
            scale (float, optional): A scale factor that scales the position embedding. The scale will be used only
                when `normalize` is True. Defaults to 2*pi.
        """
        super(PositionEmbeddingSine, self).__init__()
        self._num_pos_feats = num_pos_feats
        self._temperature = temperature
        self._normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self._scale = scale

    def call(self, input, training=None):
        not_mask = tf.ones(tf.shape(input)[:-1])  # (N,H,W)

        y_embed = tf.math.cumsum(not_mask, axis=1)  # (N,H,W)
        x_embed = tf.math.cumsum(not_mask, axis=2)  # (N,H,W)
        if self._normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self._scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self._scale

        dim_t = tf.range(self._num_pos_feats, dtype=tf.float32)  # (self._num_pos_feats)
        dim_t = self._temperature ** (2 * (dim_t // 2) / self._num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (N,H,W,self._num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t  # (N,H,W,self._num_pos_feats)
        pos_x = tf.reshape(tf.stack((tf.math.sin(pos_x[:, :, :, 0::2]), tf.math.cos(pos_x[:, :, :, 1::2])),
                                    axis=4),
                           tf.shape(pos_x))
        pos_y = tf.reshape(tf.stack((tf.math.sin(pos_y[:, :, :, 0::2]), tf.math.cos(pos_y[:, :, :, 1::2])),
                                    axis=4),
                           tf.shape(pos_y))
        pos = tf.concat((pos_y, pos_x), axis=3)  # (N,H,W,self._num_pos_feats*2)
        return pos
