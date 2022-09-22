import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers

from ..components import SuperModel


def _linear(input, weight, bias):
    """Applies a simple linear transformation to the input arguments: `y = input * weight^T + bias`"""
    return tf.nn.bias_add(tf.matmul(input, weight, transpose_b=True), bias)


def _in_projection_packed(q, k, v, w, b=None):
    """
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    See PyTorch implementation for shape information:
    https://github.com/pytorch/pytorch/blob/e86d8323cb872a835ba0845e3519079f25e204fc/torch/nn/functional.py#L4798

    Args:
        q, k, v: Query, Key and Value tensors to be projected. For self-attention, these are typically the same tensor;
            for encoder-decoder attention, k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v must share a common embedding
            dimension; otherwise their shapes may vary.
        w: Projection weights for q, k and v, packed into a single tensor. Weights are packed along dimension 0, in q,
            k, v order.
        b: (Optional) projection biases for q, k and v, packed into a single tensor in q, k, v order.

    Returns:
        A list `[q', k', v']`, where each output tensor will have the same shape as the corresponding input tensor
    """
    #E = q.shape[-1]
    E = tf.shape(q)[-1]
    if k is v:
        if q is k:
            # self-attention
            return tf.split(_linear(q, w, b), 3, axis=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = tf.split(w, [E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = tf.split(b, [E, E * 2])
            return (_linear(q, w_q, b_q),) + tf.split(_linear(k, w_kv, b_kv), 2, axis=-1)
    else:
        w_q, w_k, w_v = tf.split(w, 3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = tf.split(b, 3)
        return _linear(q, w_q, b_q), _linear(k, w_k, b_k), _linear(v, w_v, b_v)


def _in_projection(q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None):
    """
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    See PyTorch implementation for shape information:
    https://github.com/pytorch/pytorch/blob/e86d8323cb872a835ba0845e3519079f25e204fc/torch/nn/functional.py#L4854

    Args:
        q, k, v: Query, Key and Value tensors to be projected.
        w_q, w_k, w_v: Weights for q, k and v, respectively.
        b_q, b_k, b_v: (Optional) Biases for q, k and v, respectively.

    Returns: A tuple (q', k', v')
        q': `[Qdims..., Eq]`
        k': `[Kdims..., Eq]`
        v': `[Vdims..., Eq]`
    """
    #Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    #Eq, Ek, Ev = tf.shape(q)[-1], tf.shape(k)[-1], tf.shape(v)[-1]
    # assert w_q.shape == (Eq, Eq), "expecting query weights shape of {}, but got {}".format((Eq, Eq), {w_q.shape})
    # assert w_k.shape == (Eq, Ek), "expecting key weights shape of {}, but got {}".format((Eq, Ek), w_k.shape)
    # assert w_v.shape == (Eq, Ev), "expecting value weights shape of {}, but got {}".format((Eq, Ev), w_v.shape)
    # assert b_q is None or b_q.shape == (Eq,), "expecting query bias shape of {}, but got {}".format((Eq,), b_q.shape)
    # assert b_k is None or b_k.shape == (Eq,), "expecting key bias shape of {}, but got {}".format((Eq,), b_k.shape)
    # assert b_v is None or b_v.shape == (Eq,), "expecting value bias shape of {}, but got {}".format((Eq,), b_v.shape)
    return _linear(q, w_q, b_q), _linear(k, w_k, b_k), _linear(v, w_v, b_v)


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified. 
    Args:
        q, k, v: Query, Key and Value tensors. See Shape section for shape details
        attn_mask: (Optional) Tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details
        dropout_p: (Optional) Dropout probability
    Shapes:
        q: `(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        key: `(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        value: `(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        attn_mask: either a 3D tensor of shape `(B, Nt, Ns)` or a 2D tensor of
            shape `(Nt, Ns)`.
    Returns: A tuple of attended values and attention weights.
        (attention_values `(B, Nt, E)`, attention_weights `(B, Nt, Ns)`)
    """
    query_shape = tf.shape(q)
    B = query_shape[0]
    Nt = query_shape[1]
    E = query_shape[2]

    #B, Nt, E = q.shape
    q = q / tf.math.sqrt(tf.cast(E, tf.float32))
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
    if attn_mask is not None:
        attn += attn_mask
    attn = tf.nn.softmax(attn)
    if dropout_p > 0.0:
        attn = tf.nn.dropout(attn, dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = tf.matmul(attn, v)
    return output, attn


class MultiHeadAttention(keras.layers.Layer):
    """MultiHeadAttention from PyTorch, a key part of a Transformer.

    This exists for weight compatibility reasons as Tensorflow's MultiHeadAttention implementation differs from the
    PyTorch one in some areas. The weights in the TensorFlow implementation scale with the number of heads, whereas in
    PyTorch, the model's dimensionality is divided by the number of attention head and hence the shape of the weights
    are the same regardless of the number of heads. 

    Paper: https://arxiv.org/abs/1706.03762
    Implementation based on: https://github.com/pytorch/pytorch/blob/e86d8323cb872a835ba0845e3519079f25e204fc/torch/nn/modules/activation.py#L862
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, name=None):
        """
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads. Note that `embed_dim` will be split across `num_heads`
                (i.e. each head will have dimension `embed_dim // num_heads`)
            dropout: (Optional) Dropout probability on `attn_output_weights`
            bias: (Optional) If specified, adds bias to the input / output projection layers
            add_bias_kv: (Optional) If specified, adds bias to the key and value sequences at dim=0
            add_zero_attn: (Optional) If specified, adds a new batch of zeros to the key and value sequences at dim=1
            kdim: (Optional) Total number of features for keys. Will use embedding_dim if not provided
            vdim: (Optional) Total number of features for values. Will use embedding_dim if not provided
            batch_first: (Optional) If `True`, then the input and output tensors are provided
                as (batch, seq, feature). Otherwise they will be returned as (seq, batch, feature)
            name: (Optional) Name for the instantiated layer
        """
        super(MultiHeadAttention, self).__init__(name=name)
        name = name + "/" if name is not None else ""
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._dropout = dropout
        self._bias = bias
        self._add_bias_kv = add_bias_kv
        self._add_zero_attn = add_zero_attn
        self._kdim = kdim if kdim is not None else embed_dim
        self._vdim = vdim if vdim is not None else embed_dim
        self._batch_first = batch_first
        self._qkv_same_embed_dim = self._kdim == embed_dim and self._vdim == embed_dim

        self._head_dim = embed_dim // num_heads
        assert self._head_dim * num_heads == self._embed_dim, "embed_dim must be divisible by num_heads"

        # Create in_proj weights
        xavier_uniform_init = keras.initializers.GlorotUniform()
        xavier_normal = keras.initializers.GlorotNormal()
        zeros_init = keras.initializers.Zeros()
        if self._qkv_same_embed_dim is False:
            self._q_proj_weight = tf.Variable(xavier_uniform_init((embed_dim, embed_dim)),
                                              name=name + "q_proj/weight")
            self._k_proj_weight = tf.Variable(xavier_uniform_init((embed_dim, self._kdim)),
                                              name=name + "k_proj/weight")
            self._v_proj_weight = tf.Variable(xavier_uniform_init((embed_dim, self._vdim)),
                                              name=name + "v_proj/weight")
            self._in_proj_weight = None
        else:
            self._q_proj_weight = None
            self._k_proj_weight = None
            self._v_proj_weight = None
            self._in_proj_weight = tf.Variable(xavier_uniform_init((3 * embed_dim, embed_dim)),
                                               name=name + "in_proj_weight")

        # Create out_proj weights
        self._out_proj_weight = tf.Variable(xavier_uniform_init((embed_dim, embed_dim)),
                                            name=name + "out_proj/weight")
        if bias:
            self._in_proj_bias = tf.Variable(zeros_init((3 * embed_dim,)),
                                             name=name + "in_proj_bias")
            self._out_proj_bias = tf.Variable(xavier_uniform_init((embed_dim,)),
                                              name=name + "out_proj/bias")
        else:
            self._in_proj_bias = None
            self._out_proj_bias = None

        if add_bias_kv:
            self._bias_k = tf.Variable(xavier_normal((embed_dim, embed_dim)))
            self._bias_v = tf.Variable(zeros_init((embed_dim,)))
        else:
            self._bias_k = self._bias_v = None

    @staticmethod
    def multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias,
                                     bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
                                     training=False, key_padding_mask=None, need_weights=True, attn_mask=None,
                                     use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                     v_proj_weight=None, static_k=None, static_v=None):
        # Set up shape variables
        query_shape = tf.shape(query)
        tgt_len = query_shape[0]
        batch_size = query_shape[1]
        embed_dim = query_shape[2]
        src_len = tf.shape(key)[0]

        head_dim = embed_dim // num_heads
        # assert head_dim * num_heads == embed_dim, \
        #     "embed_dim {} not divisible by num_heads {}".format(embed_dim, num_heads)
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert tuple(key.shape[:2]) == tuple(value.shape[:2]), \
                "key's sequence and batch dims {} do not match value's {}".format(key.shape[:2], value.shape[:2])
        else:
            assert tuple(key.shape) == tuple(
                value.shape), "key shape {} does not match value shape {}".format(key.shape, value.shape)

        # Compute in-projection
        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = tf.split(in_proj_bias, 3)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # Prep attention mask
        if attn_mask is not None:
            # Ensure attn_mask's dim is 3
            if len(attn_mask.shape) == 2:
                # correct_2d_size = (tgt_len, src_len)
                # if tuple(attn_mask.shape) != correct_2d_size:
                #     raise RuntimeError("The shape of the 2D attn_mask is {}, but should be {}".format(
                #         attn_mask.shape, correct_2d_size))
                attn_mask = tf.expand_dims(attn_mask, 0)
            elif len(attn_mask.shape) > 3:
                #     correct_2d_size = (tgt_len, src_len)
                #     correct_2d_size2 = (1, src_len) # Allow broadcasting
                #     correct_2d_size3 = (1, None) # Allow broadcasting
                #     if ((tuple(attn_mask.shape[1:]) != correct_2d_size) and
                #         (tuple(attn_mask.shape[1:]) != correct_2d_size2) and
                #         (tuple(attn_mask.shape[1:]) != correct_2d_size3)):
                #         raise RuntimeError("The shape of the 3D attn_mask is {}, but should be {}, or {}, or {}".format(
                #             attn_mask.shape, correct_2d_size, correct_2d_size2, correct_2d_size3))
                # else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(len(attn_mask.shape)))

        # Prep key padding mask
        if key_padding_mask is not None:
            # Convert key_padding_mask to tf.bool
            key_padding_mask = tf.cast(key_padding_mask, tf.bool)

        # Add bias along the batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = tf.concat([k, tf.tile(bias_k, (1, batch_size, 1))], 0)
            v = tf.concat([v, tf.tile(bias_v, (1, batch_size, 1))], 0)
            if attn_mask is not None:
                attn_mask = tf.pad(attn_mask, [[0, 0], [0, 1]])
            if key_padding_mask is not None:
                key_padding_mask = tf.pad(attn_mask, [[0, 0], [0, 1]])
        else:
            assert bias_k is None
            assert bias_v is None

        # Reshape q, k, v for multihead attention and make them batch first
        q = tf.transpose(tf.reshape(q, (tgt_len, batch_size * num_heads, head_dim)), perm=[1, 0, 2])
        if static_k is None:
            k = tf.transpose(tf.reshape(k, (k.shape[0], batch_size * num_heads, head_dim)), perm=[1, 0, 2])
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            # assert static_k.shape[0] == batch_size * num_heads, \
            #     "expecting static_k.size(0) of {}, but got {}".format(batch_size * num_heads, static_k.shape[0])
            # assert static_k.shape[2] == head_dim, \
            #     "expecting static_k.size(2) of {}, but got {}".format(head_dim, static_k.shape[2])
            k = static_k
        if static_v is None:
            v = tf.transpose(tf.reshape(v, (v.shape[0], batch_size * num_heads, head_dim)), perm=[1, 0, 2])
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            # assert static_v.shape[0] == batch_size * num_heads, \
            #     "expecting static_v.size(0) of {}, but got {}".format(batch_size * num_heads, static_v.shape[0])
            # assert static_v.shape[2] == head_dim, \
            #     "expecting static_v.size(2) of {}, but got {}".format(head_dim, static_v.shape[2])
            v = static_v

        # Add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (batch_size * num_heads, 1, head_dim)
            k = tf.concat([k, tf.zeros(zero_attn_shape, dtype=k.dtype)], 1)
            v = tf.concat([v, tf.zeros(zero_attn_shape, dtype=v.dtype)], 1)
            if attn_mask is not None:
                attn_mask = tf.pad(attn_mask, [[0, 0], [0, 1]])
            if key_padding_mask is not None:
                key_padding_mask = tf.pad(attn_mask, [[0, 0], [0, 1]])

        # Update source sequence length after adjustments
        src_len = tf.shape(k)[1]

        # merge key padding and attention masks
        if key_padding_mask is not None:
            # assert key_padding_mask.shape == (batch_size, src_len), \
            #     "expecting key_padding_mask shape of {}, but got {}".format((batch_size, src_len),
            #                                                                 key_padding_mask.shape)
            key_padding_mask = tf.reshape(key_padding_mask, (batch_size, 1, 1, src_len))
            kpm_shape = key_padding_mask.shape
            key_padding_mask = tf.broadcast_to(key_padding_mask, (kpm_shape[0], num_heads, kpm_shape[2], kpm_shape[3]))
            key_padding_mask = tf.reshape(key_padding_mask, (batch_size * num_heads, 1, src_len))
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype.is_bool:
                attn_mask = tf.math.logical_or(attn_mask, tf.cast(key_padding_mask, dtype=tf.bool))
            else:
                attn_mask = tf.where(attn_mask, x=key_padding_mask, y=attn_mask)

        # Convert mask to float
        if attn_mask is not None and attn_mask.dtype.is_bool:
            new_attn_mask = tf.where(attn_mask,
                                     x=tf.zeros_like(attn_mask, dtype=tf.float32),
                                     y=tf.fill(tf.shape(attn_mask), -np.inf))
            attn_mask = new_attn_mask

        # Adjust dropout probability
        if not training:
            dropout_p = 0.0

        # Calculate attention and out projection
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = tf.reshape(tf.transpose(attn_output, perm=[1, 0, 2]), (tgt_len, batch_size, embed_dim))
        attn_output = _linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # Average attention weights over heads
            attn_output_weights = tf.reshape(attn_output_weights, (batch_size, num_heads, tgt_len, src_len))
            return attn_output, tf.reduce_sum(attn_output_weights, axis=1) / num_heads
        else:
            return attn_output, None

    def call(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, training=False):
        """
        Args:
            query: Query embeddings of shape `(L, N, E)` when `batch_first=False` or `(N, L, E)` when
                `batch_first=True`, where `L` is the target sequence length, `N` is the batch size, and `E` is the query
                embedding dimension `embed_dim`. Queries are compared against key-value pairs to produce the output
            key: Key embeddings of shape `(S, N, E)` when `batch_first=False` or `(N, S, E)` when `batch_first=True`,
                where `S` is the source sequence length, `N` is the batch size, and `E` is the key embedding dimension `kdim`
            value: Value embeddings of shape `(S, N, E)` when `batch_first=False` or `(N, S, E)` when
                `batch_first=True`, where `S` is the source sequence length, `N` is the batch size, and
                `E` is the value embedding dimension `vdim`
            key_padding_mask: A mask of shape `(N, S)` indicating which elements within `key` to ignore for the purpose
                of attention (i.e. treat as "padding"). Binary and byte masks are supported. For a binary mask, a `True`
                value indicates that the corresponding `key` value will be ignored for the purpose of attention.
                For a byte mask, a non-zero value indicates that the corresponding `key` value will be ignored
            need_weights: Whether to return `attn_output_weights` in addition to `attn_outputs`
            attn_mask: A 2D or 3D mask preventing attention to certain positions. Must be of shape `(L, S)` or
                `(N * num_heads, L, S)`, where `N` is the batch size, `L` is the target sequence length, and `S` is the
                source sequence length. See PyTorch documentation for more details

        Returns:
            attn_output: Attention outputs of shape `(L, N, E)` when `batch_first=False` or `(N, L, E)` when
                `batch_first=True`, where `L` is the target sequence length, `N` is the batch size, and `E` is the
                embedding dimension `embed_dim`
            attn_output_weights: Attention output weights of shape `(N, L, S)`, where `N` is the batch size, `L` is the
                target sequence length, and `S` is the source sequence length. Only returned if `need_weights=True`
        """
        if self._batch_first:
            query, key, value = [tf.transpose(x, perm=[1, 0, 2]) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = MultiHeadAttention.multi_head_attention_forward(
                query, key, value, self._num_heads,
                self._in_proj_weight, self._in_proj_bias,
                self._bias_k, self._bias_v, self._add_zero_attn,
                self._dropout, self._out_proj_weight, self._out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self._q_proj_weight, k_proj_weight=self._k_proj_weight,
                v_proj_weight=self._v_proj_weight)
        else:
            attn_output, attn_output_weights = MultiHeadAttention.multi_head_attention_forward(
                query, key, value, self._num_heads,
                self._in_proj_weight, self._in_proj_bias,
                self._bias_k, self._bias_v, self._add_zero_attn,
                self._dropout, self._out_proj_weight, self._out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self._batch_first:
            return tf.transpose(attn_output, perm=[1, 0, 2]), attn_output_weights
        else:
            return attn_output, attn_output_weights


class QuickGELU(keras.layers.Layer):
    """The QuickGELU layer used in the ResidualAttentionBlock
    Implementation reference: https://github.com/openai/CLIP/blob/e184f608c5d5e58165682f7c332c3a8b4c1545f2/clip/model.py#L162-L164
    """

    def call(self, input):
        return input * tf.sigmoid(1.702 * input)


class ResidualAttentionBlock(SuperModel):
    """A Residual Attention Block used in GPT-2. Defined as just 'block' in the original implementation at
    https://github.com/openai/gpt-2/blob/master/src/model.py#L123 but given the current name when it was re-implemented
    for CLIP at https://github.com/openai/CLIP/blob/main/clip/model.py#L167
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.0, attn_mask=None, batch_first=False, name=None):
        """
        Args:
            embedding_dim: Length of the embedding
            num_heads: Number of heads for the `keras.layers.MultiHeadAttention` layer used in this block
            name: (Optional) Name for the instantiated block
        """
        super(ResidualAttentionBlock, self).__init__(name=name)
        name = name + "/" if name is not None else ""
        self._mha = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout,
                                       batch_first=batch_first, name=name + "attn")
        self._ln_1 = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="ln_1")
        self._c_fc = keras.layers.Dense(embedding_dim * 4, name="mlp/c_fc")
        self._gelu = QuickGELU()
        self._c_proj = keras.layers.Dense(embedding_dim, name="mlp/c_proj")
        self._ln_2 = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="ln_2")
        self._attn_mask = attn_mask

    def _mlp(self, x):
        x = self._c_fc(x)
        x = self._gelu(x)
        x = self._c_proj(x)
        return x

    def _attention(self, x, attn_mask, training):
        x, _ = self._mha(x, x, x, attn_mask=attn_mask, need_weights=False, training=training)
        return x

    def call(self, input, attn_mask=None, training=False):
        if attn_mask is None:
            attn_mask = self._attn_mask
        x = input + self._attention(self._ln_1(input), attn_mask=attn_mask, training=training)
        x = x + self._mlp(self._ln_2(x))
        return x
