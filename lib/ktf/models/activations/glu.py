import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """ Gated Linear Unit(GLU) Activation following official PyTorch GLU function
    Refer to : https://pytorch.org/docs/stable/generated/torch.nn.functional.glu.html 

    Input is split in half along specified dimension, perform sigmoid on the second tensor and element wise multiplication
    with the first tensor. For a more detailed explanation, refer to: https://leimao.github.io/blog/Gated-Linear-Units/

    Adapted from : https://github.com/TensorSpeech/TensorFlowASR/blob/main/tensorflow_asr/models/activations/glu.py#L18
    """

    def __init__(self,
                 axis=-1,
                 name="glu",
                 **kwargs):
        super(GLU, self).__init__(name=name, **kwargs)
        """
        Args:
            axis : Axis to split input by.
        """
        self._axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self._axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)
