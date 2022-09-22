import tensorflow as tf
import tensorflow.keras as keras

from .components import *
from .networks import *


class Sequential(SuperModel):
    """A simplified wrapper model for defining sequential layers.
    This differs from keras.Sequential in that it supports multiple inputs and outputs for each layer, and
    through the SuperModel base class also exposes the output of each layer.s
    """

    def __init__(self,
                 layers,
                 **kwargs):
        """Initializes the model.

        Args:
            layers: A list of keras.layer.Layer objects.
            **kwargs: (Optional) Arguments for base keras.layer.Layer.
        """
        super(Sequential, self).__init__(**kwargs)

        self._num_layers = 0
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        setattr(self, "_seq_layer%d" % self._num_layers, layer)
        self._num_layers += 1

    def call(self, input, **kwargs):
        x = input
        for i in range(self._num_layers):
            layer = getattr(self, "_seq_layer%d" % i)
            x = layer(x, **kwargs)
        return x
