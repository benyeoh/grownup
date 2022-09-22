import tensorflow as tf
import tensorflow.keras as keras

from .components import *
from .networks import *


class Row(SuperModel):
    """A wrapper model for defining row-wise (parallel) layers. This is useful when we want to define
    siamese networks or similar computations that are run in parallel rather than sequentially.
    """

    def __init__(self,
                 layers,
                 broadcast_inputs=False,
                 flatten_outputs=False,
                 **kwargs):
        """Initializes the model.

        Args:
            layers: A list of keras.layer.Layer objects.
            broadcast_inputs: (Optional) Broadcast the input to all layers if True. Default is False.
            flatten_outputs: (Optional) Flatten each output if it is a list or tuple. Moreover, if the output
                after flattening is a list of 1 element, then just output the element itself.

                Example if flatten_outputs is True:
                    [[x, y, z], [1, 2, 3]] -> [x, y, z, 1, 2, 3]
                    [[x]] -> [x] -> x

            **kwargs: (Optional) Arguments for base keras.layer.Layer.
        """
        super(Row, self).__init__(**kwargs)

        self._seq_layers = []
        self._broadcast_inputs = broadcast_inputs
        self._flatten_outputs = flatten_outputs
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        setattr(self, "_seq_layer%d" % len(self._seq_layers), layer)
        self._seq_layers.append(layer)

    def call(self, inputs, training=False):
        if not self._broadcast_inputs:
            if not isinstance(inputs, (list, tuple)):
                raise ValueError("An input list or tuple of length == %d expected. Got a scalar instead"
                                 % len(self._seq_layers))
            elif len(inputs) != len(self._seq_layers):
                raise ValueError("An input list or tuple of length == %d expected. Got %d instead"
                                 % (len(self._seq_layers), len(inputs)))
            x = inputs
        else:
            x = [inputs for _ in range(len(self._seq_layers))]

        res = []
        for i, layer in enumerate(self._seq_layers):
            if layer is not None:
                res.append(layer(x[i], training=training))
            else:
                # If None, then just pass-through the input
                res.append(x[i])

        if self._flatten_outputs:
            # Flatten list of lists
            flatten_res = []
            for elem in res:
                if isinstance(elem, (list, tuple)):
                    flatten_res.extend(elem)
                else:
                    flatten_res.append(elem)
            res = flatten_res

            # Automatically convert to scalar as well if list is len 1
            # TODO: Some use cases may not want this
            if len(res) == 1:
                res = res[0]
        return res
