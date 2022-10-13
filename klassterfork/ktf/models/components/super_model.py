import math
import weakref

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class SuperModel(keras.Model):
    """This class is the KTF utility base Model class that we will use to get
    some nice features automatically that is not available in keras.Model.

    As of now, one important feature that the SuperModel class gives is
    the ability to track and expose intermediate outputs of Layers defined within
    the SuperModel class. This allows easier building of skip-connections within
    composite models.

    Example:
    ```
        class DerivedSuperModel(SuperModel):
            def __init__(self):
                # Caveat: Must assign to an object instance variable for it to be exposed
                self._dense1 = Dense(128, name='dense1")
                self._dense2 = Dense(64)

            def call(self, input):
                x = self._dense1(input)
                return self._dense2(x)

        model = DerivedSuperModel()                # Create object derived from SuperModel    
        output = model(dataset)                    # Get output from model
        layer_output = model.get_output("dense1")  # Get output from some layer in model  
    ```

    This is also particularly helpful when using the model subclassing
    method in Tensorflow 2.X to create custom models in eager mode,
    since in such cases you cannot extract intermediate Tensors from arbitrary layers, as would
    be the typical case if one were using keras functional models.
    """

    def __init__(self, *args, **kwargs):
        super(SuperModel, self).__init__(*args, **kwargs)
        # We had to introduce this hack/bypass to force TF to
        # not wrap our dictionary with tracking capabilities
        object.__setattr__(self, "_layer_proxies", {})
        self._layer_outputs = {}

    def __setattr__(self, name, value):
        super(SuperModel, self).__setattr__(name, value)

        if isinstance(value, keras.layers.Layer):
            owner = self

            # Create proxy to actual Layer object
            class _LayerProxy:
                __slots__ = ["_layer", "__weakref__"]

                def __init__(self, layer_obj):
                    object.__setattr__(self, "_layer", layer_obj)

                def __getattribute__(self, name):
                    return getattr(object.__getattribute__(self, "_layer"), name)

                def __delattr__(self, name):
                    delattr(object.__getattribute__(self, "_layer"), name)

                def __setattr__(self, name, value):
                    setattr(object.__getattribute__(self, "_layer"), name, value)

                def __nonzero__(self):
                    return bool(object.__getattribute__(self, "_layer"))

                def __str__(self):
                    return str(object.__getattribute__(self, "_layer"))

                def __repr__(self):
                    return repr(object.__getattribute__(self, "_layer"))

                def __call__(self, *args, **kwargs):
                    # We store all results in the owner
                    layer = object.__getattribute__(self, "_layer")
                    res = layer(*args, **kwargs)
                    owner._layer_outputs[layer.name] = res
                    return res

            layer_proxy = object.__getattribute__(self, "_layer_proxies")
            layer_proxy[name] = _LayerProxy(value)

    def __getattribute__(self, name):
        try:
            layer_proxy = object.__getattribute__(self, "_layer_proxies")
            if layer_proxy and name in layer_proxy:
                return layer_proxy[name]
        except:
            pass

        value = super(SuperModel, self).__getattribute__(name)
        return value

    def __delattr__(self, name):
        try:
            layer_proxy = object.__getattribute__(self, "_layer_proxies")
            if layer_proxy and name in layer_proxy:
                del layer_proxy[name]
        except:
            pass

        super(SuperModel, self).__delattr__(name)

    def get_output(self, layer_name):
        """Get most recent output results of a layer.
        Note that the layer in question must:
            (1) have been executed at least once
            (2) have been assigned to an instance variable

        Args:
            layer_name: The name of the layer whose outputs to fetch

        Returns:
            The output Tensor(s) of the layer
        """
        return self._layer_outputs[layer_name]
