import os

import tensorflow as tf
import tensorflow.keras as keras


def load_weights(model, path):
    """Convenience function to load model weights for use in dynamic config

    Args:
        model: A keras.Model object
        path: Path to the .h5 or .tf weights

    Returns:
        The same keras.Model object
    """
    model.load_weights(path)
    return model
