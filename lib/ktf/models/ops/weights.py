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


def pretrained_check_hpc_mount():
    """Convenience function for ensuring /hpc-datasets is mounted prior to loading pretrained weights"""
    if not os.path.ismount(os.sep + "hpc-datasets"):
        msg = "For automatic loading of pre-trained weights, the hpc-datasets DFS must be mounted at `/hpc-datasets` "
        msg += "otherwise the checkpoint path has to be specified explicitly.\nFor instructions on mounting "
        msg += "`/hpc-datasets`, see https://confluence.internal.klass.dev/x/H48eAQ"
        raise Exception(msg)
