import math

import numpy as np
import tensorflow as tf

from .weighted_loss import weighted_loss_wrap, CategoricalClassWeights, SparseCategoricalClassWeights


# Convenience alias for weighted CategoricalCrossentropy
CategoricalCrossentropyW = (lambda class_weights, **kwargs:
                            weighted_loss_wrap(tf.keras.losses.CategoricalCrossentropy,
                                               sample_weight_fn=CategoricalClassWeights(
                                                   class_weights),
                                               **kwargs))

SparseCategoricalCrossentropyW = (lambda class_weights, **kwargs:
                                  weighted_loss_wrap(tf.keras.losses.SparseCategoricalCrossentropy,
                                                     sample_weight_fn=SparseCategoricalClassWeights(
                                                         class_weights),
                                                     **kwargs))
