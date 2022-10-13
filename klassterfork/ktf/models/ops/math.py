import tensorflow as tf


def _is_power_of_2(n):
    # refer to https://stackoverflow.com/a/600306 for how this works
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def inverse_sigmoid(x, eps=1e-5):
    """ Adopted from https://github.com/fundamentalvision/Deformable-DETR/blob/main/util/misc.py#L513
    """
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    x1 = tf.clip_by_value(x, clip_value_min=eps, clip_value_max=1)
    x2 = tf.clip_by_value(1 - x, clip_value_min=eps, clip_value_max=1)
    return tf.math.log(x1 / x2)
