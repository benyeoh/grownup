import re

import tensorflow as tf


def diff_lr_wrap(optimizer_class, lr_list, debug_info=False, **kwargs):
    """Differential learning rate AKA discriminative learning rate wrapper
    for keras compatible optimizers.

    Example usage (using a dictionary definition to be parsed by DynamicConfig):
        {
            ...

            "optimizer": {
                "ktf.train.optimizers.diff_lr_wrap": {
                    "optimizer_class": "${tf.keras.optimizers.Adam}",
                    "learning_rate": 0.001,
                    "lr_list": [
                        ["one_head_net/base_model/conv2d/.*", 0.0],
                        ["one_head_net/base_model/batch_normalization/.*", 0.05],
                        ["res_net_basic_block/.*", 0.1],
                        ["res_net_basic_block_1/.*", 0.2],
                        ["res_net_basic_block_2/.*", 0.3],
                        ["res_net_basic_block_3/.*", 0.4],
                        ["res_net_basic_block_4/.*", 0.5],
                        ["res_net_basic_block_5/.*", 0.6],
                        ["res_net_basic_block_6/.*", 0.7],
                        ["res_net_basic_block_7/.*", 0.8],
                        ["one_head_net/base_model/dense/.*", 0.9],
                        ["one_head_net/head/.*", 1.0]
                    ]
                }
            }
        }

    See the unit test `tests/diff_lr.py` for a runtime example.    

    When defining the mapping of learning rates to apply to the layers, it is helpful to run `scripts/print_model.py`
    to dump all the layer names and weight names (in the order of creation). Then you can use that to 
    define appropriate regex / learning rate mappings.

    Args:
        optimizer_class: The actual optimizer class to wrap and apply differential learning rates
        lr_list: A list of (<regex_string_match>, <lr_factor>) pairs to apply. When determining the
            lr_factor to apply to a particular weight, the first regex expression entry completely matching
            the weight name is used.
        debug_info: (Optional) Prints debugging information showing the learning rates applied to each weight
            and also weights without any regex matches. Useful when you need to confirm weights
            are correctly applied.
        **kwargs: (Optional) Additional arguments that will be passed to the actual optimizer instance.
    Returns:
        A wrapped keras-compatible optimizer with differential learning rates
    """

    class _DiffLRWrapper(optimizer_class):
        def __init__(self,
                     diff_lr_list,
                     name=optimizer_class.__name__ + "_diff_lr",
                     **kwargs):
            super(_DiffLRWrapper, self).__init__(name=name, **kwargs)
            self._re_text_val = [(re.compile(k), k, v) for k, v in diff_lr_list]

        def _find_rate(self, name):
            for k, t, v in self._re_text_val:
                if k.fullmatch(name):
                    if debug_info:
                        print("Match Diff LR: %f to %s - [%s]" % (v, name, t))
                    return v
            if debug_info:
                print("Unmatched Diff LR: %s" % name)
            return 1.0

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            if not apply_state:
                apply_state = {(var_device, var_dtype): {}}
                self._prepare_local(var_device, var_dtype, apply_state)
            apply_state_cp = {
                (var_device, var_dtype): {k: v for k, v in apply_state[(var_device, var_dtype)].items()}
            }
            lr_t = apply_state_cp[(var_device, var_dtype)]["lr_t"] * self._find_rate(var.name)
            apply_state_cp[(var_device, var_dtype)]["lr_t"] = lr_t
            return super(_DiffLRWrapper, self)._resource_apply_dense(grad, var, apply_state_cp)

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            if not apply_state:
                apply_state = {(var_device, var_dtype): {}}
                self._prepare_local(var_device, var_dtype, apply_state)
            apply_state_cp = {
                (var_device, var_dtype): {k: v for k, v in apply_state[(var_device, var_dtype)].items()}
            }
            lr_t = apply_state_cp[(var_device, var_dtype)]["lr_t"] * self._find_rate(var.name)
            apply_state_cp[(var_device, var_dtype)]["lr_t"] = lr_t
            return super(_DiffLRWrapper, self)._resource_apply_sparse(grad, var, indices, apply_state_cp)

        def get_config(self):
            config = super(_DiffLRWrapper, self).get_config()
            config.update({
                'diff_lr_list': [(t, v) for _, t, v in self._re_text_val],
            })
            return config

    return _DiffLRWrapper(diff_lr_list=lr_list, **kwargs)
