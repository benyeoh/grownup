import importlib
import pkgutil
import copy
import os

from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import hjson

import ktf


def _import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.iter_modules(package.__path__):
        full_name = package.__name__ + '.' + name
        if recursive and is_pkg:
            results.update(_import_submodules(full_name))
    return results


# We recursively import all subpackages in the ktf namespace
# We require this to "magically" resolve functions and classes defined in this ktf package
_import_submodules(ktf)


# We store all custom user-defined functions / classes mappings here
_global_dynamic_fns = {}


def export_config(fn):
    """We use this decorator to tag user-defined functions or classes that the
       DynamicConfig class should see when it resolves symbols.

    Args:
        fn: The function or class method that we want to expose for DynamicConfig to resolve

    Returns:
        fn: The original function `fn`
    """
    global _global_dynamic_fns

    assert fn.__name__ not in _global_dynamic_fns
    _global_dynamic_fns[fn.__name__] = fn
    return fn


class DynamicConfig:
    """This class is used to dynamically resolve a specific dictionary configuration format into actual objects
    that are used in model training.

    The dictionary format is as follows:

    ```
    [
        {
            "cache": ...       <-- Optional cached custom data
            "datasets": ...    <-- Specify datasets
            "model": ...       <-- Specify model to use 
            "loss": ...        <-- Loss function
            "optimizer": ...   <-- Optimizer to use 
            "train_loop": ...  <-- Training loop
        },

        ... <-- Optionally define more than 1 config
    ]
    ```

    Example:
    ```
    [
        {
            "cache": { "my_string": "data/affnist" }
            "datasets": { "ktf.datasets.affnist.from_tfrecord": { "affnist_tfrecord_dir": "${self.get_cache('my_string')}" } },
            "model": { "ktf.models.OneHeadNet": { "num_outputs": 10 } },
            "loss": { "tf.keras.losses.SparseCategoricalCrossentropy": { "from_logits": true } },
            "optimizer": { "tf.keras.Adam": {} },
            "train_loop": { "ktf.train.KerasTrainLoop": {} }
        }
    ]
    ```

    For each key/value pair where the value is of type(dict), the key is assumed to be a python function or class constructor
    and will be resolved as such. Note that functions or constructors defined outside the scope of ktf or its imported packages
    cannot be resolved unless tagged with the @ktf.train.export_config decorator.
    """

    def __init__(self, config_list=None):
        """Initialization function.

        Args:
            config_list: (Optional) A dictionary specifying the configuration, a string specifying either a
                json-formatted configuration or a filepath to a json file. A superset of json that supports comments,
                hjson, is supported as well. 
        """
        if isinstance(config_list, str):
            self.set_config_from_json(config_list)
        else:
            self.set_config(config_list)

    def _eval(self, str):
        try:
            res = eval(str)
        except NameError as err:
            if str in _global_dynamic_fns:
                res = _global_dynamic_fns[str]
            else:
                raise err
        return res

    def _resolve_sub_objects(self, sub):
        if isinstance(sub, list):
            return [self._resolve_sub_objects(v) for v in sub]
        elif isinstance(sub, dict):
            assert len(sub) == 1
            model_fn_str, model_params_dict = next(iter(sub.items()))
            model_fn = self._eval(model_fn_str)
            assert model_fn is not None

            model_params_dict = self._resolve_parameters(model_params_dict)
            return model_fn(**model_params_dict)
        elif isinstance(sub, str):
            if sub.startswith("${") and sub.endswith("}"):
                return self._eval(sub[2:-1])
            else:
                return sub
        else:
            return sub

    def _resolve_parameters(self, params):
        assert isinstance(params, dict)
        resolved_params = {}
        for k, v in params.items():
            resolved_params[k] = self._resolve_sub_objects(v)
        return resolved_params

    def _get_object(self, name, index=0):
        for i in range(index, -1, -1):
            obj = self._config_list[i].get(name, None)
            res = self._resolve_sub_objects(obj)
            if res is not None:
                return res
        return None

    def _overwrite_config_recurse(self, d, key, val):
        if isinstance(d, (list, tuple)):
            for e in d:
                if isinstance(e, (dict, list, tuple)):
                    self._overwrite_config_recurse(e, key, val)
        elif isinstance(d, dict):
            for k, v in d.items():
                if k == key:
                    print("Overwriting %s = %s" % (k, val))
                    d[k] = val
                elif isinstance(v, (dict, list, tuple)):
                    self._overwrite_config_recurse(v, key, val)

    def set_config(self, config_list):
        if isinstance(config_list, dict):
            self._config_list = [config_list]
        else:
            self._config_list = config_list

        assert self._config_list is None or isinstance(self._config_list, list)
        self._cached_user_data = {}

    def set_config_from_json(self, json_str):
        if ((json_str.strip().startswith("{") and json_str.strip().endswith("}")) or
                (json_str.strip().startswith("[") and json_str.strip().endswith("]"))):
            # Assume it's json or hjson
            self.set_config(hjson.loads(json_str, object_pairs_hook=OrderedDict))
        else:
            # Assume it's a file path
            with open(json_str) as fd:
                config_list = hjson.load(fd, object_pairs_hook=OrderedDict)
                self.set_config(config_list)

    def dump_config(self, index=0):
        return hjson.dumps(self._config_list[index], indent=4)

    def copy_config(self):
        return copy.deepcopy(self._config_list)

    def get_num_entries(self):
        if self._config_list is None:
            return 0
        return len(self._config_list)

    def get_cache(self, name, index=0):
        model_spec = self._config_list[index]["cache"][name]
        if index not in self._cached_user_data:
            self._cached_user_data[index] = {}
        if name not in self._cached_user_data[index]:
            self._cached_user_data[index][name] = self._resolve_sub_objects(model_spec)
        return self._cached_user_data[index][name]

    def get_model(self, index=0):
        return self._get_object("model", index)

    def get_loss(self, index=0):
        return self._get_object("loss", index)

    def get_loss_weights(self, index=0):
        return self._get_object("loss_weights", index)

    def get_metrics(self, index=0):
        return self._get_object("metrics", index)

    def get_optimizer(self, index=0):
        res = self._get_object("optimizer", index)
        return res if res is not None else keras.optimizers.Adam()

    def get_datasets(self, index=0):
        return self._get_object("datasets", index)

    def get_train_loop(self, index=0):
        res = self._get_object("train_loop", index)
        return res if res is not None else KerasTrainLoop()

    def get_meta(self, index=0):
        return self._get_object("meta", index)

    def overwrite_config(self, config_name, key, val):
        """Overwrite the configuration list for the specified config object and
        parameter key

        Args:
            config_name: A string specifying the name of the
                configuration object (ie, "datasets", "model" etc)
            key: A string specifying the parameter key to overwrite
            val: The value to overwrite
        """
        for config in self._config_list:
            if config_name in config:
                self._overwrite_config_recurse(config[config_name], key, val)

    @staticmethod
    def wrap(fn, *args, **kwargs):
        """Convenience functor wrapper for DynamicConfig `get_<object_name>` functions.
        """
        def _factory():
            return fn(*args, **kwargs)
        return _factory
