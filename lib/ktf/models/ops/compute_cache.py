import types
from contextlib import contextmanager


class ComputeCache:
    """This class allows the convenient re-use of computations between layers without
    manually passing (optional) parameters or Tensors between layer call()'s.

    For example in graph computation layers, the input features are often padded to
    the maximum size allowed in a batch and computations are often repeated to extract
    the "valid" features from the padded inputs.

    One way to handle that would be to pass optional parameters between layer calls:

    ```python
    def call(inputs, valid_indices=None, valid_features=None):        
        if not valid_indices:
            valid_indices = ... do something with inputs ...
        if not valid_features:
            valid_features =  ... do something with inputs again ...
        ...
    ```

    However this gets quickly messy when layers are deep or when the number of such
    potentially re-useable computations becomes larger, or when you want to mix and
    match layers that may or may not have these kind of optional parameters in
    their call() signatures. 

    The alternative way is to use the ComputeCache. First you need to define commonly
    used operations and decorate it with `@ComputeCache.register`:

    ```python
    @ComputeCache.register()
    def get_valid_indices(adjacency_exists):
        return tf.where(tf.reduce_any(tf.reduce_any(adjacency_exists, axis=-1), axis=-1))
    ```

    Then finally, call the registered functions within a cache context:

    ```python
    with ComputeCache.push_context():   # Set a cache context
        valid_indices1 = get_valid_indices(some_tensor)
        ...
        valid_indices2 = get_valid_indices(some_tensor)  # Re-used. Same Tensor as valid_indices1
        ...
        some_other_layer(some_layer_inputs) # Nested layers can also reuse computations
        ...
    ```

    The results of a registered function will now be re-used if the same function was called
    before with the **same input parameters**, within the same ComputeCache context.

    **WARNING: Arguments to function calls are assumed to be immutable**.

    This is true for tf.Tensors and tf.RaggedTensors and tf.SparseTensors, but not true in general
    for all python objects.

    The following argument types are supported for these functions:
        1. Immutable objects like Tensors and so on
        2. List and tuples
        3. Int, float and bool
    """

    _current_contexts = []

    def __init__(self, name="cache", log_reuse=False, log_eval=False, store_params=True):
        """Initialization

        Args:
            name: (Optional) Name of the context for logging purposes
            log_reuse: (Optional) Log function calls when they are re-used. Default is False.
            log_eval: (Optional) Log function calls when they are evaluated. Default is False.
            store_params: (Optional) Store all arguments/parameters for function calls so
                they will **not** be garbage collected. Since Tensorflow will re-use Tensors
                when they are gargage collected, these could break the immutable assumption
                if these are not stored at the expense of potentially using more memory.
                Default is True.
        """
        self._map = {}
        self._name = name
        self._log_reuse = log_reuse
        self._log_eval = log_eval
        if store_params:
            self._params = {}

    @staticmethod
    @contextmanager
    def push_context(**kwargs):
        """Pushes a cache context to a context stack. All subsequent calls
        to registered functions will use the current cache in the stack.

        Args:
            **kwargs: (Optional) Arguments for ComputeCache constructor.        
        """
        ComputeCache._current_contexts.append(ComputeCache(**kwargs))
        try:
            yield
        finally:
            ComputeCache._current_contexts.pop()

    @staticmethod
    def assign(name_or_fn, args, val):
        """Utility function to manually assign a value to the cache given a
        registered function or name ID and input arguments. 

        This is an advanced use of the compute cache, but very useful in saving
        redundant computations in cases where you essentially derive the same
        result using a different set of computations vs another set of computations.
        By manually assigning the cache values, we can reuse the results on one
        set of computations on another different set of computations.

        Args:
            name_or_fn: A str or a ComputeCache registered function used for key generation
            args: A list or tuple of arguments used for key generation
            val: An object value to assign to the cache
        """
        if len(ComputeCache._current_contexts) > 0:
            cache = ComputeCache._current_contexts[-1]
            name = name_or_fn.fn_id if isinstance(name_or_fn, types.FunctionType) else name_or_fn
            key = (name, ) + args
            key_id = ComputeCache._get_key_id(key)
            if getattr(cache, "_params", None) is not None:
                # We only assert for this case because Tensorflow sometimes
                # reuses objects (ie, Tensors) if they are garbage collected
                # By storing the arguments we ensure that there is at least
                # a still a reference to the objects
                assert key_id not in cache._map, \
                    "In most cases of correct use, this should be unassigned beforehand. Please check!"
                cache._params[key_id] = args
            cache._map[key_id] = val

    @staticmethod
    def register(name_id=None):
        """Creates a lambda to register a function or method for caching and re-use.
        These functions can have arguments of type:
            1. Immutable objects like Tensors and so on
            2. List and tuples
            3. Int, float and bool.
            4. None

        Meant to be used as a decorator.

        Args:
            name_id: (Optional) A name ID to manually assign for this function. Default is None.

        Returns:
            A registration function
        """
        def _internal_reg(fn):
            fn_id = "%s.%s" % (fn.__module__, fn.__name__) if name_id is None else name_id

            def _lazy_eval(*args, **kwargs):
                if kwargs is not None and len(kwargs) != 0:
                    raise NotImplementedError("kwargs is not supported atm due to robustness issues :( .. %s" % kwargs)

                if len(ComputeCache._current_contexts) > 0:
                    # At least 1 cache is in stack
                    cache = ComputeCache._current_contexts[-1]
                    key_id = ComputeCache._get_key_id((fn_id,) + args)
                    if key_id not in cache._map:
                        res = fn(*args)
                        assert key_id not in cache._map, \
                            "Race condition detected. " + \
                            "Maybe your function calls a function that overwrites the same index?"
                        cache._map[key_id] = res
                        # Store params if required
                        if getattr(cache, "_params", None) is not None:
                            cache._params[key_id] = args
                        if cache._log_eval:
                            print("%s: Adding %s" % (cache._name, key_id))
                    else:
                        if cache._log_reuse:
                            print("%s: Reusing %s" % (cache._name, key_id))
                    return cache._map[key_id]
                else:
                    # No context in stack
                    return fn(*args)
            # We store the ID of the function in
            _lazy_eval.fn_id = fn_id
            return _lazy_eval
        return _internal_reg

    @staticmethod
    def _get_key_id(key):
        # Constructs a unique string key from a list/tuple of arguments.
        assert len(key) >= 1
        if not isinstance(key[0], str):
            raise KeyError("key[0] must be a string. Got a %s" % type(key[0]))

        convert_to_str = {
            int: lambda x: "_i%d" % x,
            float: lambda x: "_f%.9f" % x,
            bool: lambda x: "_%s" % str(x),
            type(None): lambda x: "_None"
        }

        def _convert_keys(key_list):
            str_id = ""
            for k in key_list:
                if isinstance(k, (list, tuple)):
                    str_id += "_[" + _convert_keys(k) + "]"
                else:
                    str_id += convert_to_str.get(type(k), lambda x: "_%d" % id(x))(k)
            return str_id

        str_id = _convert_keys(key[1:])
        return ("%s" % key[0]) + str_id
