import tensorflow as tf

from .dynamic_config import DynamicConfig
from .metas.default_trainer import DefaultTrainer


class Environment:
    """Convenience class to run multi-stage training using DynamicConfig
    """

    def __init__(self):
        self._dynamic_config = DynamicConfig()

    def run(self, config):
        """Runs training given the user-defined configuration

        Args:
            config: Can be a string, in which it is assumed to be a json path.
                Can also be a dictionary

        Returns:
            A list which contains the results of the TrainLoop of each stage
        """
        if type(config) is str:
            self._dynamic_config.set_config_from_json(config)
        else:
            self._dynamic_config.set_config(config)

        # We create a "proxy" dataset function that is only used to initialize models
        # from the original dataset definition
        # The "proxy" is equivalent to the original dataset except that it overwrites
        # some often-used parameters to speed things up during initialization
        #
        # NOTE: This is not the most robust since parameter names can be different
        proxy_dyn_config = DynamicConfig(self._dynamic_config.copy_config())
        proxy_dyn_config.overwrite_config("datasets", "batch_size", 1)
        proxy_dyn_config.overwrite_config("datasets", "shuffle_size", None)
        proxy_dyn_config.overwrite_config("datasets", "cache", False)

        res = []
        for i in range(self._dynamic_config.get_num_entries()):
            create_model_fn = DynamicConfig.wrap(self._dynamic_config.get_model, i)
            create_loss_fn = DynamicConfig.wrap(self._dynamic_config.get_loss, i)
            create_metrics_fn = DynamicConfig.wrap(self._dynamic_config.get_metrics, i)
            create_optimizer_fn = DynamicConfig.wrap(self._dynamic_config.get_optimizer, i)
            train_dataset, valid_dataset = self._dynamic_config.get_datasets(i)
            create_loss_weights_fn = DynamicConfig.wrap(self._dynamic_config.get_loss_weights, i)
            create_train_loop_fn = DynamicConfig.wrap(self._dynamic_config.get_train_loop, i)

            meta_trainer = self._dynamic_config.get_meta(i)
            if meta_trainer is None:
                meta_trainer = DefaultTrainer()

            # This is for generating minimal datasets sneakily to minimize model build times
            proxy_train_dataset, _ = proxy_dyn_config.get_datasets(i)

            def _create_and_build_model_fn():
                model = create_model_fn()
                # Build model with training=False to avoid a quirk of the ODAPI where the model expects truth labels to
                # be passed before the model's call if training=True
                training = True
                try:
                    if model._defer_set_training:
                        training = False
                except AttributeError:
                    pass
                model(next(iter(proxy_train_dataset))[0], training=training)
                return model

            print("\n**************** Start Run %d **********************\n" % (i + 1))
            print(self._dynamic_config.dump_config(i))
            print()

            res.append(meta_trainer(train_dataset,
                                    valid_dataset,
                                    _create_and_build_model_fn,
                                    create_loss_fn,
                                    create_loss_weights_fn,
                                    create_metrics_fn,
                                    create_optimizer_fn,
                                    create_train_loop_fn))

            print("\n**************** End Run %d **********************\n" % (i + 1))

        print("Done!")
        return res
