#!/usr/bin/env python
import os
import sys
import json
import argparse
import tensorflow as tf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from ktf.train import DynamicConfig


class LoadSaveSubModel:
    """Class to load and save submodel weights
    Script will load in model and randomly generate weights, then load in saved submodel weights,
    and save the model to the indicated saved weights path.
    Note: This script only accepts h5 files for weights.
    """

    def __init__(self, config_json):
        """Args:
            config_json: Path to config file
        """
        self._config_json = config_json

    def _get_config(self):
        dynamic_config = DynamicConfig(self._config_json)
        dynamic_config.overwrite_config("datasets", "batch_size", 1)
        dynamic_config.overwrite_config("datasets", "shuffle_size", None)
        dynamic_config.overwrite_config("datasets", "cache", False)
        return dynamic_config

    def _init_model(self):
        """ Initialize model and get the random weights from the initializiation
        """
        dynamic_config = self._get_config()
        dataset = dynamic_config.get_datasets()[0]
        model = dynamic_config.get_model()
        model(next(iter(dataset))[0], training=True)
        old_model_weights = model.get_weights()

        return model, old_model_weights

    def _find_submodel(self, model, name=None):
        if not name:
            return model
        cur_submodel = model
        for cur_submodel_str in name.split("@"):
            cur_submodel = cur_submodel.get_layer(cur_submodel_str)

        return cur_submodel

    def _load_submodel_weights(self, model, config):
        """Loads submodel weights from configurations provided

        Args:
            model: Initialized model to load weights into
            load_weights_config: Configuration list
        """
        # Get submodel if submodel name is provided, otherwise return entire model
        if len(config) < 2:
            submodel = model
            weights_path = config[0]
        else:
            submodel = self._find_submodel(model, config[0])
            weights_path = config[1]
        old_weights = submodel.get_weights()
        submodel.load_weights(weights_path)
        is_equal = (old_weights[0] == submodel.get_weights()[0])
        assert not tf.reduce_all(is_equal), "Failed to update submodel weights. Weights are equal."

        return old_weights, submodel

    def _save_submodel_weights(self, model, save_weights_config):
        """Saves submodel weights from configurations provided

        Args:
            model: initialized model with weights loaded
            save_weights_config: configuration list containing pairs of [submodel_name, weights_path]
        """
        # Get submodel if submodel name is provided, otherwise return entire model
        if len(save_weights_config) < 2:
            submodel = model
            weights_path = save_weights_config[0]
        else:
            submodel = self._find_submodel(model, save_weights_config[0])
            weights_path = save_weights_config[1]
        submodel.save_weights(weights_path)

    def load_and_save(self, load_weights_config, save_weights_config):
        print("Initializing model ...")
        model, old_model_weights = self._init_model()
        print("Loading and saving weights ...")
        self._load_submodel_weights(model, load_weights_config)
        self._save_submodel_weights(model, save_weights_config)
        print("Checking that saved weights are successfully updated ...")
        model.set_weights(old_model_weights)
        old_weights, submodel = self._load_submodel_weights(model, save_weights_config)
        print("%s - Before:" % submodel.name)
        print(old_weights[0])
        print()
        print("%s - After:" % submodel.name)
        print(submodel.get_weights()[0])
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config_json', help='Path to config json', metavar='PATH',
                        default=None)
    parser.add_argument('-l', dest='load_weights_config',
                        help='configuration to load weights from, format: "submodel_name weights_path"',
                        nargs='+', default=None)
    parser.add_argument('-s', dest='save_weights_config',
                        help='configuration to save weights from, format: "submodel_name weights_path"',
                        nargs='+', default=None)

    args = parser.parse_args()

    lsm = LoadSaveSubModel(args.config_json)
    lsm.load_and_save(args.load_weights_config, args.save_weights_config)
