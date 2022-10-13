#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import optparse

import tensorflow as tf
import numpy as np

import ktf.train


def print_layers(model):

    def _dfs_print(layer, prefix="", last=True):
        try:
            print(prefix, " └── " if last else " ├── ", layer.name, sep="")
            prefix += "    " if last else " |  "
            # for v in layer.weights:
            #    print(v.name)
            for i, l in enumerate(layer.layers):
                last = i == (len(layer.layers) - 1)
                _dfs_print(l, prefix, last)
        except AttributeError:
            pass

    _dfs_print(model)


def print_weights(model):
    for w in model.weights:
        print(w.name)


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-c', dest='config', help='Path of training config JSON file', metavar='PATH')
    opt_parser.add_option('--in-shape', dest='in_shape', help="Python input shape string for the model",
                          metavar='STR', default=None)
    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0 or not opt_args.config:
        opt_parser.print_help()
        exit()

    config = ktf.train.DynamicConfig(opt_args.config)
    config.overwrite_config("datasets", "batch_size", 1)
    config.overwrite_config("datasets", "shuffle_size", None)
    config.overwrite_config("datasets", "cache", False)

    model = config.get_model()
    if opt_args.in_shape:
        model.build(eval(opt_args.in_shape))
    else:
        ds, _ = config.get_datasets()
        # Build model with training=False to avoid a quirk of the ODAPI where the model expects truth labels to
        # be passed before the model's call if training=True
        training = True
        try:
            if model._defer_set_training:
                training = False
        except AttributeError:
            pass
        model(next(iter(ds))[0], training=training)

    print()
    print("\nLayers:")
    print_layers(model)
    print("\nWeights:")
    print_weights(model)
    print()
