#!/usr/bin/env python

"""Tiny boilerplate script to run model training with ktf.

User has to provide a config JSON file describing the training parameters.
"""
import optparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

import ktf.train


def main(config_json):
    train_env = ktf.train.Environment()
    train_env.run(config_json)


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-c', dest='config', help='Path of training config JSON file', metavar='PATH')
    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0 or not opt_args.config:
        opt_parser.print_help()
        exit()

    # Change cwd to config dir
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

    # Start training
    main(opt_args.config)
