#!/usr/bin/env python
import argparse
import os
import json
import multiprocessing
import sys

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "klassterfork"))

import ktf.datasets.web.html_to_graph
from ktf.datasets.web.html_to_graph import FAIL, WARNING, ENDC

import prepare_data_cleaneval


if __name__ == "__main__":
    # Should be self-explanatory, but yeah cos' TF uses multi-threading semaphores internally and
    # would deadlock and cause all sorts of issues if we "fork" processes that already import
    # tensorflow. So we set it to spawn instead.
    print()
    print(WARNING + "WARNING: Forcing python multiprocessing to 'spawn' to ensure TF does not hang." + ENDC)
    print()
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='data_config_path', help='Path to data config json', metavar='PATH',
                        default=None)
    parser.add_argument('-i', dest='input_dir', help='Input directory of raw dragnet files', metavar="DIR")
    parser.add_argument('-o', dest='output_dir', help='Output directory of data files', metavar="DIR")
    parser.add_argument('-n', dest='num_proc', help='Num processes for multi-processing', metavar="N",
                        type=int, default=1)
    parser.add_argument('--skip-if-fail', dest='skip_if_fail',
                        help='Skip if there are failures during feature extraction',
                        action="store_true", default=False)
    parser.add_argument('--skip-if-exists', dest='skip_if_exists',
                        help='Skip if .pkl.gz exists',
                        action="store_true", default=False)
    args = parser.parse_args()

    config_list = None
    with open(args.data_config_path) as fd:
        config_list = json.load(fd)
    os.makedirs(args.output_dir, exist_ok=True)

    tag_feats_model = ktf.datasets.web.TagFeatures(**config_list["tag_features"])
    html_file_tup = prepare_data_cleaneval.find_files(os.path.join(
        args.input_dir, "test_html"), os.path.join(args.input_dir, "test_cleaned"))
    prepare_data_cleaneval.process_html_labels(args.num_proc, args.output_dir,
                                               not args.skip_if_fail, args.skip_if_fail, args.skip_if_exists,
                                               html_file_tup, tag_feats_model)
