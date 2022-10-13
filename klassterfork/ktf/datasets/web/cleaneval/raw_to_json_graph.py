#!/usr/bin/env python
import os
import glob
import sys
import traceback
import json
import gzip
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import numpy as np
import optparse
import tensorflow as tf
import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing

import ktf.datasets
import ktf.datasets.web
import ktf.datasets.web.cleaneval


tag_features_model = None


def _worker_init(tag_features_args):
    print("Loading tag features per worker...")
    global tag_features_model
    tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)


def write_to_json_graph(args):
    orig_file, cleaned_file, out_file = args

    print("Parsing %s, %s ..." % (orig_file, cleaned_file))
    html_parser = ktf.datasets.web.cleaneval.ParserHTML(orig_file, cleaned_file, tag_features_model)
    try:
        html_parser.init_graph_samplers()
        content_config = html_parser._content_sampler.get_config()
        content_config.pop("full_graph")
        non_content_config = html_parser._non_content_sampler.get_config()
        non_content_config.pop("full_graph")

        graph_dict = {
            "graph": json_graph.node_link_data(html_parser._graph),
            "content_config": content_config,
            "non_content_config": non_content_config
        }

        json_str = json.dumps(graph_dict)

        print("Writing to %s ..." % out_file)
        with gzip.open(out_file, "wt", encoding="ascii") as fd:
            fd.write(json_str)

    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        print()
        print("Skipping file %s..." % orig_file)
        print()


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-d', dest='orig_dir', help='Original web data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-c', dest='cleaned_dir', help='Cleaned web data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-m', dest='fasttext_model_path', help='Fasttext model path',
                          metavar='PATH', default=None)
    opt_parser.add_option('-o', dest='out_dir', help='Output directory',
                          metavar='DIR', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../out'))
    opt_parser.add_option('-u', dest='use_visual_feats', help='Use visual features', action="store_true",
                          default=False)
    opt_parser.add_option('-t', dest='trunc_wordvec', help='Truncate word vec', type='int', metavar="NUM", default=None)
    opt_parser.add_option('--num-eigvec', dest='num_eigvec', help='Number of eigen vectors',
                          type='int', metavar="NUM", default=16)
    opt_parser.add_option('--num-child-pos', dest='num_child_pos',
                          help='Number of child position markers', type='int', metavar="NUM", default=20)
    # opt_parser.add_option('-g', dest='graph_params', help='Graph parameters in format "<depth>,<nodes>,<neighbours>"',
    #                      metavar='STR', default="7,450,20")
    opt_parser.add_option('--tag-file', dest='tag_file', help='Text file containing HTML tag names to supports',
                          metavar='PATH', default="")
    opt_parser.add_option('--out-config', dest='out_config', help='JSON file containing saved config',
                          metavar='PATH', default="")
    opt_parser.add_option('--debug-no-wh', dest='debug_no_wh', help='(Debug) Disable width/height feature', action="store_true",
                          default=False)
    opt_parser.add_option('-n', dest='num_proc', help='Number of parallel processors',
                          metavar='NUM', type='int', default=4)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.print_help()
        exit()

    np.seterr(all="raise")

    if len(opt_args.tag_file) > 0:
        with open(opt_args.tag_file, "r") as fd:
            print("Loading tags ...")
            tags = [t.strip() for t in fd.readlines()]
            for i, t in enumerate(tags):
                print("%d: %s" % (i, t))
            print()
    else:
        tags = []

    #graph_params = [int(p) for p in opt_args.graph_params.split(",")]

    tag_features_args = {
        "ft_model_path": opt_args.fasttext_model_path,
        "truncate_wordvec": opt_args.trunc_wordvec,
        "include_visual": opt_args.use_visual_feats,
        "graph_num_eigvec": opt_args.num_eigvec,
        "tag_num_child_pos": opt_args.num_child_pos,
        "html_tags": tags,
        "debug_no_wh": opt_args.debug_no_wh
    }

    os.makedirs(opt_args.out_dir, exist_ok=True)

    print("Loading tag features ...")
    tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)
    print("Feature size: %d" % tag_features_model.get_feature_size_and_offsets()["feature_size"])
    print()

    if len(opt_args.out_config) > 0:
        print("Writing config to: %s" % opt_args.out_config)
        config = {
            "tag_features": tag_features_args,
            "feature_desc": tag_features_model.get_feature_size_and_offsets()
        }
        print()
        print(config)
        print()
        with open(opt_args.out_config, "w") as fd:
            json.dump(config, fd, indent=4, sort_keys=True)

    all_orig = glob.glob(os.path.join(opt_args.orig_dir, "*.html"))
    all_cleaned = glob.glob(os.path.join(opt_args.cleaned_dir, "*.txt"))

    all_cleaned_map = {}
    for file in all_cleaned:
        filename = os.path.basename(file).split(".")[0].split("-")[0]
        all_cleaned_map[filename] = file

    exclude_files = [os.path.basename(f).split(".")[0] + ".html"
                     for f in glob.glob(os.path.join(opt_args.out_dir, "*.json.gz"))]

    to_process_triplets = []
    for file in all_orig:
        filename = os.path.basename(file).split(".")[0]
        if filename in all_cleaned_map and os.path.basename(file) not in exclude_files:
            to_process_triplets.append((file, all_cleaned_map[filename], os.path.join(
                opt_args.out_dir, "%s.json.gz" % filename)))
        else:
            print("Skipping parsing %s" % file)

    with multiprocessing.Pool(opt_args.num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
        res = p.map(write_to_json_graph, to_process_triplets)
    p.close()
    p.join()

    print("Done!")
