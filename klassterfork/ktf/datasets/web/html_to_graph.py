#!/usr/bin/env python

"""A generic HTML-to-graph conversion utility script to convert from a .html source to either
.pkl (pickle serialization, not recommended) or .tfrecord files mainly for unsupervised learning tasks
"""

import os
import glob
import sys
import traceback
import json
import gzip
import multiprocessing
import optparse
import pickle

import numpy as np
import bs4

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))

import ktf.datasets
import ktf.datasets.web


FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'
INFO = '\033[32m'

tag_features_model = None


def _worker_init(tag_features_args):
    """Internal function to initialize workers for multiprocessing
    """
    print("Loading tag features per worker...")
    global tag_features_model
    tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)


def write_to_pkl(args):
    """Multiprocessing-friendly function to convert a html file to a pkl files
    given some arguments.

    Args:
        args: An argument list of:
            1. The html file path
            2. The output .pkl file path
            3. A boolean to indicate whether graph nodes should be sampled exclusively
                from tags containing text strings. If False, then graph nodes are sampled from all the
                tags in the DOM.
            4. Number of iterations to expand the nodes in the graph.
            5. Maximum number of nodes the graph can contain
            6. The maximum number of neighbouring nodes that a node in the graph can have.    
    """
    orig_file, out_file, sample_text_nodes, inline_css, depth, max_num_nodes, max_neighbours = args

    print("Parsing %s ..." % (orig_file))

    try:
        count = 0
        for adj, feats, node_list in html_to_graph_tensor(orig_file,
                                                          sample_text_nodes,
                                                          depth,
                                                          max_num_nodes,
                                                          max_neighbours,
                                                          inline_css):
            output_file_path = out_file
            if count > 0:
                split_path = output_file_path.split(".")
                split_path[-3] = split_path[-3] + ("_%d" % (count + 1))
                output_file_path = ".".join(split_path)
            print("Writing to %s ..." % output_file_path)
            with gzip.open(output_file_path, "wb") as fd:
                pickle.dump((adj, feats, node_list), fd)
            count += 1
    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        print(e)
        print()
        print((FAIL + "Skipping file %s..." + ENDC) % orig_file)
        print()
        raise


def html_to_graph_tensor(file, sample_text_nodes, depth, max_num_nodes, max_neighbours, inline_css):
    """Convert from a HTML DOM to a graph tensor.

    Args:
        file: The HTML file path to process
        sample_text_nodes: A boolean to indicate whether graph nodes should be sampled exclusively
                from tags containing text strings. If False, then graph nodes are sampled from all the
                tags in the DOM.
        depth: Number of iterations to expand the nodes in the graph.
        max_num_nodes: Maximum number of nodes the graph can contain
        max_neighbours: The maximum number of neighbouring nodes that a node in the graph can have.
        inline_css: Inline CSS styles into tags.

    Returns:
        A tuple of tensors `(adjacency, features, source_nodes)` where:
            adjacency: A numpy tensor of shape (max_num_nodes, 3, max_neighbours)
            features: A numpy tensor of shape (max_num_nodes, feature_size)
            source_nodes: A numpy 1D-tensor of shape (64 or 192,)
    """
    try:
        prop_filepath = os.path.join(os.path.dirname(file), os.path.basename(file) + ".br")
        (soup, elem_props) = ktf.datasets.web.to_soup_from_file(file, inline_css, prop_filepath=prop_filepath)
        tag_features_model.set_tag_features(soup, recover_failures=False, elem_props=elem_props)

        graph = ktf.datasets.web.to_graph(soup)
        if sample_text_nodes:
            tag_tok_count = {}
            for s in soup.html.find_all_next(string=True):
                if type(s) == bs4.element.NavigableString:
                    if len(str(s).split()) > 0:
                        if s.parent.tag_id not in tag_tok_count:
                            tag_tok_count[s.parent.tag_id] = 0
                        tag_tok_count[s.parent.tag_id] += len(str(s).split())
            node_list_raw = list(tag_tok_count.keys())
            max_node_list_size = 64
        else:
            node_list_raw = list(graph.nodes)
            max_node_list_size = 192

        cur_node_set = set(node_list_raw)
        while len(cur_node_set) > 0:
            sampled_node_list = list(np.random.choice(list(cur_node_set),
                                                      replace=False,
                                                      size=min(max_node_list_size, len(cur_node_set))))
            while (len(sampled_node_list) < max_node_list_size):
                sampled_node_list += list(np.random.choice(node_list_raw,
                                                           replace=False,
                                                           size=min(max_node_list_size - len(sampled_node_list),
                                                                    len(node_list_raw))))

            graph_sampler = ktf.datasets.web.GraphNodeSampler(graph, sampled_node_list)

            try:
                adj, feats, _, id_remap = graph_sampler.to_numpy(depth,
                                                                 max_neighbours,
                                                                 max_num_nodes,
                                                                 ignore_warnings=False)
            except ValueError as e:
                print()
                print(WARNING + str(e) + "\n... Resampling and ignoring further warnings ..." + ENDC)
                print()
                adj, feats, _, id_remap = graph_sampler.to_numpy(depth,
                                                                 max_neighbours,
                                                                 max_num_nodes,
                                                                 ignore_warnings=True)

            unique_sampled_list = set(sampled_node_list)
            sampled_nodes = list(unique_sampled_list & set(id_remap.keys()))
            cur_node_set -= set(sampled_nodes)

            if len(sampled_nodes) < len(unique_sampled_list):
                print((WARNING + "Warning: Undersampled nodes in graph."
                       " You need to increase the max_depth,max_nodes,max_neighbours parameters.\n"
                       "Expected node list size: %d. Final node list: %d. Total sampled nodes: %d\n" + ENDC)
                      % (len(unique_sampled_list), len(sampled_nodes), len(id_remap.keys())))
            node_list = np.array([id_remap[n] for n in sampled_nodes] + [-1] * (192 - len(sampled_nodes)))

            yield adj, feats, node_list

    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        # print(e)
        print()
        print((FAIL + "Skipping file %s..." + ENDC) % file)
        print()
        raise
    except ValueError as e:
        print()
        # print(e)
        print()
        print((FAIL + "Skipping file %s..." + ENDC) % file)
        print()
        pass


def write_to_tfrecord(args):
    """Multiprocessing-friendly function to convert several html files to multiple tfrecord files
    given some arguments.

    Args:
        args: An argument list of:
            1. An integer ID for this process. Usually 0 if multiprocessing is not used.
            2. A list of html file path strings.
            3. An integer number of tfrecord files to generate for the converted data.
            4. A boolean to indicate whether graph nodes should be sampled exclusively
                from tags containing text strings. If False, then graph nodes are sampled from all the
                tags in the DOM.
            5. Number of iterations to expand the nodes in the graph.
            6. Maximum number of nodes the graph can contain
            7. The maximum number of neighbouring nodes that a node in the graph can have.
            8. Boolean of inline CSS into tag.   
    """
    id, all_files, out_dir, num_splits, sample_text_nodes, inline_css, depth, max_num_nodes, max_neighbours = args

    print("%d: Creating %d record writers ..." % (id, num_splits))
    record_writers = [ktf.datasets.RecordWriter(os.path.join(out_dir, "%d_%d.tfrecord" % (i, id)),
                                                ktf.datasets.web.FEATURE_DESC_GRAPH,
                                                use_compression=True) for i in range(num_splits)]

    total_count = 0
    num_errors = 0

    for orig_file in all_files:

        # TODO: Delete at some point
        # Some problem files that I found. Keeping here for posterity
        # Please ignore.
        #orig_file = "/hpc-datasets/web/commoncrawl/html/CC-MAIN-2008-2009/cdx-00184/www.takansho.com_2008_cdx-00184_320910f47cb2d229d35bcd10524f246f.html"
        #orig_file = "/hpc-datasets/web/commoncrawl/html/CC-MAIN-2008-2009/cdx-00023/www.akronin.com_2008_cdx-00023_2edeaaa66ec83ec555896733342a091b.html"
        #orig_file = "/hpc-datasets/web/commoncrawl/html/CC-MAIN-2008-2009/cdx-00123/www.lyricstag.com_2008_cdx-00123_f8e204f2ca954a06615b4a9aa62dfa8a.html"

        print("%d: Parsing %s ..." % (id, orig_file))
        count = 0
        try:
            for j, (adj, feats, node_list) in enumerate(html_to_graph_tensor(orig_file,
                                                                             sample_text_nodes,
                                                                             depth,
                                                                             max_num_nodes,
                                                                             max_neighbours,
                                                                             inline_css)):
                record_writers[total_count % num_splits].write((adj.flatten().tolist(),
                                                                feats.flatten().tolist(),
                                                                node_list.tolist()))
                count += 1
                total_count += 1
                if j == 10:
                    print("Already converted %d subgraphs. Skipping the rest ..." % j)
                    break

        except AssertionError:
            num_errors += 1

        print("%d: Written %s - %d records ..." % (id, orig_file, count))
        if count > 0:
            print("%d: Adj shape: %s, feat shape: %s, nodes shape: %s" % (id, adj.shape, feats.shape, node_list.shape))

    for record_writer in record_writers:
        record_writer.close()

    print()
    print((WARNING + "%d: Finished %d files. Written %d records ..." + ENDC) % (id, len(all_files), total_count))
    if num_errors > 0:
        print((FAIL + "%d: WARNING. %d errors encountered ..." + ENDC) % (id, num_errors))
    print()


def initialize_options(opt_parser, exclude=[]):
    """Top level utility function to define and initialize common arguments for use in other
    html->graph conversion functions.

    Args:
        opt_parser: `optparse.OptionParser` object that will be initialized with more arguments.
        exclude: A list of strings denoting options to exclude from the command line parser

    Returns:
        A tuple of `(opt_args, args)` pair from `opt_parser.parse_args()`
    """
    if "input_dir" not in exclude:
        opt_parser.add_option('-i', dest='input_dir', help='Original web data directory',
                              metavar='DIR', default=None)
    if "text_model" not in exclude:
        opt_parser.add_option('--text', dest='text_model',
                              help=('The text model to use defined by an inline json spec, like this: '
                                    '--text "{ Text2VecUSE: { use_model_path: \'/hpc-datasets/ext_models/tf_hub/use_multi\' } }"'),
                              metavar='json', default=None)
    if "font_model" not in exclude:
        opt_parser.add_option('--font', dest='font_model',
                              help=('The font model to use defined by an inline json spec, like this: '
                                    '--font "{ FontSimple: {} }"'),
                              metavar='json', default=None)
    if "visual_model" not in exclude:
        opt_parser.add_option('--visual', dest='visual_model',
                              help=('The visual model to use defined by an inline json spec, like this: '
                                    '--visual "{ VisualSimple: {} }"'),
                              metavar='json', default=None)
    if "out_dir" not in exclude:
        opt_parser.add_option('-o', dest='out_dir', help='Output directory',
                              metavar='DIR', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../out'))
    if "graph_params" not in exclude:
        opt_parser.add_option('-g', dest='graph_params', help='Graph parameters in format "<depth>,<nodes>,<neighbours>"',
                              metavar='STR', default="10,1750,40")
    if "num_eigvec" not in exclude:
        opt_parser.add_option('--num-eigvec', dest='num_eigvec', help='Number of eigen vectors',
                              type='int', metavar="NUM", default=32)
    if "num_child_pos" not in exclude:
        opt_parser.add_option('--num-child-pos', dest='num_child_pos',
                              help='Number of child position markers', type='int', metavar="NUM", default=32)
    if "tag_file" not in exclude:
        opt_parser.add_option('--tag-file', dest='tag_file', help='Text file containing HTML tag names to support',
                              metavar='PATH', default="")
    if "out_config" not in exclude:
        opt_parser.add_option('--out-config', dest='out_config', help='JSON file containing saved config',
                              metavar='PATH', default="")
    if "inline_css" not in exclude:
        opt_parser.add_option('--inline-css', dest='inline_css', help='Inline CSS styles into tags',
                              action="store_false", default=True)
    # if "debug_no_wh" not in exclude:
    #    opt_parser.add_option('--debug-no-wh', dest='debug_no_wh', help='(Debug) Disable width/height feature', action="store_true",
    #                          default=True)
    if "num_proc" not in exclude:
        opt_parser.add_option('--num-proc', dest='num_proc', help='Number of parallel processors',
                              metavar='NUM', type='int', default=1)
    if "num_tfrecords" not in exclude:
        opt_parser.add_option('--num-tfrecords', dest='num_tfrecords', help='Num of tfrecords files to store',
                              metavar='NUM', type='int', default=50)
    if "sample_text_nodes" not in exclude:
        opt_parser.add_option('--sample-text-nodes', dest='sample_text_nodes', help='Samples from text nodes',
                              action="store_true", default=False)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    return opt_args, args


def initialize_config(opt_args):
    """Top level utility function to initialize various common parameters used in
    other functions. Also writes out a config file used for book-keeping and possibly for dataset ingestion.

    Args:
        opt_args: `optparse` arguments object initialized with user arguments

    Returns:
        A tuple of `(tag_features, graph_params)` where tag_features is a `TagFeatures` object and
        `graph_params` is a list of `[depth, max_nodes, max_neighbours]`
    """
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

    graph_params = [int(p) for p in opt_args.graph_params.split(",")]

    tag_features_args = {
        "text_model": opt_args.text_model,
        "visual_model": opt_args.visual_model,
        "font_model": opt_args.font_model,
        "graph_num_eigvec": opt_args.num_eigvec,
        "tag_num_child_pos": opt_args.num_child_pos,
        "html_tags": tags
    }

    print("Loading tag features ...")
    tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)
    print("Feature size: %d" % tag_features_model.get_feature_size_and_offsets()["feature_size"])
    print()

    config_path = os.path.join(opt_args.out_dir, "config.json")
    if len(opt_args.out_config) > 0:
        config_path = opt_args.out_config

    print("Writing config to: %s" % config_path)
    config = {
        "tag_features": tag_features_args,
        "graph_params": graph_params,
        "feature_desc": tag_features_model.get_feature_size_and_offsets()
    }
    print()
    print(config)
    print()

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as fd:
        json.dump(config, fd, indent=4, sort_keys=True)

    # Remove eigen vector feature desc if used so we can use this modified config
    # to workaround not permuting eigenvectors during training
    config["feature_desc"]["feature_offsets"].pop("tag_graph_eigen", None)

    path_splits = config_path.split('.')
    config_path = '.'.join(path_splits[:-1]) + "_no_rand_eig" + '.' + path_splits[-1]
    with open(config_path, "w") as fd:
        json.dump(config, fd, indent=4, sort_keys=True)

    return tag_features_model, graph_params


def write_html_to_graph(opt_args, html_files, tag_feats_model, graph_params):
    """Top level utility function to serializes html to a graph for unsupervised learning tasks
    (using multiprocessing if required)

    Args:
        opt_args: optparse arguments object
        html_files: A list of HTML file paths to serialize
        tag_feats_model: A `ktf.datasets.web.TagFeatures` object to convert tags into features
        graph_params: A tuple of `(depth, max_nodes, max_neighbours)` parameters for graph building
    """

    # This parameter is a global because we might want to use multi-processing
    # but we can never pickle this. Each process will have to initialize its own
    # and this is a convenient way to have the same codepath for both multi and single
    # process
    global tag_features_model
    tag_features_model = tag_feats_model
    tag_features_args = tag_feats_model.get_config()

    all_orig = html_files

    os.makedirs(opt_args.out_dir, exist_ok=True)
    if opt_args.num_tfrecords == 0:
        exclude_files = [os.path.basename(f).split(".")[0] + ".html"
                         for f in glob.glob(os.path.join(opt_args.out_dir, "*.pkl.gz"))]

        to_process = []
        for file in all_orig:
            filename = os.path.basename(file).split(".")[0]
            if os.path.basename(file) not in exclude_files:
                to_process.append(tuple([file, os.path.join(opt_args.out_dir, "%s.pkl.gz" % filename)] +
                                        [opt_args.sample_text_nodes, opt_args.inline_css] +
                                        graph_params))
            else:
                print("Skipping parsing %s" % file)

        if opt_args.num_proc > 1:
            with multiprocessing.Pool(opt_args.num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
                res = p.map(write_to_pkl, to_process)
                p.close()
                p.join()
        else:
            for tup in to_process:
                write_to_pkl(tup)
    else:
        if opt_args.num_proc > 1:
            to_process = []
            num_files_per_task = int(len(all_orig) / opt_args.num_proc)
            num_files_so_far = 0
            for i in range(opt_args.num_proc - 1):
                to_process.append(tuple([i,
                                         all_orig[num_files_so_far:num_files_so_far + num_files_per_task],
                                         opt_args.out_dir,
                                         opt_args.num_tfrecords,
                                         opt_args.sample_text_nodes,
                                         opt_args.inline_css] + graph_params))
                num_files_so_far += num_files_per_task

            to_process.append(tuple([opt_args.num_proc - 1,
                                     all_orig[num_files_so_far:],
                                     opt_args.out_dir,
                                     opt_args.num_tfrecords,
                                     opt_args.sample_text_nodes,
                                     opt_args.inline_css] + graph_params))

            with multiprocessing.Pool(opt_args.num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
                res = p.map(write_to_tfrecord, to_process)
                p.close()
                p.join()
        else:
            write_to_tfrecord(tuple([0,
                                     all_orig,
                                     opt_args.out_dir,
                                     opt_args.num_tfrecords,
                                     opt_args.sample_text_nodes,
                                     opt_args.inline_css] + graph_params))


def glob_html_files(opt_args):
    """Convenience function to get all html files from the option parser.

    Args:
        opt_args: `optparse` arguments object

    Returns:
        A list of html file paths
    """
    files = glob.glob(os.path.join(opt_args.input_dir, "*.html"))
    return files


if __name__ == "__main__":
    # Should be self-explanatory, but yeah cos' TF uses multi-threading semaphores internally and
    # would deadlock and cause all sorts of issues if we "fork" processes that already import
    # tensorflow. So we set it to spawn instead.
    print()
    print(WARNING + "WARNING: Forcing python multiprocessing to 'spawn' to ensure TF does not hang." + ENDC)
    print()
    multiprocessing.set_start_method("spawn")

    opt_parser = optparse.OptionParser()
    opt_args, args = initialize_options(opt_parser)
    tag_feats_model, graph_params = initialize_config(opt_args)
    files = glob_html_files(opt_args)
    write_html_to_graph(opt_args, files, tag_feats_model, graph_params)
    print("Done!")
