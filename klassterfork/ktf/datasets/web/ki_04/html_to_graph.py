#!/usr/bin/env python
import glob
import optparse
import os
import sys
import traceback
import multiprocessing

import numpy as np

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import ktf.datasets
import ktf.datasets.web
from ktf.datasets.web.html_to_graph import FAIL, WARNING, INFO, ENDC
from ktf.datasets.web.webgenre7 import FEATURE_DESC_GRAPH

tag_feats_model = None


def _worker_init(tag_features_args):
    """Internal function to initialize workers for multiprocessing
    """
    print("Loading tag features per worker...")
    global tag_feats_model
    tag_feats_model = ktf.datasets.web.TagFeatures(**tag_features_args)


def initialize_options(opt_parser):
    """Customized initialized options (args)
    For KI-04 dataset, ideal output is .tfrecord per .html
    """
    opt_args, args = ktf.datasets.web.html_to_graph.initialize_options(opt_parser)

    return opt_args, args


def get_input_files(opt_args):
    """Get the input webpages dictionary:
    - Key: class label
    - Values: list of path for each webpage under current class
    """

    input_files = {}
    total_num_classes = 0
    total_num_pages = 0

    classes_dir = [path for path in sorted(glob.glob(os.path.join(opt_args.input_dir, "*"))) if os.path.isdir(path)]
    for class_dir in classes_dir:
        class_label = class_dir.split('/')[-1]
        total_num_classes += 1

        # Note: pay attention to the raw input file format. For KI-04 each webpage is saved as .html
        input_files[class_label] = [path for path in glob.glob(os.path.join(
            opt_args.input_dir, class_dir, "*.html")) if os.path.isfile(path)]
        total_num_pages += len(input_files[class_label])

    print("Total num classes: %d" % total_num_classes)
    print("Total num pages: %d" % total_num_pages)
    return input_files


def html_to_graph_tensor(file, depth, max_num_nodes, max_neighbours):
    """Modified from ktf.datasets.web.html_to_graph.html_to_graph_tensor with simplified logic to convert from a HTML
    DOM to a graph tensor.

    Args:
        file: The HTML file path to process
        depth: Number of iterations to expand the nodes in the graph.
        max_num_nodes: Maximum number of nodes the graph can contain
        max_neighbours: The maximum number of neighbouring nodes that a node in the graph can have.

    Returns:
        A tuple of tensors `(adjacency, features, source_nodes)` where:
            adjacency: A numpy tensor of shape (max_num_nodes, 3, max_neighbours)
            features: A numpy tensor of shape (max_num_nodes, feature_size)
    """
    try:
        soup = ktf.datasets.web.to_soup_from_file(file)
        tag_feats_model.set_tag_features(soup, recover_failures=False)
        graph = ktf.datasets.web.to_graph(soup)

        node_list_raw = list(graph.nodes)
        cur_node_set = set(node_list_raw)
        max_node_list_size = 32

        while len(cur_node_set) > 0:
            sampled_node_list = list(np.random.choice(list(cur_node_set),
                                                      replace=False,
                                                      size=min(max_node_list_size, len(cur_node_set))))
            while(len(sampled_node_list) < max_node_list_size):
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
            cur_node_set -= set(id_remap.keys())

            if len(sampled_nodes) < len(unique_sampled_list):
                print(WARNING + "Warning: Undersampled nodes in graph."
                      " You need to increase the max_depth,max_nodes,max_neighbours parameters.\n"
                      "Expected node list size: {}. Final node list: {}. Total sampled nodes: {}\n".
                      format(len(unique_sampled_list), len(sampled_nodes), len(id_remap.keys())) + ENDC)

            yield adj, feats

    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        print(e)
        print()
        print(FAIL + "Skipping file {}...".format(file) + ENDC)
        print()
        raise


def write_html_to_tfrecord(args):
    global tag_feats_mode

    id, all_inputs, out_dir, depth, max_num_nodes, max_neighbours = args

    errors = []
    for class_idx, cls, pages in all_inputs:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        record_writers = [ktf.datasets.RecordWriter(os.path.join(
            out_dir, cls, '%s.tfrecord' % (html_path.split("/")[-1])), FEATURE_DESC_GRAPH, use_compression=True)
            for html_path in pages]

        for i, html_path in enumerate(pages):
            try:
                print("%d - Writing: %s" % (id, html_path))
                for j, (adj, feats) in enumerate(html_to_graph_tensor(html_path,
                                                                      depth,
                                                                      max_num_nodes,
                                                                      max_neighbours)):
                    record_writers[i].write((adj.flatten().tolist(), feats.flatten().tolist(), class_idx))
                    # Note, currently allow convert 5 subgraphs for each webpage, corresponding to 1750 x 5 = 8750 nodes.
                    # This is derived from an estimation of extreme large webpage
                    if j == 5:
                        print("Already converted %d subgraphs. Skipping the rest ..." % j)
                        break

                print("Adj shape: %s, feat shape: %s" % (adj.shape, feats.shape))
            except AssertionError:
                errors.append(html_path)
        for record_writer in record_writers:
            record_writer.close()

        print("%d - Done: %s" % (id, cls))

    if len(errors) > 0:
        print(FAIL + ("%d - WARNING. %d errors encountered ...\n" % (id, len(errors))) + ENDC)
        for error in errors:
            print(FAIL + ("%d - Error: %s" % (id, error)) + ENDC)
    print()


def process_all(opt_args, tag_features_model, graph_params):
    global tag_feats_model
    tag_feats_model = tag_features_model

    tag_features_args = tag_feats_model.get_config()

    html_inputs = get_input_files(opt_args)
    classes = sorted(list(html_inputs.keys()))
    print("Classes: %s" % str(classes))

    # Rearrange all input tuples into groups for easier consumption downstream
    all_orig = []
    for cls, webpages in html_inputs.items():
        all_orig.append((classes.index(cls), cls, webpages))

    assert opt_args.num_tfrecords > 0, ".pkl output not supported. Only .tfrecord output is supported."
    if opt_args.num_proc > 1:
        to_process = []
        num_webpages_per_task = int(len(all_orig) / opt_args.num_proc)
        num_webpages_so_far = 0
        for i in range(opt_args.num_proc - 1):
            to_process.append(tuple([i,
                                     all_orig[num_webpages_so_far:num_webpages_so_far + num_webpages_per_task],
                                     opt_args.out_dir] + graph_params))
            num_webpages_so_far += num_webpages_per_task

        to_process.append(tuple([opt_args.num_proc - 1,
                                 all_orig[num_webpages_so_far:],
                                 opt_args.out_dir] + graph_params))

        with multiprocessing.Pool(opt_args.num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
            res = p.map(write_html_to_tfrecord, to_process)
            p.close()
            p.join()
    else:
        write_html_to_tfrecord(tuple([0,
                                      all_orig,
                                      opt_args.out_dir] + graph_params))


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
    tag_feats_model, graph_params = ktf.datasets.web.html_to_graph.initialize_config(opt_args)

    process_all(opt_args, tag_feats_model, graph_params)
