#!/usr/bin/env python
import argparse
import os
import json
import glob
import re
import traceback
import multiprocessing
import pickle
import magic
import sys
import gzip

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "klassterfork"))

import numpy as np
import bs4

import ktf.datasets.web
import ktf.datasets.web.cleaneval
import ktf.datasets.web.html_to_graph
from ktf.datasets.web.html_to_graph import FAIL, WARNING, ENDC


def _worker_init(tag_features_args):
    """Internal function to initialize workers for multiprocessing
    """
    print("Loading tag features per worker...")
    ktf.datasets.web.html_to_graph.tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)


def _read_file(path):
    with open(path, "rb") as fd:
        m = magic.Magic(mime_encoding=True)
        doc = fd.read()
        try:
            doc = doc.decode(m.from_buffer(doc))
        except:
            try:
                doc = doc.decode("cp1252")
            except UnicodeDecodeError:
                doc = doc.decode("iso-8859-1")
        return doc


def find_files(input_html_dir, gold_std_dir):
    html_test = glob.glob(os.path.join(input_html_dir, "*.html"))
    cleaned_test = glob.glob(os.path.join(gold_std_dir, "*.txt"))

    cleaned_map = {}
    for cleaned in cleaned_test:
        filename = os.path.basename(cleaned).split(".")[0].split("-")[0]
        cleaned_map[filename] = cleaned

    html_file_tup = []
    for html in html_test:
        filename = os.path.basename(html).split(".")[0]
        if filename in cleaned_map:
            html_file_tup.append((html, cleaned_map[filename], filename))
        else:
            print("Skipping parsing %s" % html)
    return html_file_tup


def initialize_options(opt_parser):
    return ktf.datasets.web.html_to_graph.initialize_options(opt_parser)


def write_graph_pkl_text(args):
    file_tup, out_dir, feat_recover_failures, skip_if_failure, skip_if_exists, graph_params = args
    html_file, cleaned_file, filename = file_tup
    output_pkl_path = os.path.join(out_dir, filename + ".pkl.gz")
    output_cleaned_path = os.path.join(out_dir, filename + ".txt")

    if skip_if_exists and (os.path.exists(output_pkl_path) and os.path.exists(output_cleaned_path)):
        print((WARNING + "Already exists. Skipping %s, %s ..." + ENDC) % (html_file, cleaned_file))
        return

    print("Parsing %s, %s ..." % (html_file, cleaned_file))

    try:
        soup = ktf.datasets.web.to_soup_from_file(html_file, inline_css=True)
        ktf.datasets.web.html_to_graph.tag_features_model.set_tag_features(soup,
                                                                           recover_failures=feat_recover_failures)
        graph = ktf.datasets.web.to_graph(soup)
        # Fetch content nodes vs non-content nodes for later sanity checks
        parser_dummy = ktf.datasets.web.cleaneval.ParserHTML(
            orig_path=None, cleaned_path=None, inline_css=False, tag_features_model=None)
        gold_std = _read_file(cleaned_file)
        gold_std_toks = parser_dummy._gold_std_to_list(gold_std)
        toks, tok_tags = parser_dummy._soup_txt_to_list(soup)
        tag_tok_count, tag_tok_lcs_count, lcs = parser_dummy._get_lcs_counts(soup, toks, tok_tags, gold_std_toks)

        content_nodes = []
        non_content_nodes = []
        other_nodes = []
        for tag in soup.find_all():
            if tag.tag_id in tag_tok_lcs_count:
                lcs_ratio = float(tag_tok_lcs_count[tag.tag_id]) / tag_tok_count[tag.tag_id]
                if lcs_ratio > 0:
                    content_nodes.append(tag.tag_id)
                else:
                    non_content_nodes.append(tag.tag_id)
            else:
                other_nodes.append(tag.tag_id)

        # Fetch all text strings and try to keep correct spacings between each neighbouring text
        blk_elems = set(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                         "ul", "ol", "td", "dl", "pre", "hr", "blockquote", "address"])
        all_strings = []
        all_tag_ids = []
        is_whitespace_last = True
        for s in soup.html.next_elements:
            if type(s) == bs4.element.Tag:
                if s.name == "br" or s.name in blk_elems:
                    is_whitespace_last = True
                elif s.tag_style:
                    if s.tag_style["display"] == "block":
                        is_whitespace_last = True
            elif type(s) == bs4.element.NavigableString:
                text_str = str(s)
                split_str = text_str.split()
                if len(split_str) > 0:
                    # We split strings into separate word tokens based on whether or not
                    # there is a whitespace either in the beginning of this string or the end of the last string
                    # or if the string starts on a new HTML block
                    append_str = str(s).strip()
                    if (is_whitespace_last or text_str[0].isspace()):
                        append_str = "\n" + append_str
                    all_strings.append(append_str)
                    if s.parent.tag_id is None:
                        print((FAIL + "No tag ID for parent: %s" + ENDC) % s.parent.name)
                    assert s.parent.tag_id
                    all_tag_ids.append(s.parent.tag_id)

                if len(text_str) > 0:
                    is_whitespace_last = text_str[-1].isspace()

        # Finally get the list of all text nodes
        unique_ids = list(set(all_tag_ids))
        assert len(set(content_nodes) - set(unique_ids)) == 0
        assert len(set(non_content_nodes) - set(unique_ids)) == 0

        if not graph_params:
            # This is the typical case. We simply binarize the graph in it's entirety
            # per html document
            graph_sampler = ktf.datasets.web.GraphNodeSampler(graph, node_list=unique_ids)
            try:
                adj, feats, _, id_remap = graph_sampler.to_numpy(1000,
                                                                 256,
                                                                 len(graph),
                                                                 ignore_warnings=False)
            except ValueError as e:
                if skip_if_failure:
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)  # Fixed format
                    print(e)
                    print()
                    print((FAIL + "Skipping file %s..." + ENDC) % html_file)
                    print()
                else:
                    print()
                    print(WARNING + str(e) + "\n... Resampling and ignoring further warnings ..." + ENDC)
                    print()
                    adj, feats, _, id_remap = graph_sampler.to_numpy(1000,
                                                                     256,
                                                                     len(graph),
                                                                     ignore_warnings=True)

            assert len(set(unique_ids) - set(id_remap.keys())) == 0
            node_list = np.array([id_remap[n] for n in unique_ids])

            print("Writing to %s, %s ..." % (output_pkl_path, output_cleaned_path))
            with gzip.open(output_pkl_path, "wb") as fd:
                adj = np.expand_dims(adj, axis=0)
                feats = np.expand_dims(feats, axis=0)
                node_list = np.expand_dims(node_list, axis=0)
                unique_ids = np.expand_dims(np.array(unique_ids), axis=0)
                pickle.dump(((adj, feats, node_list, unique_ids, None),
                             all_tag_ids, all_strings, set(content_nodes)),
                            fd,
                            protocol=4)
        else:
            # This code path uses the supplied graph_params and generates graphs in a manner
            # similar to training. Mainly for debugging use since it's a lot slower although
            # it allows more effective batching during inferencing and scales better
            # in terms of memory usage as the DOM gets larger
            max_depth, max_nodes, max_neighbours = graph_params
            graph_content = ktf.datasets.web.GraphNodeSampler(graph, node_list=content_nodes)
            graph_non_content = ktf.datasets.web.GraphNodeSampler(graph, node_list=non_content_nodes)

            all_adj = []
            all_feats = []
            all_nodes = []
            all_node_ids = []
            all_node_res = []

            print("Sampling content ...")
            for i, sample in enumerate(graph_content.iter_nodes(max_depth, max_neighbours, max_nodes)):
                adj, feats, node, _ = sample
                all_adj.append(adj)
                all_feats.append(feats)
                all_nodes.append(node)
                all_node_res.append(1)
                all_node_ids.append(content_nodes[i])

            print("Sampling non-content ...")
            for i, sample in enumerate(graph_non_content.iter_nodes(max_depth, max_neighbours, max_nodes)):
                adj, feats, node, _ = sample
                all_adj.append(adj)
                all_feats.append(feats)
                all_nodes.append(node)
                all_node_res.append(0)
                all_node_ids.append(non_content_nodes[i])

            assert len(unique_ids) == len(all_node_ids)
            assert len(set(unique_ids) - set(all_node_ids)) == 0
            assert len(set(all_node_ids) - set(unique_ids)) == 0

            print("Writing to %s, %s ..." % (output_pkl_path, output_cleaned_path))
            with gzip.open(output_pkl_path, "wb") as fd:
                adj = np.array(all_adj)
                feats = np.array(all_feats)
                node_list = np.array(all_nodes)
                node_res = np.array(all_node_res)
                node_ids = np.array(all_node_ids)
                pickle.dump(((adj, feats, node_list, node_ids, node_res),
                             all_tag_ids, all_strings, set(content_nodes)),
                            fd,
                            protocol=4)

        with open(output_cleaned_path, "w", encoding="utf-8") as fd:
            txt = re.sub("(^|\n)(URL:.*)", "", _read_file(cleaned_file))
            txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)
            fd.write(txt)

    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        print(e)
        print()
        print((FAIL + "Skipping file %s..." + ENDC) % html_file)
        print()
        if not skip_if_failure:
            raise


def process_html_labels(num_proc, output_dir, feat_recover_failures, skip_if_fail, skip_if_exists,
                        html_files_tup, tag_feats_model, graph_params=None):
    """Top level utility function to serializes html to a graph for unsupervised learning tasks
    (using multiprocessing if required)

    Args:
        opt_args: optparse arguments object
        html_files: A list of HTML file paths to serialize
        tag_feats_model: A `ktf.datasets.web.TagFeatures` object to convert tags into features
    """

    # This parameter is a global because we might want to use multi-processing
    # but we can never pickle this. Each process will have to initialize its own
    # and this is a convenient way to have the same codepath for both multi and single
    # process
    ktf.datasets.web.html_to_graph.tag_features_model = tag_feats_model
    tag_features_args = tag_feats_model.get_config()

    os.makedirs(output_dir, exist_ok=True)

    to_process = [(file_tup, output_dir, feat_recover_failures, skip_if_fail, skip_if_exists, graph_params)
                  for file_tup in html_files_tup]

    if num_proc > 1:
        with multiprocessing.Pool(num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
            res = p.map(write_graph_pkl_text, to_process)
            p.close()
            p.join()
    else:
        for tup in to_process:
            write_graph_pkl_text(tup)
    print("Done. Written %d" % (len(to_process)))


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
    parser.add_argument('-i', dest='input_dir', help='Input directory of raw files', metavar="DIR")
    parser.add_argument('-o', dest='output_dir', help='Output directory of data files', metavar="DIR")
    parser.add_argument('-n', dest='num_proc', help='Num processes for multi-processing', metavar="N",
                        type=int, default=1)
    parser.add_argument('--skip-if-fail', dest='skip_if_fail',
                        help='Skip if there are failures during feature extraction',
                        action="store_true", default=False)
    parser.add_argument('--use-graph-params', dest='use_graph_params',
                        help='Use graph parameters when binarizing graphs',
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
    html_file_tup = find_files(os.path.join(args.input_dir, "input"), os.path.join(args.input_dir, "gold_std"))
    process_html_labels(args.num_proc, args.output_dir,
                        not args.skip_if_fail, args.skip_if_fail, args.skip_if_exists,
                        html_file_tup, tag_feats_model, graph_params=config_list["graph_params"] if args.use_graph_params else None)
