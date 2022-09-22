#!/usr/bin/env python
import os
import glob
import sys
import gzip
import optparse
import pickle

import numpy as np

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import ktf.datasets
import ktf.datasets.web.cleaneval
import ktf.datasets.web.html_to_graph
from ktf.datasets.web.html_to_graph import INFO, ENDC


def write_to_tfrecords(orig_dir,
                       cleaned_dir,
                       out_dir,
                       num_splits,
                       inline_css,
                       graph_params,
                       tag_features_model):
    """Writes to `num_splits` tfrecord files in the specified `out_dir` using the affnist parser.

    Args:
        orig_dir: Path to the directory containing raw html files
        cleaned_dir: Path to the directory containing cleaned files
        out_dir: Path to the destination directory containing converted tfrecord files
        num_splits: Number of tfrecord files to write. If > 1, each entry will be inserted to each tfrecord
            in a round robin fashion. If <= 0, will create 1 tfrecord file per input html
        inline_css: Inline CSS styles into tags
        graph_params: A tuple of (max depth, max # nodes, max # neighbours)
        tag_features_model: A TagFeatures object to be used for feature extraction
    """

    if num_splits > 0:
        parser = ktf.datasets.web.cleaneval.Parser(orig_dir, cleaned_dir, inline_css, tag_features_model)

        print("Creating %d record writers ..." % num_splits)
        record_writers = [ktf.datasets.RecordWriter(os.path.join(out_dir, "%d.tfrecord" % i),
                                                    ktf.datasets.web.cleaneval.FEATURE_DESC,
                                                    use_compression=True) for i in range(num_splits)]

        print("Writing records ...")
        count = 0
        for node_adj, node_feats, node, _, _, label, _ in parser.parse(max_depth=graph_params[0],
                                                                       max_neighbours=graph_params[2],
                                                                       max_nodes=graph_params[1]):

            record_writers[count % num_splits].write(
                (node_adj.flatten().tolist(), node_feats.flatten().tolist(), node, label))
            if count > 0 and (count % 1000) == 0:
                print((INFO + "Written %d records ..." + ENDC) % count)
                print((INFO + "Adj shape: %s, feat shape: %s" + ENDC) % (node_adj.shape, node_feats.shape))
            count += 1

        for record_writer in record_writers:
            record_writer.close()
    else:
        # Export using 1 tfrecord per input file and also store a reference data in pickle format
        # This codepath is mainly used for debugging.
        exclude_files = [os.path.basename(f).split(".")[0] + ".html"
                         for f in glob.glob(os.path.join(out_dir, "*.tfrecord"))]
        parser = ktf.datasets.web.cleaneval.Parser(
            orig_dir, cleaned_dir, inline_css, tag_features_model, exclude_files=exclude_files)

        print("Writing records ...")
        count = 0
        prev_file = None
        record_writer = None
        pkl_fd = None

        all_adjs = []
        all_feats = []
        all_nodes = []
        all_node_ids = []
        all_labels = []

        for node_adj, node_feats, node, id_remap, _, label, file in parser.parse(max_depth=graph_params[0],
                                                                                 max_neighbours=graph_params[2],
                                                                                 max_nodes=graph_params[1]):

            if prev_file != file:
                filename = "".join(os.path.basename(file).split(".")[:-1])
                print((INFO + "Writing to %s.tfrecord..." + ENDC) % filename)
                if record_writer:
                    record_writer.close()
                record_writer = ktf.datasets.RecordWriter(os.path.join(out_dir, "%s.tfrecord" % filename),
                                                          ktf.datasets.web.cleaneval.FEATURE_DESC,
                                                          use_compression=True)
                prev_file = file
                if pkl_fd:
                    all_adjs = np.stack(all_adjs, axis=0)
                    all_feats = np.stack(all_feats, axis=0)
                    all_nodes = np.array(all_nodes)
                    all_node_ids = np.array(all_node_ids)
                    all_labels = np.array(all_labels)

                    pickle.dump(((all_adjs, all_feats, all_nodes, all_node_ids, all_labels),
                                 None, None, None),
                                pkl_fd,
                                protocol=4)
                    pkl_fd.close()
                    all_adjs = []
                    all_feats = []
                    all_nodes = []
                    all_node_ids = []
                    all_labels = []

                pkl_fd = gzip.open(os.path.join(out_dir, "%s.pkl.gz" % filename), "wb")

            record_writer.write((node_adj.flatten().tolist(), node_feats.flatten().tolist(), node, label))

            all_adjs.append(node_adj)
            all_feats.append(node_feats)
            all_nodes.append(node)
            id_map = {i: n for n, i in id_remap.items()}
            all_node_ids.append(id_map[node])
            all_labels.append(label)

            if count > 0 and (count % 1000) == 0:
                print((INFO + "Written %d records ..." + ENDC) % count)
                print((INFO + "Adj shape: %s, feat shape: %s" + ENDC) % (node_adj.shape, node_feats.shape))
            count += 1

        if record_writer:
            record_writer.close()

        if pkl_fd:
            all_adjs = np.stack(all_adjs, axis=0)
            all_feats = np.stack(all_feats, axis=0)
            all_nodes = np.array(all_nodes)
            all_node_ids = np.array(all_node_ids)
            all_labels = np.array(all_labels)

            pickle.dump(((all_adjs, all_feats, all_nodes, all_node_ids, all_labels),
                         None, None, None),
                        pkl_fd,
                        protocol=4)
            pkl_fd.close()
            all_adjs = []
            all_feats = []
            all_nodes = []
            all_node_ids = []
            all_labels = []

    print("Written %d records ..." % count)
    print("Done!")


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-c', dest='cleaned_dir', help='Cleaned web data directory',
                          metavar='DIR', default=None)

    opt_args, args = ktf.datasets.web.html_to_graph.initialize_options(
        opt_parser, exclude=["num_proc", "sample_text_nodes"])
    tag_feats_model, graph_params = ktf.datasets.web.html_to_graph.initialize_config(opt_args)
    os.makedirs(opt_args.out_dir, exist_ok=True)
    write_to_tfrecords(opt_args.input_dir, opt_args.cleaned_dir, opt_args.out_dir, opt_args.num_tfrecords,
                       opt_args.inline_css, graph_params, tag_feats_model)
