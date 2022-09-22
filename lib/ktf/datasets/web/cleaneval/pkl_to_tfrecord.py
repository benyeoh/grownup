#!/usr/bin/env python
import os
import glob
import sys
import json
import shutil
import gzip
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import numpy as np
import optparse
import tensorflow as tf

import ktf.datasets
import ktf.datasets.web
import ktf.datasets.web.cleaneval


def write_to_tfrecords(pkl_dir,
                       out_dir,
                       num_splits):

    print("Creating %d record writers ..." % num_splits)
    record_writers = [ktf.datasets.RecordWriter(os.path.join(out_dir, "%d.tfrecord" % i),
                                                ktf.datasets.web.FEATURE_DESC_GRAPH,
                                                use_compression=True) for i in range(num_splits)]
    print("Writing records ...")
    count = 0
    all_pkl_files = [f for f in glob.glob(os.path.join(pkl_dir, "*.pkl.gz"))]

    for file in all_pkl_files:
        with gzip.open(file, "rb") as fd:
            adj, feats, nodes = pickle.load(fd)
            record_writers[count % num_splits].write(
                (adj.flatten().tolist(), feats.flatten().tolist(), nodes.tolist()))
            if count > 0 and (count % 100) == 0:
                print("Written %d records ..." % count)
                print("Adj shape: %s, feat shape: %s, nodes shape: %s" % (adj.shape, feats.shape, nodes.shape))
            count += 1

    for record_writer in record_writers:
        record_writer.close()

    print("Written %d records ..." % count)

    if os.path.exists(os.path.join(pkl_dir, "config.json")):
        print("Copying config.json ...")
        shutil.copy(os.path.join(pkl_dir, "config.json"), os.path.join(out_dir, "config.json"))
    print("Done!")


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-i', dest='pkl_dir', help='Original pkl data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-o', dest='out_dir', help='Output directory',
                          metavar='DIR', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../out'))
    opt_parser.add_option('-s', dest='num_splits', help='Num splits of the dataset', type='int',
                          metavar='NUM', default=20)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.print_help()
        exit()

    np.seterr(all="raise")

    os.makedirs(opt_args.out_dir, exist_ok=True)

    write_to_tfrecords(opt_args.pkl_dir, opt_args.out_dir, opt_args.num_splits)
