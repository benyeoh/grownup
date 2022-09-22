#!/usr/bin/env python
import os
import sys
import argparse
import glob

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "web2text", "src", "main", "python"))

import tensorflow as tf
import numpy as np

# import main
from forward import EDGE_VARIABLES, UNARY_VARIABLES, edge, unary
from viterbi import viterbi


PATCH_SIZE = 9
N_FEATURES = 128
N_EDGE_FEATURES = 25
EDGE_LAMBDA = 1


def get_input_output_paths(input_dir, output_dir):
    all_dirs = glob.glob(os.path.join(input_dir, "*/"))
    input_output_paths = []
    for dir in all_dirs:
        edge_feats_path = os.path.join(dir, "edge_features.csv")
        block_feats_path = os.path.join(dir, "block_features.csv")
        if (os.path.exists(edge_feats_path) and os.path.exists(block_feats_path)):
            output_path = os.path.join(output_dir, os.path.basename(os.path.dirname(dir)) + ".csv")
            input_output_paths.append((block_feats_path, edge_feats_path, output_path))
        else:
            print("Missing edge/block feats. Skipping %s." % dir)
    return input_output_paths


def classify(input_dir, output_dir, ckpt_dir, lamb=EDGE_LAMBDA):
    io_paths = get_input_output_paths(input_dir, output_dir)

    unary_features = tf.placeholder(tf.float32, shape=[1, None, 1, N_FEATURES])  # tf.constant(block_features)
    edge_features = tf.placeholder(tf.float32, shape=[1, None, 1, N_EDGE_FEATURES])  # tf.constant(edge_features)

    unary_logits = unary(unary_features, is_training=False)
    edge_logits = edge(edge_features, is_training=False)

    unary_saver = tf.train.Saver(tf.get_collection(UNARY_VARIABLES))
    edge_saver = tf.train.Saver(tf.get_collection(EDGE_VARIABLES))

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        unary_saver.restore(session, os.path.join(ckpt_dir, "unary.ckpt"))
        edge_saver.restore(session, os.path.join(ckpt_dir, "edge.ckpt"))

        from time import time
        for i, (block_feats_path, edge_feats_path, output_path) in enumerate(io_paths):
            block_feats = np.genfromtxt(block_feats_path, delimiter=',')
            edge_feats = np.genfromtxt(edge_feats_path, delimiter=',')

            # Reshape
            try:
                block_feats = block_feats.T[np.newaxis, :, np.newaxis, :].astype(np.float32)
                edge_feats = edge_feats.T[np.newaxis, :, np.newaxis, :].astype(np.float32)

                start = time()

                unary_lgts = session.run(unary_logits, feed_dict={unary_features: block_feats})
                edge_lgts = session.run(edge_logits, feed_dict={edge_features: edge_feats})
                labels = viterbi(unary_lgts.reshape([-1, 2]), edge_lgts.reshape([-1, 4]), lam=lamb).astype(np.int32)

                duration = time() - start
                # print("Done. Classification took %.2f seconds " % duration)
                with open(output_path, 'w') as fp:
                    fp.write(",".join('%d' % label for label in labels))
                print("%d: Written %s. Duration = %.2fs" % (i, output_path, duration))
            except IndexError:
                print("%d: Skipping %s." % (i, output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Dir of inputs for extractions', metavar="DIR")
    parser.add_argument('output_dir', help='Output directory of labels', metavar="DIR")
    parser.add_argument('ckpt_dir', help='Dir of saved checkpoints', metavar="DIR")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # print(args.output_dir)
    classify(args.input_dir, args.output_dir, args.ckpt_dir)


if __name__ == "__main__":
    main()
