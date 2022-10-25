#!/usr/bin/env python
import argparse
import os
import glob
import gzip
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "klassterfork"))

import numpy as np
import tensorflow as tf

import ktf.train
import ktf.datasets.web.cleaneval


def compute_precision_recall_f1(num_tp, num_fp, num_fn):
    if num_tp == 0:
        prec = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        prec = float(num_tp) / (num_tp + num_fp)
        recall = float(num_tp) / (num_tp + num_fn)
        f1 = 2.0 * (prec * recall) / (prec + recall)
    return prec, recall, f1


def extract_content(input_files,
                    output_files,
                    infer_model_config_path,
                    infer_model_weights_path):

    dyn_config = ktf.train.DynamicConfig(infer_model_config_path)

    print("Creating model ...")
    infer_model = dyn_config.get_model()

    num_tp = 0
    num_fp = 0
    num_fn = 0

    total_node_tp = 0
    total_node_fp = 0
    total_node_fn = 0

    for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
        print("%d: Processing %s ..." % (i, input_file))
        with gzip.open(input_file, "rb") as fd:
            data = pickle.loads(fd.read())
            # if len(data) == 4:
            tensors, all_tag_ids, all_strings, gt_content = data
            if len(tensors) == 5:
                adj, feats, node_list, unique_ids, node_gt = tensors
            else:
                adj, feats, node_list, unique_ids = tensors
                node_gt = None

            # else:
             #   (adj, feats, node_list, unique_ids), all_tag_ids, all_strings = data
             #   gt_content = None

        # if len(adj.shape) == 3:
        #    adj = np.expand_dims(adj, axis=0)
        # if len(feats.shape) == 2:
        #    feats = np.expand_dims(feats, axis=0)
        # if len(node_list.shape) == 1:
        #    node_list = np.expand_dims(node_list, axis=-1 if adj.shape[0] > 1 else 0)

        # if isinstance(unique_ids, list):
        #    unique_ids = np.array(unique_ids)
        if len(unique_ids.shape) == 1:
            unique_ids = np.expand_dims(unique_ids, axis=-1 if adj.shape[0] > 1 else 0)

        if node_gt is not None and len(node_gt.shape) == 1:
            node_gt = np.expand_dims(node_gt, axis=-1 if adj.shape[0] > 1 else 0)

        if not infer_model.built:
            print("Building model ...")
            infer_model((adj, feats, node_list))
            print("Loading weights ...")
            infer_model.load_weights(infer_model_weights_path)
            print()
            infer_model.summary()

        res = infer_model((adj, feats, node_list))
        print(res.shape)
        print(node_list.shape)

        if len(node_list.shape) == 2 and len(res.shape) == 3:
            res = tf.squeeze(res, axis=-1)

        node_tp = 0
        node_fp = 0
        node_fn = 0

        content_set = set()
        assert unique_ids.shape == res.shape
        for batch in range(res.shape[0]):
            for idx in range(res.shape[1]):
                threshold = res[batch][idx]
                if threshold >= 0.5:
                    content_set.add(unique_ids[batch][idx])
                    if node_gt is not None:
                        node_tp += node_gt[batch][idx]
                        node_fp += 1 - node_gt[batch][idx]
                else:
                    if node_gt is not None:
                        node_fn += node_gt[batch][idx]

        if gt_content:
            tp = content_set & gt_content
            fp = content_set - gt_content
            fn = gt_content - content_set
            prec, recall, f1 = compute_precision_recall_f1(len(tp), len(fp), len(fn))
            print("%d: Prec: %f, Recall: %f, F1: %f" % (i, prec, recall, f1))
            num_tp += len(tp)
            num_fp += len(fp)
            num_fn += len(fn)
        else:
            print("%d: No content: %s" % (i, gt_content))

        if node_gt is not None:
            prec, recall, f1 = compute_precision_recall_f1(node_tp, node_fp, node_fn)
            print("Node GT %d: Prec: %f, Recall: %f, F1: %f" % (i, prec, recall, f1))
            total_node_tp += node_tp
            total_node_fp += node_fp
            total_node_fn += node_fn

        if all_strings is not None:
            print("%d: Dumping cleaned output %s ..." % (i, output_file))
            with open(output_file, "w", encoding="utf-8") as fd:
                res_txt = []
                for id, txt in zip(all_tag_ids, all_strings):
                    if id in content_set:
                        res_txt.append(txt)
                fd.write("".join(res_txt))
        print()

    if num_tp != 0 or num_fp != 0 or num_fn != 0:
        prec, recall, f1 = compute_precision_recall_f1(num_tp, num_fp, num_fn)
        print("Total -- Prec: %f, Recall: %f, F1: %f" % (prec, recall, f1))

    if node_gt is not None:
        prec, recall, f1 = compute_precision_recall_f1(total_node_tp, total_node_fp, total_node_fn)
        print("Total Node GT %d: Prec: %f, Recall: %f, F1: %f" % (i, prec, recall, f1))


def _debug_extract_content_tfrecord(tfrecord_dir,
                                    infer_model_config_path,
                                    infer_model_weights_path):

    ds, _ = ktf.datasets.web.cleaneval.from_tfrecord(tfrecord_dir,
                                                     expand_binary_class=False,
                                                     max_nodes_neighbours_feat_size=None,
                                                     validation_split=0.0,
                                                     batch_size=64,
                                                     shuffle_size=None,
                                                     repeat=False,
                                                     config_path=os.path.join(tfrecord_dir, "config_no_rand_eig.json"),
                                                     cache=False)

    dyn_config = ktf.train.DynamicConfig(infer_model_config_path)

    num_tp = 0
    num_fp = 0
    num_fn = 0

    print("Creating model ...")
    infer_model = dyn_config.get_model()

    for i, (data, label) in enumerate(iter(ds)):
        adj, feats, node_list = data

        if not infer_model.built:
            print("Building model ...")
            infer_model((adj, feats, node_list))
            print("Loading weights ...")
            infer_model.load_weights(infer_model_weights_path)
            print()
            infer_model.summary()

        res = infer_model((adj, feats, node_list))

        if len(node_list.shape) == 2 and len(res.shape) == 3:
            res = tf.squeeze(res, axis=-1)

        if len(label.shape) == 1:
            label = tf.expand_dims(label, axis=-1)

        node_tp = 0
        node_fp = 0
        node_fn = 0

        content_set = set()
        for batch in range(res.shape[0]):
            for idx in range(res.shape[1]):
                threshold = res[batch][idx]
                if threshold >= 0.5:
                    node_tp += tf.cast(label[batch][idx], tf.float32)
                    node_fp += 1.0 - tf.cast(label[batch][idx], tf.float32)
                else:
                    node_fn += tf.cast(label[batch][idx], tf.float32)

        prec, recall, f1 = compute_precision_recall_f1(node_tp, node_fp, node_fn)
        print("Node GT %d: Prec: %f, Recall: %f, F1: %f" % (i, prec, recall, f1))

        num_tp += node_tp
        num_fp += node_fp
        num_fn += node_fn

    prec, recall, f1 = compute_precision_recall_f1(num_tp, num_fp, num_fn)
    print("Total -- Prec: %f, Recall: %f, F1: %f" % (prec, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='Directory of input files', metavar='DIR')
    parser.add_argument('-o', dest='output_dir', help='Directory of extracted files', metavar='DIR', default=None)
    parser.add_argument('--model-config-path', dest='model_config_path',
                        help='Path to json config for model', metavar='PATH')
    parser.add_argument('--model-weights-path', dest='model_weights_path',
                        help='Path to weights for model', metavar='PATH')
    args = parser.parse_args()

    if not args.output_dir:
        print("No output dir given. Running in validation debug mode (expect input dir to contain tfrecords)")
        _debug_extract_content_tfrecord(args.input_dir,
                                        args.model_config_path,
                                        args.model_weights_path)
    else:
        all_input_files = []
        all_output_files = []
        all_input_files.extend(glob.glob(os.path.join(args.input_dir, "*.pkl.gz")))
        for path in all_input_files:
            out_filename = os.path.basename(path).split(".")[0] + ".txt"
            outpath = os.path.join(args.output_dir, out_filename)
            all_output_files.append(outpath)

        os.makedirs(args.output_dir, exist_ok=True)

        extract_content(all_input_files,
                        all_output_files,
                        args.model_config_path,
                        args.model_weights_path)
