#!/usr/bin/env python
import glob
import argparse
import os
import sys
import json

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import numpy as np
import tensorflow as tf

import ktf
import ktf.train
import ktf.datasets
import ktf.datasets.web


def parse_config(config_path):
    if config_path:
        with open(config_path, "r") as fd:
            config_json = json.loads(fd.read())
            return ktf.datasets.web.parse_graph_config_json(config_json)
    return ktf.datasets.web.parse_graph_config_json(None)


def read_dataset(config_path, files):
    config = parse_config(config_path)
    max_nodes_neighbours_feat_size = config["max_nodes_neighbours_feat_size"]

    def _decode_example(examples):
        adj = tf.cast(examples["adj"], tf.int32)
        feat = tf.cast(examples["feat"], tf.float32)
        label_id = examples["label_id"]
        return ((adj, feat), label_id)

    feature_desc = [
        ('adj', int, [max_nodes_neighbours_feat_size[0], 3, max_nodes_neighbours_feat_size[1]]),
        ('feat', float, [max_nodes_neighbours_feat_size[0], max_nodes_neighbours_feat_size[2]]),
        ('label_id', np.int32, [])]

    record_io = ktf.datasets.RecordDatasetIO(feature_desc)
    ds = record_io.read_batch(files,
                              4,
                              shuffle_size=None,
                              repeat=False,
                              parse_fn=_decode_example,
                              use_compression=True)
    return ds


def evaluate(tfrecord_dir,
             infer_model_config_path,
             infer_model_weights_path,
             data_config_path):
    """Derive prediction of whole webpage, based on inference outcome from single subgraph
    - Each .tfrecord is converted from single webpage and contains graph features from several subgraphs
    - Prediction per subgraph was saved as one-hot vector
    - Prediction for whole webpage is accumulated from prediction per subgraph
    - Label with max value will be used as the final class_label for this webpage. It means model predicts most
    subgraphs into this class 
    """
    input_files = [path for path in glob.glob(os.path.join(
        tfrecord_dir, "**", "*.tfrecord"), recursive=True) if os.path.isfile(path)]
    dyn_config = ktf.train.DynamicConfig(infer_model_config_path)

    print("Creating model ...")
    infer_model = dyn_config.get_model()

    correct = 0
    total = 0
    for i, input_file in enumerate(input_files):
        print("%d: Processing: %s" % (i, input_file))
        ds = read_dataset(data_config_path, [input_file])

        num_preds = 0
        preds = None
        input_label = None
        for (data, label) in iter(ds):
            adj, feats = data
            if not infer_model.built:
                print("Building model ...")
                infer_model((adj, feats))
                print("Loading weights ...")
                infer_model.load_weights(infer_model_weights_path)
                print()
                infer_model.summary()

            res = infer_model((adj, feats)).numpy()
            res_idx = np.argmax(res, axis=-1)

            if preds is None:
                preds = np.array([0] * res.shape[1])
            for batch in range(res.shape[0]):
                num_preds += 1
                preds[res_idx[batch]] += 1

            assert input_label is None or label[0] == input_label
            input_label = label[0]

        if preds is not None and input_label is not None:
            cls = np.argmax(preds)
            if cls == input_label:
                correct += 1
            print("Pred: %s, Num: %d, Match Label: %d" % (preds, num_preds, cls == input_label))
            total += 1
        else:
            print("Skipping %s" % input_file)

    accuracy = float(correct) / total
    print("Total: %d, Accuracy: %f" % (total, accuracy))
    return accuracy


def evaluate_all(tfrecord_kfold_dirs,
                 infer_model_config_paths,
                 infer_model_weights_dirs):
    """Calculate accuracies for all tfrecord over kfold splits
    """
    accuracies = []
    for i, (tfrecord_kfold_dir, infer_model_config_path, infer_model_weights_dir) \
            in enumerate(zip(tfrecord_kfold_dirs.split(","), infer_model_config_paths.split(","), infer_model_weights_dirs.split(","))):
        weights_map = {}
        for path in glob.glob(os.path.join(infer_model_weights_dir, "*.h5")):
            if os.path.isfile(path):
                weights_map[int(path.split(".")[0].split("_")[-1])] = path

        print("Found weights for splits: %s" % sorted(weights_map.keys()))

        tfrecords_map = {}
        for dir in glob.glob(os.path.join(tfrecord_kfold_dir, "*")):
            if os.path.isdir(dir):
                tfrecords_map[int(dir.split("/")[-1].split("split")[-1])] = dir

        print("Found tfrecords for splits: %s" % sorted(tfrecords_map.keys()))

        splits = sorted(list(set(tfrecords_map.keys()) & set(weights_map.keys())))
        total_acc = 0.0
        for split in splits:
            print("Evaluating split %d" % split)
            accuracy = evaluate(tfrecords_map[split], infer_model_config_path,
                                weights_map[split], os.path.join(tfrecord_kfold_dir, "config_no_eig.json"))
            total_acc += accuracy
            accuracies += [accuracy]

        print("%d: Accuracy %f for %d splits" % (i, total_acc / len(splits), len(splits)))

    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)
    print("Accuracies: %s" % accuracies)
    print("Total Mean Accuracy %f +- %f across %d folds" % (mean_accuracy, std_dev, len(accuracies)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='Directory of input files', metavar='DIR')
    parser.add_argument('--model-config-path', dest='model_config_path',
                        help='Path to json config for model', metavar='PATH')
    parser.add_argument('--model-weights-path', dest='model_weights_path',
                        help='Path to weights for model', metavar='PATH')
    args = parser.parse_args()

    evaluate_all(args.input_dir, args.model_config_path, args.model_weights_path)
