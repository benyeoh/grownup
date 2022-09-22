#!/usr/bin/env python
import os
import sys
import argparse

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "web2text", "src", "main", "python"))

import tensorflow as tf
import numpy as np

#import main
from forward import EDGE_VARIABLES, UNARY_VARIABLES, edge, loss, unary
from shuffle_queue import ShuffleQueue
from viterbi import viterbi


BATCH_SIZE = 128
PATCH_SIZE = 9
N_FEATURES = 128
N_EDGE_FEATURES = 25
TRAIN_STEPS = 8000
LEARNING_RATE = 1e-3
DROPOUT_KEEP_PROB = 0.8
REGULARIZATION_STRENGTH = 0.000
EDGE_LAMBDA = 1


def get_batch(q, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE):
    """Takes a batch from a ShuffleQueue of documents"""
    # Define empty matrices for the return data

    batch = np.zeros((BATCH_SIZE, PATCH_SIZE, 1, N_FEATURES), dtype=np.float32)
    labels = np.zeros((BATCH_SIZE, PATCH_SIZE, 1, 1), dtype=np.float32)
    edge_batch = np.zeros((BATCH_SIZE, PATCH_SIZE - 1, 1, N_EDGE_FEATURES), dtype=np.float32)
    edge_labels = np.zeros((BATCH_SIZE, PATCH_SIZE - 1, 1, 1), dtype=np.int64)

    for entry in range(BATCH_SIZE):
        # Find an entry that is long enough (at least one patch size)
        while True:
            doc = q.takeOne()
            # print(doc)
            length = doc['data'].shape[0]
            if length > PATCH_SIZE + 1:
                break

        # Select a random patch
        i = np.random.random_integers(length - PATCH_SIZE - 1)

        # Add it to the tensors
        batch[entry, :, 0, :] = doc['data'][i:i + PATCH_SIZE, :]
        edge_batch[entry, :, 0, :] = doc['edge_data'][i:i + PATCH_SIZE - 1, :]
        labels[entry, :, 0, 0] = doc['labels'][i:i + PATCH_SIZE]  # {0,1}
        edge_labels[entry, :, 0, 0] = doc['edge_labels'][i:i + PATCH_SIZE - 1]  # {0,1,2,3} = {00,01,10,11}

    return batch, edge_batch, labels, edge_labels


def evaluate_edge(dataset, prediction_fn):
    correct, incorrect = 0, 0
    for doc in dataset:
        predictions = prediction_fn(doc['edge_data'])

        for i, lab in enumerate(doc['edge_labels']):
            if predictions[i] == lab:
                correct += 1
            else:
                incorrect += 1

    return float(correct) / (correct + incorrect)


def train_edge(train_path, output_dir, valid_ratio=0.1):
    dev_data = np.load(train_path, encoding="bytes", allow_pickle=True)
    num_valid = int(dev_data.shape[0] * valid_ratio)
    train_data = dev_data[num_valid:]
    valid_data = dev_data[:num_valid]
    print("Num train data: %d, num valid data: %d" % (train_data.shape[0], valid_data.shape[0]))
    training_queue = ShuffleQueue(train_data)

    data_shape = [BATCH_SIZE, PATCH_SIZE - 1, 1, N_EDGE_FEATURES]
    labs_shape = [BATCH_SIZE, PATCH_SIZE - 1, 1, 1]
    train_features = tf.placeholder(tf.float32, shape=data_shape)
    train_labels = tf.placeholder(tf.int64, shape=labs_shape)

    logits = edge(train_features,
                  is_training=True,
                  conv_weight_decay=REGULARIZATION_STRENGTH,
                  dropout_keep_prob=DROPOUT_KEEP_PROB)
    l = loss(tf.reshape(logits, [-1, 4]), tf.reshape(train_labels, [-1]))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(l)

    test_features = tf.placeholder(tf.float32)
    tf.get_variable_scope().reuse_variables()
    test_logits = edge(test_features, is_training=False)

    saver = tf.train.Saver(tf.get_collection(EDGE_VARIABLES))
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        # Initialize
        session.run(init_op)

        def prediction(features):
            features = features[np.newaxis, :, np.newaxis, :]
            logits = session.run(test_logits, feed_dict={test_features: features})
            return np.argmax(logits, axis=-1).flatten()

        BEST_VAL_SO_FAR = 0
        for step in range(TRAIN_STEPS + 1):
            # Construct a bs-length numpy array
            _, edge_features, labels, edge_labels = get_batch(training_queue)
            # Run a training step
            loss_val, _ = session.run(
                [l, train_op],
                feed_dict={train_features: edge_features, train_labels: edge_labels}
            )

            if step % 100 == 0:
                accuracy_validation = evaluate_edge(valid_data, prediction)
                accuracy_train = evaluate_edge(train_data, prediction)
                if accuracy_validation > BEST_VAL_SO_FAR:
                    best = True
                    saver.save(session, os.path.join(output_dir, 'edge.ckpt'))
                    BEST_VAL_SO_FAR = accuracy_validation
                else:
                    best = False
                print("%10d: train=%.4f, val=%.4f %s" % (step, accuracy_train, accuracy_validation, '*' if best else ''))
        # saver.save(session, os.path.join(CHECKPOINT_DIR, 'edge.ckpt'))
        return accuracy_validation


def evaluate_unary(dataset, prediction_fn):
    fp, fn, tp, tn = 0, 0, 0, 0
    for doc in dataset:
        predictions = prediction_fn(doc['data'], doc['edge_data'])

        for i, lab in enumerate(doc['labels']):
            if predictions[i] == 1 and lab == 1:
                tp += 1
            elif predictions[i] == 1 and lab == 0:
                fp += 1
            elif predictions[i] == 0 and lab == 1:
                fn += 1
            else:
                tn += 1

    n = fp + fn + tp + tn
    accuracy = float(tp + tn) / n
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    if (precision + recall) > 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return accuracy, precision, recall, f1


def train_unary(train_path, output_dir, valid_ratio=0.1):
    dev_data = np.load(train_path, encoding="bytes", allow_pickle=True)
    num_valid = int(dev_data.shape[0] * valid_ratio)
    train_data = dev_data[num_valid:]
    valid_data = dev_data[:num_valid]
    print("Num train data: %d, num valid data: %d" % (train_data.shape[0], valid_data.shape[0]))
    training_queue = ShuffleQueue(train_data)

    data_shape = [BATCH_SIZE, PATCH_SIZE, 1, N_FEATURES]
    labs_shape = [BATCH_SIZE, PATCH_SIZE, 1, 1]
    train_features = tf.placeholder(tf.float32, shape=data_shape)
    train_labels = tf.placeholder(tf.int64, shape=labs_shape)

    logits = unary(train_features,
                   is_training=True,
                   conv_weight_decay=REGULARIZATION_STRENGTH,
                   dropout_keep_prob=DROPOUT_KEEP_PROB)
    l = loss(tf.reshape(logits, [-1, 2]), tf.reshape(train_labels, [-1]))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(l)

    test_features = tf.placeholder(tf.float32)
    tf.get_variable_scope().reuse_variables()
    test_logits = unary(test_features, is_training=False)

    saver = tf.train.Saver(tf.get_collection(UNARY_VARIABLES))
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        # Initialize
        session.run(init_op)

        def prediction(features, edge_features):
            features = features[np.newaxis, :, np.newaxis, :]
            logits = session.run(test_logits, feed_dict={test_features: features})
            return np.argmax(logits, axis=-1).flatten()

        BEST_VAL_SO_FAR = 0
        for step in range(TRAIN_STEPS + 1):
            # Construct a bs-length numpy array
            features, _, labels, edge_labels = get_batch(training_queue)
            # Run a training step
            loss_val, _ = session.run(
                [l, train_op],
                feed_dict={train_features: features, train_labels: labels}
            )

            if step % 100 == 0:
                _, _, _, f1_validation = evaluate_unary(valid_data, prediction)
                _, _, _, f1_train = evaluate_unary(train_data, prediction)
                if f1_validation > BEST_VAL_SO_FAR:
                    best = True
                    saver.save(session, os.path.join(output_dir, 'unary.ckpt'))
                    BEST_VAL_SO_FAR = f1_validation
                else:
                    best = False                
                print("%10d: train=%.4f, val=%.4f %s" % (step, f1_train, f1_validation, '*' if best else ''))
        # saver.save(session, os.path.join(CHECKPOINT_DIR, 'unary.ckpt'))
        return f1_validation


def test_structured(test_path, ckpt_dir, lamb=EDGE_LAMBDA):
    test_data = np.load(test_path, encoding="bytes", allow_pickle=True)
    print("Num test data: %d" % test_data.shape[0])
    unary_features = tf.placeholder(tf.float32)
    edge_features = tf.placeholder(tf.float32)

    # hack to get the right shape weights
    _ = unary(tf.placeholder(tf.float32, shape=[1, PATCH_SIZE, 1, N_FEATURES]), False)
    _ = edge(tf.placeholder(tf.float32, shape=[1, PATCH_SIZE, 1, N_EDGE_FEATURES]), False)

    tf.get_variable_scope().reuse_variables()
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

        start = time()

        def prediction_structured(features, edge_feat):
            features = features[np.newaxis, :, np.newaxis, :]
            edge_feat = edge_feat[np.newaxis, :, np.newaxis, :]

            unary_lgts = session.run(unary_logits, feed_dict={unary_features: features})
            edge_lgts = session.run(edge_logits, feed_dict={edge_features: edge_feat})

            return viterbi(unary_lgts.reshape([-1, 2]), edge_lgts.reshape([-1, 4]), lam=lamb)

        def prediction_unary(features, _):
            features = features[np.newaxis, :, np.newaxis, :]
            logits = session.run(unary_logits, feed_dict={unary_features: features})
            return np.argmax(logits, axis=-1).flatten()

        accuracy, precision, recall, f1 = evaluate_unary(test_data, prediction_structured)
        accuracy_u, precision_u, recall_u, f1_u = evaluate_unary(test_data, prediction_unary)
        end = time()
        print('duration', end - start)
        print('size', len(test_data))
        print("Structured: Accuracy=%.5f, precision=%.5f, recall=%.5f, F1=%.5f" % (accuracy, precision, recall, f1))
        print("Just unary: Accuracy=%.5f, precision=%.5f, recall=%.5f, F1=%.5f" %
              (accuracy_u, precision_u, recall_u, f1_u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path of training .npy ', metavar="PATH")
    parser.add_argument('output_dir', help='Output directory of .ckpt files', metavar="DIR")
    parser.add_argument('test', help='Path of testing .npy', metavar="TEST_PATH")
    parser.add_argument('dropout', help='Dropout keep prob', nargs="?", type=float, metavar="dropout", default=0.8)
    #parser.add_argument('test', help='Path of testing .npy', nargs="?", metavar="TEST_PATH")
    args = parser.parse_args()

    global DROPOUT_KEEP_PROB
    DROPOUT_KEEP_PROB = args.dropout
    print("Dropout: %f" % DROPOUT_KEEP_PROB)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Training unary ...")
    train_unary(args.input, args.output_dir)

    print("Resetting default graph")
    tf.reset_default_graph()

    print("Training edge ...")
    train_edge(args.input, args.output_dir)

    print("Resetting default graph")
    tf.reset_default_graph()

    print("Evaluating test")
    test_structured(args.test, args.output_dir)


if __name__ == "__main__":
    main()
