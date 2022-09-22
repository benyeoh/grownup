#!/usr/bin/env python
import os
import glob
import sys
import json
import argparse


def get_bag_of_words(filepath):
    with open(filepath, "r", encoding="utf-8") as fd:
        text = fd.read()
        return set(text.split())


def get_bag_of_unique_words(filepath):
    with open(filepath, "r", encoding="utf-8") as fd:
        text = fd.read()
        text_map = {}
        for token in text.split():
            if token not in text_map:
                text_map[token] = 0
            text_map[token] += 1
        unique_word_set = set()
        for token, count in text_map.items():
            for i in range(count):
                unique_word_set.add(token + ("__#$@_%d_@$#" % i))
        return unique_word_set


def get_confusion_matrix(filepath_pred, filepath_gt, use_unique_words=True):
    bag_of_words_fn = get_bag_of_unique_words if use_unique_words else get_bag_of_words
    bag_of_words_pred = bag_of_words_fn(filepath_pred)
    bag_of_words_gt = bag_of_words_fn(filepath_gt)
    true_positives = bag_of_words_pred & bag_of_words_gt
    false_positives = bag_of_words_pred - true_positives
    false_negatives = bag_of_words_gt - true_positives
    true_negatives = set()
    return true_positives, true_negatives, false_positives, false_negatives


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='extracted_dir',
                        help='Directory containing extracted text', metavar='DIR',
                        default=None)
    parser.add_argument('-g', dest='gt_dir', help='Directory to ground-truth text', metavar='DIR',
                        default=None)
    parser.add_argument('--unique', dest='unique_words', help='Use unique words to compute BOW',
                        action="store_true", default=False)

    args = parser.parse_args()

    all_gt = glob.glob(os.path.join(args.gt_dir, "*.txt"))
    all_extracted = sorted(glob.glob(os.path.join(args.extracted_dir, "*.txt")))

    gt_map = {}
    for gt in all_gt:
        gt_map[os.path.basename(gt).split(".")[0]] = gt

    all_extracted_set = set()

    # Compute micro-f1
    num_tp = 0
    num_fp = 0
    num_tn = 0
    num_fn = 0

    computed = set()
    skipped = set()
    for extracted in all_extracted:
        name = os.path.basename(extracted).split(".")[0]
        all_extracted_set.add(name)
        if name in gt_map:
            assert extracted not in computed
            computed.add(extracted)

            tp, tn, fp, fn = get_confusion_matrix(extracted, gt_map[name], args.unique_words)
            num_tp += len(tp)
            num_tn += len(tn)
            num_fp += len(fp)
            num_fn += len(fn)
        else:
            assert extracted not in skipped
            print("Skipped: %s" % extracted)
            skipped.add(extracted)

    print("Processed: %d, Skipped: %d" % (len(computed), len(skipped)))
    not_in_extracted = set(gt_map.keys()) - all_extracted_set
    print("Not in extracted: %d, %s" % (len(not_in_extracted), list(not_in_extracted)))
    prec, recall, f1 = compute_precision_recall_f1(num_tp, num_fp, num_fn)
    print("Precision: %.5f, Recall: %.5f, F1: %.5f" % (prec, recall, f1))
