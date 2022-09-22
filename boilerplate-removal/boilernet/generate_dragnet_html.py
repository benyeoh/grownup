#!/usr/bin/env python
import os
import sys
import glob
import random
import argparse

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ""))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net", "misc"))

import generate_cleaneval_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='Input top level directory of raw files', metavar="DIR")
    parser.add_argument('-o', dest='output_dir', help='Output directory of processed html files', metavar="DIR")
    parser.add_argument('--val-ratio', dest='val_ratio', help='Validation ratio', metavar="NUM", type=float)
    args = parser.parse_args()

    out_train_dir = os.path.join(args.output_dir, "train")
    out_test_dir = os.path.join(args.output_dir, "test")
    out_split_dir = os.path.join(args.output_dir, "split")

    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_split_dir, exist_ok=True)

    train_html_dir = os.path.join(args.input_dir, "train_html")
    train_cleaned_dir = os.path.join(args.input_dir, "train_cleaned")
    html_file_tup = generate_cleaneval_html.find_files(train_html_dir, train_cleaned_dir)
    random.shuffle(html_file_tup)

    num_valid = int(len(html_file_tup) * args.val_ratio)
    train_dataset = html_file_tup[num_valid:]
    valid_dataset = html_file_tup[:num_valid]

    print("Train / Dev Set")
    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        html_out_path = os.path.join(out_train_dir, filename + ".html")
        generate_cleaneval_html.write_generated_html(html_path, cleaned_path, html_out_path)

    out_train_set_path = os.path.join(out_split_dir, "train_set.txt")
    out_valid_set_path = os.path.join(out_split_dir, "dev_set.txt")

    with open(out_train_set_path, "w", encoding="utf-8") as fd:
        for _, _, filename in train_dataset:
            fd.write(filename + ".html" + "\n")

    with open(out_valid_set_path, "w", encoding="utf-8") as fd:
        for _, _, filename in valid_dataset:
            fd.write(filename + ".html" + "\n")

    # Test
    print("Test")
    test_html_dir = os.path.join(args.input_dir, "test_html")
    test_cleaned_dir = os.path.join(args.input_dir, "test_cleaned")
    html_file_tup = generate_cleaneval_html.find_files(test_html_dir, test_cleaned_dir)

    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        html_out_path = os.path.join(out_test_dir, filename + ".html")
        generate_cleaneval_html.write_generated_html(html_path, cleaned_path, html_out_path)

    test_dataset = html_file_tup
    out_test_set_path = os.path.join(out_split_dir, "test_set.txt")
    with open(out_test_set_path, "w", encoding="utf-8") as fd:
        for _, _, filename in test_dataset:
            fd.write(filename + ".html" + "\n")

    print("Done!")
