#!/usr/bin/env python
import glob
import optparse
import os
import sys
import shutil
import random

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))


def get_input_files(input_dir):
    """Get the input webpages dictionary:
    - Key: class label
    - Values: list of path for each webpage under current class
    """
    input_files = {}
    total_num_classes = 0
    total_num_pages = 0

    classes_dir = [path for path in sorted(glob.glob(os.path.join(input_dir, "*"))) if os.path.isdir(path)]

    for class_dir in classes_dir:
        class_label = class_dir.split('/')[-1]
        total_num_classes += 1

        paths = (glob.glob(os.path.join(input_dir, class_dir, "*.html")) +
                 glob.glob(os.path.join(input_dir, class_dir, "*.tfrecord")))

        input_files[class_label] = [path for path in paths if os.path.isfile(path)]
        total_num_pages += len(input_files[class_label])

    print("Total num classes: %d" % total_num_classes)
    print("Total num pages: %d" % total_num_pages)
    return input_files


def split_k_folds(k, input_files, seed=None):
    """Random split all webpages into k folders
    - random.seed: ensure that the data is divided the same way every time the code is run
    - random.randint: ensure the starting subdir to save webpage are randomized
    - random.shuffle: ensure the raw input are shuffled and then splitted
    """
    folds = [[] for _ in range(k)]

    if seed:
        random.seed(seed)

    start_idx = random.randint(0, k - 1)
    for _, pages in input_files.items():
        random.shuffle(pages)
        for page in pages:
            folds[start_idx % k].append(page)
            start_idx += 1

    for i in range(len(folds)):
        print("%d fold: %d" % (i, len(folds[i])))
    return folds


def write_outputs(folds, root_dir, out_dir):
    for i, fold in enumerate(folds):
        for path in fold:
            subdir = "/".join(path.split(os.path.join(root_dir, ""))[-1].split("/")[0:-1])
            out_subdir = os.path.join(out_dir, "split%d" % i, subdir, "")
            os.makedirs(out_subdir, exist_ok=True)
            print("%d fold - Copying %s to %s" % (i, path, out_subdir))
            shutil.copy(path, out_subdir)


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-i', dest='input_dir', help='Root directory for source files organized by class / website',
                          metavar='DIR', default=None)
    opt_parser.add_option('-k', dest='k', help='Num folds',
                          metavar='INT', type='int', default=10)
    opt_parser.add_option('-s', dest='seed', help='Random seed',
                          metavar='INT', type='int', default=None)
    opt_parser.add_option('-o', dest='out_dir', help='Output directory',
                          metavar='DIR', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../out'))

    (opt_args, args) = opt_parser.parse_args()

    input_files = get_input_files(opt_args.input_dir)
    folds = split_k_folds(opt_args.k, input_files, seed=opt_args.seed)
    write_outputs(folds, opt_args.input_dir, opt_args.out_dir)
    print("Done")
