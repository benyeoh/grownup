#!/usr/bin/env python
import os
import glob
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import numpy as np
import optparse
import tensorflow as tf

import ktf.datasets
import ktf.datasets.web


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-l', dest='html_dir', help='Original html directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-c', dest='cleaned_dir', help='Cleaned txt data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-o', dest='out_html_dir', help='Output HTML directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-p', dest='out_cleaned_dir', help='Output cleaned txt data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-f', dest='split_txt_path', help='Input txt file with splits',
                          metavar='PATH', default=None)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.print_help()
        exit()

    np.seterr(all="raise")

    html_files = {}
    print("Collecting HTML at %s ..." % opt_args.html_dir)
    for filepath in glob.glob(os.path.join(opt_args.html_dir, "*.html")):
        filename = os.path.basename(filepath).split(".")[0]
        html_files[filename] = filepath

    cleaned_files = {}
    print("Collecting cleaned txt at %s ..." % opt_args.cleaned_dir)
    for filepath in glob.glob(os.path.join(opt_args.cleaned_dir, "*.txt")):
        filename = os.path.basename(filepath).split(".")[0]
        cleaned_files[filename] = filepath

    if not os.path.exists(opt_args.out_html_dir):
        os.makedirs(opt_args.out_html_dir)

    if not os.path.exists(opt_args.out_cleaned_dir):
        os.makedirs(opt_args.out_cleaned_dir)

    with open(opt_args.split_txt_path, "r") as fd:
        for i, file_id in enumerate(fd.readlines()):
            file_id = file_id.strip()
            src_html = html_files[file_id]
            dst_html = os.path.join(opt_args.out_html_dir, os.path.basename(src_html))
            print("%d: Copying %s to %s" % (i, src_html, dst_html))
            shutil.copyfile(src_html, dst_html)

            src_txt = cleaned_files[file_id]
            dst_txt = os.path.join(opt_args.out_cleaned_dir, os.path.basename(src_txt))
            print("%d: Copying %s to %s" % (i, src_txt, dst_txt))
            shutil.copyfile(src_txt, dst_txt)

    print("Done!")
