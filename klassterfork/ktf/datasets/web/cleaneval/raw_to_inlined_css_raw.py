#!/usr/bin/env python
import os
import glob
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import multiprocessing
import numpy as np
import optparse

import ktf.datasets
import ktf.datasets.web


def inline_css(path_pair):
    html_path = path_pair[0]
    out_path = path_pair[1]

    print("Reading file: %s" % html_path)
    soup = ktf.datasets.web.to_soup_from_file(html_path, inline_css=True)
    with open(out_path, "w") as fd:
        print("Writing to: %s" % out_path)
        fd.write(soup.prettify())

    return 1


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-d', dest='orig_dir', help='Original web data directory',
                          metavar='DIR', default=None)
    opt_parser.add_option('-n', dest='num_proc', help='Number of parallel processors',
                          metavar='NUM', type='int', default=16)
    opt_parser.add_option('-o', dest='out_dir', help='Output directory',
                          metavar='DIR', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../out'))

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.print_help()
        exit()

    all_orig = glob.glob(os.path.join(opt_args.orig_dir, "*.html"))
    os.makedirs(opt_args.out_dir, exist_ok=True)

    all_output = []
    for path in all_orig:
        out_filename = os.path.basename(path).split(".")[0] + "_inlined.html"
        outpath = os.path.join(opt_args.out_dir, out_filename)
        all_output.append(outpath)

    with multiprocessing.Pool(opt_args.num_proc) as p:
        res = p.map(inline_css, list(zip(all_orig, all_output)))
    p.close()
    p.join()
    print("Done! Written: %d" % sum(res))
