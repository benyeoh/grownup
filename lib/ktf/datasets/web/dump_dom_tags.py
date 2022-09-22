#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))

import glob
import optparse

import ktf.datasets.web


def dump_dom_tags(html_files, threshold, path=None):
    tags, counts, unsup_tags, unsup_counts = ktf.datasets.web.from_file_filter_dom_tags(html_files, threshold=threshold)

    print("======= Tags =========")
    for i, (tag, count) in enumerate(zip(tags, counts)):
        print("%d: %s, %d" % (i, tag, count))
    print()
    print("======= Unsupported Tags =========")
    for i, (tag, count) in enumerate(zip(unsup_tags, unsup_counts)):
        print("%d: %s, %d" % (i, tag, count))

    if path:
        with open(path, "w") as fd:
            fd.writelines(["%s\n" % tag for tag in tags])


if __name__ == "__main__":

    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-d', dest='dirs', help='HTML data directories (comma delimited)', metavar='DIR')
    opt_parser.add_option('-t', dest='threshold', help='Min threshold of occurences to filter',
                          metavar='NUM', default=0)
    opt_parser.add_option('-o', dest='path', help='Output tags to file', metavar='PATH', default=None)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.print_help()
        exit()

    all_files = []
    for dir in opt_args.dirs.split(","):
        print("Processing directory: %s" % dir)
        all_files.extend([path for path in glob.glob(os.path.join(
            dir, "**", "*.html"), recursive=True) if os.path.isfile(path)])
    dump_dom_tags(all_files, int(opt_args.threshold), opt_args.path)
    print("\nDone!\n")
