#!/usr/bin/env python
import os
import sys
import argparse
import json

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ""))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net", "misc"))

import generate_cleaneval_html
import generate_cleaneval_extract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='Input directory of raw files', metavar="DIR")
    parser.add_argument('-o', dest='output_dir', help='Output directory of processed files', metavar="DIR")
    parser.add_argument('--word-map', dest='word_map', help='Path to the word map .json', metavar="PATH")
    parser.add_argument('--tag-map', dest='tag_map', help='Path to the tag map .json', metavar="PATH")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.word_map), 'r', encoding='utf-8') as fp:
        word_map = json.load(fp)

    with open(os.path.join(args.tag_map), 'r', encoding='utf-8') as fp:
        tag_map = json.load(fp)

    test_html_dir = os.path.join(args.input_dir, "test_html")
    test_cleaned_dir = os.path.join(args.input_dir, "test_cleaned")
    html_file_tup = generate_cleaneval_html.find_files(test_html_dir, test_cleaned_dir)

    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        generate_cleaneval_extract.write_extraction_inputs(word_map,
                                                           tag_map,
                                                           html_path,
                                                           cleaned_path,
                                                           os.path.join(args.output_dir, filename + ".pkl.gz"),
                                                           os.path.join(args.output_dir, filename + ".txt"))
    print("Done!")
