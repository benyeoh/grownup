#!/usr/bin/env python
import os
import sys
import glob
import pickle
import gzip
import argparse
import json

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net", "misc"))

import bs4
from bs4 import BeautifulSoup
import numpy as np

import net.misc.prepare_dataset
import net.preprocess
import generate_cleaneval_html


def write_extraction_inputs(word_map,
                            tag_map,
                            in_html_path,
                            in_cleaned_path,
                            out_pkl_path,
                            out_cleaned_path):
    soup = generate_cleaneval_html.read_soup(in_html_path)
    net.misc.prepare_dataset.remove_comments(soup)

    # Fetch all text strings and try to keep correct spacings between each neighbouring text
    blk_elems = set(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                     "ul", "ol", "td", "dl", "pre", "hr", "blockquote", "address"])

    id_to_string = {}
    ordering_ids = []
    is_whitespace_last = True    
    for s in soup.html.next_elements:
        if type(s) == bs4.element.Tag:
            if s.name == "br" or s.name in blk_elems:
                is_whitespace_last = True
            elif s.tag_style:
                if s.tag_style["display"] == "block":
                    is_whitespace_last = True
        elif type(s) == bs4.element.NavigableString:
            text_str = str(s)
            split_str = text_str.split()
            if len(split_str) > 0:
                # We split strings into separate word tokens based on whether or not
                # there is a whitespace either in the beginning of this string or the end of the last string
                # or if the string starts on a new HTML block
                append_str = str(s).strip()
                if (is_whitespace_last or text_str[0].isspace()):
                    append_str = "\n" + append_str
                ordering_ids.append(s.id)
                id_to_string[s.id] = append_str
                
            if len(text_str) > 0:
                is_whitespace_last = text_str[-1].isspace()

    doc_feature_list = []
    id_list = []
    for leaf, tag_list, is_content in net.preprocess.get_leaves(soup.find_all('html')[0]):
        leaf_representation = net.preprocess.get_leaf_representation(leaf, tag_list, is_content)
        words_dict, tags_dict, _ = leaf_representation
        feature_vector = net.preprocess.get_feature_vector(words_dict, tags_dict, word_map, tag_map)
        doc_feature_list.append(feature_vector)
        id_list.append(leaf.id)

    with gzip.open(out_pkl_path, "wb") as fd:
        doc_feature_list = np.expand_dims(np.array(doc_feature_list), axis=0)
        pickle.dump((doc_feature_list, id_list, ordering_ids, id_to_string), fd, protocol=4)

    with open(out_cleaned_path, "w", encoding="utf-8") as fd:
        cleaned_text = generate_cleaneval_html.read_gold_std(in_cleaned_path)
        fd.write(cleaned_text)


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

    test_html_dir = os.path.join(args.input_dir, "test", "input")
    test_cleaned_dir = os.path.join(args.input_dir, "test", "gold_std")
    html_file_tup = generate_cleaneval_html.find_files(test_html_dir, test_cleaned_dir)

    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        write_extraction_inputs(word_map,
                                tag_map,
                                html_path,
                                cleaned_path,
                                os.path.join(args.output_dir, filename + ".pkl.gz"),
                                os.path.join(args.output_dir, filename + ".txt"))
    print("Done!")
