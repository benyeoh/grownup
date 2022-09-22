#!/usr/bin/env python
import os
import sys
import difflib
import glob
import argparse
import re
import random

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net", "misc"))

import bs4
from bs4 import BeautifulSoup

import net.misc.prepare_dataset


def read_file(path):
    with open(path, "rb") as fd:
        doc = fd.read()
        try:
            res = doc.decode("utf-8")
        except:
            try:
                res = doc.decode("cp1252")
            except UnicodeDecodeError:
                res = doc.decode("iso-8859-1")
    return res


def read_gold_std(path):
    txt = read_file(path)
    txt = re.sub("(^|\n)(URL:.*)", "", txt)
    txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)
    return txt


def soup_txt_to_list(soup):
    blk_elems = set(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                     "ul", "ol", "td", "dl", "pre", "hr", "blockquote", "address"])

    filtered_node = []
    filtered_str = []
    is_whitespace_last = True
    #last_blk_parent = None
    for s in soup.html.next_elements:
        if type(s) == bs4.element.Tag:
            if s.name == "br" or s.name in blk_elems:
                is_whitespace_last = True
            elif s.tag_style:
                if s.tag_style["display"] == "block":
                    is_whitespace_last = True

        elif type(s) == bs4.element.NavigableString:
            text_str = str(s)
            split_str = str(s).split()
            #blk_parent = _find_blk_parent(s)
            if len(split_str) > 0:
                # We split strings into separate word tokens based on whether or not
                # there is a whitespace either in the beginning of this string or the end of the last string
                # or if the string starts on a new HTML block
                if (is_whitespace_last or text_str[0].isspace()):  # or blk_parent != last_blk_parent):
                    filtered_str.extend(split_str)
                    # We also store owner tags for each word token that this string belongs to
                    filtered_node.extend([s for _ in range(len(split_str))])
                else:
                    # We assume that there is no whitespace between this string and the previous
                    filtered_str[-1] += split_str[0]
                    if not isinstance(filtered_node[-1], list):
                        filtered_node[-1] = [filtered_node[-1]]
                    filtered_node[-1].append(s)

                    if len(split_str) > 1:
                        filtered_str.extend(split_str[1:])
                        filtered_node.extend([s for _ in range(len(split_str[1:]))])

            #last_blk_parent = blk_parent
            if len(text_str) > 0:
                is_whitespace_last = text_str[-1].isspace()

    return filtered_str, filtered_node


def get_lcs_counts(soup, toks, tok_tags, gold_std_toks):
    """Get the longest common subsequence (LCS) comparing word tokens from `gold_std_toks` to `toks`, then
    count tokens owned by HMTL tags that appear in the LCS.
    """
    seq_matcher = difflib.SequenceMatcher(None, toks, gold_std_toks, autojunk=False)

    # Get the LCS assemble the word tokens in a list
    lcs = [tok for block in seq_matcher.get_matching_blocks() for tok in toks[block.a:(block.a + block.size)]]

    # Get the same LCS but assemble the relevant owner tags of each token in a list
    lcs_tags = [tag for block in seq_matcher.get_matching_blocks()
                for tag in tok_tags[block.a:(block.a + block.size)]]

    # Now we count the total number of tokens that each tag owns, and how many of these tokens belong in the LCS
    tag_tok_count = {}
    tag_tok_lcs_count = {}
    for s in soup.html.find_all_next(string=True):
        if type(s) == bs4.element.NavigableString:
            if len(str(s).split()) > 0:
                if s.id not in tag_tok_count:
                    tag_tok_count[s.id] = 0
                    tag_tok_lcs_count[s.id] = 0
                tag_tok_count[s.id] += len(str(s).split())

    for tag in lcs_tags:
        if isinstance(tag, list):
            for t in tag:
                tag_tok_lcs_count[t.id] += 1
        else:
            tag_tok_lcs_count[tag.id] += 1

    return tag_tok_count, tag_tok_lcs_count, lcs


def read_soup(html_path):
    soup = BeautifulSoup(read_file(html_path), features='html5lib')
    net.misc.prepare_dataset.remove_comments(soup)

    for i, s in enumerate(soup.html.find_all_next(string=True)):
        if type(s) == bs4.element.NavigableString:
            setattr(s, "id", i)
    return soup


def write_generated_html(in_html_path, in_cleaned_path, out_html_path):
    soup = read_soup(in_html_path)
    gold_std = read_gold_std(in_cleaned_path)
    gold_std_toks = gold_std.split()
    toks, tok_tags = soup_txt_to_list(soup)
    tag_tok_count, tag_tok_lcs_count, lcs = get_lcs_counts(soup, toks, tok_tags, gold_std_toks)

    content_nodes = []
    non_content_nodes = []
    for s in soup.html.find_all_next(string=True):
        if type(s) == bs4.element.NavigableString:
            if s.id in tag_tok_lcs_count:
                lcs_ratio = float(tag_tok_lcs_count[s.id]) / tag_tok_count[s.id]
                if lcs_ratio > 0:
                    content_nodes.append(s)
                else:
                    non_content_nodes.append(s)

    for s in content_nodes:
        span = soup.new_tag('span')
        s.wrap(span)
        span["__boilernet_label"] = 1

    for s in non_content_nodes:
        span = soup.new_tag('span')
        s.wrap(span)
        span["__boilernet_label"] = 0

    with open(out_html_path, "w", encoding="utf-8") as fd:
        fd.write(soup.prettify())
    print("%d vs %d" % (len(content_nodes), len(non_content_nodes)))


def find_files(input_html_dir, gold_std_dir):
    html_test = glob.glob(os.path.join(input_html_dir, "*.html"))
    cleaned_test = glob.glob(os.path.join(gold_std_dir, "*.txt"))

    cleaned_map = {}
    for cleaned in cleaned_test:
        filename = os.path.basename(cleaned).split(".")[0].split("-")[0]
        cleaned_map[filename] = cleaned

    html_file_tup = []
    for html in html_test:
        filename = os.path.basename(html).split(".")[0]
        if filename in cleaned_map:
            html_file_tup.append((html, cleaned_map[filename], filename))
        else:
            print("Skipping parsing %s" % html)
    return html_file_tup


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

    train_html_dir = os.path.join(args.input_dir, "dev", "en_original")
    train_cleaned_dir = os.path.join(args.input_dir, "dev", "en_cleaned")
    html_file_tup = find_files(train_html_dir, train_cleaned_dir)
    random.shuffle(html_file_tup)

    num_valid = int(len(html_file_tup) * args.val_ratio)
    train_dataset = html_file_tup[num_valid:]
    valid_dataset = html_file_tup[:num_valid]

    print("Train / Dev Set")
    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        html_out_path = os.path.join(out_train_dir, filename + ".html")
        write_generated_html(html_path, cleaned_path, html_out_path)

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
    test_html_dir = os.path.join(args.input_dir, "test", "input")
    test_cleaned_dir = os.path.join(args.input_dir, "test", "gold_std")
    html_file_tup = find_files(test_html_dir, test_cleaned_dir)

    for i, (html_path, cleaned_path, filename) in enumerate(html_file_tup):
        print("%d - Processing: %s, %s" % (i + 1, html_path, cleaned_path))
        html_out_path = os.path.join(out_test_dir, filename + ".html")
        write_generated_html(html_path, cleaned_path, html_out_path)

    test_dataset = html_file_tup
    out_test_set_path = os.path.join(out_split_dir, "test_set.txt")
    with open(out_test_set_path, "w", encoding="utf-8") as fd:
        for _, _, filename in test_dataset:
            fd.write(filename + ".html" + "\n")

    print("Done!")
