#!/usr/bin/env python
import argparse
import os
import shutil
import glob
import magic
import re
import logging

from dragnet.data_processing import extract_all_gold_standard_data


def _read_file(path):
    with open(path, "rb") as fd:
        m = magic.Magic(mime_encoding=True)
        doc = fd.read()
        try:
            doc = doc.decode(m.from_buffer(doc))
        except:
            try:
                doc = doc.decode("cp1252")
            except UnicodeDecodeError:
                doc = doc.decode("iso-8859-1")
        return doc


def _process_cleaneval(raw_dir, out_dir, dev_subdir, test_subdir):
    html_dev = glob.glob(os.path.join(raw_dir, dev_subdir, "en_original", "*.html"))
    cleaned_dev = glob.glob(os.path.join(raw_dir, dev_subdir, "en_cleaned", "*.txt"))

    html_test = glob.glob(os.path.join(raw_dir, test_subdir, "input", "*.html"))
    cleaned_test = glob.glob(os.path.join(raw_dir, test_subdir, "gold_std", "*.txt"))

    def _process_files(html_files, cleaned_files, out_subdir):
        corrected_path = os.path.join(out_dir, out_subdir, "Corrected")
        html_path = os.path.join(out_dir, out_subdir, "HTML")

        os.makedirs(corrected_path, exist_ok=True)
        os.makedirs(html_path, exist_ok=True)

        cleaned_map = {}
        for cleaned in cleaned_files:
            filename = os.path.basename(cleaned).split(".txt")[0].split("-")[0]
            cleaned_map[filename] = cleaned

        html_file_tup = []
        for html in html_files:
            filename = os.path.basename(html).split(".html")[0]
            if filename in cleaned_map:
                html_file_tup.append((html, cleaned_map[filename], filename))
            else:
                print("Skipping parsing %s" % html)

        for html, cleaned, filename in html_file_tup:
            txt = re.sub("(^|\n)(URL:.*)", "", _read_file(cleaned))
            txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)

            out_cleaned = os.path.join(corrected_path, filename + ".html.corrected.txt")
            with open(out_cleaned, "w", encoding="utf8") as fd:
                fd.write(txt)

            html_txt = _read_file(html)
            out_html = os.path.join(html_path, filename + ".html")
            with open(out_html, "w", encoding='utf8') as fd:
                fd.write(html_txt)

    print("Processing %s ..." % dev_subdir)
    _process_files(html_dev, cleaned_dev, dev_subdir)
    print("Processing %s ..." % test_subdir)
    _process_files(html_test, cleaned_test, test_subdir)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='raw_dir', help='Path to CleanEval raw directory', metavar='PATH',
                        default=None)
    parser.add_argument('-d', dest='dev_subdir', help='Name of CleanEval dev sub-directory', metavar='NAME',
                        default="dev")
    parser.add_argument('-t', dest='test_subdir', help='Name of CleanEval test sub-directory', metavar='NAME',
                        default="test")
    parser.add_argument('-o', dest='out_dir', help='Path to output directory', metavar='PATH',
                        default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    _process_cleaneval(args.raw_dir, args.out_dir, dev_subdir=args.dev_subdir, test_subdir=args.test_subdir)

    print("Extracting %s" % args.dev_subdir)
    os.makedirs(os.path.join(args.out_dir, args.dev_subdir, "block_corrected"), exist_ok=True)
    extract_all_gold_standard_data(os.path.join(args.out_dir, args.dev_subdir))

    print("Extracting %s" % args.test_subdir)
    os.makedirs(os.path.join(args.out_dir, args.test_subdir, "block_corrected"), exist_ok=True)
    extract_all_gold_standard_data(os.path.join(args.out_dir, args.test_subdir))
