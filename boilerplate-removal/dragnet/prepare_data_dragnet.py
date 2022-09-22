#!/usr/bin/env python
import argparse
import os
import shutil
import logging
import magic

from dragnet.data_processing import extract_all_gold_standard_data


def _read_file(path):
    with open(path, "rb") as fd:
        m = magic.Magic(mime_encoding=True)
        doc = fd.read()
        try:
            doc = doc.decode(m.from_buffer(doc))
        except:
            doc = doc.decode("iso-8859-1")
        return doc


def _parse_txt_list(filepath):
    src_dir = os.path.dirname(filepath)
    corrected_dir = os.path.join(src_dir, "Corrected")
    html_dir = os.path.join(src_dir, "HTML")

    with open(filepath, "r") as fd:
        all_files = []
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            corrected_filepath = os.path.join(corrected_dir, "%s.html.corrected.txt" % line)
            html_filepath = os.path.join(html_dir, "%s.html" % line)
            if os.path.exists(corrected_filepath) and os.path.exists(html_filepath):
                all_files.append((html_filepath, corrected_filepath))
            else:
                print("Skipping %s" % line)
        print("Total %s: %d" % (filepath, len(all_files)))
        return all_files


def _copy_data(data, out_dir):
    out_html_dir = os.path.join(out_dir, "HTML")
    out_corrected_dir = os.path.join(out_dir, "Corrected")
    os.makedirs(out_html_dir, exist_ok=True)
    os.makedirs(out_corrected_dir, exist_ok=True)

    for html_filepath, corrected_filepath in data:
        print("Copying %s and %s to %s" % (html_filepath, corrected_filepath, out_dir))
        doc = _read_file(html_filepath)
        filename = os.path.basename(html_filepath)
        with open(os.path.join(out_html_dir, filename), "w", encoding="utf-8") as fd:
            fd.write(doc)
        doc = _read_file(corrected_filepath)
        filename = os.path.basename(corrected_filepath)
        with open(os.path.join(out_corrected_dir, filename), "w", encoding="utf-8") as fd:
            fd.write(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='raw_dir', help='Path to data directory', metavar='PATH',
                        default=None)
    parser.add_argument('-o', dest='out_dir', help='Path to output directory', metavar='PATH',
                        default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print("Copying training data")
    train_out_dir = os.path.join(args.out_dir, "train")
    training_data = _parse_txt_list(os.path.join(args.raw_dir, "training.txt"))
    _copy_data(training_data, train_out_dir)

    print("Copying test data")
    test_out_dir = os.path.join(args.out_dir, "test")
    test_data = _parse_txt_list(os.path.join(args.raw_dir, "test.txt"))
    _copy_data(test_data, test_out_dir)

    os.makedirs(os.path.join(train_out_dir, "block_corrected"), exist_ok=True)
    os.makedirs(os.path.join(test_out_dir, "block_corrected"), exist_ok=True)

    print("Extracting train data")
    extract_all_gold_standard_data(train_out_dir)

    print("Extracting test data")
    extract_all_gold_standard_data(test_out_dir)
