#!/usr/bin/env python
import argparse
import os
import glob
import magic
import re
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "python-goose"))

import goose


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


def extract_content(html_dir, output_dir, cleaned_dir=None):
    htmls = glob.glob(os.path.join(html_dir, "*.html"))
    cleaned = None
    if cleaned_dir:
        cleaned = glob.glob(os.path.join(cleaned_dir, "*.txt"))
        gt_map = {os.path.basename(clean).split(".")[0].split("-cleaned")[0]: clean for clean in cleaned}
        filtered_htmls = []
        for html in htmls:
            if os.path.basename(html).split(".html")[0] in gt_map:
                filtered_htmls.append(html)
        htmls = filtered_htmls

    if not os.path.exists(os.path.join(output_dir, "extracted")):
        os.makedirs(os.path.join(output_dir, "extracted"))

    g = goose.Goose(config={"enable_image_fetching": False})

    print("Writing extracted text ...")
    for i, html in enumerate(htmls):
        print("%d: Processing %s ..." % (i, html))
        try:
            article = g.extract(raw_html=_read_file(html))
            outpath = os.path.join(output_dir, "extracted", os.path.basename(html).split(".html")[0] + ".txt")
            with open(outpath, "w") as fd:
                fd.write(article.cleaned_text.encode("utf-8"))
        except AssertionError:
            print("Skipping %s" % html)

    if cleaned:
        print("Copying cleaned text ...")
        if not os.path.exists(os.path.join(output_dir, "gt")):
            os.makedirs(os.path.join(output_dir, "gt"))
        for i, clean in enumerate(cleaned):
            print("%d: Processing %s ..." % (i, clean))
            txt = _read_file(clean)
            txt = re.sub("(^|\n)(URL:.*)", "", txt)
            txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)
            outpath = os.path.join(output_dir, "gt", os.path.basename(
                clean).split(".")[0].split("-cleaned")[0] + ".txt")
            with open(outpath, "w") as fd:
                fd.write(txt.encode("utf-8"))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='html_dir', help='Path to html directory', metavar='PATH')
    parser.add_argument('-c', dest='cleaned_dir', help='Path to cleaned directory', metavar='NAME',
                        default=None)
    parser.add_argument('-o', dest='out_dir', help='Path to output directory', metavar='PATH')
    args = parser.parse_args()

    extract_content(args.html_dir, args.out_dir, args.cleaned_dir)
