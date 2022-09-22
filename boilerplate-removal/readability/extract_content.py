#!/usr/bin/env python
import argparse
import os
import glob
import magic
import re

import readability
from readabilipy import simple_json_from_html_string
import bs4
from bs4 import BeautifulSoup


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


class ReadabilityPy:

    def extract(self, html):
        doc = readability.Document(_read_file(html))
        soup = BeautifulSoup(doc.summary(), "html.parser")
        if soup.html:
            # for s in soup.html.find_all_next(string=True):
            #     if type(s) == bs4.element.NavigableString:
            #         if len(str(s).split()) > 0:
            #             text.append(str(s))
            return soup.get_text()
        return None


class ReadabilityJS:

    def extract(self, html):
        try:
            article = simple_json_from_html_string(_read_file(html), use_readability=True)
            soup = BeautifulSoup(article["plain_content"], "html.parser")
            if soup:
                # for s in soup.html.find_all_next(string=True):
                #     if type(s) == bs4.element.NavigableString:
                #         if len(str(s).split()) > 0:
                #             text.append(str(s))
                return soup.get_text()
        except:
            pass
        return None


def extract_content(html_dir, output_dir, cleaned_dir=None, extractor=ReadabilityPy()):
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

    os.makedirs(os.path.join(output_dir, "extracted"), exist_ok=True)

    print("Writing extracted text ...")
    for i, html in enumerate(htmls):
        print("%d: Processing %s ..." % (i, html))
        text = extractor.extract(html)
        if text:
            outpath = os.path.join(output_dir, "extracted", os.path.basename(html).split(".html")[0] + ".txt")
            with open(outpath, "w", encoding="utf-8") as fd:
                # fd.write("\n".join(text))
                fd.write(text)
        else:
            print("Skipping %s" % html)

    if cleaned:
        print("Copying cleaned text ...")
        os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
        for i, clean in enumerate(cleaned):
            print("%d: Processing %s ..." % (i, clean))
            txt = _read_file(clean)
            txt = re.sub("(^|\n)(URL:.*)", "", txt)
            txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)
            outpath = os.path.join(output_dir, "gt", os.path.basename(
                clean).split(".")[0].split("-cleaned")[0] + ".txt")
            with open(outpath, "w", encoding="utf-8") as fd:
                fd.write(txt)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--js', dest='use_js', help='Use Readability JS version',
                        action='store_true', default=False)
    parser.add_argument('-i', dest='html_dir', help='Path to html directory', metavar='PATH')
    parser.add_argument('-c', dest='cleaned_dir', help='Path to cleaned directory', metavar='NAME',
                        default=None)
    parser.add_argument('-o', dest='out_dir', help='Path to output directory', metavar='PATH')
    args = parser.parse_args()

    if args.use_js:
        extract_content(args.html_dir, args.out_dir, args.cleaned_dir, extractor=ReadabilityJS())
    else:
        extract_content(args.html_dir, args.out_dir, args.cleaned_dir, extractor=ReadabilityPy())
