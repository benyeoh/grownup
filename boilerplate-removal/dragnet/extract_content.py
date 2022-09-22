#!/usr/bin/env python
import argparse
import logging
import os
import glob

from dragnet.blocks import BlockifyError
from dragnet.util import load_pickled_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_path', help='Path to model .pkl', metavar='PATH',
                        default=None)
    parser.add_argument('-i', dest='html_dir', help='Path to html dir', metavar='DIR',
                        default=None)
    parser.add_argument('-o', dest='out_dir', help='Path to out dir', metavar='DIR',
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model: %s" % args.model_path)
    extractor = load_pickled_model(args.model_path)

    print("Fetching HTMLs ...")
    total_skipped = []

    htmls = glob.glob(os.path.join(args.html_dir, "*.html"))
    for html in htmls:
        out_filepath = os.path.join(args.out_dir,
                                    os.path.basename(html).split(".html")[0] + ".txt")
        print("Extracting %s to %s" % (html, out_filepath))
        with open(html, "r") as fd:
            html_data = fd.read()
            try:
                extracted = extractor.extract(html_data)
                with open(out_filepath, "w", encoding="utf-8") as fd_out:
                    fd_out.write(extracted)
            except BlockifyError as e:
                print(e)
                print("Error encountered. Skipping %s." % html)
                total_skipped.append(html)

    for skipped in total_skipped:
        print("Skipped: %s" % skipped)

    print("Total: %d, Skipped: %d" % (len(htmls) - len(total_skipped), len(total_skipped)))
    print("Done")
