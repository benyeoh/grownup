#!/usr/bin/env python
import os
import sys
import glob
import pickle
import gzip
import argparse
import csv

# Hack to allow this script to either be run independently or imported as a module
# if __name__ == "__main__":
#    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet"))
#    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net"))
#    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "boilernet", "net", "misc"))

#import bs4
#from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf

#import net.misc.prepare_dataset
#import net.preprocess
#import generate_cleaneval_html
#import net.leaf_classifier


def extract_content(input_files, output_files, model_path):
    print("Creating model ...")
    """
    with open(params_path, "r") as fd:
        params = {row[0]: row[1] for row in csv.reader(fd)}
    
    info_file = os.path.join(params["DATA_DIR"], 'info.pkl')
    with open(info_file, 'rb') as fd:
        info = pickle.load(fd)    
    
    kwargs = {'input_size': info['num_words'] + info['num_tags'],
              'hidden_size': params["hidden_units"],
              'num_layers': params["num_layers"],
              'dropout': params["dropout"],
              'dense_size': params["dense_size"]}     
    classifier = net.leaf_classifier.LeafClassifier(**kwargs)
    """
    infer_model = tf.keras.models.load_model(model_path)
    infer_model.summary()

    for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
        print("%d: Processing %s ..." % (i, input_file))
        with gzip.open(input_file, "rb") as fd:
            data = pickle.loads(fd.read())
            doc_feature_list, id_list, ordering_ids, id_to_string = data

        doc_feature_list = doc_feature_list.astype(np.float32)
        res = infer_model(doc_feature_list, training=False)
        print(res.shape)

        content_set = set()
        for idx in range(res.shape[1]):
            threshold = res[0][idx]
            if threshold >= 0.5:
                content_set.add(id_list[idx])

        with open(output_file, "w", encoding="utf-8") as fd:
            print("%d: Dumping cleaned output %s ..." % (i, output_file))
            for id in ordering_ids:
                if id in content_set:
                    fd.write(id_to_string[id])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', help='Directory of input files', metavar='DIR')
    parser.add_argument('-o', dest='output_dir', help='Directory of extracted files', metavar='DIR', default=None)
    parser.add_argument('--model-path', dest='model_path', help='Path to saved .h5 model', metavar='PATH')
    args = parser.parse_args()

    all_input_files = []
    all_output_files = []
    all_input_files.extend(glob.glob(os.path.join(args.input_dir, "*.pkl.gz")))
    for path in all_input_files:
        out_filename = os.path.basename(path).split(".")[0] + ".txt"
        outpath = os.path.join(args.output_dir, out_filename)
        all_output_files.append(outpath)

    os.makedirs(args.output_dir, exist_ok=True)

    extract_content(all_input_files, all_output_files, args.model_path)
