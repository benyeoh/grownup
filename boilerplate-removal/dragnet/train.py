#!/usr/bin/env python
import argparse
import logging
import os

from sklearn.ensemble import ExtraTreesClassifier

from dragnet.extractor import Extractor
from dragnet.model_training import train_model
from dragnet.data_processing import prepare_all_data
from dragnet.model_training import evaluate_model_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='train_dir', help='Path to train directory', metavar='DIR',
                        default=None)
    parser.add_argument('-e', dest='test_dir', help='Path to test directory', metavar='DIR',
                        default=None)
    parser.add_argument('-o', dest='out_dir', help='Path to model output directory', metavar='DIR',
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = ExtraTreesClassifier(
        n_estimators=128,
        max_features=None,
        min_samples_leaf=1
    )

    features = ['kohlschuetter', 'weninger', 'readability']
    to_extract = ['content', 'comments']
    base_extractor = Extractor(
        features=features,
        to_extract=to_extract,
        model=model
    )

    print("Training ...")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    extractor = train_model(base_extractor, args.train_dir, output_dir=args.out_dir)
    print("Testing ...")
    test_data = prepare_all_data(args.test_dir)
    test_html, test_labels, _ = extractor.get_html_labels_weights(test_data)

    predictions = extractor.predict(test_html)
    # print(predictions)
    # print()
    # print(test_labels)
    scores = evaluate_model_predictions(test_labels, predictions)
    print(scores)
