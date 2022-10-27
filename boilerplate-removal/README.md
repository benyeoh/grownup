# GROWN+UP Webpage Boilerplate Removal Benchmarks
This folder collects the material required to reproduce the web boilerplate removal benchmark results stated in the paper.

## Pre-requisites
* Python 3.7 or newer

## Description
There are only 2 utility scripts in this root folder to evaluate the scores from the extracted text compared to the ground truth. They are:

1. `eval_bagofwords.py` Computes the F1 score using a bag of words model
2. `eval_lcs.py` Computes the F1 score using the least common subsequence of the extracted text vs ground truth. This is the scoring method referenced in the paper.

The various subfolders store self-contained material required to extract the text from webpages sans boilerplate for GROWN+UP and each baseline mentioned in the paper.

These are:
1. [GROWN+UP Feature Extractor](grownup_fe)
2. [BoilerNet](boilernet)
3. [CETD-DS](cetd)
4. [Dragnet](dragnet)
5. [Goose](goose)
6. [Readability](readability)
7. [Web2Text](web2text)

## Steps to use

1. Extract the text sans boilerplate + labels using one of the above methods
2. Run `eval_lcs.py -e <EXTRACTED_DIR> -g <LABELS_DIR>`.

