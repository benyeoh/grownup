#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RAW_DIR=
EXTRACT_DATA_OUT=/tmp/boilernet/extract_data/
HTML_OUT=/tmp/boilernet/html/
PREPRO_OUT=/tmp/boilernet/prepro/

#POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -r|--raw-dir)
      RAW_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--train-out)
      TRAIN_OUT="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--extract-data-out)
      EXTRACT_DATA_OUT="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

#set -- "${POSITIONAL[@]}" # restore positional parameters

pushd $DIR/web2text
echo -e "Generating training dev data  ...\n"
echo -e "Output directory: ${TRAIN_OUT}/dev/ \n"
sbt "runMain bench.ExtractTraining ${RAW_DIR}/train_html ${RAW_DIR}/train_cleaned ${TRAIN_OUT}/dev"

echo -e "Generating training test data ...\n"
echo -e "Output directory: ${TRAIN_OUT}/test/ \n"
sbt "runMain bench.ExtractTraining ${RAW_DIR}/test_html ${RAW_DIR}/test_cleaned ${TRAIN_OUT}/test"

echo -e "Preprocessing data for training ...\n"
echo -e "Output directory: ${TRAIN_OUT}/dev\n"
./src/main/python/data/convert_scala_csv.py ${TRAIN_OUT}/dev/block_features.csv ${TRAIN_OUT}/dev/edge_features.csv ${TRAIN_OUT}/dev/block_features.npy
echo -e "Output directory: ${TRAIN_OUT}/test\n"
./src/main/python/data/convert_scala_csv.py ${TRAIN_OUT}/test/block_features.csv ${TRAIN_OUT}/test/edge_features.csv ${TRAIN_OUT}/test/block_features.npy

echo -e "Generating data for extraction ...\n"
echo -e "Output directory: ${EXTRACT_DATA_OUT}\n"
sbt "runMain bench.ExtractInferencing  ${RAW_DIR}/test_html ${RAW_DIR}/test_cleaned ${EXTRACT_DATA_OUT}"

echo -e "Done!\n"
popd