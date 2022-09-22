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
    -h|--html-out)
      HTML_OUT="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--prepro-out)
      PREPRO_OUT="$2"
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

pushd $DIR
echo -e "Generating htmls for preprocessing ...\n"
echo -e "Output directory: ${HTML_OUT}\n"
./generate_dragnet_html.py -i $RAW_DIR -o $HTML_OUT --val-ratio 0.1

echo -e "Preprocessing data for training ...\n"
echo -e "Output directory: ${PREPRO_OUT}\n"
python $DIR/boilernet/net/preprocess.py -s $HTML_OUT/split/ -w 1000 -t 50 $HTML_OUT/train $HTML_OUT/test --save $PREPRO_OUT

echo -e "Generating data for extraction ...\n"
echo -e "Output directory: ${EXTRACT_DATA_OUT}\n"
./generate_dragnet_extract.py -i $RAW_DIR -o $EXTRACT_DATA_OUT --word-map $PREPRO_OUT/words.json --tag-map $PREPRO_OUT/tags.json

echo -e "Done!\n"
popd