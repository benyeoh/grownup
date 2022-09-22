#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

HTML_DIR=
DATA_DIR=
LABEL_DIR=
MODEL_DIR=
EXTRACT_OUT=/tmp/web2text/extract

#POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--html-dir)
      HTML_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--data-dir)
      DATA_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--label-dir)
      LABEL_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model-dir)
      MODEL_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--extract-out)
      EXTRACT_OUT="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

pushd $DIR
echo -e "Extracting labels ...\n"
echo -e "Output directory: ${LABEL_DIR}/ \n"
./extract_content.py $DATA_DIR $LABEL_DIR $MODEL_DIR
popd

pushd $DIR/web2text
echo -e "Extracting text ...\n"
echo -e "Output directory: ${EXTRACT_OUT}/ \n"
sbt "runMain bench.ApplyLabels ${HTML_DIR} ${LABEL_DIR} ${EXTRACT_OUT}"
popd
