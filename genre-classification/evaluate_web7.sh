#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_TFRECORD_DIR=
INPUT_WEIGHTS_DIR=

usage() {
    echo "Usage:"
    echo "    ./evaluate_web7.sh <OPTIONS>"
    echo ""         
    echo "        -h            Display this help message"
    echo "        -i <DIR>      Root input folder path for 10-fold CV training data."
    echo "        -w <DIR>      Folder for saved 10-fold CV trained weights .h5"
}

while getopts ":hi:w:" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        i )
            INPUT_TFRECORD_DIR=$OPTARG
            ;;
        w )
            INPUT_WEIGHTS_DIR=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$INPUT_TFRECORD_DIR" ] || [ -z "$INPUT_WEIGHTS_DIR" ]; then
    usage
    exit 1
fi

$DIR/klassterfork/ktf/datasets/web/web7/evaluate_page_from_graph.py -i $INPUT_TFRECORD_DIR --model-config-path $DIR/klassterfork/runs/web/web7/web7.hjson --model-weights-path $INPUT_WEIGHTS_DIR