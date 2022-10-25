#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_TFRECORD_DIR=
OUTPUT_WEIGHTS_DIR=

usage() {
    echo "Usage:"
    echo "    ./train_web7.sh <OPTIONS>"
    echo ""         
    echo "        -h            Display this help message"
    echo "        -i <DIR>      Root input folder path for 10-fold CV training data."
    echo "        -o <DIR>      Output folder for saved 10-fold CV trained weights .h5"
}

while getopts ":hi:o:" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        i )
            INPUT_TFRECORD_DIR=$OPTARG
            ;;
        o )
            OUTPUT_WEIGHTS_DIR=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$INPUT_TFRECORD_DIR" ] || [ -z "$OUTPUT_WEIGHTS_DIR" ]; then
    usage
    exit 1
fi

KTF_TRAIN_DIR=$INPUT_TFRECORD_DIR KTF_SAVED_WEIGHTS_DIR=$OUTPUT_WEIGHTS_DIR $DIR/klassterfork/ktf/train/trainer.py -c runs/web/web7/web7.hjson
