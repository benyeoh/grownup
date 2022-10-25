#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_TFRECORD_DIR=
OUTPUT_WEIGHTS_PATH=

usage() {
    echo "Usage:"
    echo "    ./train_dragnet.sh <OPTIONS>"
    echo ""         
    echo "        -h            Display this help message"
    echo "        -i <DIR>      Root input folder path for Dragnet training data."
    echo "        -o <DIR>      Output filepath for saved Dragnet trained weights .h5"
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
            OUTPUT_WEIGHTS_PATH=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$INPUT_TFRECORD_DIR" ] || [ -z "$OUTPUT_WEIGHTS_PATH" ]; then
    usage
    exit 1
fi

KTF_TRAIN_DIR=$INPUT_TFRECORD_DIR KTF_SAVED_WEIGHTS=$OUTPUT_WEIGHTS_PATH $DIR/klassterfork/ktf/train/trainer.py -c runs/web/dragnet/dragnet.hjson
