#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_DIR=
OUTPUT_DIR=
NUM_PROCESSES=4

usage() {
    echo "Usage:"
    echo "    ./prepare_data_cleaneval.sh <OPTIONS>"
    echo ""
    echo "        -h            Display this help message."
    echo "        -i <DIR>      Root input folder path for raw CleanEval data."
    echo "        -o <DIR>      Root output folder path for processed CleanEval data."
}

while getopts ":hi:o:" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        i )
            INPUT_DIR=$OPTARG
            ;;
        o )
            OUTPUT_DIR=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
    exit 1
fi

echo "Generating CleanEval .tfrecords for training ... This may take a couple of hours ..."
CUDA_VISIBLE_DEVICES= $DIR/klassterfork/ktf/datasets/web/cleaneval/raw_to_tfrecord.py -i $INPUT_DIR/dev/en_original/ -c $INPUT_DIR/dev/en_cleaned/ \
   --font="{ FontSimple: {}}" \
   --text="{ Text2VecUSE: { use_model_path: '$DIR/klassterfork/runs/web/data_common/use_multi' } }" \
   --visual="{ VisualSimple: {} }" \
   --num-eigvec=32 --num-child-pos=32 -g 10,750,30 \
   --tag-file=$DIR/klassterfork/runs/web/data_common/tags.txt \
   --inline-css \
   -o $OUTPUT_DIR/train --out-config $OUTPUT_DIR/train/config.json
echo "Done!"
echo ""

echo "Generating CleanEval .pkl for testing and using $NUM_PROCESSES processes ... This may take several hours ..."
CUDA_VISIBLE_DEVICES= $DIR/prepare_data_cleaneval.py -c $OUTPUT_DIR/train/config.json -i $INPUT_DIR/test -o $OUTPUT_DIR/test -n $NUM_PROCESSES
echo "Done!"

echo "All done!"