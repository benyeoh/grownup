#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_DIR=
OUTPUT_DIR=
NUM_PROCESSES=4
SEED=9999

usage() {
    echo "Usage:"
    echo "    ./prepare_data_ki04.sh <OPTIONS>"
    echo ""
    echo "        -h            Display this help message."
    echo "        -i <DIR>      Root input folder path for raw 7-Web-Genre html data."
    echo "        -o <DIR>      Root output folder path for processed 7-Web-Genre data."
    echo "        -s <INT>      Random seed for splitting data 10 fold."
}

while getopts ":hi:o:s:" opt; do
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
        s )
            SEED=$OPTARG
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

echo "Generating 7-Web-Genre .tfrecords for training ... This may take a few hours ..."
CUDA_VISIBLE_DEVICES= $DIR/klassterfork/ktf/datasets/web/web7/html_to_graph.py -i $INPUT_DIR \
   --font="{ FontSimple: {}}" \
   --text="{ Text2VecUSE: { use_model_path: '$DIR/klassterfork/runs/web/data_common/use_multi' } }" \
   --visual="{ VisualSimple: {} }" \
   --num-eigvec=32 --num-child-pos=32 -g 10,750,30 \
   --tag-file=$DIR/klassterfork/runs/web/data_common/tags.txt \
   --num-proc $NUM_PROCESSES \
   --inline-css \
   -o $OUTPUT_DIR/full --out-config $OUTPUT_DIR/full/config.json   
echo "Done!"
echo ""

echo "Splitting 7-Web-Genre .tfrecords for 10-fold CV ... This may take a few hours ..."
CUDA_VISIBLE_DEVICES= $DIR/klassterfork/ktf/datasets/web/web7/split_k_folds.py -i $OUTPUT_DIR/full -k 10 -o $OUTPUT_DIR/split_$SEED
cp $OUTPUT_DIR/full/*.json $OUTPUT_DIR/split_$SEED/
echo "Done!"

echo "All done!"