#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INPUT_TEST_DIR=
INPUT_WEIGHTS_PATH=
OUTPUT_EXTRACTED_DIR=

usage() {
    echo "Usage:"
    echo "    ./extract_dragnet.sh <OPTIONS>"
    echo ""         
    echo "        -h            Display this help message"
    echo "        -i <DIR>      Input folder path for processed dragnet test data."
    echo "        -w <DIR>      Input path for saved dragnet trained weights .h5"
    echo "        -o <DIR>      Output folder for extracted content"
}

while getopts ":hi:o:w:" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        i )
            INPUT_TEST_DIR=$OPTARG
            ;;
        w )
            INPUT_WEIGHTS_PATH=$OPTARG
            ;;
        o )
            OUTPUT_EXTRACTED_DIR=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$INPUT_TEST_DIR" ] || [ -z "$INPUT_WEIGHTS_PATH" ] || [ -z $OUTPUT_EXTRACTED_DIR ]; then
    usage
    exit 1
fi

$DIR/extract_content.py --model-config-path $DIR/klassterfork/runs/web/dragnet/dragnet.hjson --model-weights-path $INPUT_WEIGHTS_PATH -i $INPUT_TEST_DIR -o $OUTPUT_EXTRACTED_DIR
