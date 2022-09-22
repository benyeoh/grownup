 #!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

EPOCHS=50
DROPOUT=0.5
INPUT_DIR=
OUTPUT_DIR=

#POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--dropout)
      DROPOUT="$2"
      shift # past argument
      shift # past value
      ;;    
    -i|--input)
      INPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
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
echo -e "Training for ${EPOCHS} epochs with dropout ${DROPOUT}...\n"
echo -e "Input: ${INPUT_DIR}, Output: ${OUTPUT_DIR}\n"

TF_FORCE_GPU_ALLOW_GROWTH=true python boilernet/net/train.py -l 2 -u 256 -d $DROPOUT -s 256 -e $EPOCHS -b 16 --interval 1 --working_dir $OUTPUT_DIR $INPUT_DIR

echo -e "Done\n"
popd
