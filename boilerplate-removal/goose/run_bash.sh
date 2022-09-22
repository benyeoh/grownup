#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo nvidia-docker run -it --rm --net=host $@ -v $DIR:/goose goose /bin/bash
