#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo docker run -it --rm --net=host $@ -v $DIR:/dragnet dragnet_bench /bin/bash
