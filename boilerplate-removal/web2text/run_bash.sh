#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo nvidia-docker run -it --rm --net=host $@ -v $DIR:/web2text web2text /bin/bash
