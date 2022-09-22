#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VER=$(sudo docker version -f "{{.Server.Version}}")
MAJOR="$(echo ${VER} | cut -d '.' -f 1)"
MINOR="$(echo ${VER} | cut -d '.' -f 2)"

if [ $((10#${MAJOR} * 100 + 10#${MINOR})) -ge 1903 ] ; then
   CMD="docker run --gpus all"
else
    CMD="nvidia-docker run"
fi

sudo $CMD --privileged --rm -it --net=host --shm-size=1gb \
-v /dev:/dev:rw \
-v $HOME/.Xauthority:/root/.Xauthority \
-v $HOME/.Xauthority:/home/klass/.Xauthority \
-e DISPLAY \
-e TERM=xterm-256color \
$@ \
--entrypoint=/bin/bash \
boilernet:latest