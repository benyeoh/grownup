#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Building docker image ..."
sudo docker build --network=host --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t cetd \
    -f $DIR/Dockerfile \
    $DIR/
