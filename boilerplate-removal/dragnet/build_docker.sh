#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Building docker image for dragnet ..."
pushd dragnet
sudo docker build --network=host -t dragnet .
popd

echo "Building docker image for ..."
sudo docker build --network=host -t dragnet_bench .
