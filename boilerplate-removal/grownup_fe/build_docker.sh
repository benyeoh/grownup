#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Building docker image for klassterfork ..."
pushd klassterfork/docker
./build_docker.sh
popd

echo "Building docker image for ktfnet_bench ..."
sudo docker build --network=host -t ktfnet_bench .

