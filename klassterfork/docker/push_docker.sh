#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

repo="dr1.klass.dev:5000"
repo_img="${repo}/$(whoami)/klassterfork_min"

echo -e "Tagging image as ${repo_img} ...\n"
sudo docker tag klassterfork_min $repo_img

echo -e "Pushing image ${repo_img} ...\n"
sudo docker push $repo_img
