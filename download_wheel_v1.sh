#!/usr/bin/env bash
set -euo pipefail

# Define target directory relative to current working directory
WHEELS_DIR="./wheels"

# Make sure the directory exists
mkdir -p "$WHEELS_DIR"

# Download the wheel into the directory
curl -L -o "$WHEELS_DIR/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl" \
  https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl

echo "Wheel downloaded to $WHEELS_DIR"
