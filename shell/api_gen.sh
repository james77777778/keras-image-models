#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
python3 "${base_dir}"/api_gen.py

echo "Formatting api directory..."
bash "${base_dir}"/shell/format.sh

echo -e "\nAPI generation finish!"
