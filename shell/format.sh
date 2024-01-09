#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))
isort --sp "${base_dir}/pyproject.toml" .
black --config "${base_dir}/pyproject.toml" .
ruff --config "${base_dir}/pyproject.toml" .
