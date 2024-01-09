#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))
isort --check --sp "${base_dir}/pyproject.toml" .
black --check --config "${base_dir}/pyproject.toml" .
ruff check --config "${base_dir}/pyproject.toml" .
