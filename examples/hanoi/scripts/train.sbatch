#!/usr/bin/env bash
set -euo pipefail
source examples/hanoi/scripts/env.sh
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
exec .venv/bin/python -m examples.hanoi.training.train "$@"
