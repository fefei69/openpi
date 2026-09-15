#!/usr/bin/env bash
set -euo pipefail
source examples/hanoi/env.sh
export JAX_PLATFORMS=cpu
exec .venv/bin/python -m examples.hanoi.manage --submit --watch "$@"
