#!/usr/bin/env bash
set -euo pipefail
source examples/hanoi/scripts/env.sh
export JAX_PLATFORMS=cpu
exec .venv/bin/python -m examples.hanoi.pipeline.manage --submit --watch "$@"
