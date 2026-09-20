#!/usr/bin/env bash
# Start the pi0.5 waypoint_v4 checkpoint for the waypoint client; arguments go to serve_waypoint.py.
set -euo pipefail

hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"

hanoi_python="$hanoi_repo_root/.venv/bin/python"
if [[ ! -x "$hanoi_python" ]]; then
    printf 'Missing model environment: %s\nSee %s/examples/hanoi/deployment/README.md\n' "$hanoi_python" "$hanoi_repo_root" >&2
    exit 1
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$hanoi_repo_root/.cache/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONUNBUFFERED=1

exec "$hanoi_python" -m examples.hanoi.deployment.serve_waypoint "$@"
