#!/usr/bin/env bash
# Align above rod A, check feedback, then return the arm home.
set -euo pipefail

hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
hanoi_python="$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python"
if [[ ! -x "$hanoi_python" ]]; then
    printf 'Missing client environment: %s\n' "$hanoi_python" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
exec "$hanoi_python" -m examples.hanoi.deployment.initialize "$@"
