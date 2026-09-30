#!/usr/bin/env bash
# Regenerate the six-task scoreboard from all recorded runs (robot venv).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
exec "$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python" -m examples.hanoi.deployment.scoreboard "$@"
