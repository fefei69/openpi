#!/usr/bin/env bash
# Progress table over a series of trials (robot venv).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
exec "$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python" -m examples.hanoi.deployment.trial_report "$@"
