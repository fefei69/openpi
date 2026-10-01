#!/usr/bin/env bash
# Average-task-progress bar chart from the six-task trials (model venv: it has matplotlib).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
exec "$hanoi_repo_root/.venv/bin/python" -m examples.hanoi.deployment.plot_comparison "$@"
