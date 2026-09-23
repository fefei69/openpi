#!/usr/bin/env bash
# Compose the dream-first video for a published Cosmos run (robot venv + ROS environment).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
set +u
# shellcheck disable=SC1090
source "${HANOI_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
set -u
exec "$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python" -m examples.hanoi.deployment.dream_story "$@"
