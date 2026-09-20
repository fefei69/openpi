#!/usr/bin/env bash
# Publish a deployment run into exp_vid/ (needs the robot venv and, for camera bags, the ROS environment).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
set +u
# shellcheck disable=SC1090
source "${HANOI_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
set -u
exec "$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python" -m examples.hanoi.deployment.publish_run "$@"
