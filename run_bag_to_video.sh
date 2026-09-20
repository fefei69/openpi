#!/usr/bin/env bash
# Render a run's camera bag to MP4 (needs the robot venv and the ROS environment).
set -euo pipefail
hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"
hanoi_python="$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python"
hanoi_ros_setup="${HANOI_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
set +u
# shellcheck disable=SC1090
source "$hanoi_ros_setup"
set -u
exec "$hanoi_python" -m examples.hanoi.deployment.bag_to_video "$@"
