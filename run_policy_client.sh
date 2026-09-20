#!/usr/bin/env bash
# Initialize the robot, then run live policy control for 30 seconds by default.
# Current diagnostic: recorded velocity inputs with live images, XYZ, and jaw.
set -euo pipefail

hanoi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$hanoi_repo_root"

hanoi_python="$hanoi_repo_root/.cache/hanoi-robot-venv/bin/python"
if [[ ! -x "$hanoi_python" ]]; then
    printf 'Missing client environment: %s\nSee %s/examples/hanoi/deployment/README.md\n' "$hanoi_python" "$hanoi_repo_root" >&2
    exit 1
fi

hanoi_args=("$@")
hanoi_mode=live
hanoi_explicit_mode=false
hanoi_explicit_velocity=false
hanoi_help=false
for ((hanoi_i = 0; hanoi_i < ${#hanoi_args[@]}; hanoi_i++)); do
    case "${hanoi_args[hanoi_i]}" in
        --mode)
            hanoi_explicit_mode=true
            hanoi_mode="${hanoi_args[hanoi_i + 1]:-}"
            ;;
        --mode=*)
            hanoi_explicit_mode=true
            hanoi_mode="${hanoi_args[hanoi_i]#--mode=}"
            ;;
        --velocity-source|--velocity-source=*)
            hanoi_explicit_velocity=true
            ;;
        -h|--help)
            hanoi_help=true
            ;;
    esac
done
if [[ "$hanoi_explicit_mode" == false ]]; then
    hanoi_args=(--mode live "${hanoi_args[@]}")
fi
if [[ "$hanoi_explicit_velocity" == false && "$hanoi_mode" == live ]]; then
    hanoi_args=(--velocity-source recorded "${hanoi_args[@]}")
fi

# Replay and CLI help don't need a ROS installation. ROS setup scripts may read
# unset variables, so nounset is disabled only while sourcing their environment.
if [[ "$hanoi_help" == false && ( "$hanoi_mode" == shadow || "$hanoi_mode" == live ) ]]; then
    hanoi_ros_setup="${HANOI_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    if [[ ! -r "$hanoi_ros_setup" ]]; then
        printf 'Missing ROS setup: %s\nSet HANOI_ROS_SETUP to your ROS setup.bash.\n' "$hanoi_ros_setup" >&2
        exit 1
    fi
    set +u
    # shellcheck disable=SC1090
    source "$hanoi_ros_setup"
    set -u
fi

export PYTHONUNBUFFERED=1
exec "$hanoi_python" -m examples.hanoi.deployment.client "${hanoi_args[@]}"
