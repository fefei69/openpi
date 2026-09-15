#!/usr/bin/env bash
set -euo pipefail
source examples/hanoi/env.sh
export JAX_PLATFORMS=cpu
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
.venv/bin/python -m examples.hanoi.convert_hanoi_data_to_lerobot --resume
.venv/bin/python -m examples.hanoi.compute_norm_stats
.venv/bin/python -m examples.hanoi.verify_execution
.venv/bin/python -m examples.hanoi.verify_data
