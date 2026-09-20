#!/usr/bin/env bash
set -euo pipefail
source examples/hanoi/scripts/env.sh
export JAX_PLATFORMS=cpu
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
.venv/bin/python -m examples.hanoi.data.convert_hanoi_data_to_lerobot --resume
.venv/bin/python -m examples.hanoi.data.compute_norm_stats
.venv/bin/python -m examples.hanoi.evaluation.verify_execution
.venv/bin/python -m examples.hanoi.data.verify_data
