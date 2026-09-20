#!/usr/bin/env bash
# Source from the repository root before running Hanoi tools or batch jobs.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PWD/.cache/openpi}"
export HF_HOME="$PWD/.cache/huggingface"
# Cluster-wide datasets/hub cache overrides can point outside the writable workspace.
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD/data/lerobot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.cache/jax}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Avoid long kernel compaction stalls while NumPy allocates checkpoint transfer buffers.
export NUMPY_MADVISE_HUGEPAGE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
