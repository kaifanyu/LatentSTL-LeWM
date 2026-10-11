#!/usr/bin/env bash
# Source this file before manual collection, training, or evaluation.
LEWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$LEWM_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$LEWM_ROOT:$LEWM_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_ROOT="$LEWM_ROOT/third_party/LIBERO"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LEWM_ROOT/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="$LEWM_ROOT/.cache/numba"
export MPLCONFIGDIR="$LEWM_ROOT/.cache/matplotlib"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
