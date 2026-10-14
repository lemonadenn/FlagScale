#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
FLAGSCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BSA_PATH="flagscale/inference/core/diffusion/horizonkv/third_party/Block-Sparse-Attention"
SOURCE="${HORIZONKV_BLOCK_SPARSE_SOURCE:-$FLAGSCALE_ROOT/$BSA_PATH}"
PYTHON_BIN="${PYTHON_BIN:-${HORIZONKV_ENV:-$FLAGSCALE_ROOT/.venvs/horizonkv-worker}/bin/python}"
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-4}"
[[ -x "$PYTHON_BIN" ]] || { echo "HorizonKV Python not found: $PYTHON_BIN; run setup_env.sh first." >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys, torch; assert sys.version_info[:2] == (3, 10) and torch.__version__.split("+")[0] == "2.5.1" and torch.version.cuda == "12.4", "Use the Python 3.10 / torch 2.5.1 cu124 worker"'
command -v nvcc >/dev/null || { echo 'nvcc is required to compile BSA.' >&2; exit 1; }
nvcc --version | grep -q 'release 12.4' || { echo 'Use CUDA toolkit 12.4 for this worker.' >&2; exit 1; }
if [[ -z "${HORIZONKV_BLOCK_SPARSE_SOURCE:-}" ]]; then
  git -C "$FLAGSCALE_ROOT" submodule update --init --recursive -- "$BSA_PATH"
else
  git -C "$SOURCE" submodule update --init --recursive
fi
[[ -f "$SOURCE/setup.py" && -f "$SOURCE/csrc/cutlass/include/cutlass/numeric_types.h" ]] || {
  echo "BSA/CUTLASS source is missing in $SOURCE; initialize the recursive submodules." >&2; exit 1;
}
echo "[horizonkv] building modified Block-Sparse-Attention from $SOURCE"
# The upstream wheel does not contain our kernels; always rebuild from source.
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE "$PYTHON_BIN" -m pip install \
  --no-build-isolation --no-cache-dir --force-reinstall --no-deps "$SOURCE"
"$PYTHON_BIN" "$FLAGSCALE_ROOT/examples/horizonkv/check_worker.py"
