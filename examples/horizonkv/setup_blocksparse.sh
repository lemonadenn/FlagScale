#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

FLAG_SCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${HORIZONKV_BLOCK_SPARSE_SOURCE:-$FLAG_SCALE_ROOT/flagscale/inference/core/diffusion/horizonkv/third_party/Block-Sparse-Attention}"
PYTHON_BIN="${PYTHON_BIN:-${HORIZONKV_ENV:-$FLAG_SCALE_ROOT/.venvs/horizonkv-worker}/bin/python}"

[[ -d "$SOURCE" ]] || { echo "Block-Sparse-Attention source not found: $SOURCE" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "HorizonKV Python not found: $PYTHON_BIN" >&2; exit 1; }

echo "[horizonkv] installing bundled Block-Sparse-Attention from $SOURCE"
exec "$PYTHON_BIN" -m pip install --no-build-isolation "$SOURCE"
