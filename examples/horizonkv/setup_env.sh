#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
FLAGSCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZONKV_ENV="${HORIZONKV_ENV:-$FLAGSCALE_ROOT/.venvs/horizonkv-worker}"
HORIZONKV_PYTHON="${HORIZONKV_PYTHON:-${HORIZONKV_BASE_PYTHON:-python3.10}}"
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-4}"
command -v nvcc >/dev/null || { echo 'Install CUDA toolkit 12.4 (including nvcc), and add its bin directory to PATH.' >&2; exit 1; }
nvcc --version | grep -q 'release 12.4' || { echo 'This worker build requires CUDA toolkit 12.4.' >&2; exit 1; }
if [[ ! -e "$HORIZONKV_ENV/bin/python" ]]; then
  "$HORIZONKV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
  "$HORIZONKV_PYTHON" -m venv "$HORIZONKV_ENV"
fi
PYTHON_BIN="$HORIZONKV_ENV/bin/python"
"$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
assert sys.version_info[:2] == (3, 10), 'Use a Python 3.10 worker'
assert sys.prefix != sys.base_prefix, 'HORIZONKV_ENV must be a virtual environment'
config = Path(sys.prefix) / 'pyvenv.cfg'
assert 'include-system-site-packages = true' not in config.read_text().lower(), (
    'Use a new isolated HORIZONKV_ENV; inherited site-packages can supply stale CUDA extensions'
)
PY
"$PYTHON_BIN" -m pip install --upgrade pip 'setuptools>=77' wheel
"$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_BIN" -m pip install -r "$FLAGSCALE_ROOT/examples/horizonkv/requirements-worker.txt"
"$PYTHON_BIN" -c 'import torch; assert torch.version.cuda == "12.4", "Install the cu124 torch wheel"'
# Build extensions only after torch and build dependencies have been installed.
FLASH_ATTENTION_FORCE_BUILD=TRUE "$PYTHON_BIN" -m pip install \
  --no-build-isolation --no-cache-dir --force-reinstall --no-deps flash-attn==2.8.3
# Install the FlagScale CLI/Runner without the incompatible cuda-inference extra.
"$PYTHON_BIN" -m pip install --no-deps -e "$FLAGSCALE_ROOT"
HORIZONKV_ENV="$HORIZONKV_ENV" bash "$FLAGSCALE_ROOT/examples/horizonkv/setup_blocksparse.sh"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" "$FLAGSCALE_ROOT/examples/horizonkv/check_worker.py" --cuda
printf 'HorizonKV worker ready: %s\n' "$HORIZONKV_ENV"
