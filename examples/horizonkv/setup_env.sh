#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
FLAGSCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZONKV_ENV="${HORIZONKV_ENV:-$FLAGSCALE_ROOT/.venvs/horizonkv-worker}"
if [[ ! -e "$HORIZONKV_ENV/bin/python" ]]; then
  : "${HORIZONKV_BASE_PYTHON:?Set HORIZONKV_BASE_PYTHON to a Python 3.10 / PyTorch 2.5.1 interpreter when rebuilding the worker}"
  "$HORIZONKV_BASE_PYTHON" -c 'import sys, torch; assert sys.version_info[:2] == (3, 10) and torch.__version__.split("+")[0] == "2.5.1", "Use a Python 3.10 / torch 2.5.1 HorizonKV environment as the base"'
fi
if [[ ! -e "$HORIZONKV_ENV/bin/python" ]]; then
  "$HORIZONKV_BASE_PYTHON" -m venv --system-site-packages "$HORIZONKV_ENV"
fi
"$HORIZONKV_ENV/bin/python" -m pip install -r "$FLAGSCALE_ROOT/examples/horizonkv/requirements-worker.txt"
echo "HorizonKV worker: $HORIZONKV_ENV/bin/python"
echo 'Run the preflight described in examples/horizonkv/README.md before generation.'
