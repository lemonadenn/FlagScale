#!/usr/bin/env bash
# Source from the benchmark launchers; keep generated caches and temporary files local.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$FLAG_SCALE_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export HORIZONKV_FUSED_ONLOAD_BUILD_DIR="$TORCH_EXTENSIONS_DIR/horizonkv_transfer"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$HORIZONKV_FUSED_ONLOAD_BUILD_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
