#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify the worker ABI and optional 64x64 CUDA execution without model weights."""

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Run a sparse CUDA correctness check")
    args = parser.parse_args()
    # Load torch's shared libraries before importing the CUDA extension.
    import torch
    import block_sparse_attn
    import block_sparse_attn_cuda
    import flash_attn

    from flagscale.inference.core.diffusion.horizonkv.runtime import check_environment

    report = check_environment()
    if torch.version.cuda != "12.4":
        raise RuntimeError("Use the torch 2.5.1 cu124 wheel")
    parameters = inspect.signature(block_sparse_attn.block_sparse_attn_func).parameters
    if not {"m_block_dim", "n_block_dim"}.issubset(parameters):
        raise RuntimeError("Stock BSA detected; run setup_blocksparse.sh to build the modified fork")
    report.update(
        python_executable=sys.executable,
        bsa_python=block_sparse_attn.__file__,
        bsa_cuda=block_sparse_attn_cuda.__file__,
        flash_attention=flash_attn.__file__,
    )
    if args.cuda:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("An NVIDIA GPU with compute capability >= 8.0 is required")
        from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_backend import (
            stage2_blocksparse_attention,
        )

        generator = torch.Generator(device="cuda").manual_seed(42)
        # Unequal, non-aligned lengths exercise the modified block rounding.
        q = torch.randn(1, 73, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        k = torch.randn(1, 137, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        v = torch.randn(1, 137, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
        with torch.inference_mode():
            actual = stage2_blocksparse_attention(q, k, v, block_mask=mask)
            token_mask = mask.repeat_interleave(64, 0).repeat_interleave(64, 1)[:73, :137]
            scores = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2)
            scores = (scores / 128**0.5).masked_fill(~token_mask, float("-inf"))
            expected = (scores.softmax(-1) @ v.transpose(1, 2).float()).transpose(1, 2)
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
            # Also execute FlashAttention's compiled extension.
            flash_attn.flash_attn_func(q, k, v)
            torch.cuda.synchronize()
        report["gpu"] = torch.cuda.get_device_name()
        report["cuda_kernels"] = "BSA 64x64 sparse/unequal-length correctness and FlashAttention passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
