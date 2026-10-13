#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the native HorizonKV ABI, resources, and optional GPU kernels."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flagscale.inference.core.diffusion.horizonkv.model_spec import kv_memory_estimate, model_config
from flagscale.inference.core.diffusion.horizonkv.runtime import HorizonKVRuntime, check_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-root",
        default=os.environ.get("HORIZONKV_ASSETS"),
        required="HORIZONKV_ASSETS" not in os.environ,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("HORIZONKV_CHECKPOINT"),
        required="HORIZONKV_CHECKPOINT" not in os.environ,
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--model-variant", choices=("self-forcing-1.3b", "krea-14b"), default="self-forcing-1.3b"
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--cache", type=int, default=12)
    args = parser.parse_args()
    runtime = HorizonKVRuntime(assets_root=args.assets_root, checkpoint=args.checkpoint)
    try:
        with runtime.activated():
            report = check_environment(require_cuda=args.cuda)
            from flagscale.inference.core.diffusion.horizonkv.pipeline import HorizonKVPipeline

            checkpoint = HorizonKVPipeline._validate_assets(runtime, args.model_variant)
            report["assets"] = "present"
            report["model_variant"] = args.model_variant
            report["geometry"] = model_config(args.model_variant)
            report["checkpoint"] = checkpoint
            report["raw_bf16_kv_estimate"] = kv_memory_estimate(
                args.model_variant,
                latent_frames=args.frames,
                cache_frames=args.cache,
            )
            print(json.dumps(report, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
