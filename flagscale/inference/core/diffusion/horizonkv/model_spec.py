# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""HorizonKV model specifications shared with the AR+Diffusion baseline."""

# Keep the HorizonKV import path stable while the model geometry and checkpoint
# validation live at the diffusion layer shared by dense and sparse pipelines.
from flagscale.inference.core.diffusion.ar_diffusion_model_spec import (  # noqa: F401
    MODEL_CONFIGS,
    denoising_steps,
    kv_memory_estimate,
    model_config,
    validate_krea_checkpoint,
)

__all__ = [
    "MODEL_CONFIGS",
    "denoising_steps",
    "kv_memory_estimate",
    "model_config",
    "validate_krea_checkpoint",
]
