# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model-size adaptations around the optional native HorizonKV implementation.

Import only inside an activated runtime after checking the native CUDA ABI.
"""

from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn

from flagscale.inference.core.diffusion.horizonkv.algorithm.backbone import WanDiffusionWrapper
from flagscale.inference.core.diffusion.horizonkv.algorithm.pipeline import CausalInferencePipeline
from flagscale.inference.core.diffusion.horizonkv.algorithm.scheduler import FlowMatchScheduler
from flagscale.inference.core.diffusion.horizonkv.wan.modules.causal_model import CausalWanModel

from .model_spec import model_config, validate_krea_checkpoint


def load_krea_model(checkpoint, *, dtype, local_attn_size, sink_size, long_video_gen):
    """Load all distilled tensors strictly without loading the 57 GB teacher first."""
    validate_krea_checkpoint(checkpoint)
    with init_empty_weights():
        model = CausalWanModel(
            **model_config("krea-14b"),
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            long_video_gen=long_video_gen,
        )
    expected = model.state_dict()
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        actual = {key.removeprefix("model.") for key in tensors.keys()}
        if actual != set(expected):
            raise ValueError(
                f"Krea checkpoint keys mismatch: missing={sorted(set(expected) - actual)}, "
                f"unexpected={sorted(actual - set(expected))}"
            )
        # Validate EVERY shape before reading tens of gigabytes of tensor data.
        for name, tensor in expected.items():
            shape = tuple(tensors.get_slice(f"model.{name}").get_shape())
            if shape != tuple(tensor.shape):
                raise ValueError(f"Krea checkpoint shape mismatch for {name}: {shape}")
        weights = {name: tensors.get_tensor(f"model.{name}").to(dtype=dtype) for name in expected}
    model.load_state_dict(weights, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Krea loading left uninitialized parameters")
    return model.eval().requires_grad_(False)


class LoadedWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.uniform_timestep = False
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()


class SizedCausalInferencePipeline(CausalInferencePipeline):
    """Compatibility name for the geometry-aware causal pipeline.

    Cache allocation is implemented once by :class:`CausalInferencePipeline`.
    Keeping this adapter avoids changing existing configuration paths while
    preventing the 14B and 1.3B implementations from drifting apart.
    """
