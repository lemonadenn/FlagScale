# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native CUDA parity is exercised by the validation script."""

import os
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from flagscale.inference.core.diffusion.horizonkv import pipeline as pipeline_module
from flagscale.inference.core.diffusion.horizonkv.pipeline import HorizonKVPipeline, generator_scope
from flagscale.inference.core.diffusion.horizonkv.runtime import HorizonKVRuntime
from flagscale.inference.core.diffusion.horizonkv_transformation import HorizonKVTransformation
from flagscale.inference.core.inference_engine import EngineConfig, InferenceEngine
from flagscale.inference.runtime_context import current_ctx
from flagscale.transformations import create_transformations_from_config


def test_generator_controls_all_native_draws_and_restores_global_state():
    original = torch.get_rng_state().clone()
    expected_rng = torch.Generator().manual_seed(123)
    expected = [torch.randn(7, generator=expected_rng) for _ in range(2)]
    rng = torch.Generator().manual_seed(123)
    with generator_scope(rng, "cpu"):
        first = torch.randn(7)
        second = torch.randn_like(first)
    torch.testing.assert_close(first, expected[0])
    torch.testing.assert_close(second, expected[1])
    assert torch.equal(rng.get_state(), expected_rng.get_state())
    assert torch.equal(torch.get_rng_state(), original)
    with pytest.raises(RuntimeError, match="interrupted"), generator_scope(rng, "cpu"):
        torch.randn(7)
        raise RuntimeError("interrupted")
    assert torch.equal(torch.get_rng_state(), original)


def test_unindexed_cuda_generator_resolves_to_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert pipeline_module._concrete_device("cuda") == torch.device("cuda:2")
    assert pipeline_module._concrete_device("cuda:0") == torch.device("cuda:0")


def test_environment_is_scoped_even_on_error(monkeypatch):
    runtime = object.__new__(HorizonKVRuntime)
    runtime.env = {"HORIZONKV_OVERLAP_ATTN": "1", "ATTN_DUMP": "0"}
    monkeypatch.setenv("HORIZONKV_OVERLAP_ATTN", "0")
    monkeypatch.setenv("HORIZONKV_UNRELATED_EXPERIMENT", "stale")
    monkeypatch.setenv("ATTN_DUMP", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError), runtime.activated():
        assert os.environ["HORIZONKV_OVERLAP_ATTN"] == "1"
        assert "HORIZONKV_UNRELATED_EXPERIMENT" not in os.environ
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
        raise ValueError("failure")
    assert os.environ["HORIZONKV_OVERLAP_ATTN"] == "0"
    assert os.environ["HORIZONKV_UNRELATED_EXPERIMENT"] == "stale"
    assert os.environ["ATTN_DUMP"] == "1"


def test_runtime_is_native_and_does_not_import_external_namespaces(tmp_path, monkeypatch):
    imported = ModuleType("wan")
    imported.__file__ = "/other/project/wan/__init__.py"
    monkeypatch.setitem(__import__("sys").modules, "wan", imported)
    before = list(__import__("sys").path)
    runtime = HorizonKVRuntime(assets_root=tmp_path, checkpoint=tmp_path / "model.safetensors")
    try:
        snapshot = runtime.snapshot()
        assert snapshot["adaptive"] and snapshot["overlap"]
        assert snapshot["topk"] == 7 and snapshot["min_frames"] == 5
        assert snapshot["rasa_policy"]["min_config"] == {"frames": 5, "block_sparsity": 0.3}
        assert snapshot["kv_config"]["mandatory_recent_kv_frames"] == 4
        assert "command" not in snapshot and "source_root" not in snapshot
        assert __import__("sys").path == before
        assert __import__("sys").modules["wan"] is imported
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "options",
    [
        dict(min_frames=4),
        dict(topk=8),
        dict(cache=6),
        dict(sparsity=float("nan")),
        dict(split_min_misses=0),
    ],
)
def test_runtime_rejects_inconsistent_policy_before_allocation(tmp_path, options):
    with pytest.raises(ValueError):
        HorizonKVRuntime(assets_root=tmp_path, checkpoint=tmp_path / "model.safetensors", **options)


def test_rasa_switch_is_inclusive_and_preserves_compulsory_frame_budget(tmp_path):
    from flagscale.inference.core.diffusion.horizonkv.algorithm.rasa_policy import select_endpoint

    runtime = HorizonKVRuntime(assets_root=tmp_path, checkpoint=tmp_path / "model.safetensors")
    try:
        policy = runtime.snapshot()["rasa_policy"]
        gate = policy["switch_miss_items"]
        best, minimum = select_endpoint(policy, gate - 1), select_endpoint(policy, gate)
        assert (best["frames"], minimum["frames"]) == (7, 5)
        assert best["frames"] * (1 - best["block_sparsity"]) == pytest.approx(
            minimum["frames"] * (1 - minimum["block_sparsity"])
        )
    finally:
        runtime.close()


def test_transform_requires_native_adapter():
    transforms = create_transformations_from_config(
        OmegaConf.create({"HorizonKVTransformation": {}})
    )
    assert isinstance(transforms[0], HorizonKVTransformation)
    with pytest.raises(ValueError, match="native causal backbone"):
        transforms[0].apply(torch.nn.Identity())


def test_incompatible_torch_fails_before_loading_cuda(monkeypatch):
    from flagscale.inference.core.diffusion.horizonkv.runtime import check_environment

    monkeypatch.setattr(torch, "__version__", "2.9.0")
    cuda_probe = Mock(side_effect=AssertionError("CUDA should not be touched"))
    monkeypatch.setattr(torch.cuda, "is_available", cuda_probe)
    with pytest.raises(RuntimeError, match="Python 3.10 / PyTorch 2.5.1 worker"):
        check_environment(require_cuda=True)
    cuda_probe.assert_not_called()


def test_reset_drains_each_layer_and_clears_conditioning(monkeypatch):
    resets = []
    fake_oae = ModuleType("flagscale.inference.core.diffusion.horizonkv.algorithm.oae")
    fake_oae.reset_kv_cache_state = resets.append
    monkeypatch.setitem(
        __import__("sys").modules,
        "flagscale.inference.core.diffusion.horizonkv.algorithm.oae",
        fake_oae,
    )
    native = torch.nn.Module()
    native.generator = torch.nn.Module()
    native.generator.model = torch.nn.Linear(1, 1)
    native.kv_cache1 = [{"layer": 0}, {"layer": 1}]
    native.crossattn_cache = [{"is_init": True}, {"is_init": True}]
    native.vae = SimpleNamespace(model=SimpleNamespace(clear_cache=Mock()))
    pipe = HorizonKVPipeline(native, SimpleNamespace())
    pipe.reset_request()
    assert resets == native.kv_cache1
    assert all(not cache["is_init"] for cache in native.crossattn_cache)
    native.vae.model.clear_cache.assert_called_once()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_frames": 4},
        {"latent_frames": 0},
        {"height": 481},
        {"prompt": ["a", "b"]},
        {"output_type": "unknown"},
    ],
)
def test_invalid_requests_fail_before_allocation(kwargs):
    pipe = object.__new__(HorizonKVPipeline)
    request = {"prompt": "landscape", **kwargs}
    with pytest.raises(ValueError):
        pipe.forward(**request)


def test_forward_cleans_native_state_after_failure(monkeypatch):
    native = torch.nn.Module()
    native.generator = torch.nn.Module()
    native.generator.model = torch.nn.Linear(1, 1)
    native.inference = Mock(side_effect=RuntimeError("denoise failed"))
    pipe = HorizonKVPipeline(
        native,
        SimpleNamespace(
            installed=True, activated=nullcontext, plan={"model_variant": "self-forcing-1.3b"}
        ),
    )
    pipe.reset_request = Mock()
    monkeypatch.setattr(HorizonKVPipeline, "device", property(lambda self: torch.device("cuda:0")))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        pipeline_module, "generator_scope", lambda *args: nullcontext(torch.Generator())
    )
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.zeros(1))
    from flagscale.inference.core.diffusion.horizonkv.algorithm import rasa

    monkeypatch.setattr(rasa, "reset_stats", Mock())
    with pytest.raises(RuntimeError, match="denoise failed"):
        pipe.forward("landscape")
    pipe.reset_request.assert_called_once()


class ProbePipeline(torch.nn.Module):
    """Exercise the actual engine loader contract without native weights."""

    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Linear(1, 1)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert path == "native-assets"
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert kwargs["checkpoint"] == "ema.pt"
        return cls()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, **kwargs):
        assert kwargs["return_dict"]
        assert kwargs["generator"].initial_seed() == 42
        assert current_ctx() is not None
        return SimpleNamespace(frames=[["frame"]])


def test_engine_custom_class_load_generator_context_and_fps(tmp_path, monkeypatch):
    from flagscale.inference.core import inference_engine

    export = Mock()
    monkeypatch.setattr(inference_engine, "export_to_video", export)
    engine = InferenceEngine(
        OmegaConf.create(
            {
                "model": "native-assets",
                "loader": "diffusers",
                "device": "cpu",
                "pipeline": {
                    "class": f"{__name__}.ProbePipeline",
                    "from_pretrained": {"torch_dtype": "bfloat16", "checkpoint": "ema.pt"},
                },
                "results_path": str(tmp_path),
                "output_format": "video",
                "output_fps": 16,
            }
        )
    )
    result = engine.generate(generator=OmegaConf.create({"seed": 42, "device": "cpu"}))
    assert current_ctx() is None
    engine.save(result)
    assert export.call_args.kwargs["fps"] == 16
    assert EngineConfig("outputs", "video").output_fps == 15


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(fps):
    with pytest.raises(ValueError, match="output_fps"):
        EngineConfig("outputs", "video", output_fps=fps)


def test_krea_requires_explicit_model_identity():
    from flagscale.inference.core.diffusion.horizonkv.model_spec import (
        denoising_steps,
        model_config,
        validate_krea_checkpoint,
    )

    assert denoising_steps("self-forcing-1.3b") == [1000, 750, 500, 250]
    # Match the default GenerationSession: five denoising evaluations including zero.
    assert denoising_steps("krea-14b") == torch.linspace(1000, 0, 5).long().tolist()
    assert model_config("krea-14b")["num_layers"] == 40
    with pytest.raises(ValueError, match="Unknown model_variant"):
        model_config("wan2.1-teacher")
    with pytest.raises(ValueError, match="krea-realtime-video-14b.safetensors"):
        validate_krea_checkpoint("self_forcing_dmd.pt")


def test_krea_rejects_raw_teacher_or_incomplete_weights(tmp_path):
    from safetensors.torch import save_file

    from flagscale.inference.core.diffusion.horizonkv.model_spec import validate_krea_checkpoint

    checkpoint = tmp_path / "teacher.safetensors"
    save_file({"patch_embedding.weight": torch.zeros(1)}, str(checkpoint))
    with pytest.raises(ValueError, match="Not a Krea 14B wrapper checkpoint"):
        validate_krea_checkpoint(checkpoint)


def test_14b_kv_memory_scales_with_all_layers_and_heads():
    from flagscale.inference.core.diffusion.horizonkv.model_spec import kv_memory_estimate

    memory = kv_memory_estimate("krea-14b", latent_frames=24, cache_frames=12)
    assert memory["bytes_per_latent_frame"] == 2 * 2 * 40 * 40 * 128 * 30 * 52
    assert memory["host_history_bytes"] == 2 * memory["gpu_kv_bytes"]


@pytest.fixture
def native_adapter():
    from flagscale.inference.core.diffusion.horizonkv import native

    return native


@pytest.mark.parametrize("mode", ["dense", "offload", "local"])
def test_14b_all_cache_modes_and_cross_attention_use_model_geometry(
    native_adapter, monkeypatch, mode
):
    from flagscale.inference.core.diffusion.horizonkv.algorithm import pipeline as pipeline_module

    cls = native_adapter.SizedCausalInferencePipeline
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    config = SimpleNamespace(cpu_offload_kv=mode == "offload", layer_has_sparse=lambda *_: True)
    model = SimpleNamespace(num_heads=40, dim=5120, text_len=3, kv_sparse_config=config)
    pipe.generator = SimpleNamespace(model=model)
    pipe.num_transformer_blocks = 40
    pipe.frame_seq_length = 2
    pipe.local_attn_size = 6
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {"allocated": True}

    monkeypatch.setattr(pipeline_module, "build_kv_cache_state", build)
    monkeypatch.setattr(pipeline_module, "build_local_kv_cache_state", build)
    monkeypatch.setattr(pipeline_module, "uses_fast_lf_rope_local", lambda _: mode == "local")
    pipe._initialize_kv_cache(1, torch.bfloat16, "cpu", frame_capacity=6, kv_cache_size_override=12)
    pipe._initialize_crossattn_cache(1, torch.bfloat16, "cpu")
    assert len(pipe.kv_cache1) == len(pipe.crossattn_cache) == 40
    if mode == "dense":
        assert pipe.kv_cache1[-1]["k"].shape == (1, 12, 40, 128)
        assert pipe.kv_cache1[-1]["v"].shape == (1, 12, 40, 128)
    else:
        assert len(calls) == 40
        assert all(c["num_heads"] == 40 and c["head_dim"] == 128 for c in calls)
        assert all(c["frame_capacity"] == 6 for c in calls)
    assert all(c["k"].shape == (1, 3, 40, 128) and not c["is_init"] for c in pipe.crossattn_cache)


def test_default_config_is_14b_with_rasa_and_oae():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[4] / "examples/horizonkv/conf"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="inference")
        assert cfg.inference.engine.pipeline.from_pretrained.model_variant == "krea-14b"
        assert cfg.inference.engine.pipeline.from_pretrained.horizonkv.cache == 15
        assert not cfg.inference.engine.pipeline.from_pretrained.horizonkv.fixed
        assert cfg.experiment.exp_name == "horizonkv"
        original = compose(config_name="inference")
        assert original.inference.engine.pipeline.from_pretrained.model_variant == "krea-14b"
        assert original.inference.engine.pipeline.from_pretrained.horizonkv.min_frames == 5
        assert not original.inference.engine.pipeline.from_pretrained.horizonkv.no_overlap
