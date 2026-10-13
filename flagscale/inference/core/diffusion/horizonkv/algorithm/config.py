# Modified for the HorizonKV inference artifact; see NOTICE.
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

DEFAULT_ATTN_BLOCK_SIZE = 64
DEFAULT_IMPORTANCE_Q_POOL = DEFAULT_ATTN_BLOCK_SIZE
DEFAULT_IMPORTANCE_K_POOL = DEFAULT_ATTN_BLOCK_SIZE

# Only explicit configs for the supported inference paths are accepted.

@dataclass
class KVSparseConfig:
    enabled: bool
    chunk_k: int
    respect_sink: bool
    granularity: str
    rule: str
    frame_selection_score: str
    frame_selection_strategy: str
    target_density: float
    target_recall: float | None
    max_kv_frames: int | None
    frame_selection_score_threshold_ratio: float | None
    low_recall_threshold: float | None
    expand_target_recall: float | None
    block_target_density: float
    block_rule: str
    block_target_recall: float | None
    block_density_allocation: str
    block_density_scale_by_max_kv_frames: bool
    block_force_dense_compulsory_frames: bool
    block_compute_backend: str
    attn_block_q: int
    attn_block_k: int
    layers: dict[str, Any]
    importance_q_pool: int
    importance_k_pool: int
    forced_outside_budget: bool
    cpu_offload_kv: bool
    gpu_cache_frames: int | None
    cache_full_flash_attn: bool
    overlap_block_sparsity: float
    mandatory_recent_kv_frames: int | None
    mandatory_preceding_kv_frames: int | None
    reuse_frame_selection: bool
    prefetch_next_layer_kv: bool
    predict_next_layer_query: bool
    reindex_rope_positions: bool
    score_with_unroped_qk: bool
    rope_semantics: str
    selector: str
    lightforcing_sparsity: float
    lightforcing_sparsity_base: float | None
    lightforcing_local_attn_size: int
    lightforcing_num_output_frames: int
    lightforcing_frame_seq: int
    lightforcing_keep_frames: int
    lightforcing_keep_sink: int
    lightforcing_keep_near: int
    visualize_attn_mask: bool
    visualize_output_dir: str | None
    visualize_layers: set[int] | None
    visualize_heads: set[int] | None
    visualize_timesteps: set[str] | None
    visualize_full_indices: bool
    source_path: str | None
    source_stem: str | None
    visualize_run_name: str | None
    profile_cache_hit: bool
    profile_onload: bool
    profile_onload_sync_cuda: bool
    profile_onload_log_every: int
    profile_frame_selection: bool
    profile_stage2: bool
    profile_chunk: bool
    profile_chunk_sync_cuda: bool
    profile_nvtx: bool
    profile_sync_cuda: bool
    profile_log_every: int
    profile_verbose: bool
    _visualization_index: int = field(default=0, init=False, repr=False)
    _profile_index: int = field(default=0, init=False, repr=False)
    _profile_score_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_rule_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_head_select_calls: int = field(default=0, init=False, repr=False)
    _profile_selected_frames_total: int = field(default=0, init=False, repr=False)
    _profile_stage2_compact_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_stage2_block_mass_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_stage2_block_select_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_stage2_mask_build_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_stage2_kernel_ms_total: float = field(default=0.0, init=False, repr=False)
    _profile_stage2_head_calls: int = field(default=0, init=False, repr=False)
    _profile_registered_atexit: bool = field(default=False, init=False, repr=False)

    def _get_layer_head_spec(self, layer: int, head: int) -> Any:
        lk = str(layer)
        if lk not in self.layers:
            return None
        spec = self.layers[lk]
        if isinstance(spec, (int, float)):
            return spec
        if not isinstance(spec, dict):
            return None
        if "all" in spec:
            return spec["all"]
        hk = str(head)
        if hk in spec:
            return spec[hk]
        return None

    def get_density(self, layer: int, head: int) -> float:
        spec = self._get_layer_head_spec(layer, head)
        if spec is None:
            return float(self.target_density)
        if isinstance(spec, dict):
            return float(spec.get("density", self.target_density))
        return float(spec)

    def get_max_kv_frames(self, layer: int, head: int) -> int | None:
        spec = self._get_layer_head_spec(layer, head)
        value = self.max_kv_frames
        if isinstance(spec, dict):
            value = spec.get(
                "max_kv_frames",
                spec.get(
                    "max_stage1_kv_frames",
                    spec.get(
                        "target_max_kv_frames",
                        spec.get("top_k", spec.get("stage1_top_k", value)),
                    ),
                ),
            )
        if value is None:
            return None
        return max(0, int(value))

    def all_heads_max_kv_frames_at_least(self, layer: int, heads: Any, n_frames: int) -> bool:
        has_head = False
        target = int(n_frames)
        for head in heads:
            has_head = True
            value = self.get_max_kv_frames(layer, int(head))
            if value is None or int(value) < target:
                return False
        return has_head

    def shared_max_kv_frames(self, layer: int, heads: Any) -> int | None:
        shared: int | None = None
        has_head = False
        for head in heads:
            has_head = True
            value = self.get_max_kv_frames(layer, int(head))
            if value is None:
                return None
            value = int(value)
            if shared is None:
                shared = value
            elif shared != value:
                return None
        if not has_head:
            return None
        return shared

    def get_block_density(self, layer: int, head: int) -> float:
        spec = self._get_layer_head_spec(layer, head)
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return float(spec)
        if isinstance(spec, dict):
            if "block_density" in spec:
                return float(spec["block_density"])
            if "density" in spec:
                return float(spec["density"])
        return float(self.block_target_density)

    def get_block_rule(self, layer: int, head: int) -> str:
        spec = self._get_layer_head_spec(layer, head)
        value = self.block_rule
        if isinstance(spec, dict):
            value = spec.get(
                "block_rule",
                spec.get("second_level_rule", spec.get("stage2_rule", self.block_rule)),
            )
        value = str(value).strip().lower()
        if value in ("recall", "mass_recall", "target_recall"):
            return "recall"
        return "density"

    def get_block_recall(self, layer: int, head: int) -> float | None:
        spec = self._get_layer_head_spec(layer, head)
        value = self.block_target_recall
        if isinstance(spec, dict):
            value = spec.get(
                "block_target_recall",
                spec.get("block_recall", spec.get("second_level_recall", value)),
            )
        return float(value) if value is not None else None

    def get_block_density_allocation(self, layer: int, head: int) -> str:
        spec = self._get_layer_head_spec(layer, head)
        value = self.block_density_allocation
        if isinstance(spec, dict):
            value = spec.get(
                "block_density_allocation",
                spec.get(
                    "block_allocation",
                    spec.get("stage2_density_allocation", value),
                ),
            )
        value = str(value).strip().lower()
        if value in (
            "frame_score_weighted",
            "per_frame_score_weighted",
            "weighted_by_frame_score",
            "weighted",
        ):
            return "frame_score_weighted"
        return "global"

    def get_block_compute_backend(self, layer: int, head: int) -> str:
        spec = self._get_layer_head_spec(layer, head)
        value = self.block_compute_backend
        if isinstance(spec, dict):
            value = spec.get(
                "block_compute_backend",
                spec.get("second_level_backend", spec.get("stage2_backend", value)),
            )
        value = str(value).strip().lower()
        if value in (
            "",
            "block_sparse_attention",
            "blocksparse_attention",
            "blocksparse_kernel",
            "kernel",
            "bsa",
            "block-sparse-attention",
            "flash_attn",
            "flash-attn",
            "masked_sdpa",
            "sdpa",
        ):
            return "block_sparse_attention"
        if value in ("lightforcing_triton", "lightforcing", "lf_triton"):
            return "lightforcing_triton"
        raise ValueError(
            f"Unsupported block_compute_backend '{value}'. Supported backends are "
            "block_sparse_attention and lightforcing_triton."
        )

    def _frame_level_has_sparse(self, layer: int, num_heads: int) -> bool:
        r = self.rule
        if r == "recall":
            return self.target_recall is not None and float(self.target_recall) < 1.0 - 1e-6
        if r == "max_kv_frames":
            return any(self.get_max_kv_frames(layer, h) is not None for h in range(num_heads))
        if r == "topk_plus_threshold":
            return any(self.get_max_kv_frames(layer, h) is not None for h in range(num_heads))
        if r == "density_if_low_recall":
            ex = self.expand_target_recall
            if ex is not None and float(ex) < 1.0 - 1e-6:
                return True
            if self.target_density < 1.0 - 1e-6:
                return True
            for h in range(num_heads):
                if self.get_density(layer, h) < 1.0 - 1e-6:
                    return True
            return False
        if r == "density":
            for h in range(num_heads):
                if self.get_density(layer, h) < 1.0 - 1e-6:
                    return True
            return False
        return True

    def layer_has_sparse(self, layer: int, num_heads: int) -> bool:
        if self.selector == "lightforcing":
            # LightForcing block sparsity is independent of frame density.
            return float(self.lightforcing_sparsity) > 1e-6
        frame_sparse = self._frame_level_has_sparse(layer, num_heads)
        if self.granularity != "kv_frame_then_attn_block":
            return frame_sparse
        if frame_sparse:
            return True
        for h in range(num_heads):
            if self.get_block_rule(layer, h) == "recall":
                rec = self.get_block_recall(layer, h)
                if rec is not None and rec < 1.0 - 1e-6:
                    return True
            elif self.get_block_density(layer, h) < 1.0 - 1e-6:
                return True
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_index_filter(value: Any) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("all", "*"):
            return None
        items = [part.strip() for part in text.split(",") if part.strip()]
        return {int(part) for part in items}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    return {int(value)}


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _scalarize_timestep(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().flatten()[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalarize_timestep(value[0])
    return value


def _normalize_timestep_token(value: Any) -> str:
    value = _scalarize_timestep(value)
    if value is None:
        return "unknown"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "unknown"
        try:
            value = float(text)
        except ValueError:
            return text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isfinite(val) and abs(val - round(val)) <= 1e-6:
            return str(int(round(val)))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _parse_timestep_filter(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("all", "*"):
            return None
        items = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    return {_normalize_timestep_token(item) for item in items}


def _format_filter(values: set[int] | set[str] | None) -> str:
    if values is None:
        return "all"
    if not values:
        return "(empty)"
    return ",".join(str(v) for v in sorted(values))


def _sanitize_token(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-._")
    if not text:
        return default
    return text[:80]


def load_kv_sparse_config(path: str | Path) -> KVSparseConfig | None:
    path = Path(path)
    if not str(path).strip() or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("preset") or data.get("config_preset"):
        raise ValueError("Built-in sparse presets are no longer supported; use an explicit JSON config")
    if not data.get("enabled", False):
        return None

    gran = str(data.get("granularity", "kv_frame")).strip().lower()
    if gran in ("two_level", "frame_then_attn_block", "kv_frame+attn_block"):
        gran = "kv_frame_then_attn_block"
    if gran not in ("kv_frame", "attn_block", "kv_frame_then_attn_block"):
        gran = "kv_frame"

    raw_rule = str(data.get("rule", "density")).strip().lower()
    frame_selection_score = str(
        data.get("frame_selection_score", data.get("frame_score_mode", ""))
    ).strip().lower()
    if raw_rule == "simple_mean" and not frame_selection_score:
        frame_selection_score = "pooled_qk"
        rule = "density"
    else:
        rule = raw_rule
    if rule not in ("density", "max_kv_frames"):
        raise ValueError(f"Unsupported sparse rule: {rule!r}; use density or max_kv_frames")
    if frame_selection_score in ("", "attn_mass", "attention_mass", "softmax_mass"):
        frame_selection_score = "attn_mass"
    elif frame_selection_score in ("pooled_qk", "simple_mean", "frame_mean_dot", "pooled_dot"):
        frame_selection_score = "pooled_qk"
    else:
        raise ValueError(f"Unsupported frame_selection_score: {frame_selection_score!r}")

    raw_selection_strategy = str(
        data.get(
            "frame_selection_strategy",
            data.get("selection_strategy", data.get("frame_sampling_strategy", "max_score")),
        )
    ).strip().lower()
    if raw_selection_strategy in {"", "max", "topk", "top_k", "max_score", "argmax"}:
        frame_selection_strategy = "max_score"
    elif raw_selection_strategy in {
        "weighted_random",
        "weighted_sampling",
        "probability",
        "probabilistic",
        "sample",
    }:
        frame_selection_strategy = "weighted_random"
    else:
        raise ValueError(
            "frame_selection_strategy must be 'max_score' or 'weighted_random', "
            f"got {raw_selection_strategy!r}"
        )
    if frame_selection_strategy == "weighted_random" and frame_selection_score != "pooled_qk":
        raise ValueError(
            "frame_selection_strategy=weighted_random requires "
            "frame_selection_score=pooled_qk"
        )
    if frame_selection_strategy == "weighted_random" and rule != "max_kv_frames":
        raise ValueError(
            "frame_selection_strategy=weighted_random requires rule=max_kv_frames"
        )

    td = float(data.get("target_density", data.get("default_density", 1.0)))
    tr = data.get("target_recall", None)
    mkf = data.get(
        "max_stage1_kv_frames",
        data.get("max_kv_frames", data.get("target_max_kv_frames", data.get("top_k", None))),
    )
    fs_thr = data.get(
        "frame_selection_score_threshold_ratio",
        data.get("frame_selection_threshold_ratio", data.get("score_threshold_ratio", None)),
    )
    lr_gate = data.get("low_recall_threshold", None)
    ex_rec = data.get("expand_target_recall", None)
    block_target_density = float(
        data.get(
            "block_target_density",
            data.get("second_level_density", data.get("stage2_density", 1.0)),
        )
    )
    block_rule_raw = str(
        data.get("block_rule", data.get("second_level_rule", data.get("stage2_rule", "density")))
    ).strip().lower()
    if block_rule_raw not in {"", "density"}:
        raise ValueError("Only density block selection is supported")
    block_rule = "density"
    block_target_recall = None
    block_density_allocation = str(
        data.get(
            "block_density_allocation",
            data.get("block_allocation", data.get("stage2_density_allocation", "global")),
        )
    ).strip().lower()
    if block_density_allocation in (
        "frame_score_weighted",
        "per_frame_score_weighted",
        "weighted_by_frame_score",
        "weighted",
    ):
        block_density_allocation = "frame_score_weighted"
    else:
        block_density_allocation = "global"
    block_compute_backend = str(
        data.get(
            "block_compute_backend",
            data.get("second_level_backend", data.get("stage2_backend", "block_sparse_attention")),
        )
    ).strip().lower()
    if block_compute_backend in (
        "",
        "block_sparse_attention",
        "blocksparse_attention",
        "blocksparse_kernel",
        "kernel",
        "bsa",
        "block-sparse-attention",
        "flash_attn",
        "flash-attn",
        "masked_sdpa",
        "sdpa",
    ):
        block_compute_backend = "block_sparse_attention"
    elif block_compute_backend in ("lightforcing_triton", "lightforcing", "lf_triton"):
        block_compute_backend = "lightforcing_triton"
    else:
        raise ValueError(
            f"Unsupported block_compute_backend '{block_compute_backend}'. Supported backends are "
            "block_sparse_attention and lightforcing_triton."
        )
    block_force_dense_compulsory_frames = bool(
        data.get("block_force_dense_compulsory_frames", True)
    )
    block_density_scale_by_max_kv_frames = _truthy(
        data.get(
            "block_density_scale_by_max_kv_frames",
            data.get(
                "stage2_density_scale_by_max_kv_frames",
                data.get("block_fixed_compute_budget", False),
            ),
        )
    )
    if mkf is not None:
        mkf = max(0, int(mkf))
    if "rule" not in data and mkf is not None:
        rule = "max_kv_frames"

    block_default = max(1, int(data.get("block_size", DEFAULT_ATTN_BLOCK_SIZE)))
    abq = max(1, int(data.get("attn_block_q", block_default)))
    abk = max(1, int(data.get("attn_block_k", block_default)))
    attn_block_override_raw = str(os.getenv("HORIZONKV_ATTN_BLOCK_SIZE", "")).strip()
    if attn_block_override_raw:
        try:
            attn_block_override = int(attn_block_override_raw)
        except ValueError as exc:
            raise ValueError("HORIZONKV_ATTN_BLOCK_SIZE must be 64") from exc
        if attn_block_override != 64:
            raise ValueError("HORIZONKV_ATTN_BLOCK_SIZE must be 64")
        block_default = attn_block_override
        abq = attn_block_override
        abk = attn_block_override
    if (abq, abk) != (64, 64):
        raise ValueError("This release uses 64x64 sparse attention blocks")
    importance_q_pool_default = (
        abq if gran in ("attn_block", "kv_frame_then_attn_block") else DEFAULT_IMPORTANCE_Q_POOL
    )
    importance_k_pool_default = (
        abk if gran in ("attn_block", "kv_frame_then_attn_block") else DEFAULT_IMPORTANCE_K_POOL
    )
    importance_q_pool = max(1, int(data.get("importance_q_pool", importance_q_pool_default)))
    importance_k_pool = max(1, int(data.get("importance_k_pool", importance_k_pool_default)))
    if attn_block_override_raw and gran in ("attn_block", "kv_frame_then_attn_block"):
        # Keep block scoring pool and kernel block size locked together.
        importance_q_pool = abq
        importance_k_pool = abk
    visualize_attn_mask = _truthy(
        os.getenv("HORIZONKV_VIS_ENABLED", data.get("visualize_attn_mask", False))
    )
    visualize_output_dir_raw = os.getenv(
        "HORIZONKV_VIS_DIR",
        str(data.get("visualize_output_dir", "") or "").strip(),
    )
    visualize_output_dir = str(visualize_output_dir_raw).strip() or None
    if visualize_attn_mask and visualize_output_dir is None:
        visualize_output_dir = str(Path("logs/kv_sparse_attn_mask") / path.stem)
    visualize_layers = _parse_index_filter(
        os.getenv("HORIZONKV_VIS_LAYERS", data.get("visualize_layers", None))
    )
    visualize_heads = _parse_index_filter(
        os.getenv("HORIZONKV_VIS_HEADS", data.get("visualize_heads", None))
    )
    visualize_timesteps = _parse_timestep_filter(
        os.getenv("HORIZONKV_VIS_TIMESTEPS", data.get("visualize_timesteps", None))
    )
    visualize_full_indices = _truthy(
        os.getenv("HORIZONKV_VIS_FULL_INDICES", data.get("visualize_full_indices", False))
    )
    visualize_run_name = str(os.getenv("HORIZONKV_VIS_RUN_NAME", "")).strip() or None
    profile_frame_selection = _truthy(
        os.getenv("HORIZONKV_PROFILE_FRAME_SELECTION", data.get("profile_frame_selection", False))
    )
    profile_stage2 = _truthy(
        os.getenv("HORIZONKV_PROFILE_STAGE2", data.get("profile_stage2", False))
    )
    profile_cache_hit = _truthy(
        os.getenv(
            "HORIZONKV_PROFILE_CACHE_HIT",
            data.get("profile_cache_hit", data.get("profile_gpu_cache_hit", False)),
        )
    )
    profile_onload = _truthy(
        os.getenv("HORIZONKV_PROFILE_ONLOAD", data.get("profile_onload", False))
    )
    profile_onload_sync_cuda = _truthy(
        os.getenv(
            "HORIZONKV_PROFILE_ONLOAD_SYNC",
            data.get("profile_onload_sync_cuda", data.get("profile_sync_cuda", False)),
        )
    )
    profile_onload_log_every = _nonnegative_int(
        os.getenv(
            "HORIZONKV_PROFILE_ONLOAD_LOG_EVERY",
            data.get("profile_onload_log_every", data.get("profile_log_every", 0)),
        ),
        0,
    )
    profile_chunk = _truthy(
        os.getenv("HORIZONKV_PROFILE_CHUNK", data.get("profile_chunk", False))
    )
    profile_chunk_sync_cuda = _truthy(
        os.getenv(
            "HORIZONKV_PROFILE_CHUNK_SYNC_CUDA",
            data.get("profile_chunk_sync_cuda", data.get("profile_sync_cuda", False)),
        )
    )
    profile_nvtx = _truthy(
        os.getenv("HORIZONKV_NVTX", data.get("profile_nvtx", False))
    )
    profile_sync_cuda = _truthy(
        os.getenv("HORIZONKV_PROFILE_SYNC_CUDA", data.get("profile_sync_cuda", True))
    )
    profile_log_every = _nonnegative_int(
        os.getenv("HORIZONKV_PROFILE_LOG_EVERY", data.get("profile_log_every", 0)),
        0,
    )
    profile_verbose = _truthy(
        os.getenv("HORIZONKV_PROFILE_VERBOSE", data.get("profile_verbose", False))
    )
    mandatory_recent_kv_frames_raw = os.getenv("HORIZONKV_MANDATORY_RECENT_KV_FRAMES")
    if mandatory_recent_kv_frames_raw is None:
        mandatory_recent_kv_frames_raw = os.getenv("HORIZONKV_KEEP_RECENT_KV_FRAMES")
    if mandatory_recent_kv_frames_raw is None:
        mandatory_recent_kv_frames_raw = data.get(
            "mandatory_recent_kv_frames",
            data.get(
                "keep_recent_kv_frames",
                data.get("recent_kv_frames", data.get("forced_recent_kv_frames", None)),
            ),
        )
    mandatory_recent_kv_frames = _optional_nonnegative_int(mandatory_recent_kv_frames_raw)
    mandatory_preceding_kv_frames_raw = os.getenv(
        "HORIZONKV_MANDATORY_PRECEDING_KV_FRAMES"
    )
    if mandatory_preceding_kv_frames_raw is None:
        mandatory_preceding_kv_frames_raw = data.get(
            "mandatory_preceding_kv_frames",
            data.get("keep_preceding_kv_frames", None),
        )
    mandatory_preceding_kv_frames = _optional_nonnegative_int(
        mandatory_preceding_kv_frames_raw
    )
    cpu_offload_kv_env = os.getenv("HORIZONKV_CPU_OFFLOAD")
    cpu_offload_kv_cfg = data.get("cpu_offload_kv", None)
    gpu_cache_frames_raw = os.getenv(
        "HORIZONKV_GPU_CACHE_FRAMES",
        data.get(
            "gpu_cache_frames",
            data.get("gpu_kv_cache_frames", data.get("kv_gpu_cache_frames", None)),
        ),
    )
    cache_full_flash_attn = _truthy(
        os.getenv(
            "HORIZONKV_CACHE_FULL_FLASH_ATTN",
            data.get(
                "cache_full_flash_attn",
                data.get(
                    "cache_flash_attn",
                    data.get("gpu_cache_flash_attn", False),
                ),
            ),
        )
    )
    overlap_block_sparsity_raw = os.getenv("HORIZONKV_OVERLAP_BLOCK_SPARSITY")
    if overlap_block_sparsity_raw is None:
        overlap_block_sparsity_raw = os.getenv("HORIZONKV_FLASH_BLOCK_SPARSITY")
    if overlap_block_sparsity_raw is None:
        overlap_block_sparsity_raw = data.get(
            "overlap_block_sparsity",
            data.get(
                "flash_block_sparsity",
                data.get("block_sparsity", data.get("sparsity", None)),
            ),
        )
    overlap_block_sparse_enabled_raw = os.getenv("HORIZONKV_OVERLAP_BLOCK_SPARSE_ATTN")
    if overlap_block_sparse_enabled_raw is None:
        overlap_block_sparse_enabled_raw = data.get(
            "overlap_block_sparse_attn",
            data.get("flash_block_sparse_attn", None),
        )
    if overlap_block_sparsity_raw is None:
        overlap_block_sparse_enabled = (
            overlap_block_sparse_enabled_raw is not None
            and _truthy(overlap_block_sparse_enabled_raw)
        )
        overlap_block_sparsity = (
            0.5 if overlap_block_sparse_enabled else 0.0
        )
    else:
        try:
            overlap_block_sparsity = max(
                0.0,
                min(1.0, float(overlap_block_sparsity_raw)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "overlap_block_sparsity / flash_block_sparsity / sparsity must be a float in [0, 1]"
            ) from exc
    if (
        overlap_block_sparse_enabled_raw is not None
        and not _truthy(overlap_block_sparse_enabled_raw)
    ):
        overlap_block_sparsity = 0.0
    prefetch_next_layer_kv = _truthy(
        os.getenv(
            "HORIZONKV_PREFETCH_NEXT_LAYER",
            data.get(
                "prefetch_next_layer_kv",
                data.get("prefetch_next_layer", data.get("layer_prefetch", False)),
            ),
        )
    )
    reuse_frame_selection_raw = os.getenv("HORIZONKV_REUSE_FRAME_SELECTION")
    if reuse_frame_selection_raw is None:
        reuse_frame_selection_raw = os.getenv("HORIZONKV_REUSE_SELECTION")
    if reuse_frame_selection_raw is None:
        reuse_frame_selection_raw = data.get(
            "reuse_frame_selection",
            data.get(
                "reuse_kv_frame_selection",
                data.get("reuse_selection_across_steps", False),
            ),
        )
    reuse_frame_selection = _truthy(reuse_frame_selection_raw)
    if isinstance(gpu_cache_frames_raw, str) and not gpu_cache_frames_raw.strip():
        gpu_cache_frames_raw = None
    cpu_offload_disabled_by_env = False
    if cpu_offload_kv_env is not None:
        cpu_offload_kv = _truthy(cpu_offload_kv_env)
        cpu_offload_disabled_by_env = not cpu_offload_kv
    elif cpu_offload_kv_cfg is None:
        cpu_offload_kv = False
    else:
        cpu_offload_kv = _truthy(cpu_offload_kv_cfg)
    if gpu_cache_frames_raw is None and cpu_offload_kv:
        gpu_cache_frames_raw = mkf if mkf is not None else 8
    gpu_cache_frames = (
        max(0, int(gpu_cache_frames_raw))
        if gpu_cache_frames_raw is not None
        else None
    )
    if gran == "attn_block":
        cpu_offload_kv = False
        gpu_cache_frames = None
    if cpu_offload_disabled_by_env:
        cpu_offload_kv = False
        gpu_cache_frames = None
    elif gpu_cache_frames is not None:
        cpu_offload_kv = True
    if prefetch_next_layer_kv:
        reuse_frame_selection = True
    if not cpu_offload_kv:
        prefetch_next_layer_kv = False
        cache_full_flash_attn = False
    reindex_rope_positions = _truthy(
        os.getenv(
            "HORIZONKV_REINDEX_ROPE",
            data.get(
                "reindex_rope_positions",
                data.get(
                    "rope_reindex",
                    data.get("reencode_rope_positions", False),
                ),
            ),
        )
    )
    score_with_unroped_qk = _truthy(
        os.getenv(
            "HORIZONKV_SCORE_UNROPED_QK",
            data.get(
                "score_with_unroped_qk",
                data.get("score_with_pre_rope_qk", False),
            ),
        )
    )
    rope_semantics = str(
        data.get("rope_semantics", data.get("cache_rope_semantics", "absolute"))
    ).strip().lower()
    if rope_semantics in ("", "absolute", "standard", "longlive", "global_absolute"):
        rope_semantics = "absolute"
    elif rope_semantics in (
        "horizonkv_compacted",
        "lightforcing_compacted_fast",
        "lf_compacted_window_fast",
        "lf_window_fast",
    ):
        # The fast variant keeps the same raw-K/packed-window RoPE semantics,
        # but routes frame selection and attention through the regular
        # CPU-offload overlap implementation.
        rope_semantics = "horizonkv_compacted"
    else:
        raise ValueError(
            "Unsupported rope_semantics: "
            f"{rope_semantics!r}; expected 'absolute' or "
            "'horizonkv_compacted'"
        )
    selector = str(data.get("selector", data.get("block_selector", "longlive"))).strip().lower()
    if selector == "horizonkv":
        selector = "longlive"
    if selector in ("lf", "light_forcing", "lightforcing_hierarchical"):
        selector = "lightforcing"
    if selector not in ("longlive", "lightforcing"):
        raise ValueError(f"Unsupported kv sparse selector: {selector!r}")
    lightforcing_sparsity = float(data.get("sparsity", data.get("lightforcing_sparsity", 0.85)))
    lightforcing_sparsity_base_raw = data.get(
        "sparsity_base", data.get("lightforcing_sparsity_base", None)
    )
    lightforcing_sparsity_base = (
        float(lightforcing_sparsity_base_raw)
        if lightforcing_sparsity_base_raw is not None
        else None
    )
    lightforcing_local_attn_size = int(
        data.get("local_attn_size", data.get("lightforcing_local_attn_size", 12))
    )
    lightforcing_num_output_frames = max(
        1,
        int(data.get("num_output_frames", data.get("lightforcing_num_output_frames", 120))),
    )
    lightforcing_frame_seq = max(1, int(data.get("frame_seq", data.get("lightforcing_frame_seq", 1560))))
    lightforcing_keep_frames = max(0, int(data.get("keep_frames", data.get("lightforcing_keep_frames", 3))))
    lightforcing_keep_sink = max(0, int(data.get("keep_sink", data.get("lightforcing_keep_sink", 0))))
    lightforcing_keep_near = max(0, int(data.get("keep_near", data.get("lightforcing_keep_near", 1))))
    predict_next_layer_query = _truthy(
        os.getenv(
            "HORIZONKV_PREDICT_NEXT_LAYER_QUERY",
            data.get("predict_next_layer_query", False),
        )
    )
    if predict_next_layer_query and frame_selection_score != "pooled_qk":
        raise ValueError("predict_next_layer_query requires frame_selection_score=pooled_qk")

    return KVSparseConfig(
        enabled=True,
        chunk_k=int(data.get("chunk_k", 8192)),
        respect_sink=bool(data.get("respect_sink", True)),
        granularity=gran,
        rule=rule,
        frame_selection_score=frame_selection_score,
        frame_selection_strategy=frame_selection_strategy,
        target_density=td,
        target_recall=float(tr) if tr is not None else None,
        max_kv_frames=int(mkf) if mkf is not None else None,
        frame_selection_score_threshold_ratio=float(fs_thr) if fs_thr is not None else None,
        low_recall_threshold=float(lr_gate) if lr_gate is not None else None,
        expand_target_recall=float(ex_rec) if ex_rec is not None else None,
        block_target_density=block_target_density,
        block_rule=block_rule,
        block_target_recall=float(block_target_recall) if block_target_recall is not None else None,
        block_density_allocation=block_density_allocation,
        block_density_scale_by_max_kv_frames=block_density_scale_by_max_kv_frames,
        block_force_dense_compulsory_frames=block_force_dense_compulsory_frames,
        block_compute_backend=block_compute_backend,
        attn_block_q=abq,
        attn_block_k=abk,
        layers=dict(data.get("layers", {})),
        importance_q_pool=importance_q_pool,
        importance_k_pool=importance_k_pool,
        forced_outside_budget=bool(data.get("forced_outside_budget", True)),
        cpu_offload_kv=bool(cpu_offload_kv),
        gpu_cache_frames=gpu_cache_frames,
        cache_full_flash_attn=bool(cache_full_flash_attn),
        overlap_block_sparsity=float(overlap_block_sparsity),
        mandatory_recent_kv_frames=mandatory_recent_kv_frames,
        mandatory_preceding_kv_frames=mandatory_preceding_kv_frames,
        reuse_frame_selection=bool(reuse_frame_selection),
        prefetch_next_layer_kv=bool(prefetch_next_layer_kv),
        predict_next_layer_query=bool(predict_next_layer_query),
        reindex_rope_positions=bool(reindex_rope_positions),
        score_with_unroped_qk=bool(score_with_unroped_qk),
        rope_semantics=rope_semantics,
        selector=selector,
        lightforcing_sparsity=lightforcing_sparsity,
        lightforcing_sparsity_base=lightforcing_sparsity_base,
        lightforcing_local_attn_size=lightforcing_local_attn_size,
        lightforcing_num_output_frames=lightforcing_num_output_frames,
        lightforcing_frame_seq=lightforcing_frame_seq,
        lightforcing_keep_frames=lightforcing_keep_frames,
        lightforcing_keep_sink=lightforcing_keep_sink,
        lightforcing_keep_near=lightforcing_keep_near,
        visualize_attn_mask=visualize_attn_mask,
        visualize_output_dir=visualize_output_dir,
        visualize_layers=visualize_layers,
        visualize_heads=visualize_heads,
        visualize_timesteps=visualize_timesteps,
        visualize_full_indices=visualize_full_indices,
        source_path=str(path),
        source_stem=path.stem,
        visualize_run_name=visualize_run_name,
        profile_cache_hit=profile_cache_hit,
        profile_onload=profile_onload,
        profile_onload_sync_cuda=profile_onload_sync_cuda,
        profile_onload_log_every=profile_onload_log_every,
        profile_frame_selection=profile_frame_selection,
        profile_stage2=profile_stage2,
        profile_chunk=profile_chunk,
        profile_chunk_sync_cuda=profile_chunk_sync_cuda,
        profile_nvtx=profile_nvtx,
        profile_sync_cuda=profile_sync_cuda,
        profile_log_every=profile_log_every,
        profile_verbose=profile_verbose,
    )


def attach_kv_sparse_to_model(model: torch.nn.Module, path: str | Path | None) -> None:
    p = str(path or "").strip()
    cfg = load_kv_sparse_config(p) if p else None
    model.kv_sparse_config = cfg
    if cfg is not None:
        c = cfg
        extra = f"rule={c.rule} target_density={c.target_density}"
        if c.selector != "longlive":
            extra += f" selector={c.selector}"
            extra += (
                f" lf_sparsity={c.lightforcing_sparsity}"
                f" lf_keep_frames={c.lightforcing_keep_frames}"
                f" lf_keep_sink={c.lightforcing_keep_sink}"
                f" lf_keep_near={c.lightforcing_keep_near}"
            )
        extra += f" cpu_offload_kv={c.cpu_offload_kv}"
        if c.gpu_cache_frames is not None:
            extra += f" gpu_cache_frames={c.gpu_cache_frames}"
        if c.cache_full_flash_attn:
            extra += " cache_full_flash_attn=True"
        if float(getattr(c, "overlap_block_sparsity", 0.0) or 0.0) > 0.0:
            extra += f" overlap_block_sparsity={c.overlap_block_sparsity}"
        if c.mandatory_recent_kv_frames is not None:
            extra += f" mandatory_recent_kv_frames={c.mandatory_recent_kv_frames}"
        if c.mandatory_preceding_kv_frames is not None:
            extra += f" mandatory_preceding_kv_frames={c.mandatory_preceding_kv_frames}"
        if c.predict_next_layer_query:
            extra += " predict_next_layer_query=True"
        if c.reuse_frame_selection:
            extra += " reuse_frame_selection=True"
        if c.prefetch_next_layer_kv:
            extra += " prefetch_next_layer_kv=True"
        if c.reindex_rope_positions:
            extra += " reindex_rope_positions=True"
        if c.score_with_unroped_qk:
            extra += " score_with_unroped_qk=True"
        if c.rope_semantics != "absolute":
            extra += f" rope_semantics={c.rope_semantics}"
        if c.target_recall is not None:
            extra += f" target_recall={c.target_recall}"
        if c.max_kv_frames is not None:
            extra += f" max_kv_frames={c.max_kv_frames}"
        block_override_raw = str(os.getenv("HORIZONKV_ATTN_BLOCK_SIZE", "")).strip()
        if block_override_raw:
            extra += f" env_attn_block_size={block_override_raw}"
        if c.frame_selection_score_threshold_ratio is not None:
            extra += (
                f" frame_selection_score_threshold_ratio="
                f"{c.frame_selection_score_threshold_ratio}"
            )
        if c.rule == "density_if_low_recall":
            if c.low_recall_threshold is not None:
                extra += f" low_recall_threshold={c.low_recall_threshold}"
            if c.expand_target_recall is not None:
                extra += f" expand_target_recall={c.expand_target_recall}"
        print(
            f"[horizonkv] granularity={c.granularity} frame_selection_score={c.frame_selection_score} "
            f"frame_selection_strategy={c.frame_selection_strategy} "
            f"{extra} "
            f"chunk_k={c.chunk_k} importance_q_pool={c.importance_q_pool} "
            f"importance_k_pool={c.importance_k_pool}"
        )
        if c.granularity == "attn_block":
            budget = (
                f"target_recall={c.target_recall}"
                if c.rule == "recall"
                else f"target_density={c.target_density}"
            )
            print(
                f"[horizonkv] attn_block_tile=({c.attn_block_q}x{c.attn_block_k}) "
                f"rule={c.rule} {budget} "
                f"block_compute_backend={c.block_compute_backend}"
            )
            if c.selector == "lightforcing":
                print(
                    "[horizonkv] lightforcing_compute_backend=lightforcing_triton "
                    "(selector + sparse attention kernel)"
                )
        if c.granularity == "kv_frame_then_attn_block":
            block_budget = (
                f"block_target_recall={c.block_target_recall}"
                if c.block_rule == "recall"
                else f"block_target_density={c.block_target_density}"
            )
            print(
                f"[horizonkv] two_level attn_block_tile=({c.attn_block_q}x{c.attn_block_k}) "
                f"block_rule={c.block_rule} {block_budget} "
                f"block_density_allocation={c.block_density_allocation} "
                f"block_density_scale_by_max_kv_frames={c.block_density_scale_by_max_kv_frames} "
                f"block_compute_backend={c.block_compute_backend} "
                f"block_force_dense_compulsory_frames={c.block_force_dense_compulsory_frames}"
            )
        if c.visualize_attn_mask:
            print(
                f"[horizonkv] visualize_attn_mask=ON dir={c.visualize_output_dir} "
                f"run={c.visualize_run_name or 'run'} cfg={c.source_stem or 'cfg'} "
                f"layers={_format_filter(c.visualize_layers)} "
                f"heads={_format_filter(c.visualize_heads)} "
                f"timesteps={_format_filter(c.visualize_timesteps)} "
                f"full_indices={c.visualize_full_indices}"
            )
        if c.profile_frame_selection:
            print(
                "[horizonkv] profile_frame_selection=ON "
                f"sync_cuda={c.profile_sync_cuda} log_every={c.profile_log_every} "
                f"verbose={c.profile_verbose}"
            )
        if c.profile_stage2:
            print(
                "[horizonkv] profile_stage2=ON "
                f"sync_cuda={c.profile_sync_cuda} log_every={c.profile_log_every} "
                f"verbose={c.profile_verbose}"
            )
        if c.profile_cache_hit:
            print(
                "[horizonkv] profile_cache_hit=ON "
                f"log_every={c.profile_log_every} verbose={c.profile_verbose}"
            )
        if c.profile_onload:
            print(
                "[horizonkv] profile_onload=ON "
                f"sync_cuda={c.profile_onload_sync_cuda} log_every={c.profile_onload_log_every}"
            )
        if c.profile_chunk:
            print(
                "[horizonkv] profile_chunk=ON "
                f"sync_cuda={c.profile_chunk_sync_cuda} log_every={c.profile_log_every} "
                f"nvtx={c.profile_nvtx}"
            )
    elif p:
        print(f"[horizonkv] WARNING: path set but sparse OFF: {p!r}")


__all__ = [
    "DEFAULT_ATTN_BLOCK_SIZE",
    "DEFAULT_IMPORTANCE_K_POOL",
    "DEFAULT_IMPORTANCE_Q_POOL",
    "KVSparseConfig",
    "_format_filter",
    "_normalize_timestep_token",
    "_parse_index_filter",
    "_parse_timestep_filter",
    "_sanitize_token",
    "_scalarize_timestep",
    "_truthy",
    "attach_kv_sparse_to_model",
    "load_kv_sparse_config",
]
