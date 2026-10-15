# Modified for the HorizonKV inference artifact; see NOTICE.
"""Lazy CUDA extension for fused sparse KV CPU onload + GPU slot materialization."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def _enabled_by_env() -> bool:
    value = str(os.getenv("HORIZONKV_FUSED_ONLOAD", "1")).strip().lower()
    return value not in ("0", "false", "no", "off", "disable", "disabled")


def _plan_enabled_by_env() -> bool:
    value = str(os.getenv("HORIZONKV_CPP_PLAN", "0")).strip().lower()
    return value not in ("0", "false", "no", "off", "disable", "disabled")


def _batched_h2d_enabled_by_env() -> bool:
    value = str(os.getenv("HORIZONKV_BATCHED_H2D", "0")).strip().lower()
    return value not in ("0", "false", "no", "off", "disable", "disabled")


def _batched_h2d_min_copy_items_by_env() -> int:
    raw = str(os.getenv("HORIZONKV_BATCHED_H2D_MIN_COPIES", "8")).strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 8


def _cpu_cache_is_head_major(
    cpu_k: torch.Tensor,
    gpu_k: torch.Tensor,
    *,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> bool:
    """Infer layout from GPU slot geometry; a plan may contain only some heads."""
    if cpu_k.ndim != 4 or gpu_k.ndim != 4:
        return False
    tokens_per_head = int(frame_seqlen) * int(gpu_cache_frames)
    if tokens_per_head <= 0 or int(gpu_k.shape[1]) % tokens_per_head:
        return False
    gpu_heads = int(gpu_k.shape[1]) // tokens_per_head
    return int(cpu_k.shape[1]) == gpu_heads and int(cpu_k.shape[2]) != gpu_heads


@lru_cache(maxsize=1)
def _load_module() -> Any | None:
    if not _enabled_by_env():
        return None
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception:
        return None
    if CUDA_HOME is None:
        return None

    src_dir = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(os.getenv("HORIZONKV_FUSED_ONLOAD_BUILD_DIR", "/tmp/horizonkv_transfer"))
    build_dir.mkdir(parents=True, exist_ok=True)
    verbose = str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        return load(
            name="horizonkv_transfer",
            sources=[
                str(src_dir / "kv_sparse_onload.cpp"),
                str(src_dir / "kv_sparse_onload_cuda.cu"),
            ],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            print(f"[horizonkv] fused onload extension disabled after build failure: {exc}")
        return None


def fused_kv_frame_materialize(
    *,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int = 0,
) -> bool:
    """Materialize compact KV from GPU slots/live KV after miss frames are copied."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_frame_materialize"):
        return False
    device = k_out.device
    empty_live = torch.empty(0, device=device, dtype=k_out.dtype)
    has_live = live_k is not None and live_v is not None
    module.kv_sparse_frame_materialize(
        gpu_k,
        gpu_v,
        k_out,
        v_out,
        head_indices,
        frame_indices,
        slot_indices,
        live_k if has_live else empty_live,
        live_v if has_live else empty_live,
        int(frame_seqlen),
        int(total_tokens),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        bool(has_live),
    )
    return True


def fused_kv_frame_materialize_varlen(
    *,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int = 0,
) -> bool:
    """Materialize compact KV directly into FlashAttention varlen packed layout."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_frame_materialize_varlen"):
        return False
    device = k_out.device
    empty_live = torch.empty(0, device=device, dtype=k_out.dtype)
    has_live = live_k is not None and live_v is not None
    module.kv_sparse_frame_materialize_varlen(
        gpu_k,
        gpu_v,
        k_out,
        v_out,
        head_indices,
        frame_indices,
        slot_indices,
        cu_seqlens_k,
        live_k if has_live else empty_live,
        live_v if has_live else empty_live,
        int(frame_seqlen),
        int(total_tokens),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        bool(has_live),
    )
    return True


def select_pooled_qk_topk(
    *,
    q: torch.Tensor,
    frame_k: torch.Tensor,
    head_indices: torch.Tensor,
    compulsory_indices: torch.Tensor,
    frame_seqlen: int,
    max_kv_frames: int,
) -> torch.Tensor | None:
    """Fused pooled-QK score + max-frame topK selection."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_select_pooled_qk_topk"):
        return None
    return module.kv_sparse_select_pooled_qk_topk(
        q,
        frame_k,
        head_indices,
        compulsory_indices,
        int(frame_seqlen),
        int(max_kv_frames),
    )


def lookup_frame_slots_all_hit(
    *,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_to_slot: torch.Tensor,
    total_tokens: int,
    frame_seqlen: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_start_index: int,
    live_len: int,
    has_live: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Look up selected frame slots on GPU and return (slot_indices, miss_flag)."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_lookup_frame_slots"):
        return None
    result = module.kv_sparse_lookup_frame_slots(
        head_indices,
        frame_indices,
        frame_to_slot,
        int(total_tokens),
        int(frame_seqlen),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        int(live_len),
        bool(has_live),
    )
    if result is None or len(result) != 2:
        return None
    return result[0], result[1]


def update_frame_slot_map(
    *,
    frame_to_slot: torch.Tensor,
    head_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    evicted_frame_ids: torch.Tensor,
) -> bool:
    """Apply batched frame->slot metadata updates on GPU."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_update_frame_slot_map"):
        return False
    module.kv_sparse_update_frame_slot_map(
        frame_to_slot,
        head_ids,
        frame_ids,
        slot_ids,
        evicted_frame_ids,
    )
    return True


def copy_kv_frame_plan_to_gpu_slots(
    *,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_plan: torch.Tensor,
    copy_plan: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> int:
    """Copy miss frames from a CPU frame plan into GPU slots."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_copy_frame_plan_to_slots"):
        return 0
    return int(module.kv_sparse_copy_frame_plan_to_slots(cpu_k, cpu_v, gpu_k, gpu_v, frame_indices, slot_plan, copy_plan, int(frame_seqlen), int(gpu_cache_frames)))


def copy_kv_frame_plan_to_gpu_slots_batched(
    *,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_indices: torch.Tensor,
    slot_plan: torch.Tensor,
    copy_plan: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> int:
    """Submit pitched CPU KV copies with one CUDA driver batch API call."""
    if not _batched_h2d_enabled_by_env():
        return 0
    head_major = _cpu_cache_is_head_major(
        cpu_k, gpu_k, frame_seqlen=frame_seqlen, gpu_cache_frames=gpu_cache_frames
    )
    if not head_major:
        copy_count = int(copy_plan.ne(0).sum().item())
        if copy_count < _batched_h2d_min_copy_items_by_env():
            return 0
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_copy_frame_plan_to_slots_batched"):
        return 0
    return int(
        module.kv_sparse_copy_frame_plan_to_slots_batched(
            cpu_k,
            cpu_v,
            gpu_k,
            gpu_v,
            frame_indices,
            slot_plan,
            copy_plan,
            int(frame_seqlen),
            int(gpu_cache_frames),
        )
    )


def copy_kv_frames_to_gpu_compact(
    *,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    head_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_seqlen: int,
    cpu_head_major: bool = False,
) -> int:
    """Copy selected CPU KV frames into FlashAttention packed compact layout."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_copy_frames_to_compact"):
        return 0
    return int(
        module.kv_sparse_copy_frames_to_compact(
            cpu_k,
            cpu_v,
            compact_k,
            compact_v,
            head_ids,
            frame_ids,
            cu_seqlens,
            int(frame_seqlen),
            bool(cpu_head_major),
        )
    )


def copy_compact_to_gpu_slots_d2d(
    *,
    compact_k: torch.Tensor,
    compact_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    src_rows: torch.Tensor,
    src_cols: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
) -> int:
    """Fill GPU slots from packed miss KV with asynchronous DtoD memcpy."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_copy_compact_to_slots_d2d"):
        return 0
    return int(
        module.kv_sparse_copy_compact_to_slots_d2d(
            compact_k,
            compact_v,
            gpu_k,
            gpu_v,
            head_ids,
            slot_ids,
            src_rows,
            src_cols,
            cu_seqlens,
            int(frame_seqlen),
            int(gpu_cache_frames),
        )
    )


def copy_kv_span_to_cpu_cache(
    *,
    gpu_k_span: torch.Tensor,
    gpu_v_span: torch.Tensor,
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    start_token: int,
) -> int:
    """Async DtoH copy of a contiguous KV token span into pinned CPU cache."""
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_copy_kv_span_to_cpu"):
        return 0
    return int(
        module.kv_sparse_copy_kv_span_to_cpu(
            gpu_k_span,
            gpu_v_span,
            cpu_k,
            cpu_v,
            int(start_token),
        )
    )


def plan_kv_frame_slots_tensor(
    *,
    frame_indices: torch.Tensor,
    frame_to_slot: torch.Tensor,
    slot_frame_ids: torch.Tensor,
    slot_frame_lens: torch.Tensor,
    slot_last_used: torch.Tensor,
    total_tokens: int,
    frame_seqlen: int,
    num_heads: int,
    gpu_cache_frames: int,
    live_start_index: int,
    live_len: int,
    has_live: bool,
    gpu_clock: int,
    build_compact_plans: bool = True,
    split_min_miss_items: int = 0,
    split_min_miss_ratio: float = 0.0,
) -> dict[str, Any] | None:
    """Run tensor-native CPU-side slot planning inside the extension."""
    if not _plan_enabled_by_env():
        return None
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_plan_slots_tensor"):
        return None
    compact_options = (
        (bool(build_compact_plans), int(split_min_miss_items), float(split_min_miss_ratio))
        if getattr(module, "slot_plan_version", 1) >= 2
        else ()
    )
    return module.kv_sparse_plan_slots_tensor(
        frame_indices,
        frame_to_slot,
        slot_frame_ids,
        slot_frame_lens,
        slot_last_used,
        int(total_tokens),
        int(frame_seqlen),
        int(num_heads),
        int(gpu_cache_frames),
        int(live_start_index),
        int(live_len),
        bool(has_live),
        int(gpu_clock),
        *compact_options,
    )
