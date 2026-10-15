# Modified for the HorizonKV inference artifact; see NOTICE.
"""Host KV history, bounded GPU slots and overlapped hit/miss attention."""
from __future__ import annotations
import atexit
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import torch
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig
from flagscale.inference.core.diffusion.horizonkv.wan.attention import attention
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_kernels import (
    compacted_rope_apply,
    compacted_rope_apply_compact,
    compacted_rope_apply_packed_rows,
)
import flash_attn
from flagscale.inference.core.diffusion.horizonkv.algorithm.transfer import (
    copy_kv_frame_plan_to_gpu_slots,
    copy_kv_frame_plan_to_gpu_slots_batched,
    copy_compact_to_gpu_slots_d2d,
    copy_kv_span_to_cpu_cache,
    copy_kv_frames_to_gpu_compact,
    fused_kv_frame_materialize,
    fused_kv_frame_materialize_varlen,
    plan_kv_frame_slots_tensor,
    select_pooled_qk_topk,
    update_frame_slot_map,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.triton_kernels import (
    blocksparse_attention_pooled_triton,
    merge_attention_lse_parts_triton,
)


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _overlap_attn_enabled_by_env() -> bool:
    return _truthy_env("HORIZONKV_OVERLAP_ATTN", "0")


def _overlap_flash_split_attention_enabled_by_env() -> bool:
    default = os.getenv("HORIZONKV_OVERLAP_FLASH_LSE", "1")
    return _truthy_env("HORIZONKV_OVERLAP_FLASH_SPLIT_ATTN", default)


def _overlap_flash_materialize_enabled_by_env() -> bool:
    return True


def _overlap_effective_block_sparsity(
    kv_sparse_config: KVSparseConfig | None, selected_frames: int | None
) -> float:
    return kv_sparse_config.overlap_block_sparsity


def _overlap_blocksparse_block_size(kv_sparse_config: KVSparseConfig | None, attr: str) -> int:
    return 64


def _overlap_split_min_miss_items_by_env() -> int:
    return int(os.environ["HORIZONKV_OVERLAP_SPLIT_MIN_MISS_ITEMS"])


def _overlap_split_worthwhile_for_miss_counts(
    *, hit_item_count: int, miss_item_count: int, miss_copy_count: int
) -> bool:
    miss_item_count = max(0, int(miss_item_count))
    miss_copy_count = max(0, int(miss_copy_count))
    if miss_item_count <= 0 or miss_copy_count <= 0:
        return False
    if miss_item_count < _overlap_split_min_miss_items_by_env():
        return False
    return True


_NO_NVTX_CONTEXT = nullcontext()
_SHARED_CUDA_STREAMS: dict[tuple[str, int, str], torch.cuda.Stream] = {}
_ALL_HEAD_INDICES_DEVICE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}
_COMPULSORY_INDICES_DEVICE_CACHE: dict[tuple[str, int, int, tuple[int, ...]], torch.Tensor] = {}
_COMPULSORY_INDICES_DEVICE_READY: dict[
    tuple[str, int, int, tuple[int, ...]], tuple[torch.cuda.Event, int]
] = {}
_PREFETCH_SUBMIT_EXECUTOR: ThreadPoolExecutor | None = None


def _shutdown_submit_executors() -> None:
    global _PREFETCH_SUBMIT_EXECUTOR
    prefetch_executor = _PREFETCH_SUBMIT_EXECUTOR
    _PREFETCH_SUBMIT_EXECUTOR = None
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True, cancel_futures=False)


atexit.register(_shutdown_submit_executors)


def _can_pin_cpu_memory(device: torch.device | str) -> bool:
    dev = torch.device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def _zeros_cpu(
    shape: tuple[int, ...] | list[int], *, dtype: torch.dtype, pin_memory: bool
) -> torch.Tensor:
    try:
        return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    except RuntimeError:
        return torch.zeros(shape, dtype=dtype, device="cpu")


def _pin_cpu_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return tensor
    try:
        if tensor.is_pinned() or not torch.cuda.is_available():
            return tensor
    except Exception:
        return tensor
    try:
        return tensor.pin_memory()
    except Exception:
        return tensor


def build_kv_cache_state(
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    frame_capacity: int,
    frame_seqlen: int,
    num_heads: int,
    head_dim: int,
    kv_sparse_config: KVSparseConfig | None = None,
) -> dict[str, Any]:
    frame_capacity = max(1, int(frame_capacity))
    frame_seqlen = max(1, int(frame_seqlen))
    token_capacity = frame_capacity * frame_seqlen
    num_heads = max(1, int(num_heads))
    head_dim = max(1, int(head_dim))
    device = torch.device(device)
    gpu_cache_frames = getattr(kv_sparse_config, "gpu_cache_frames", None)
    gpu_cache_frames = max(0, int(gpu_cache_frames or 0))
    cache_full_flash_attn = bool(gpu_cache_frames > 0 and False)
    base: dict[str, Any] = {
        "cache_mode": "cpu_offload",
        "cpu_offload_kv": True,
        "frame_capacity": int(frame_capacity),
        "frame_seqlen": int(frame_seqlen),
        "token_capacity": int(token_capacity),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    }
    pin_memory = _can_pin_cpu_memory(device)
    selected_frame_capacity = 0
    if kv_sparse_config is not None:
        try:
            selected_frame_capacity = max(
                0, int(getattr(kv_sparse_config, "max_kv_frames", 0) or 0)
            )
        except (TypeError, ValueError):
            selected_frame_capacity = 0
    frame_indices_cpu_workspace = None
    if pin_memory and selected_frame_capacity > 0:
        try:
            frame_indices_cpu_workspace = torch.empty(
                (num_heads, selected_frame_capacity),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            frame_indices_cpu_workspace = None
    cpu_kv_shape = [batch_size, num_heads, token_capacity, head_dim]
    base.update(
        {
            "cpu_k": _zeros_cpu(cpu_kv_shape, dtype=dtype, pin_memory=pin_memory),
            "cpu_v": _zeros_cpu(cpu_kv_shape, dtype=dtype, pin_memory=pin_memory),
            "cpu_cache_head_major": True,
            "frame_k_mean": torch.zeros(
                [batch_size, frame_capacity, num_heads, head_dim], dtype=dtype, device=device
            ),
            "gpu_cache_frames": int(gpu_cache_frames),
            "gpu_frame_to_slot_dense": (
                torch.full((num_heads, frame_capacity), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_frame_ids_tensor_cpu": (
                torch.full((num_heads, gpu_cache_frames), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_frame_lens_tensor_cpu": (
                torch.zeros((num_heads, gpu_cache_frames), dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_slot_last_used_tensor_cpu": (
                torch.full((num_heads, gpu_cache_frames), -1, dtype=torch.long)
                if gpu_cache_frames > 0
                else None
            ),
            "gpu_identity_filled_frames": 0,
            "_kv_sparse_clean_gpu_identity_fill_scheduled_frames": 0,
            "gpu_clock": 0,
            "_kv_sparse_overlap_frame_indices_cpu_workspace": {
                "buffer": frame_indices_cpu_workspace,
                "event": None,
            },
        }
    )
    if gpu_cache_frames > 0:
        gpu_k_slots = torch.empty(
            [batch_size, num_heads, gpu_cache_frames, frame_seqlen, head_dim],
            dtype=dtype,
            device=device,
        )
        gpu_v_slots = torch.empty_like(gpu_k_slots)
        base["gpu_k_slots"] = gpu_k_slots
        base["gpu_v_slots"] = gpu_v_slots
        base["gpu_k"] = gpu_k_slots.reshape(
            batch_size, num_heads * gpu_cache_frames * frame_seqlen, 1, head_dim
        )
        base["gpu_v"] = gpu_v_slots.reshape(
            batch_size, num_heads * gpu_cache_frames * frame_seqlen, 1, head_dim
        )
        gpu_frame_to_slot_device = torch.full(
            (num_heads, frame_capacity), -1, dtype=torch.long, device=device
        )
        identity_count = min(frame_capacity, gpu_cache_frames)
        if identity_count > 0:
            identity_slots = torch.arange(identity_count, dtype=torch.long, device=device)
            gpu_frame_to_slot_device[:, :identity_count] = identity_slots.view(1, -1)
        base["gpu_frame_to_slot_device"] = gpu_frame_to_slot_device
    else:
        base["gpu_k"] = None
        base["gpu_v"] = None
        base["gpu_k_slots"] = None
        base["gpu_v_slots"] = None
        base["gpu_frame_to_slot_device"] = None
    return base


def reset_kv_cache_state(kv_cache: dict[str, Any]) -> None:
    _kv_cache_wait_pending_slot_metadata_commit(kv_cache)
    _kv_cache_wait_pending_cpu_copy(kv_cache)
    if "global_end_index" in kv_cache:
        kv_cache["global_end_index"].fill_(0)
    if "local_end_index" in kv_cache:
        kv_cache["local_end_index"].fill_(0)
    gpu_k = kv_cache.get("gpu_k", None)
    stream = None
    if isinstance(gpu_k, torch.Tensor) and gpu_k.device.type == "cuda":
        stream = _shared_cuda_stream(device=gpu_k.device, name="gpu_prefetch")
    if stream is not None and torch.cuda.is_available():
        try:
            stream.synchronize()
        except Exception:
            pass
        for stream_name in (
            "overlap_compact_onload",
            "overlap_compact_slot_fill",
            "overlap_frame_indices_d2h",
            "overlap_onload",
            "next_layer_prefetch",
        ):
            try:
                extra_stream = _shared_cuda_stream(device=gpu_k.device, name=stream_name)
                if extra_stream is not None:
                    extra_stream.synchronize()
            except Exception:
                pass
    dense_frame_to_slot = kv_cache.get("gpu_frame_to_slot_dense", None)
    if isinstance(dense_frame_to_slot, torch.Tensor):
        dense_frame_to_slot.fill_(-1)
    slot_frame_ids_tensor = kv_cache.get("gpu_slot_frame_ids_tensor_cpu", None)
    if isinstance(slot_frame_ids_tensor, torch.Tensor):
        slot_frame_ids_tensor.fill_(-1)
    slot_frame_lens_tensor = kv_cache.get("gpu_slot_frame_lens_tensor_cpu", None)
    if isinstance(slot_frame_lens_tensor, torch.Tensor):
        slot_frame_lens_tensor.zero_()
    slot_last_used_tensor = kv_cache.get("gpu_slot_last_used_tensor_cpu", None)
    if isinstance(slot_last_used_tensor, torch.Tensor):
        slot_last_used_tensor.fill_(-1)
    kv_cache["gpu_clock"] = 0
    kv_cache["gpu_identity_filled_frames"] = 0
    kv_cache["_kv_sparse_clean_gpu_identity_fill_scheduled_frames"] = 0
    gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
    if isinstance(gpu_frame_to_slot_device, torch.Tensor):
        gpu_frame_to_slot_device.fill_(-1)
        identity_count = min(
            int(gpu_frame_to_slot_device.shape[1]), int(kv_cache.get("gpu_cache_frames", 0) or 0)
        )
        if identity_count > 0:
            identity_slots = torch.arange(
                identity_count, dtype=torch.long, device=gpu_frame_to_slot_device.device
            )
            gpu_frame_to_slot_device[:, :identity_count] = identity_slots.view(1, -1)
    kv_cache["_kv_sparse_gpu_hit_spec_cooldown"] = 0
    kv_cache.pop("_kv_sparse_skip_gpu_hit_plan_once", None)
    kv_cache.pop("_kv_sparse_pending_next_selection", None)
    frame_indices_workspace = kv_cache.get("_kv_sparse_overlap_frame_indices_cpu_workspace", None)
    if isinstance(frame_indices_workspace, dict):
        frame_indices_workspace["event"] = None


def _kv_cache_slot_state_tensors(
    kv_cache: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    return tuple(
        (
            kv_cache[k]
            for k in (
                "gpu_frame_to_slot_dense",
                "gpu_slot_frame_ids_tensor_cpu",
                "gpu_slot_frame_lens_tensor_cpu",
                "gpu_slot_last_used_tensor_cpu",
            )
        )
    )


def _kv_cache_wait_pending_slot_metadata_commit(kv_cache: dict[str, Any] | None) -> bool:
    if not isinstance(kv_cache, dict):
        return True
    pending = kv_cache.pop("_kv_sparse_pending_slot_metadata_commits", None)
    if pending is None:
        kv_cache.pop("_kv_sparse_pending_slot_metadata_commit", None)
        return True
    if not isinstance(pending, list):
        return True
    ok = True
    for item in pending:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage", None)
        if not isinstance(stage, dict):
            continue
        ready_event = item.get("ready_event", None)
        if ready_event is not None and torch.cuda.is_available():
            try:
                device = item.get("device", None)
                device = torch.device(device) if device is not None else None
                current_stream = (
                    torch.cuda.current_stream(device)
                    if device is not None
                    else torch.cuda.current_stream()
                )
                current_stream.wait_event(ready_event)
            except Exception:
                pass
        committed = _kv_cache_commit_compact_stage_slots(
            kv_cache,
            stage,
            frame_seqlen=int(item.get("frame_seqlen", 0) or 0),
            total_tokens=int(item.get("total_tokens", 0) or 0),
            device_wait_event=ready_event,
        )
        ok = bool(ok and committed)
        if committed and item.get("identity_fill_end", None) is not None:
            identity_fill_end = int(item["identity_fill_end"])
            kv_cache["gpu_identity_filled_frames"] = max(
                int(kv_cache.get("gpu_identity_filled_frames", 0) or 0), identity_fill_end
            )
            kv_cache["_kv_sparse_clean_gpu_identity_fill_scheduled_frames"] = max(
                int(kv_cache.get("_kv_sparse_clean_gpu_identity_fill_scheduled_frames", 0) or 0),
                identity_fill_end,
            )
    kv_cache.pop("_kv_sparse_pending_slot_metadata_commit", None)
    return ok


def _kv_cache_commit_incremental_tensor_slot_plan_state(
    kv_cache: dict[str, Any],
    plan_result: dict[str, Any],
    next_clock: int,
    *,
    frame_seqlen: int,
    total_tokens: int,
) -> bool:
    if not bool(plan_result.get("tensor_slot_state_incremental", False)):
        return False

    def _cpu_long_1d(name: str) -> torch.Tensor | None:
        value = plan_result.get(name, None)
        if (
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and (value.dtype == torch.long)
            and (value.ndim == 1)
        ):
            return _pin_cpu_tensor_if_available(value.contiguous())
        return None

    fill_head_ids = _cpu_long_1d("fill_head_ids")
    fill_frame_ids = _cpu_long_1d("fill_frame_ids")
    fill_slot_ids = _cpu_long_1d("fill_slot_ids")
    fill_last_used = _cpu_long_1d("fill_last_used")
    hit_head_ids = _cpu_long_1d("hit_head_ids")
    hit_slot_ids = _cpu_long_1d("hit_slot_ids")
    hit_last_used = _cpu_long_1d("hit_last_used")
    required = (
        fill_head_ids,
        fill_frame_ids,
        fill_slot_ids,
        fill_last_used,
        hit_head_ids,
        hit_slot_ids,
        hit_last_used,
    )
    if any((value is None for value in required)):
        return False
    assert fill_head_ids is not None
    assert fill_frame_ids is not None
    assert fill_slot_ids is not None
    assert fill_last_used is not None
    assert hit_head_ids is not None
    assert hit_slot_ids is not None
    assert hit_last_used is not None
    fill_count = int(fill_head_ids.numel())
    hit_count = int(hit_head_ids.numel())
    if (
        int(fill_frame_ids.numel()) != fill_count
        or int(fill_slot_ids.numel()) != fill_count
        or int(fill_last_used.numel()) != fill_count
        or (int(hit_slot_ids.numel()) != hit_count)
        or (int(hit_last_used.numel()) != hit_count)
    ):
        return False
    slot_state = _kv_cache_slot_state_tensors(kv_cache)
    if slot_state is None:
        return False
    dense, slot_frame_ids_t, slot_frame_lens_t, slot_last_used_t = slot_state
    num_heads = int(slot_frame_ids_t.shape[0])
    gpu_cache_frames = int(slot_frame_ids_t.shape[1])
    frame_capacity = int(dense.shape[1])
    if hit_count > 0:
        hit_valid = (
            hit_head_ids.ge(0)
            & hit_head_ids.lt(num_heads)
            & hit_slot_ids.ge(0)
            & hit_slot_ids.lt(gpu_cache_frames)
        )
        if bool(hit_valid.any().item()):
            write_hit_heads = hit_head_ids[hit_valid]
            write_hit_slots = hit_slot_ids[hit_valid]
            write_hit_last = hit_last_used[hit_valid]
            slot_last_used_t[write_hit_heads, write_hit_slots] = write_hit_last
    if fill_count <= 0:
        kv_cache["gpu_clock"] = max(int(kv_cache.get("gpu_clock", 0) or 0), int(next_clock))
        return True
    heads = fill_head_ids.contiguous()
    frames = fill_frame_ids.contiguous()
    slots = fill_slot_ids.contiguous()
    last_used = fill_last_used.contiguous()
    head_slot_valid = heads.ge(0) & heads.lt(num_heads) & slots.ge(0) & slots.lt(gpu_cache_frames)
    frame_valid = head_slot_valid & frames.ge(0) & frames.lt(frame_capacity)
    if not bool(frame_valid.any().item()):
        kv_cache["gpu_clock"] = max(int(kv_cache.get("gpu_clock", 0) or 0), int(next_clock))
        return True
    valid_heads = heads[frame_valid]
    valid_slots = slots[frame_valid]
    valid_frames = frames[frame_valid]
    valid_last_used = last_used[frame_valid]
    unique_head_slots = torch.unique(valid_heads * gpu_cache_frames + valid_slots)
    if int(unique_head_slots.numel()) != int(valid_heads.numel()):
        return False
    old_frames = slot_frame_ids_t[valid_heads, valid_slots].clone()
    old_valid = old_frames.ge(0) & old_frames.lt(frame_capacity)
    if bool(old_valid.any().item()):
        old_heads = valid_heads[old_valid]
        old_slots = valid_slots[old_valid]
        old_frames_valid = old_frames[old_valid]
        old_matches = dense[old_heads, old_frames_valid].eq(old_slots)
        if bool(old_matches.any().item()):
            dense[old_heads[old_matches], old_frames_valid[old_matches]] = -1
    frame_lens = (int(total_tokens) - valid_frames * int(frame_seqlen)).clamp(
        min=0, max=int(frame_seqlen)
    )
    nonempty = frame_lens.gt(0)
    if bool(nonempty.any().item()):
        write_heads = valid_heads[nonempty]
        write_slots = valid_slots[nonempty]
        write_frames = valid_frames[nonempty]
        write_lens = frame_lens[nonempty]
        write_last_used = valid_last_used[nonempty]
        dense[write_heads, write_frames] = write_slots
        slot_frame_ids_t[write_heads, write_slots] = write_frames
        slot_frame_lens_t[write_heads, write_slots] = write_lens
        slot_last_used_t[write_heads, write_slots] = write_last_used
        gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
        if (
            isinstance(gpu_frame_to_slot_device, torch.Tensor)
            and gpu_frame_to_slot_device.device.type == "cuda"
        ):
            write_heads = _pin_cpu_tensor_if_available(write_heads.contiguous())
            write_frames = _pin_cpu_tensor_if_available(write_frames.contiguous())
            write_slots = _pin_cpu_tensor_if_available(write_slots.contiguous())
            write_old_frames = _pin_cpu_tensor_if_available(old_frames[nonempty].contiguous())
            try:
                update_frame_slot_map(
                    frame_to_slot=gpu_frame_to_slot_device,
                    head_ids=write_heads.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    frame_ids=write_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    slot_ids=write_slots.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    evicted_frame_ids=write_old_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                )
            except Exception:
                gpu_frame_to_slot_device.fill_(-1)
    kv_cache["gpu_clock"] = max(int(kv_cache.get("gpu_clock", 0) or 0), int(next_clock))
    return True


def _sink_frame_indices(Lk: int, kv_start: int, frame_seqlen: int, sink_tokens: int) -> set[int]:
    if sink_tokens <= 0:
        return set()
    sink_len = min(Lk, max(0, int(sink_tokens) - int(kv_start)))
    if sink_len <= 0:
        return set()
    last_frame = (sink_len - 1) // frame_seqlen
    return set(range(last_frame + 1))


def _mandatory_kv_frame_indices(
    Lk: int,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    num_frame_per_block: int,
) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0:
        return set()
    chunk_tokens = max(1, int(num_frame_per_block) * int(frame_seqlen))
    chunk_lo = int(current_start) // chunk_tokens * chunk_tokens
    chunk_hi = chunk_lo + chunk_tokens
    cs, ce = (int(current_start), int(current_end))
    base = int(current_end) - int(local_end_index) + int(kv_start)
    intervals = [(chunk_lo - base, chunk_hi - base), (cs - base, ce - base)]
    out: set[int] = set()
    for a, b in intervals:
        lo = max(0, int(a))
        hi = min(Lk, int(b))
        if lo >= hi:
            continue
        f0 = lo // frame_seqlen
        f1 = (hi - 1) // frame_seqlen
        out.update(range(f0, f1 + 1))
    return out


def _preceding_kv_frame_indices(
    Lk: int,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    num_frame_per_block: int,
    preceding_frames: int | None,
) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0 or preceding_frames is None or (int(preceding_frames) <= 0):
        return set()
    chunk_tokens = max(1, int(num_frame_per_block) * int(frame_seqlen))
    chunk_start = int(current_start) // chunk_tokens * chunk_tokens
    window_global_start = int(current_end) - int(local_end_index) + int(kv_start)
    local_chunk_start = int(chunk_start) - int(window_global_start)
    end_frame = local_chunk_start // int(frame_seqlen)
    n_frames = max(1, (int(Lk) + int(frame_seqlen) - 1) // int(frame_seqlen))
    end_frame = max(0, min(int(end_frame), n_frames))
    start_frame = max(0, end_frame - int(preceding_frames))
    return set(range(start_frame, end_frame))


def _recent_kv_frame_indices(Lk: int, frame_seqlen: int, recent_frames: int | None) -> set[int]:
    if Lk <= 0 or frame_seqlen <= 0 or recent_frames is None or (int(recent_frames) <= 0):
        return set()
    n_frames = max(1, (int(Lk) + int(frame_seqlen) - 1) // int(frame_seqlen))
    start_frame = max(0, n_frames - int(recent_frames))
    return set(range(start_frame, n_frames))


def _kv_cache_update_slot_metadata_fast(
    kv_cache: dict[str, Any],
    *,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    device_wait_event: torch.cuda.Event | None = None,
) -> bool:
    if not (
        isinstance(head_ids, torch.Tensor)
        and isinstance(slot_ids, torch.Tensor)
        and isinstance(frame_ids, torch.Tensor)
        and (head_ids.device.type == "cpu")
        and (slot_ids.device.type == "cpu")
        and (frame_ids.device.type == "cpu")
        and (head_ids.dtype == torch.long)
        and (slot_ids.dtype == torch.long)
        and (frame_ids.dtype == torch.long)
        and (head_ids.ndim == 1)
        and (slot_ids.ndim == 1)
        and (frame_ids.ndim == 1)
        and (int(head_ids.numel()) == int(slot_ids.numel()))
        and (int(frame_ids.numel()) == int(head_ids.numel()))
    ):
        return False
    slot_state = _kv_cache_slot_state_tensors(kv_cache)
    if slot_state is None:
        return False
    dense, slot_frame_ids_t, slot_frame_lens_t, slot_last_used_t = slot_state
    num_heads = int(slot_frame_ids_t.shape[0])
    gpu_cache_frames = int(slot_frame_ids_t.shape[1])
    frame_capacity = int(dense.shape[1])
    n = int(head_ids.numel())
    if n <= 0:
        return True

    def _keep_last_by_head_slot(
        heads_t: torch.Tensor, slots_t: torch.Tensor, *payload_tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if int(heads_t.numel()) <= 1:
            return (heads_t, slots_t, *payload_tensors)
        keys = (heads_t * gpu_cache_frames + slots_t).tolist()
        last_positions: dict[int, int] = {}
        for pos, key in enumerate(keys):
            last_positions[int(key)] = pos
        if len(last_positions) == len(keys):
            return (heads_t, slots_t, *payload_tensors)
        keep_positions = torch.tensor(sorted(last_positions.values()), dtype=torch.long)
        tensors = (heads_t, slots_t, *payload_tensors)
        return tuple((tensor[keep_positions] for tensor in tensors))

    clock = int(kv_cache.get("gpu_clock", 0) or 0)
    heads = head_ids.contiguous()
    slots = slot_ids.contiguous()
    frames = frame_ids.contiguous()
    head_slot_valid = heads.ge(0) & heads.lt(num_heads) & slots.ge(0) & slots.lt(gpu_cache_frames)
    frame_valid = head_slot_valid & frames.ge(0) & frames.lt(frame_capacity)
    if not bool(frame_valid.any().item()):
        kv_cache["gpu_clock"] = max(clock, clock + n)
        return True
    valid_heads = heads[frame_valid]
    valid_slots = slots[frame_valid]
    valid_frames = frames[frame_valid]
    valid_orig_indices = torch.arange(n, dtype=torch.long)[frame_valid]
    valid_heads, valid_slots, valid_frames, valid_orig_indices = _keep_last_by_head_slot(
        valid_heads, valid_slots, valid_frames, valid_orig_indices
    )
    old_frames = slot_frame_ids_t[valid_heads, valid_slots].clone()
    old_valid = old_frames.ge(0) & old_frames.lt(frame_capacity)
    if bool(old_valid.any().item()):
        old_heads = valid_heads[old_valid]
        old_slots = valid_slots[old_valid]
        old_frames_valid = old_frames[old_valid]
        old_matches = dense[old_heads, old_frames_valid].eq(old_slots)
        if bool(old_matches.any().item()):
            dense[old_heads[old_matches], old_frames_valid[old_matches]] = -1
    frame_lens = (int(total_tokens) - valid_frames * int(frame_seqlen)).clamp(
        min=0, max=int(frame_seqlen)
    )
    nonempty = frame_lens.gt(0)
    if bool(nonempty.any().item()):
        write_heads = valid_heads[nonempty]
        write_slots = valid_slots[nonempty]
        write_frames = valid_frames[nonempty]
        write_lens = frame_lens[nonempty]
        write_last_used = clock + valid_orig_indices[nonempty]
        dense[write_heads, write_frames] = write_slots
        slot_frame_ids_t[write_heads, write_slots] = write_frames
        slot_frame_lens_t[write_heads, write_slots] = write_lens
        slot_last_used_t[write_heads, write_slots] = write_last_used
        gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
        if (
            isinstance(gpu_frame_to_slot_device, torch.Tensor)
            and gpu_frame_to_slot_device.device.type == "cuda"
        ):
            write_heads = _pin_cpu_tensor_if_available(write_heads.contiguous())
            write_frames = _pin_cpu_tensor_if_available(write_frames.contiguous())
            write_slots = _pin_cpu_tensor_if_available(write_slots.contiguous())
            write_old_frames = _pin_cpu_tensor_if_available(old_frames[nonempty].contiguous())
            try:
                update_frame_slot_map(
                    frame_to_slot=gpu_frame_to_slot_device,
                    head_ids=write_heads.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    frame_ids=write_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    slot_ids=write_slots.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                    evicted_frame_ids=write_old_frames.to(
                        device=gpu_frame_to_slot_device.device, non_blocking=True
                    ),
                )
            except Exception:
                gpu_frame_to_slot_device.fill_(-1)
    kv_cache["gpu_clock"] = max(int(kv_cache.get("gpu_clock", 0) or 0), clock + n)
    _ = device_wait_event
    return True


def _kv_cache_commit_tensor_slot_plan_state(
    kv_cache: dict[str, Any], plan_result: dict[str, Any], next_clock: int
) -> bool:
    if not bool(plan_result.get("tensor_slot_state", False)):
        return False
    frame_to_slot = plan_result.get("frame_to_slot", None)
    slot_frame_ids = plan_result.get("slot_frame_ids", None)
    slot_frame_lens = plan_result.get("slot_frame_lens", None)
    slot_last_used = plan_result.get("slot_last_used", None)
    tensors = (frame_to_slot, slot_frame_ids, slot_frame_lens, slot_last_used)
    if not all((isinstance(tensor, torch.Tensor) for tensor in tensors)):
        return False
    if not all((tensor.device.type == "cpu" and tensor.dtype == torch.long for tensor in tensors)):
        return False
    if frame_to_slot.ndim != 2 or slot_frame_ids.ndim != 2:
        return False
    if (
        slot_frame_lens.shape != slot_frame_ids.shape
        or slot_last_used.shape != slot_frame_ids.shape
    ):
        return False
    kv_cache["gpu_frame_to_slot_dense"] = _pin_cpu_tensor_if_available(frame_to_slot.contiguous())
    kv_cache["gpu_slot_frame_ids_tensor_cpu"] = _pin_cpu_tensor_if_available(
        slot_frame_ids.contiguous()
    )
    kv_cache["gpu_slot_frame_lens_tensor_cpu"] = _pin_cpu_tensor_if_available(
        slot_frame_lens.contiguous()
    )
    kv_cache["gpu_slot_last_used_tensor_cpu"] = _pin_cpu_tensor_if_available(
        slot_last_used.contiguous()
    )
    kv_cache["gpu_clock"] = int(next_clock)
    gpu_frame_to_slot_device = kv_cache.get("gpu_frame_to_slot_device", None)
    if (
        isinstance(gpu_frame_to_slot_device, torch.Tensor)
        and gpu_frame_to_slot_device.device.type == "cuda"
        and (tuple(gpu_frame_to_slot_device.shape) == tuple(frame_to_slot.shape))
    ):
        gpu_frame_to_slot_device.copy_(frame_to_slot, non_blocking=True)
    return True


def _kv_cache_commit_compact_stage_slots(
    kv_cache: dict[str, Any],
    stage: dict[str, Any],
    *,
    frame_seqlen: int,
    total_tokens: int,
    device_wait_event: torch.cuda.Event | None = None,
) -> bool:
    return _kv_cache_update_slot_metadata_fast(
        kv_cache,
        head_ids=stage["scatter_head_ids_cpu"],
        slot_ids=stage["scatter_slot_ids_cpu"],
        frame_ids=stage["scatter_frame_ids_cpu"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
        device_wait_event=device_wait_event,
    )


def _kv_cache_queue_compact_stage_slots_commit(
    kv_cache: dict[str, Any],
    stage: dict[str, Any],
    *,
    frame_seqlen: int,
    total_tokens: int,
    ready_event: torch.cuda.Event | None = None,
    device: torch.device | None = None,
) -> bool:
    commit_stage = {
        "scatter_head_ids_cpu": stage["scatter_head_ids_cpu"],
        "scatter_slot_ids_cpu": stage["scatter_slot_ids_cpu"],
        "scatter_frame_ids_cpu": stage["scatter_frame_ids_cpu"],
    }
    pending = kv_cache.setdefault("_kv_sparse_pending_slot_metadata_commits", [])
    if not isinstance(pending, list):
        pending = []
        kv_cache["_kv_sparse_pending_slot_metadata_commits"] = pending
    pending.append(
        {
            "stage": commit_stage,
            "frame_seqlen": int(frame_seqlen),
            "total_tokens": int(total_tokens),
            "ready_event": ready_event,
            "device": device,
        }
    )
    return True


def _kv_cache_queue_slot_metadata_commit(
    kv_cache: dict[str, Any],
    *,
    head_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_seqlen: int,
    total_tokens: int,
    ready_event: torch.cuda.Event | None = None,
    device: torch.device | None = None,
    identity_fill_end: int | None = None,
) -> bool:
    if not (
        isinstance(head_ids, torch.Tensor)
        and isinstance(slot_ids, torch.Tensor)
        and isinstance(frame_ids, torch.Tensor)
        and (head_ids.device.type == "cpu")
        and (slot_ids.device.type == "cpu")
        and (frame_ids.device.type == "cpu")
        and (head_ids.dtype == torch.long)
        and (slot_ids.dtype == torch.long)
        and (frame_ids.dtype == torch.long)
        and (head_ids.ndim == 1)
        and (slot_ids.ndim == 1)
        and (frame_ids.ndim == 1)
        and (int(head_ids.numel()) == int(slot_ids.numel()))
        and (int(frame_ids.numel()) == int(head_ids.numel()))
    ):
        return False
    pending = kv_cache.setdefault("_kv_sparse_pending_slot_metadata_commits", [])
    if not isinstance(pending, list):
        pending = []
        kv_cache["_kv_sparse_pending_slot_metadata_commits"] = pending
    pending.append(
        {
            "stage": {
                "scatter_head_ids_cpu": head_ids.contiguous(),
                "scatter_slot_ids_cpu": slot_ids.contiguous(),
                "scatter_frame_ids_cpu": frame_ids.contiguous(),
            },
            "frame_seqlen": int(frame_seqlen),
            "total_tokens": int(total_tokens),
            "ready_event": ready_event,
            "device": device,
            "identity_fill_end": identity_fill_end,
        }
    )
    return True


def _shared_cuda_stream(*, device: torch.device, name: str) -> torch.cuda.Stream | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = ("cuda", device_index, str(name))
    stream = _SHARED_CUDA_STREAMS.get(key, None)
    if stream is not None:
        try:
            if torch.device(stream.device) == torch.device("cuda", device_index):
                return stream
        except Exception:
            pass
    stream = torch.cuda.Stream(device=torch.device("cuda", device_index))
    _SHARED_CUDA_STREAMS[key] = stream
    return stream


def _stage_overlap_frame_indices_to_cpu_async(
    kv_cache: dict[str, Any], frame_indices: torch.Tensor
) -> dict[str, Any] | None:
    if (
        frame_indices.device.type != "cuda"
        or frame_indices.dtype != torch.long
        or frame_indices.ndim != 2
        or (not torch.cuda.is_available())
    ):
        return None
    copy_stream = _shared_cuda_stream(device=frame_indices.device, name="overlap_frame_indices_d2h")
    if copy_stream is None:
        return None
    workspace = kv_cache.setdefault(
        "_kv_sparse_overlap_frame_indices_cpu_workspace", {"buffer": None, "event": None}
    )
    if not isinstance(workspace, dict):
        workspace = {"buffer": None, "event": None}
        kv_cache["_kv_sparse_overlap_frame_indices_cpu_workspace"] = workspace
    pending = workspace.setdefault("pending", [])
    if pending:
        pending[:] = [(buf, ev) for buf, ev in pending if not ev.query()]
    prior_event = workspace.get("event", None)
    buffer = workspace.get("buffer", None)
    if prior_event is not None:
        try:
            if not prior_event.query():
                if isinstance(buffer, torch.Tensor):
                    pending.append((buffer, prior_event))
                buffer = None
            else:
                workspace["event"] = None
        except Exception:
            buffer = None
    required = int(frame_indices.numel())
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.device.type != "cpu"
        or buffer.dtype != torch.long
        or (not buffer.is_pinned())
        or (int(buffer.numel()) < required)
    ):
        capacity = 1 << max(0, required - 1).bit_length()
        try:
            buffer = torch.empty((capacity,), dtype=torch.long, device="cpu", pin_memory=True)
        except RuntimeError:
            return None
    workspace["buffer"] = buffer
    cpu_view = buffer.reshape(-1).narrow(0, 0, required).view(frame_indices.shape)
    try:
        with torch.cuda.device(frame_indices.device):
            producer_stream = torch.cuda.current_stream(frame_indices.device)
            producer_event = torch.cuda.Event()
            producer_event.record(producer_stream)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(producer_event)
                cpu_view.copy_(frame_indices.detach(), non_blocking=True)
                done_event = torch.cuda.Event()
                done_event.record(copy_stream)
        frame_indices.detach().record_stream(copy_stream)
    except Exception:
        return None
    workspace["event"] = done_event
    return {"tensor": cpu_view, "event": done_event, "workspace": workspace}


def _resolve_staged_overlap_frame_indices_cpu(staged: dict[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(staged, dict):
        return None
    tensor = staged.get("tensor", None)
    if not (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and (tensor.dtype == torch.long)
        and tensor.is_contiguous()
    ):
        return None
    event = staged.get("event", None)
    if event is not None:
        try:
            if not event.query():
                event.synchronize()
        except Exception:
            return None
    workspace = staged.get("workspace", None)
    if isinstance(workspace, dict) and workspace.get("event", None) is event:
        workspace["event"] = None
    return tensor


def _all_head_indices_device(num_heads: int, device: torch.device) -> torch.Tensor:
    num_heads = int(num_heads)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", int(torch.cuda.current_device()))
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, num_heads)
    cached = _ALL_HEAD_INDICES_DEVICE_CACHE.get(key, None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.device == device
        and (cached.dtype == torch.long)
        and (int(cached.numel()) == num_heads)
    ):
        return cached
    tensor = torch.arange(num_heads, device=device, dtype=torch.long)
    _ALL_HEAD_INDICES_DEVICE_CACHE[key] = tensor
    return tensor


def _kv_cache_cpu_copy_stream(
    kv_cache: dict[str, Any], device: torch.device
) -> torch.cuda.Stream | None:
    _ = kv_cache
    return _shared_cuda_stream(device=device, name="cpu_copy")


def _kv_cache_stream_ready_event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


def update_gpu_kv_cache_frames(
    kv_cache: dict[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    source_start_index: int,
    current_start: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    frame_k: torch.Tensor,
    frame_v: torch.Tensor,
    producer_event: torch.cuda.Event | None = None,
) -> None:
    _ = (current_start, num_frame_per_block)
    if not isinstance(kv_cache, dict):
        return
    gpu_cache_frames = int(kv_cache.get("gpu_cache_frames", 0) or 0)
    if gpu_cache_frames <= 0:
        return
    committed_identity = max(0, int(kv_cache.get("gpu_identity_filled_frames", 0) or 0))
    scheduled_identity = max(
        committed_identity,
        int(
            kv_cache.get("_kv_sparse_clean_gpu_identity_fill_scheduled_frames", committed_identity)
            or 0
        ),
    )
    identity_start = min(scheduled_identity, gpu_cache_frames)
    if identity_start >= gpu_cache_frames:
        return
    gpu_k = kv_cache.get("gpu_k", None)
    gpu_v = kv_cache.get("gpu_v", None)
    if (
        not isinstance(gpu_k, torch.Tensor)
        or not isinstance(gpu_v, torch.Tensor)
        or (not isinstance(frame_k, torch.Tensor))
        or (not isinstance(frame_v, torch.Tensor))
        or (frame_k.device != gpu_k.device)
        or (frame_v.device != gpu_v.device)
    ):
        return
    slot_state = _kv_cache_slot_state_tensors(kv_cache)
    if slot_state is None:
        return
    cpu_k = kv_cache.get("cpu_k", None)
    if not isinstance(cpu_k, torch.Tensor):
        return
    frame_seqlen = int(frame_seqlen)
    if frame_seqlen <= 0 or frame_k.ndim != 4 or frame_v.ndim != 4:
        return
    if int(frame_k.shape[1]) <= 0 or int(frame_k.shape[2]) <= 0:
        return
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    if start >= end:
        return
    total_tokens = int(cpu_k.shape[2])
    source_start = int(source_start_index)
    source_token_count = int(frame_k.shape[1])
    num_heads = min(
        int(kv_cache.get("num_heads", 0) or 0),
        int(frame_k.shape[2]),
        int(frame_v.shape[2]),
        int(slot_state[1].shape[0]),
    )
    if num_heads <= 0:
        return
    frame_capacity = int(slot_state[0].shape[1])
    fill_start = int(identity_start)
    if start > fill_start:
        return
    fill_end = min(end, gpu_cache_frames, frame_capacity)
    if fill_start >= fill_end:
        return
    abs_start = fill_start * frame_seqlen
    rel_start = abs_start - source_start
    source_token_count = min(source_token_count, int(frame_v.shape[1]))
    if rel_start < 0 or rel_start >= source_token_count:
        return
    frame_count = min(fill_end - fill_start, (source_token_count - rel_start) // frame_seqlen)
    if frame_count <= 0:
        return
    fill_end = fill_start + frame_count
    rel_end = rel_start + frame_count * frame_seqlen
    identity_fill_end = fill_end
    frame_ids_cpu = torch.arange(fill_start, fill_end, dtype=torch.long)
    heads_cpu = torch.arange(num_heads, dtype=torch.long).repeat(frame_count)
    frames_cpu = frame_ids_cpu.repeat_interleave(num_heads)
    slots_cpu = frames_cpu.clone()
    copy_stream = _shared_cuda_stream(device=gpu_k.device, name="clean_kv_gpu_fill")
    copy_done: torch.cuda.Event | None = None
    gpu_k_slots = kv_cache.get("gpu_k_slots", None)
    gpu_v_slots = kv_cache.get("gpu_v_slots", None)
    use_slot_layout = (
        isinstance(gpu_k_slots, torch.Tensor)
        and isinstance(gpu_v_slots, torch.Tensor)
        and (gpu_k_slots.device == gpu_k.device)
        and (gpu_v_slots.device == gpu_v.device)
        and (gpu_k_slots.ndim == 5)
        and (gpu_v_slots.ndim == 5)
    )

    def _record_scheduled_identity_fill() -> None:
        kv_cache["_kv_sparse_clean_gpu_identity_fill_scheduled_frames"] = max(
            int(kv_cache.get("_kv_sparse_clean_gpu_identity_fill_scheduled_frames", 0) or 0),
            int(identity_fill_end),
        )

    def _copy_identity_frames() -> None:
        if use_slot_layout:
            assert isinstance(gpu_k_slots, torch.Tensor)
            assert isinstance(gpu_v_slots, torch.Tensor)
            batch_size = int(frame_k.shape[0])
            k_src = (
                frame_k[:, rel_start:rel_end, :num_heads, :]
                .reshape(batch_size, frame_count, frame_seqlen, num_heads, int(frame_k.shape[-1]))
                .permute(0, 3, 1, 2, 4)
            )
            v_src = (
                frame_v[:, rel_start:rel_end, :num_heads, :]
                .reshape(batch_size, frame_count, frame_seqlen, num_heads, int(frame_v.shape[-1]))
                .permute(0, 3, 1, 2, 4)
            )
            gpu_k_slots[:, :num_heads, fill_start:fill_end].copy_(k_src, non_blocking=True)
            gpu_v_slots[:, :num_heads, fill_start:fill_end].copy_(v_src, non_blocking=True)
            return
        slot_len = frame_count * frame_seqlen
        for head_id in range(num_heads):
            slot_start = (head_id * gpu_cache_frames + fill_start) * frame_seqlen
            gpu_k[:, slot_start : slot_start + slot_len].copy_(
                frame_k[:, rel_start:rel_end, head_id : head_id + 1, :], non_blocking=True
            )
            gpu_v[:, slot_start : slot_start + slot_len].copy_(
                frame_v[:, rel_start:rel_end, head_id : head_id + 1, :], non_blocking=True
            )

    if copy_stream is not None and torch.cuda.is_available() and (gpu_k.device.type == "cuda"):
        if producer_event is None:
            producer_event = _kv_cache_stream_ready_event(frame_k.device)
        with torch.cuda.device(gpu_k.device):
            with torch.cuda.stream(copy_stream):
                if producer_event is not None:
                    copy_stream.wait_event(producer_event)
                _copy_identity_frames()
                copy_done = torch.cuda.Event()
                copy_done.record(copy_stream)
        frame_k.record_stream(copy_stream)
        frame_v.record_stream(copy_stream)
        _record_scheduled_identity_fill()
        if _kv_cache_queue_slot_metadata_commit(
            kv_cache,
            head_ids=heads_cpu,
            slot_ids=slots_cpu,
            frame_ids=frames_cpu,
            frame_seqlen=frame_seqlen,
            total_tokens=total_tokens,
            ready_event=copy_done,
            device=gpu_k.device,
            identity_fill_end=identity_fill_end,
        ):
            return
    _copy_identity_frames()
    metadata_committed = _kv_cache_update_slot_metadata_fast(
        kv_cache,
        head_ids=heads_cpu,
        slot_ids=slots_cpu,
        frame_ids=frames_cpu,
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
        device_wait_event=copy_done,
    )
    if metadata_committed and identity_fill_end is not None:
        kv_cache["gpu_identity_filled_frames"] = max(
            int(kv_cache.get("gpu_identity_filled_frames", 0) or 0), identity_fill_end
        )
        _record_scheduled_identity_fill()


def _kv_cache_pending_cpu_copy_records(
    kv_cache: dict[str, Any]
) -> list[tuple[torch.cuda.Event, tuple[int, int]]]:
    records = kv_cache.get("cpu_copy_records", None)
    if isinstance(records, list) and records:
        out: list[tuple[torch.cuda.Event, tuple[int, int]]] = []
        for item in records:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            event, span = item
            if event is None or span is None:
                continue
            out.append((event, (int(span[0]), int(span[1]))))
        if out:
            active: list[tuple[torch.cuda.Event, tuple[int, int]]] = []
            changed = False
            for event, span in out:
                keep = True
                if torch.cuda.is_available():
                    try:
                        keep = not bool(event.query())
                    except Exception:
                        keep = True
                if keep:
                    active.append((event, span))
                else:
                    changed = True
            if changed:
                _kv_cache_set_pending_cpu_copy_records(kv_cache, active)
            if active:
                return active
            return []
    event = kv_cache.get("cpu_copy_event", None)
    if event is None:
        return []
    ranges = kv_cache.get("cpu_copy_ranges", None)
    if not ranges:
        pending = kv_cache.get("cpu_copy_range", None)
        ranges = [pending] if pending is not None else []
    out = []
    for pending in ranges:
        if pending is None:
            continue
        out.append((event, (int(pending[0]), int(pending[1]))))
    if not out:
        return []
    active: list[tuple[torch.cuda.Event, tuple[int, int]]] = []
    changed = False
    for pending_event, span in out:
        keep = True
        if torch.cuda.is_available():
            try:
                keep = not bool(pending_event.query())
            except Exception:
                keep = True
        if keep:
            active.append((pending_event, span))
        else:
            changed = True
    if changed:
        _kv_cache_set_pending_cpu_copy_records(kv_cache, active)
    return active


def _kv_cache_set_pending_cpu_copy_records(
    kv_cache: dict[str, Any], records: list[tuple[torch.cuda.Event, tuple[int, int]]]
) -> None:
    if records:
        kv_cache["cpu_copy_records"] = [
            (event, (int(span[0]), int(span[1]))) for event, span in records
        ]
        last_event, last_span = records[-1]
        kv_cache["cpu_copy_event"] = last_event
        kv_cache["cpu_copy_range"] = tuple((int(v) for v in last_span))
        kv_cache["cpu_copy_ranges"] = [tuple((int(v) for v in span)) for _, span in records]
    else:
        kv_cache["cpu_copy_records"] = []
        kv_cache["cpu_copy_event"] = None
        kv_cache["cpu_copy_range"] = None
        kv_cache["cpu_copy_ranges"] = []


def _kv_cache_wait_pending_cpu_copy(kv_cache: dict[str, Any]) -> None:
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    if not records or not torch.cuda.is_available():
        return
    gpu_k = kv_cache.get("gpu_k", None)
    device = (
        gpu_k.device if isinstance(gpu_k, torch.Tensor) and gpu_k.device.type == "cuda" else None
    )
    if device is None:
        frame_stats = kv_cache.get("frame_k_mean", None)
        if isinstance(frame_stats, torch.Tensor) and frame_stats.device.type == "cuda":
            device = frame_stats.device
    if device is None:
        return
    current_stream = torch.cuda.current_stream(device)
    for event, _ in records:
        current_stream.wait_event(event)
    _kv_cache_set_pending_cpu_copy_records(kv_cache, [])


def copy_kv_to_cpu_cache_async(
    kv_cache: dict[str, Any],
    *,
    local_start_index: int,
    local_end_index: int,
    key: torch.Tensor,
    value: torch.Tensor,
    producer_event: torch.cuda.Event | None = None,
) -> None:
    start = int(local_start_index)
    end = int(local_end_index)
    if start >= end:
        return
    cpu_k = kv_cache["cpu_k"]
    cpu_v = kv_cache["cpu_v"]

    def copy_fallback() -> None:
        cpu_k[:, :, start:end].copy_(key.detach().permute(0, 2, 1, 3), non_blocking=True)
        cpu_v[:, :, start:end].copy_(value.detach().permute(0, 2, 1, 3), non_blocking=True)

    if (
        key.device.type != "cuda"
        or value.device != key.device
        or (not torch.cuda.is_available())
        or (not (cpu_k.is_pinned() and cpu_v.is_pinned()))
    ):
        copy_fallback()
        _kv_cache_set_pending_cpu_copy_records(kv_cache, [])
        return
    stream = _kv_cache_cpu_copy_stream(kv_cache, key.device)
    if stream is None:
        copy_fallback()
        _kv_cache_set_pending_cpu_copy_records(kv_cache, [])
        return
    if producer_event is None:
        producer_event = _kv_cache_stream_ready_event(key.device)
    key_detached = key.detach()
    value_detached = value.detach()
    if producer_event is not None:
        stream.wait_event(producer_event)
    with torch.cuda.stream(stream):
        copied = 0
        try:
            copied = copy_kv_span_to_cpu_cache(
                gpu_k_span=key_detached,
                gpu_v_span=value_detached,
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                start_token=start,
            )
        except Exception:
            copied = 0
        if copied <= 0:
            cpu_k[:, :, start:end].copy_(key_detached.permute(0, 2, 1, 3), non_blocking=True)
            cpu_v[:, :, start:end].copy_(value_detached.permute(0, 2, 1, 3), non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    key_detached.record_stream(stream)
    value_detached.record_stream(stream)
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    records.append((event, (start, end)))
    _kv_cache_set_pending_cpu_copy_records(kv_cache, records)
    kv_cache["cpu_copy_last_submitted_range"] = (start, end)


def queue_deferred_clean_kv_to_cpu_cache(
    kv_cache: dict[str, Any],
    next_kv_cache: dict[str, Any] | None,
    *,
    local_start_index: int,
    local_end_index: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    queue_kv_cache = next_kv_cache if isinstance(next_kv_cache, dict) else kv_cache
    if (
        not isinstance(queue_kv_cache, dict)
        or key.device.type != "cuda"
        or value.device != key.device
        or (not torch.cuda.is_available())
    ):
        return False
    start = int(local_start_index)
    end = int(local_end_index)
    if start >= end:
        return True
    producer_event = _kv_cache_stream_ready_event(key.device)
    record = {
        "target_kv_cache": kv_cache,
        "local_start_index": start,
        "local_end_index": end,
        "key": key.detach(),
        "value": value.detach(),
        "producer_event": producer_event,
    }
    queue_kv_cache.setdefault("_deferred_clean_kv_offloads", []).append(record)
    return True


def flush_deferred_clean_kv_offloads(kv_cache: dict[str, Any] | None) -> None:
    if not isinstance(kv_cache, dict):
        return
    records = kv_cache.pop("_deferred_clean_kv_offloads", None)
    if not records:
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        target = record.get("target_kv_cache", None)
        key = record.get("key", None)
        value = record.get("value", None)
        if (
            not isinstance(target, dict)
            or not isinstance(key, torch.Tensor)
            or (not isinstance(value, torch.Tensor))
        ):
            continue
        copy_kv_to_cpu_cache_async(
            target,
            local_start_index=int(record.get("local_start_index", 0)),
            local_end_index=int(record.get("local_end_index", 0)),
            key=key,
            value=value,
            producer_event=record.get("producer_event", None),
        )


def _kv_cache_copy_frame_tensor_plan_from_cpu_to_slots(
    kv_cache: dict[str, Any],
    *,
    frame_indices_cpu: torch.Tensor,
    slot_plan_cpu: torch.Tensor,
    copy_plan_cpu: torch.Tensor,
    expected_copies: int | None = None,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> bool:
    gpu_k = kv_cache["gpu_k"]
    gpu_v = kv_cache["gpu_v"]
    cpu_k = kv_cache["cpu_k"]
    cpu_v = kv_cache["cpu_v"]
    frame_seqlen = int(frame_seqlen)
    frame_indices_cpu = frame_indices_cpu.contiguous()
    slot_plan_cpu = slot_plan_cpu.contiguous()
    copy_plan_cpu = copy_plan_cpu.contiguous()
    if expected_copies is None:
        expected_copies = int(copy_plan_cpu.ne(0).sum().item())
    else:
        expected_copies = int(expected_copies)
    if expected_copies <= 0:
        return True
    if gpu_k.device.type == "cuda":
        try:
            with torch.cuda.stream(stream):
                copied = copy_kv_frame_plan_to_gpu_slots_batched(
                    cpu_k=cpu_k,
                    cpu_v=cpu_v,
                    gpu_k=gpu_k,
                    gpu_v=gpu_v,
                    frame_indices=frame_indices_cpu,
                    slot_plan=slot_plan_cpu,
                    copy_plan=copy_plan_cpu,
                    frame_seqlen=frame_seqlen,
                    gpu_cache_frames=int(kv_cache.get("gpu_cache_frames", 0) or 0),
                )
            if int(copied) == expected_copies:
                return True
            if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                print(
                    f"[horizonkv] head-major batch H2D did not copy full plan: copied={int(copied)} expected={expected_copies}"
                )
        except Exception as exc:
            if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                print(f"[horizonkv] CUDA batch H2D fallback after failure: {exc}")
    if gpu_k.device.type == "cuda":
        try:
            with torch.cuda.stream(stream):
                copied = copy_kv_frame_plan_to_gpu_slots(
                    cpu_k=cpu_k,
                    cpu_v=cpu_v,
                    gpu_k=gpu_k,
                    gpu_v=gpu_v,
                    frame_indices=frame_indices_cpu,
                    slot_plan=slot_plan_cpu,
                    copy_plan=copy_plan_cpu,
                    frame_seqlen=frame_seqlen,
                    gpu_cache_frames=int(kv_cache.get("gpu_cache_frames", 0) or 0),
                )
            if int(copied) == expected_copies:
                return True
        except Exception as exc:
            if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                print(f"[horizonkv] batched frame copy fallback after failure: {exc}")
    return False


def _overlay_live_frame_stats(
    cached_stats: torch.Tensor | None,
    live_stats: torch.Tensor | None,
    *,
    live_start_index: int | None,
    frame_seqlen: int,
    n_frames: int,
) -> torch.Tensor | None:
    if cached_stats is None:
        return None
    out = cached_stats[:, :n_frames]
    if live_stats is None or live_start_index is None or live_stats.shape[1] == 0:
        return out
    start_frame = int(live_start_index) // int(frame_seqlen)
    end_frame = min(int(n_frames), start_frame + int(live_stats.shape[1]))
    if start_frame < 0 or start_frame >= end_frame:
        return out
    out = out.clone()
    out[:, start_frame:end_frame] = live_stats[:, : end_frame - start_frame]
    return out


@dataclass(slots=True)
class _OverlapRowPlan:
    active_rows_cpu: torch.Tensor
    frame_plan_cpu: torch.Tensor
    slot_plan_cpu: torch.Tensor
    k_lens_cpu: torch.Tensor
    item_count: int
    cu_seqlens_cpu: torch.Tensor | None = None
    frame_positions_cpu: torch.Tensor | None = None
    all_full_length: bool = False
    covers_all_heads: bool | None = None
    control_long_cpu: torch.Tensor | None = None
    control_int_cpu: torch.Tensor | None = None


@dataclass(slots=True)
class _OverlapSlotPlan:
    slot_plan_cpu: torch.Tensor
    fill_plan_cpu: torch.Tensor
    copy_plan_cpu: torch.Tensor
    hit_item_count: int
    miss_item_count: int
    fill_count: int
    hit_row_plan: _OverlapRowPlan | None = None
    miss_row_plan: _OverlapRowPlan | None = None


def _row_plan_from_cpp_dict(raw: Any) -> _OverlapRowPlan | None:
    if raw is None:
        return None
    active_count = raw["active_rows"].numel()
    shape = raw["frame_plan"].shape
    plan_items = raw["frame_plan"].numel()
    control_long = _pin_cpu_tensor_if_available(raw["control_long"])
    control_int = _pin_cpu_tensor_if_available(raw["control_int"])
    return _OverlapRowPlan(
        active_rows_cpu=control_long[:active_count],
        frame_plan_cpu=control_long.narrow(0, active_count, plan_items).view(shape),
        slot_plan_cpu=control_long.narrow(0, active_count + plan_items, plan_items).view(shape),
        frame_positions_cpu=control_long.narrow(0, active_count + 2 * plan_items, plan_items).view(
            shape
        ),
        k_lens_cpu=control_int[:active_count],
        cu_seqlens_cpu=control_int[active_count:],
        item_count=raw["item_count"],
        all_full_length=raw["all_full_length"],
        covers_all_heads=raw["covers_all_heads"],
        control_long_cpu=control_long,
        control_int_cpu=control_int,
    )


def _overlap_row_plan_covers_all_heads(row_plan: _OverlapRowPlan, num_heads: int) -> bool:
    return row_plan.covers_all_heads


def _overlap_row_plan_covers_full_query_rows(
    row_plan: _OverlapRowPlan, *, query_rows: int, B: int, num_heads: int
) -> bool:
    return int(query_rows) == int(B) * int(num_heads) and _overlap_row_plan_covers_all_heads(
        row_plan, int(num_heads)
    )


def _build_overlap_slot_plan(
    *, plan_result: dict[str, Any], frame_indices_cpu: torch.Tensor
) -> _OverlapSlotPlan | None:
    return _OverlapSlotPlan(
        slot_plan_cpu=_pin_cpu_tensor_if_available(plan_result["slot_plan"]),
        fill_plan_cpu=_pin_cpu_tensor_if_available(plan_result["fill_plan"]),
        copy_plan_cpu=_pin_cpu_tensor_if_available(plan_result["copy_plan"]),
        hit_item_count=plan_result["readable_item_count"] - plan_result["miss_item_count"],
        miss_item_count=plan_result["miss_item_count"],
        fill_count=plan_result["copy_item_count"],
        hit_row_plan=_row_plan_from_cpp_dict(plan_result["hit_row_plan"]),
        miss_row_plan=_row_plan_from_cpp_dict(plan_result["miss_row_plan"]),
    )


def _apply_fast_lf_rope_to_compact_k(
    k_compact: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    frame_seqlen: int,
    selected_frames: int,
    rope_grid_sizes: torch.Tensor | None,
    rope_freqs: torch.Tensor | None,
    start_frame: int = 0,
) -> torch.Tensor | None:
    if not (
        isinstance(k_compact, torch.Tensor)
        and k_compact.ndim == 4
        and isinstance(rope_grid_sizes, torch.Tensor)
        and isinstance(rope_freqs, torch.Tensor)
    ):
        return None
    batch_size = int(batch_size)
    num_heads = int(num_heads)
    frame_seqlen = int(frame_seqlen)
    selected_frames = int(selected_frames)
    if batch_size <= 0 or num_heads <= 0 or frame_seqlen <= 0 or (selected_frames <= 0):
        return None
    rows, token_count, one, head_dim = (int(v) for v in k_compact.shape)
    expected_rows = batch_size * num_heads
    expected_tokens = selected_frames * frame_seqlen
    if rows != expected_rows or one != 1 or token_count != expected_tokens:
        return None
    if int(rope_grid_sizes.ndim) != 2 or int(rope_grid_sizes.shape[0]) != batch_size:
        return None
    compact_rope = compacted_rope_apply_compact(
        k_compact,
        num_heads=num_heads,
        selected_frames=selected_frames,
        frame_seqlen=frame_seqlen,
        grid_sizes=rope_grid_sizes,
        freqs=rope_freqs,
        start_frame=int(start_frame),
    )
    if compact_rope is not None:
        return compact_rope
    grid_sizes_kv = rope_grid_sizes.detach().clone()
    grid_sizes_kv[:, 0] = selected_frames
    k_blhd = k_compact.reshape(batch_size, num_heads, token_count, head_dim)
    k_blhd = k_blhd.permute(0, 2, 1, 3).contiguous()
    k_blhd = compacted_rope_apply(k_blhd, grid_sizes_kv, rope_freqs, start_frame=int(start_frame))
    return k_blhd.permute(0, 2, 1, 3).reshape(rows, token_count, 1, head_dim).contiguous()


def _apply_fast_lf_rope_to_row_materialized_k(
    k_compact: torch.Tensor,
    *,
    row_plan: _OverlapRowPlan,
    batch_size: int,
    frame_seqlen: int,
    k_lens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    rope_grid_sizes: torch.Tensor | None,
    rope_freqs: torch.Tensor | None,
    inplace: bool = False,
    frame_positions_device: torch.Tensor | None = None,
    start_frame: int = 0,
) -> torch.Tensor | None:
    if not (
        isinstance(k_compact, torch.Tensor)
        and isinstance(row_plan, _OverlapRowPlan)
        and isinstance(row_plan.frame_positions_cpu, torch.Tensor)
        and isinstance(k_lens, torch.Tensor)
        and isinstance(rope_grid_sizes, torch.Tensor)
        and isinstance(rope_freqs, torch.Tensor)
    ):
        return None
    if k_compact.ndim not in (3, 4) or int(k_compact.shape[-2]) != 1:
        return None
    active_count = int(row_plan.active_rows_cpu.numel())
    batch_size = int(batch_size)
    if active_count <= 0 or batch_size <= 0:
        return None
    frame_positions = frame_positions_device
    if frame_positions is None:
        frame_positions_cpu = row_plan.frame_positions_cpu
        if frame_positions_cpu.ndim != 2:
            return None
        if batch_size > 1:
            frame_positions_cpu = frame_positions_cpu.repeat(batch_size, 1)
        frame_positions_cpu = _pin_cpu_tensor_if_available(frame_positions_cpu.contiguous())
        frame_positions = frame_positions_cpu.to(
            device=k_compact.device, dtype=torch.long, non_blocking=True
        ).contiguous()
    row_lens = k_lens.to(device=k_compact.device, dtype=torch.int32, non_blocking=True).contiguous()
    if k_compact.ndim == 3:
        if not (
            isinstance(cu_seqlens, torch.Tensor)
            and cu_seqlens.ndim == 1
            and (int(cu_seqlens.numel()) == int(row_lens.numel()) + 1)
        ):
            return None
        cu_device = cu_seqlens.to(
            device=k_compact.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
    else:
        cu_device = None
    expected_rows = int(k_compact.shape[0]) if k_compact.ndim == 4 else int(row_lens.numel())
    if int(frame_positions.shape[0]) != expected_rows or int(row_lens.numel()) != expected_rows:
        return None
    return compacted_rope_apply_packed_rows(
        k_compact,
        frame_positions,
        row_lens,
        cu_device,
        frame_seqlen=int(frame_seqlen),
        grid_sizes=rope_grid_sizes,
        freqs=rope_freqs,
        start_frame=int(start_frame),
        inplace=bool(inplace),
    )


@torch.no_grad()
def _run_materialized_slot_plan_flash_attention(
    kv_cache: dict[str, Any],
    *,
    q_group: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_indices_cpu: torch.Tensor | None,
    slot_plan_cpu: torch.Tensor | None,
    B: int,
    D: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    frame_seqlen: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int | None = None,
    frame_indices_device: torch.Tensor | None = None,
    slot_indices_device: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    kv_sparse_config: KVSparseConfig | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
    rope_freqs: torch.Tensor | None = None,
    layer_idx: int | None = None,
    num_frame_per_block: int = 1,
    sink_tokens: int = 0,
    lf_k_start_frame: int = 0,
) -> torch.Tensor | None:
    if bool(kv_cache.get("fused_onload_disabled", False)):
        return None
    if int(D) <= 0 or int(D) > 256:
        return None
    device = gpu_k.device
    rows = int(q_group.shape[0])
    Lq = int(q_group.shape[1])
    plan_shape: tuple[int, int] | None = None
    if (
        isinstance(frame_indices_cpu, torch.Tensor)
        and isinstance(slot_plan_cpu, torch.Tensor)
        and (frame_indices_cpu.device.type == "cpu")
        and (slot_plan_cpu.device.type == "cpu")
        and (frame_indices_cpu.dtype == torch.long)
        and (slot_plan_cpu.dtype == torch.long)
        and (frame_indices_cpu.ndim == 2)
        and (tuple(slot_plan_cpu.shape) == tuple(frame_indices_cpu.shape))
    ):
        plan_shape = tuple((int(v) for v in frame_indices_cpu.shape))
    elif (
        isinstance(frame_indices_device, torch.Tensor)
        and isinstance(slot_indices_device, torch.Tensor)
        and (frame_indices_device.device == device)
        and (slot_indices_device.device == device)
        and (frame_indices_device.dtype == torch.long)
        and (slot_indices_device.dtype == torch.long)
        and (frame_indices_device.ndim == 2)
        and (tuple(slot_indices_device.shape) == tuple(frame_indices_device.shape))
    ):
        plan_shape = tuple((int(v) for v in frame_indices_device.shape))
    if plan_shape is None:
        return None
    n_selected = int(plan_shape[1])
    token_count = int(n_selected) * int(frame_seqlen)
    if rows != int(B) * int(num_heads) or Lq <= 0 or token_count <= 0:
        return None
    has_live = live_k is not None or live_v is not None or live_start_index is not None
    if has_live:
        if live_k is None or live_v is None or live_start_index is None:
            return None
        if live_k.device != device or live_v.device != device:
            return None
        if live_k.dtype != gpu_k.dtype or live_v.dtype != gpu_k.dtype:
            return None
        if not (live_k.is_contiguous() and live_v.is_contiguous()):
            return None
    try:
        q_contig = q_group.contiguous()
        head_indices_device = _all_head_indices_device(num_heads, device).contiguous()
        if (
            not isinstance(frame_indices_device, torch.Tensor)
            or frame_indices_device.device != device
            or frame_indices_device.dtype != torch.long
            or (tuple(frame_indices_device.shape) != plan_shape)
        ):
            if frame_indices_cpu is None:
                return None
            frame_indices_device = frame_indices_cpu.to(
                device=device, dtype=torch.long, non_blocking=True
            ).contiguous()
        else:
            frame_indices_device = frame_indices_device.contiguous()
        if (
            not isinstance(slot_indices_device, torch.Tensor)
            or slot_indices_device.device != device
            or slot_indices_device.dtype != torch.long
            or (tuple(slot_indices_device.shape) != plan_shape)
        ):
            if slot_plan_cpu is None:
                return None
            slot_indices_device = slot_plan_cpu.to(
                device=device, dtype=torch.long, non_blocking=True
            ).contiguous()
        else:
            slot_indices_device = slot_indices_device.contiguous()
        k_compact = torch.empty((rows, token_count, 1, D), device=device, dtype=gpu_k.dtype)
        v_compact = torch.empty((rows, token_count, 1, D), device=device, dtype=gpu_v.dtype)
        ok = fused_kv_frame_materialize(
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            k_out=k_compact,
            v_out=v_compact,
            head_indices=head_indices_device,
            frame_indices=frame_indices_device,
            slot_indices=slot_indices_device,
            frame_seqlen=frame_seqlen,
            total_tokens=total_tokens,
            num_heads=num_heads,
            gpu_cache_frames=gpu_cache_frames,
            live_k=live_k if has_live else None,
            live_v=live_v if has_live else None,
            live_start_index=int(live_start_index or 0),
        )
        if not ok:
            return None
        k_compact = _apply_fast_lf_rope_to_compact_k(
            k_compact,
            batch_size=B,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            selected_frames=n_selected,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            start_frame=int(lf_k_start_frame),
        )
        if k_compact is None:
            return None
        flush_deferred_clean_kv_offloads(kv_cache)
        block_sparsity = _overlap_effective_block_sparsity(kv_sparse_config, n_selected)
        if block_sparsity > 0.0 and (not bool(kv_cache.get("overlap_blocksparse_disabled", False))):
            try:
                sparse_out = blocksparse_attention_pooled_triton(
                    q=q_contig,
                    k=k_compact,
                    v=v_compact,
                    sparsity=block_sparsity,
                    block_m=_overlap_blocksparse_block_size(kv_sparse_config, "attn_block_q"),
                    block_n=_overlap_blocksparse_block_size(kv_sparse_config, "attn_block_k"),
                    force_sparse=True,
                )
                if isinstance(sparse_out, torch.Tensor):
                    return sparse_out.to(out_dtype or q_group.dtype)
            except Exception as exc:
                raise SparseAttentionError(
                    "materialized sparse attention failed; refusing dense fallback"
                ) from exc
        if block_sparsity > 0.0:
            raise SparseAttentionError(
                f"materialized sparse attention unavailable (sparsity={block_sparsity}, selected_frames={n_selected}, tokens={token_count}); refusing dense fallback"
            )
        if hasattr(flash_attn, "flash_attn_func"):
            try:
                out = flash_attn.flash_attn_func(
                    q_contig,
                    k_compact,
                    v_compact,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False,
                    window_size=(-1, -1),
                    deterministic=False,
                )
                return out.to(out_dtype or q_group.dtype)
            except Exception:
                pass
        return attention(q_contig, k_compact, v_compact).to(out_dtype or q_group.dtype)
    except SparseAttentionError:
        raise
    except Exception as exc:
        if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[horizonkv] materialized overlap flash attention fallback after failure: {exc}")
        return None


def _copy_overlap_control_to_device(
    kv_cache: dict[str, Any], *, workspace_name: str, cpu_tensor: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    cpu_tensor = _pin_cpu_tensor_if_available(cpu_tensor.contiguous())
    cache = kv_cache.setdefault("_kv_sparse_overlap_control_workspaces", {})
    if not isinstance(cache, dict):
        cache = {}
        kv_cache["_kv_sparse_overlap_control_workspaces"] = cache
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = (str(workspace_name), device_index, cpu_tensor.dtype)
    entry = cache.get(key, None)
    current_stream = torch.cuda.current_stream(device)
    if isinstance(entry, dict):
        last_use_event = entry.get("last_use_event", None)
        if last_use_event is not None:
            current_stream.wait_event(last_use_event)
    else:
        entry = {}
    required = int(cpu_tensor.numel())
    buffer = entry.get("buffer", None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.device != device
        or buffer.dtype != cpu_tensor.dtype
        or (int(buffer.numel()) < required)
    ):
        capacity = 1 << max(0, required - 1).bit_length()
        buffer = torch.empty((capacity,), device=device, dtype=cpu_tensor.dtype)
        entry = {"buffer": buffer, "last_use_event": None}
        cache[key] = entry
    view = buffer.narrow(0, 0, required)
    view.copy_(cpu_tensor, non_blocking=True)
    return (view, key)


def _mark_overlap_control_workspaces_used(
    kv_cache: dict[str, Any], controls: dict[str, Any], stream: torch.cuda.Stream
) -> None:
    cache = kv_cache.get("_kv_sparse_overlap_control_workspaces", None)
    keys = controls.get("workspace_keys", ())
    if not isinstance(cache, dict) or not isinstance(keys, tuple) or (not keys):
        return
    for name in ("control_long", "control_int", "frame_positions"):
        tensor = controls.get(name, None)
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)
    event = torch.cuda.Event()
    event.record(stream)
    for key in keys:
        entry = cache.get(key, None)
        if isinstance(entry, dict):
            entry["last_use_event"] = event


def _prepare_compact_slot_plan_controls(
    kv_cache: dict[str, Any],
    *,
    device: torch.device,
    row_plan: _OverlapRowPlan,
    B: int,
    num_heads: int,
    frame_seqlen: int,
    workspace_name: str = "default",
) -> dict[str, Any] | None:
    if row_plan.item_count <= 0:
        return None
    active_count = row_plan.active_rows_cpu.numel()
    max_selected = row_plan.frame_plan_cpu.shape[1]
    plan_items = active_count * max_selected
    control_long, long_key = _copy_overlap_control_to_device(
        kv_cache,
        workspace_name=f"{workspace_name}.long",
        cpu_tensor=row_plan.control_long_cpu,
        device=device,
    )
    control_int, int_key = _copy_overlap_control_to_device(
        kv_cache,
        workspace_name=f"{workspace_name}.int",
        cpu_tensor=row_plan.control_int_cpu,
        device=device,
    )
    active_rows = control_long[:active_count]
    shape = (active_count, max_selected)
    return dict(
        active_count=active_count,
        max_selected=max_selected,
        token_count=max_selected * frame_seqlen,
        head_indices=active_rows,
        frame_indices=control_long.narrow(0, active_count, plan_items).view(shape),
        slot_indices=control_long.narrow(0, active_count + plan_items, plan_items).view(shape),
        frame_positions=control_long.narrow(0, active_count + 2 * plan_items, plan_items).view(
            shape
        ),
        full_row_indices=None if row_plan.covers_all_heads else active_rows,
        full_q_rows=row_plan.covers_all_heads,
        k_lens=control_int[:active_count],
        cu_seqlens_k=control_int[active_count:],
        all_full_length=row_plan.all_full_length,
        control_long=control_long,
        control_int=control_int,
        workspace_keys=(long_key, int_key),
    )


def _materialize_compact_slot_plan(
    kv_cache: dict[str, Any],
    *,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    row_plan: _OverlapRowPlan,
    B: int,
    D: int,
    total_tokens: int,
    num_heads: int,
    gpu_cache_frames: int,
    frame_seqlen: int,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int | None = None,
    controls: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if row_plan.item_count <= 0:
        return None
    device = gpu_k.device
    if controls is None:
        controls = _prepare_compact_slot_plan_controls(
            kv_cache,
            device=device,
            row_plan=row_plan,
            B=B,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
        )
    ready_event = controls.get("ready_event")
    if ready_event is not None:
        torch.cuda.current_stream(device).wait_event(ready_event)
    k_compact = torch.empty(
        (row_plan.item_count * frame_seqlen, 1, D), device=device, dtype=gpu_k.dtype
    )
    v_compact = torch.empty_like(k_compact)
    fused_kv_frame_materialize_varlen(
        gpu_k=gpu_k,
        gpu_v=gpu_v,
        k_out=k_compact,
        v_out=v_compact,
        head_indices=controls["head_indices"],
        frame_indices=controls["frame_indices"],
        slot_indices=controls["slot_indices"],
        cu_seqlens_k=controls["cu_seqlens_k"],
        frame_seqlen=frame_seqlen,
        total_tokens=total_tokens,
        num_heads=num_heads,
        gpu_cache_frames=gpu_cache_frames,
        live_k=live_k,
        live_v=live_v,
        live_start_index=int(live_start_index or 0),
    )
    _mark_overlap_control_workspaces_used(kv_cache, controls, torch.cuda.current_stream(device))
    return dict(
        controls=controls,
        full_row_indices=controls["full_row_indices"],
        full_q_rows=controls["full_q_rows"],
        k_lens=controls["k_lens"],
        cu_seqlens_k=controls["cu_seqlens_k"],
        max_seqlen_k=controls["token_count"],
        all_full_length=True,
        frame_seqlen=frame_seqlen,
        k_compact=k_compact,
        v_compact=v_compact,
    )


class SparseAttentionError(RuntimeError):
    pass


def _run_blocksparse_attention_part_with_lse(
    kv_cache: dict[str, Any],
    *,
    label_prefix: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    uniform_full_length: bool,
    k_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_k: int | None,
    kv_sparse_config: KVSparseConfig | None,
    selected_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    block_sparsity = _overlap_effective_block_sparsity(kv_sparse_config, selected_frames)
    if block_sparsity <= 0.0:
        return None
    failure = f"{label_prefix}: sparse attention failed (sparsity={block_sparsity}, q={tuple(q.shape)}, k={tuple(k.shape)}, max_seqlen_k={max_seqlen_k}); refusing dense fallback"
    if (
        bool(kv_cache.get("overlap_blocksparse_disabled", False))
        or q.ndim != 4
        or int(q.shape[2]) != 1
        or (k.ndim not in (3, 4))
        or (tuple(k.shape) != tuple(v.shape))
    ):
        raise SparseAttentionError(failure)
    rows = int(q.shape[0])
    packed_varlen = False
    if k.ndim == 4:
        if int(k.shape[0]) != rows:
            raise SparseAttentionError(failure)
        k_rows = k
        v_rows = v
    else:
        token_count = int(max_seqlen_k or 0)
        if uniform_full_length and (
            token_count > 0 and int(k.shape[0]) == rows * token_count and (int(k.shape[1]) == 1)
        ):
            k_rows = k.view(rows, token_count, int(k.shape[1]), int(k.shape[2]))
            v_rows = v.view(rows, token_count, int(v.shape[1]), int(v.shape[2]))
        elif (
            token_count <= 0
            or int(k.shape[1]) != 1
            or (not isinstance(k_lens, torch.Tensor))
            or (not isinstance(cu_seqlens_k, torch.Tensor))
            or (k_lens.device != q.device)
            or (cu_seqlens_k.device != q.device)
            or (k_lens.dtype != torch.int32)
            or (cu_seqlens_k.dtype != torch.int32)
            or (int(k_lens.numel()) != rows)
            or (int(cu_seqlens_k.numel()) != rows + 1)
        ):
            raise SparseAttentionError(failure)
        else:
            packed_varlen = True
            k_rows = k
            v_rows = v
    try:
        sparse_part = blocksparse_attention_pooled_triton(
            q=q.contiguous(),
            k=k_rows.contiguous(),
            v=v_rows.contiguous(),
            sparsity=block_sparsity,
            block_m=_overlap_blocksparse_block_size(kv_sparse_config, "attn_block_q"),
            block_n=_overlap_blocksparse_block_size(kv_sparse_config, "attn_block_k"),
            return_lse=True,
            k_lens=k_lens if packed_varlen else None,
            cu_seqlens_k=cu_seqlens_k if packed_varlen else None,
            max_seqlen_k=max_seqlen_k if packed_varlen else None,
            force_sparse=True,
        )
    except Exception as exc:
        raise SparseAttentionError(failure) from exc
    if not (
        isinstance(sparse_part, tuple)
        and len(sparse_part) == 2
        and isinstance(sparse_part[0], torch.Tensor)
        and isinstance(sparse_part[1], torch.Tensor)
        and (tuple(sparse_part[0].shape) == tuple(q.shape))
        and (tuple(sparse_part[1].shape) == (rows, int(q.shape[2]), int(q.shape[1])))
    ):
        raise SparseAttentionError(failure)
    return (sparse_part[0], sparse_part[1])


def _run_compact_slot_plan_flash_attention_part_from_materialized(
    kv_cache: dict[str, Any],
    *,
    label_prefix: str,
    q_group: torch.Tensor,
    row_plan: _OverlapRowPlan,
    materialized_part: dict[str, Any],
    B: int,
    D: int,
    num_heads: int,
    out_dtype: torch.dtype | None = None,
    kv_sparse_config: KVSparseConfig | None = None,
    selected_frames: int | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
    rope_freqs: torch.Tensor | None = None,
    lf_k_start_frame: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    out_dtype = out_dtype or q_group.dtype
    Lq = int(q_group.shape[1])
    q_heads = int(q_group.shape[2])
    full_q_rows = bool(
        materialized_part.get("full_q_rows", False)
    ) and _overlap_row_plan_covers_full_query_rows(
        row_plan, query_rows=int(q_group.shape[0]), B=int(B), num_heads=int(num_heads)
    )
    full_row_indices = materialized_part.get("full_row_indices", None)
    k_compact = materialized_part["k_compact"]
    v_compact = materialized_part["v_compact"]
    k_lens = materialized_part["k_lens"]
    row_count = (
        int(q_group.shape[0])
        if full_q_rows
        else int(full_row_indices.numel()) if isinstance(full_row_indices, torch.Tensor) else 0
    )
    if (
        not isinstance(k_lens, torch.Tensor)
        or not isinstance(k_compact, torch.Tensor)
        or (not isinstance(v_compact, torch.Tensor))
        or (k_lens.device != q_group.device)
        or (k_lens.dtype != torch.int32)
        or (row_count <= 0)
        or (int(k_lens.numel()) != row_count)
        or (k_compact.ndim not in (3, 4))
        or (tuple(k_compact.shape) != tuple(v_compact.shape))
        or (k_compact.ndim == 4 and int(k_compact.shape[0]) != row_count)
    ):
        return None
    if not full_q_rows and (
        not isinstance(full_row_indices, torch.Tensor)
        or full_row_indices.device != q_group.device
        or full_row_indices.dtype != torch.long
    ):
        return None
    k_rope = _apply_fast_lf_rope_to_row_materialized_k(
        k_compact,
        row_plan=row_plan,
        batch_size=B,
        frame_seqlen=int(materialized_part.get("frame_seqlen", 0) or 0),
        k_lens=k_lens,
        cu_seqlens=materialized_part.get("cu_seqlens_k", None),
        rope_grid_sizes=rope_grid_sizes,
        rope_freqs=rope_freqs,
        inplace=True,
        frame_positions_device=materialized_part.get("controls", {}).get("frame_positions"),
        start_frame=int(lf_k_start_frame),
    )
    if k_rope is None:
        return None
    k_compact = k_rope
    if full_q_rows:
        q_part = q_group
    else:
        q_part = q_group.index_select(0, full_row_indices)
    flash_part = _run_blocksparse_attention_part_with_lse(
        kv_cache,
        label_prefix=label_prefix,
        q=q_part,
        k=k_compact,
        v=v_compact,
        uniform_full_length=bool(row_plan.all_full_length),
        k_lens=k_lens,
        cu_seqlens_k=materialized_part.get("cu_seqlens_k", None),
        max_seqlen_k=materialized_part.get("max_seqlen_k", None),
        kv_sparse_config=kv_sparse_config,
        selected_frames=selected_frames,
    )
    if flash_part is None:
        flash_part = _flash_attention_varlen_with_lse_padded(
            kv_cache=kv_cache,
            q=q_part,
            k=k_compact,
            v=v_compact,
            k_lens=k_lens,
            cu_seqlens_k=materialized_part.get("cu_seqlens_k", None),
            max_seqlen_k=materialized_part.get("max_seqlen_k", None),
            all_full_length=bool(materialized_part.get("all_full_length", False)),
        )
    if flash_part is None:
        kv_cache["overlap_flash_lse_disabled"] = True
        return None
    out_part, lse_part = flash_part
    controls = materialized_part.get("controls")
    if isinstance(controls, dict):
        _mark_overlap_control_workspaces_used(
            kv_cache, controls, torch.cuda.current_stream(q_group.device)
        )
    if tuple(out_part.shape) != tuple(q_part.shape):
        kv_cache["overlap_flash_lse_disabled"] = True
        return None
    if tuple(lse_part.shape) != (int(q_part.shape[0]), q_heads, Lq):
        kv_cache["overlap_flash_lse_disabled"] = True
        return None
    if full_q_rows:
        return (out_part.to(dtype=out_dtype), lse_part.to(dtype=torch.float32))
    full_out = torch.zeros(
        (int(q_group.shape[0]), Lq, q_heads, int(D)), device=q_group.device, dtype=out_dtype
    )
    full_lse = torch.full(
        (int(q_group.shape[0]), q_heads, Lq), -1e309, device=q_group.device, dtype=torch.float32
    )
    full_out.index_copy_(0, full_row_indices, out_part.to(dtype=out_dtype))
    full_lse.index_copy_(0, full_row_indices, lse_part.to(dtype=torch.float32))
    return (full_out, full_lse)


def _cached_flash_varlen_q_controls(
    kv_cache: dict[str, Any] | None, device: torch.device, rows: int, Lq: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(rows)
    Lq = int(Lq)
    if kv_cache is not None:
        device_index = int(device.index) if device.index is not None else -1
        key = (str(device.type), device_index, rows, Lq)
        cache = kv_cache.setdefault("_kv_sparse_flash_varlen_q_controls", {})
        cached = cache.get(key, None) if isinstance(cache, dict) else None
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and isinstance(cached[0], torch.Tensor)
            and isinstance(cached[1], torch.Tensor)
            and (cached[0].device == device)
            and (cached[1].device == device)
            and (cached[0].dtype == torch.int32)
            and (cached[1].dtype == torch.int32)
            and (int(cached[0].numel()) == rows)
            and (int(cached[1].numel()) == rows + 1)
        ):
            return (cached[0], cached[1])
    q_lens = torch.full((rows,), Lq, dtype=torch.int32, device=device)
    cu_seqlens_q = (torch.arange(rows + 1, dtype=torch.int32, device=device) * Lq).contiguous()
    if kv_cache is not None and isinstance(cache, dict):
        cache[key] = (q_lens, cu_seqlens_q)
    return (q_lens, cu_seqlens_q)


def _flash_attention_varlen_with_lse_padded(
    *,
    kv_cache: dict[str, Any] | None = None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    all_full_length: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        return None
    if q.ndim != 4 or k.ndim not in (3, 4) or v.ndim != k.ndim:
        return None
    if tuple(k.shape) != tuple(v.shape):
        return None
    rows = int(q.shape[0])
    Lq = int(q.shape[1])
    q_heads = int(q.shape[2])
    if k.ndim == 4:
        token_count = int(k.shape[1])
        kv_heads = int(k.shape[2])
    else:
        if max_seqlen_k is None:
            return None
        token_count = int(max_seqlen_k)
        kv_heads = int(k.shape[1])
    if rows <= 0 or Lq <= 0 or token_count <= 0 or (q_heads <= 0) or (kv_heads <= 0):
        return None
    if int(k.shape[-1]) != int(q.shape[-1]) or q_heads % kv_heads != 0:
        return None
    if k.ndim == 4 and (int(k.shape[0]) != rows or int(v.shape[0]) != rows):
        return None
    if int(k_lens.numel()) != rows:
        return None
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    dtype = q.dtype if q.dtype in half_dtypes else torch.bfloat16
    try:
        _, cu_seqlens_q = _cached_flash_varlen_q_controls(kv_cache, q.device, rows, Lq)
        k_lens = k_lens.to(device=k.device, dtype=torch.int32, non_blocking=True).contiguous()
        q_flat = q.flatten(0, 1)
        if q_flat.dtype not in half_dtypes:
            q_flat = q_flat.to(dtype)
        if k.ndim == 3:
            k_flat = k
            v_flat = v
        elif bool(all_full_length):
            k_flat = k.flatten(0, 1)
            v_flat = v.flatten(0, 1)
        else:
            token_offsets = torch.arange(token_count, device=k.device).view(1, token_count)
            valid_mask = token_offsets < k_lens.view(rows, 1)
            k_flat = k[valid_mask]
            v_flat = v[valid_mask]
        if k_flat.dtype not in half_dtypes:
            k_flat = k_flat.to(dtype)
        if v_flat.dtype not in half_dtypes:
            v_flat = v_flat.to(dtype)
        q_flat = q_flat.to(v_flat.dtype)
        k_flat = k_flat.to(v_flat.dtype)
        if (
            not isinstance(cu_seqlens_k, torch.Tensor)
            or cu_seqlens_k.device != k.device
            or cu_seqlens_k.dtype != torch.int32
            or (int(cu_seqlens_k.numel()) != rows + 1)
        ):
            cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        else:
            cu_seqlens_k = cu_seqlens_k.contiguous()
        out, softmax_lse, _ = flash_attn.flash_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=Lq,
            max_seqlen_k=token_count,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=True,
        )
        out = out.unflatten(0, (rows, Lq)).type(out_dtype)
        softmax_lse = (
            softmax_lse.transpose(0, 1).contiguous().view(rows, Lq, q_heads).permute(0, 2, 1)
        )
        return (out, softmax_lse)
    except Exception:
        return None


def _merge_attention_partials(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    try:
        merged = merge_attention_lse_parts_triton(
            out_a.contiguous(), lse_a.contiguous(), out_b.contiguous(), lse_b.contiguous()
        )
    except Exception:
        merged = None
    if merged is not None:
        return merged.to(dtype=out_dtype)
    merge_max = torch.maximum(lse_a, lse_b)
    both_invalid = torch.isneginf(merge_max)
    safe_merge_max = torch.where(both_invalid, torch.zeros_like(merge_max), merge_max)
    weight_a = torch.exp(lse_a - safe_merge_max).transpose(1, 2).unsqueeze(-1)
    weight_b = torch.exp(lse_b - safe_merge_max).transpose(1, 2).unsqueeze(-1)
    invalid_mask = both_invalid.transpose(1, 2).unsqueeze(-1)
    weight_a = torch.where(invalid_mask, torch.zeros_like(weight_a), weight_a)
    weight_b = torch.where(invalid_mask, torch.zeros_like(weight_b), weight_b)
    denom = (weight_a + weight_b).clamp_min_(1e-20)
    merged = (
        out_a.to(dtype=torch.float32) * weight_a + out_b.to(dtype=torch.float32) * weight_b
    ) / denom
    return merged.to(dtype=out_dtype)


def _kv_cache_wait_pending_cpu_copy_for_frames_on_stream(
    kv_cache: dict[str, Any],
    frame_ids: list[int] | torch.Tensor,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> None:
    records = _kv_cache_pending_cpu_copy_records(kv_cache)
    if not records:
        return
    targets = [
        (int(frame_id) * int(frame_seqlen), (int(frame_id) + 1) * int(frame_seqlen))
        for frame_id in frame_ids
        if int(frame_id) >= 0
    ]
    if not targets:
        return
    for event, span in records:
        p0, p1 = (int(span[0]), int(span[1]))
        if any((start < p1 and p0 < end for start, end in targets)):
            stream.wait_event(event)


def _build_compact_miss_stage(
    *,
    miss_row_plan: _OverlapRowPlan,
    compact_scatter_plan: dict[str, Any] | None,
    fill_count: int,
    total_tokens: int,
    B: int,
    D: int,
    frame_seqlen: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    active_rows_cpu = miss_row_plan.active_rows_cpu
    frame_plan_cpu = miss_row_plan.frame_plan_cpu
    active_count = int(active_rows_cpu.numel())
    if active_count <= 0 or miss_row_plan.item_count <= 0 or frame_plan_cpu.ndim != 2:
        return None
    if not isinstance(compact_scatter_plan, dict):
        return None
    valid_frames: torch.Tensor | None = None
    all_frames_full = compact_scatter_plan.get("all_frames_full", None)
    if not isinstance(all_frames_full, bool):
        valid_frames = frame_plan_cpu[frame_plan_cpu.ge(0)]
        if int(valid_frames.numel()) != int(miss_row_plan.item_count):
            return None
        all_frames_full = not bool(
            ((valid_frames + 1) * int(frame_seqlen) > int(total_tokens)).any().item()
        )
    if not all_frames_full:
        return None
    B = int(B)
    base_cu_seqlens_cpu = compact_scatter_plan.get("cu_seqlens_long", None)
    if (
        B == 1
        and isinstance(base_cu_seqlens_cpu, torch.Tensor)
        and (base_cu_seqlens_cpu.device.type == "cpu")
        and (base_cu_seqlens_cpu.dtype == torch.long)
        and (base_cu_seqlens_cpu.ndim == 1)
        and (int(base_cu_seqlens_cpu.numel()) == active_count + 1)
    ):
        cu_seqlens_cpu = base_cu_seqlens_cpu.contiguous()
    else:
        k_lens_cpu = miss_row_plan.k_lens_cpu.to(dtype=torch.long)
        if B > 1:
            k_lens_cpu = k_lens_cpu.repeat(B)
        k_lens_cpu = k_lens_cpu.contiguous()
        cu_seqlens_cpu = torch.empty((int(k_lens_cpu.numel()) + 1,), dtype=torch.long)
        cu_seqlens_cpu[0] = 0
        torch.cumsum(k_lens_cpu, dim=0, out=cu_seqlens_cpu[1:])
    cu_seqlens_cpu = _pin_cpu_tensor_if_available(cu_seqlens_cpu)
    packed_tokens = int(cu_seqlens_cpu[-1].item())
    expected_tokens = B * int(miss_row_plan.item_count) * int(frame_seqlen)
    if packed_tokens != expected_tokens or packed_tokens <= 0:
        return None
    scatter_head_ids_cpu = compact_scatter_plan.get("scatter_head_ids", None)
    scatter_slot_ids_cpu = compact_scatter_plan.get("scatter_slot_ids", None)
    scatter_src_rows_cpu = compact_scatter_plan.get("scatter_src_rows", None)
    scatter_src_cols_cpu = compact_scatter_plan.get("scatter_src_cols", None)
    scatter_frame_ids_cpu = compact_scatter_plan.get("scatter_frame_ids", None)
    scatter_tensors = (
        scatter_head_ids_cpu,
        scatter_slot_ids_cpu,
        scatter_src_rows_cpu,
        scatter_src_cols_cpu,
    )
    expected = int(fill_count)
    if not all(
        (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and (tensor.dtype == torch.long)
            and (tensor.ndim == 1)
            and (int(tensor.numel()) == expected)
            for tensor in scatter_tensors
        )
    ):
        return None
    scatter_head_ids_cpu = _pin_cpu_tensor_if_available(scatter_head_ids_cpu.contiguous())
    scatter_slot_ids_cpu = _pin_cpu_tensor_if_available(scatter_slot_ids_cpu.contiguous())
    scatter_src_rows_cpu = _pin_cpu_tensor_if_available(scatter_src_rows_cpu.contiguous())
    scatter_src_cols_cpu = _pin_cpu_tensor_if_available(scatter_src_cols_cpu.contiguous())
    scatter_frames_from_planner = (
        isinstance(scatter_frame_ids_cpu, torch.Tensor)
        and scatter_frame_ids_cpu.device.type == "cpu"
        and (scatter_frame_ids_cpu.dtype == torch.long)
        and (scatter_frame_ids_cpu.ndim == 1)
        and (int(scatter_frame_ids_cpu.numel()) == expected)
    )
    if not scatter_frames_from_planner:
        scatter_frame_ids_cpu = frame_plan_cpu[scatter_src_rows_cpu, scatter_src_cols_cpu]
    scatter_frame_ids_cpu = _pin_cpu_tensor_if_available(scatter_frame_ids_cpu.contiguous())
    if not scatter_frames_from_planner and bool(scatter_frame_ids_cpu.lt(0).any().item()):
        return None
    frame_ids_for_wait = compact_scatter_plan.get("frame_ids_for_wait", None)
    if not (
        isinstance(frame_ids_for_wait, torch.Tensor)
        and frame_ids_for_wait.device.type == "cpu"
        and (frame_ids_for_wait.dtype == torch.long)
        and (frame_ids_for_wait.ndim == 1)
    ):
        if valid_frames is None:
            valid_frames = frame_plan_cpu[frame_plan_cpu.ge(0)]
        frame_ids_for_wait = _pin_cpu_tensor_if_available(torch.unique(valid_frames))
    else:
        frame_ids_for_wait = _pin_cpu_tensor_if_available(frame_ids_for_wait.contiguous())
    compact_k = torch.empty((packed_tokens, 1, int(D)), device=device, dtype=dtype)
    compact_v = torch.empty_like(compact_k)
    return {
        "compact_k": compact_k,
        "compact_v": compact_v,
        "row_plan": miss_row_plan,
        "frame_seqlen": int(frame_seqlen),
        "active_head_ids_cpu": active_rows_cpu.contiguous(),
        "frame_plan_cpu": frame_plan_cpu.contiguous(),
        "cu_seqlens_cpu": cu_seqlens_cpu,
        "max_seqlen_k": int(frame_plan_cpu.shape[1]) * int(frame_seqlen),
        "item_count": int(miss_row_plan.item_count),
        "active_count": active_count,
        "batch_size": int(B),
        "scatter_head_ids_cpu": scatter_head_ids_cpu,
        "scatter_slot_ids_cpu": scatter_slot_ids_cpu,
        "scatter_src_rows_cpu": scatter_src_rows_cpu,
        "scatter_src_cols_cpu": scatter_src_cols_cpu,
        "scatter_frame_ids_cpu": scatter_frame_ids_cpu,
        "frame_ids_for_wait": frame_ids_for_wait,
        "copy_done": None,
    }


def _submit_compact_miss_h2d(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    cpu_k: torch.Tensor,
    cpu_v: torch.Tensor,
    frame_seqlen: int,
    stream: torch.cuda.Stream,
) -> bool:
    copy_done: torch.cuda.Event | None = None
    try:
        with torch.cuda.stream(stream):
            _kv_cache_wait_pending_cpu_copy_for_frames_on_stream(
                kv_cache, stage["frame_ids_for_wait"], frame_seqlen, stream
            )
            copied = copy_kv_frames_to_gpu_compact(
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                compact_k=stage["compact_k"],
                compact_v=stage["compact_v"],
                head_ids=stage["active_head_ids_cpu"],
                frame_ids=stage["frame_plan_cpu"],
                cu_seqlens=stage["cu_seqlens_cpu"],
                frame_seqlen=frame_seqlen,
                cpu_head_major=True,
            )
            copy_done = torch.cuda.Event()
            copy_done.record(stream)
        stage["copy_done"] = copy_done
        stage["compact_k"].record_stream(stream)
        stage["compact_v"].record_stream(stream)
        expected = int(cpu_k.shape[0]) * int(stage["item_count"])
        return int(copied) == expected
    except Exception as exc:
        if copy_done is None:
            try:
                copy_done = torch.cuda.Event()
                copy_done.record(stream)
                stage["copy_done"] = copy_done
                stage["compact_k"].record_stream(stream)
                stage["compact_v"].record_stream(stream)
            except Exception:
                pass
        if str(os.getenv("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[horizonkv] compact miss H2D fallback after failure: {exc}")
        return False


def _submit_compact_miss_lf_rope(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    controls: dict[str, Any],
    rope_grid_sizes: torch.Tensor | None,
    rope_freqs: torch.Tensor | None,
    stream: torch.cuda.Stream,
    producer_event: torch.cuda.Event | None = None,
    lf_k_start_frame: int = 0,
) -> bool:
    copy_done = stage.get("copy_done", None)
    if copy_done is None or not isinstance(controls, dict):
        return False
    row_plan = stage.get("row_plan", None)
    if not isinstance(row_plan, _OverlapRowPlan):
        return False
    k_lens = controls.get("k_lens", None)
    cu_seqlens = controls.get("cu_seqlens_k", None)
    if not isinstance(k_lens, torch.Tensor) or not isinstance(cu_seqlens, torch.Tensor):
        return False
    if not isinstance(rope_grid_sizes, torch.Tensor) or not isinstance(rope_freqs, torch.Tensor):
        return False
    try:
        with torch.cuda.stream(stream):
            if producer_event is not None:
                stream.wait_event(producer_event)
            stream.wait_event(copy_done)
            controls_ready_event = controls.get("ready_event", None)
            if controls_ready_event is not None:
                stream.wait_event(controls_ready_event)
            attention_k = _apply_fast_lf_rope_to_row_materialized_k(
                stage["compact_k"],
                row_plan=row_plan,
                batch_size=int(stage.get("batch_size", 0) or 0),
                frame_seqlen=int(stage.get("frame_seqlen", 0) or 0),
                k_lens=k_lens,
                cu_seqlens=cu_seqlens,
                rope_grid_sizes=rope_grid_sizes,
                rope_freqs=rope_freqs,
                inplace=False,
                frame_positions_device=controls.get("frame_positions"),
                start_frame=int(lf_k_start_frame),
            )
            if not isinstance(attention_k, torch.Tensor):
                return False
            _mark_overlap_control_workspaces_used(kv_cache, controls, stream)
            attention_k.record_stream(stream)
            stage["compact_k"].record_stream(stream)
            rope_done = torch.cuda.Event()
            rope_done.record(stream)
        stage["attention_k"] = attention_k
        stage["attention_k_ready"] = rope_done
        return True
    except Exception as exc:
        if _truthy_env("HORIZONKV_FUSED_ONLOAD_VERBOSE", "0"):
            print(f"[horizonkv] async compact miss LF-RoPE skipped after failure: {exc}")
        return False


def _run_compact_miss_attention_part(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    controls: dict[str, Any],
    q_group: torch.Tensor,
    D: int,
    out_dtype: torch.dtype,
    kv_sparse_config: KVSparseConfig | None = None,
    selected_frames: int | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
    rope_freqs: torch.Tensor | None = None,
    lf_k_start_frame: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    copy_done = stage.get("copy_done", None)
    if copy_done is None:
        return None
    active_count = int(stage.get("active_count", 0) or 0)
    B = int(stage.get("batch_size", 0) or 0)
    full_row_indices = controls.get("full_row_indices", None)
    full_q_rows = bool(controls.get("full_q_rows", False))
    k_lens_device = controls.get("k_lens", None)
    cu_seqlens_device = controls.get("cu_seqlens_k", None)
    if not (
        active_count > 0
        and B > 0
        and isinstance(k_lens_device, torch.Tensor)
        and isinstance(cu_seqlens_device, torch.Tensor)
        and (k_lens_device.device == q_group.device)
        and (cu_seqlens_device.device == q_group.device)
        and (k_lens_device.dtype == torch.int32)
        and (cu_seqlens_device.dtype == torch.int32)
        and (int(k_lens_device.numel()) == B * active_count)
        and (int(cu_seqlens_device.numel()) == B * active_count + 1)
    ):
        return None
    if not full_q_rows and (
        not (
            isinstance(full_row_indices, torch.Tensor)
            and full_row_indices.device == q_group.device
            and (full_row_indices.dtype == torch.long)
            and (int(full_row_indices.numel()) == B * active_count)
        )
    ):
        return None
    current_stream = torch.cuda.current_stream(q_group.device)
    controls_ready_event = controls.get("ready_event", None)
    if controls_ready_event is not None:
        current_stream.wait_event(controls_ready_event)
    current_stream.wait_event(copy_done)
    attention_k_ready_event = stage.get("attention_k_ready", None)
    if attention_k_ready_event is not None:
        current_stream.wait_event(attention_k_ready_event)
    Lq = int(q_group.shape[1])
    q_heads = int(q_group.shape[2])
    full_q_rows = full_q_rows and B * active_count == int(q_group.shape[0])
    if full_q_rows:
        full_row_indices = None
        q_part = q_group
    else:
        q_part = q_group.index_select(0, full_row_indices)
    attention_k = stage.get("attention_k", None)
    if not isinstance(attention_k, torch.Tensor):
        row_plan = stage.get("row_plan", None)
        if not isinstance(row_plan, _OverlapRowPlan):
            return None
        attention_k = _apply_fast_lf_rope_to_row_materialized_k(
            stage["compact_k"],
            row_plan=row_plan,
            batch_size=B,
            frame_seqlen=int(stage.get("frame_seqlen", 0) or 0),
            k_lens=k_lens_device,
            cu_seqlens=cu_seqlens_device,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            frame_positions_device=controls.get("frame_positions"),
            start_frame=int(lf_k_start_frame),
        )
        if attention_k is None:
            return None
        stage["attention_k"] = attention_k
    attention_k.record_stream(current_stream)
    flash_part = _run_blocksparse_attention_part_with_lse(
        kv_cache,
        label_prefix="overlap_compact_miss",
        q=q_part,
        k=attention_k,
        v=stage["compact_v"],
        uniform_full_length=bool(controls.get("all_full_length", False)),
        k_lens=k_lens_device,
        cu_seqlens_k=cu_seqlens_device,
        max_seqlen_k=stage.get("max_seqlen_k", None),
        kv_sparse_config=kv_sparse_config,
        selected_frames=selected_frames,
    )
    if flash_part is None:
        flash_part = _flash_attention_varlen_with_lse_padded(
            kv_cache=kv_cache,
            q=q_part,
            k=attention_k,
            v=stage["compact_v"],
            k_lens=k_lens_device,
            cu_seqlens_k=cu_seqlens_device,
            max_seqlen_k=stage["max_seqlen_k"],
            all_full_length=True,
        )
    if flash_part is None:
        return None
    out_part, lse_part = flash_part
    if tuple(out_part.shape) != tuple(q_part.shape) or tuple(lse_part.shape) != (
        int(q_part.shape[0]),
        q_heads,
        Lq,
    ):
        return None
    if full_q_rows:
        _mark_overlap_control_workspaces_used(kv_cache, controls, current_stream)
        stage["compact_k"].record_stream(current_stream)
        if isinstance(attention_k, torch.Tensor) and attention_k is not stage["compact_k"]:
            attention_k.record_stream(current_stream)
        stage["compact_v"].record_stream(current_stream)
        return (out_part.to(dtype=out_dtype), lse_part.to(dtype=torch.float32))
    full_out = torch.zeros(
        (int(q_group.shape[0]), Lq, q_heads, int(D)), device=q_group.device, dtype=out_dtype
    )
    full_lse = torch.full(
        (int(q_group.shape[0]), q_heads, Lq), -1e309, device=q_group.device, dtype=torch.float32
    )
    full_out.index_copy_(0, full_row_indices, out_part.to(dtype=out_dtype))
    full_lse.index_copy_(0, full_row_indices, lse_part.to(dtype=torch.float32))
    stage["compact_k"].record_stream(current_stream)
    _mark_overlap_control_workspaces_used(kv_cache, controls, current_stream)
    if isinstance(attention_k, torch.Tensor) and attention_k is not stage["compact_k"]:
        attention_k.record_stream(current_stream)
    stage["compact_v"].record_stream(current_stream)
    return (full_out, full_lse)


def _schedule_compact_miss_slot_fill(
    kv_cache: dict[str, Any],
    *,
    stage: dict[str, Any],
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
    source_ready_event: torch.cuda.Event | None = None,
) -> bool:
    fill_stream = _shared_cuda_stream(device=gpu_k.device, name="overlap_compact_slot_fill")
    copy_done = stage.get("copy_done", None)
    if fill_stream is None or copy_done is None:
        return False
    expected = int(stage["scatter_head_ids_cpu"].numel())
    if expected <= 0:
        return False
    with torch.cuda.stream(fill_stream):
        ready_event = source_ready_event if source_ready_event is not None else copy_done
        fill_stream.wait_event(ready_event)
        copied = copy_compact_to_gpu_slots_d2d(
            compact_k=stage["compact_k"],
            compact_v=stage["compact_v"],
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            head_ids=stage["scatter_head_ids_cpu"],
            slot_ids=stage["scatter_slot_ids_cpu"],
            src_rows=stage["scatter_src_rows_cpu"],
            src_cols=stage["scatter_src_cols_cpu"],
            cu_seqlens=stage["cu_seqlens_cpu"],
            frame_seqlen=frame_seqlen,
            gpu_cache_frames=gpu_cache_frames,
        )
        slot_fill_done = torch.cuda.Event()
        slot_fill_done.record(fill_stream)
    if int(copied) != expected:
        raise RuntimeError(f"compact miss D2D slot fill copied {int(copied)} of {expected} items")
    stage["compact_k"].record_stream(fill_stream)
    stage["compact_v"].record_stream(fill_stream)
    stage["slot_fill_done"] = slot_fill_done
    stage["slot_fill_scheduled"] = True
    return True


def _identity_slot_indices_if_capacity_resident(
    kv_cache: dict[str, Any],
    *,
    frame_indices: torch.Tensor,
    frame_seqlen: int,
    gpu_cache_frames: int,
    live_start_index: int | None,
    live_len: int,
) -> torch.Tensor | None:
    if (
        not isinstance(frame_indices, torch.Tensor)
        or frame_indices.device.type != "cuda"
        or frame_indices.dtype != torch.long
        or (frame_indices.ndim != 2)
        or (int(frame_indices.shape[0]) <= 0)
        or (int(frame_indices.shape[1]) <= 0)
        or (int(frame_seqlen) <= 0)
        or (int(gpu_cache_frames) <= 0)
        or (live_start_index is None)
        or (int(live_len) <= 0)
    ):
        return None
    live_start = int(live_start_index)
    live_len = int(live_len)
    frame_seqlen = int(frame_seqlen)
    if live_start < 0 or live_start % frame_seqlen != 0 or live_len % frame_seqlen != 0:
        return None
    history_frames = live_start // frame_seqlen
    if history_frames > int(gpu_cache_frames):
        return None
    identity_filled = kv_cache.get("gpu_identity_filled_frames", None)
    if not isinstance(identity_filled, int) or int(identity_filled) < history_frames:
        return None
    if _truthy_env("HORIZONKV_COMMIT_HIT_LAST_USED", "0"):
        return None
    frame_indices = frame_indices.contiguous()
    live_first = history_frames
    live_last = live_first + live_len // frame_seqlen
    live_mask = frame_indices.ge(0) & frame_indices.ge(live_first) & frame_indices.lt(live_last)
    return torch.where(
        live_mask, torch.full((), -2, dtype=torch.long, device=frame_indices.device), frame_indices
    ).contiguous()


def _overlapped_attention_from_gpu_slots_if_possible_impl(
    kv_cache: dict[str, Any],
    q_group: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_seqlen: int,
    *,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int | None = None,
    out_dtype: torch.dtype | None = None,
    slot_metadata_ready: bool = False,
    kv_sparse_config: KVSparseConfig | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
    rope_freqs: torch.Tensor | None = None,
    layer_idx: int | None = None,
    num_frame_per_block: int = 1,
    sink_tokens: int = 0,
    staged_frame_indices: dict[str, Any] | None = None,
    lf_k_start_frame: int = 0,
) -> torch.Tensor | None:
    if not slot_metadata_ready:
        _kv_cache_wait_pending_slot_metadata_commit(kv_cache)
    gpu_cache_frames = int(kv_cache.get("gpu_cache_frames", 0) or 0)
    gpu_k = kv_cache.get("gpu_k", None)
    gpu_v = kv_cache.get("gpu_v", None)
    cpu_k = kv_cache["cpu_k"]
    cpu_v = kv_cache["cpu_v"]
    total_tokens = int(cpu_k.shape[2])
    B = int(cpu_k.shape[0])
    num_heads = int(cpu_k.shape[1])
    D = int(cpu_k.shape[-1])
    frame_seqlen = int(frame_seqlen)
    has_live = live_k is not None and live_v is not None and (live_start_index is not None)
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)
    if frame_indices.shape[0] == 1 and num_heads > 1:
        frame_indices = frame_indices.expand(num_heads, -1)
    if (
        isinstance(gpu_k, torch.Tensor)
        and isinstance(gpu_v, torch.Tensor)
        and (gpu_k.device.type == "cuda")
    ):
        identity_slots = _identity_slot_indices_if_capacity_resident(
            kv_cache,
            frame_indices=frame_indices,
            frame_seqlen=frame_seqlen,
            gpu_cache_frames=gpu_cache_frames,
            live_start_index=live_start_index if has_live else None,
            live_len=int(live_k.shape[1]) if live_k is not None else 0,
        )
        if identity_slots is not None:
            if not q_group.is_contiguous():
                q_group = q_group.contiguous()
            identity_out = _run_materialized_slot_plan_flash_attention(
                kv_cache,
                q_group=q_group,
                gpu_k=gpu_k,
                gpu_v=gpu_v,
                frame_indices_cpu=None,
                slot_plan_cpu=None,
                B=B,
                D=D,
                total_tokens=total_tokens,
                num_heads=num_heads,
                gpu_cache_frames=gpu_cache_frames,
                frame_seqlen=frame_seqlen,
                live_k=live_k,
                live_v=live_v,
                live_start_index=live_start_index,
                frame_indices_device=frame_indices,
                slot_indices_device=identity_slots,
                out_dtype=out_dtype,
                kv_sparse_config=kv_sparse_config,
                fast_lf_rope=True,
                rope_grid_sizes=rope_grid_sizes,
                rope_freqs=rope_freqs,
                layer_idx=layer_idx,
                num_frame_per_block=num_frame_per_block,
                sink_tokens=sink_tokens,
                lf_k_start_frame=int(lf_k_start_frame),
            )
            if identity_out is not None:
                return identity_out
    if staged_frame_indices is None:
        staged_frame_indices = _stage_overlap_frame_indices_to_cpu_async(kv_cache, frame_indices)
    if not q_group.is_contiguous():
        q_group = q_group.contiguous()
    frame_indices_cpu = _resolve_staged_overlap_frame_indices_cpu(staged_frame_indices)
    if frame_indices_cpu is None:
        try:
            frame_indices_cpu = torch.empty(
                tuple(frame_indices.shape), dtype=torch.long, device="cpu", pin_memory=True
            )
        except RuntimeError:
            frame_indices_cpu = torch.empty(
                tuple(frame_indices.shape), dtype=torch.long, device="cpu"
            )
        frame_indices_cpu.copy_(frame_indices.detach(), non_blocking=False)
    live_start = int(live_start_index) if live_start_index is not None else None
    slot_state_tensors = _kv_cache_slot_state_tensors(kv_cache)
    if slot_state_tensors is None:
        return None
    frame_to_slot_tensor, slot_frame_ids_tensor, slot_frame_lens_tensor, slot_last_used_tensor = (
        slot_state_tensors
    )
    plan_result = plan_kv_frame_slots_tensor(
        frame_indices=frame_indices_cpu,
        frame_to_slot=frame_to_slot_tensor,
        slot_frame_ids=slot_frame_ids_tensor,
        slot_frame_lens=slot_frame_lens_tensor,
        slot_last_used=slot_last_used_tensor,
        total_tokens=total_tokens,
        frame_seqlen=frame_seqlen,
        num_heads=num_heads,
        gpu_cache_frames=gpu_cache_frames,
        live_start_index=int(live_start or 0),
        live_len=int(live_k.shape[1]) if live_k is not None else 0,
        has_live=bool(has_live),
        gpu_clock=int(kv_cache.get("gpu_clock", 0) or 0),
        build_compact_plans=_overlap_flash_split_attention_enabled_by_env()
        and (not bool(kv_cache.get("fused_onload_disabled", False)))
        and bool(has_live)
        and (0 < D <= 256),
        split_min_miss_items=_overlap_split_min_miss_items_by_env(),
        split_min_miss_ratio=0.0,
    )
    overlap_plan = (
        _build_overlap_slot_plan(plan_result=plan_result, frame_indices_cpu=frame_indices_cpu)
        if plan_result is not None
        else None
    )
    if plan_result is None:
        return None
    if overlap_plan is None:
        return None
    hit_item_count = int(overlap_plan.hit_item_count)
    miss_item_count = int(overlap_plan.miss_item_count)
    slot_plan_cpu = overlap_plan.slot_plan_cpu
    use_materialized_flash = not bool(kv_cache.get("fused_onload_disabled", False))
    can_flash_split_attention = (
        use_materialized_flash
        and _overlap_flash_split_attention_enabled_by_env()
        and bool(has_live)
        and (live_k is not None)
        and (live_v is not None)
        and (live_start_index is not None)
        and (int(D) > 0)
        and (int(D) <= 256)
        and (hit_item_count > 0)
        and (miss_item_count > 0)
    )
    split_miss_worthwhile = _overlap_split_worthwhile_for_miss_counts(
        hit_item_count=hit_item_count,
        miss_item_count=miss_item_count,
        miss_copy_count=int(overlap_plan.fill_count),
    )
    if not split_miss_worthwhile:
        can_flash_split_attention = False
    can_compact_onload = can_flash_split_attention and int(overlap_plan.fill_count) > 0
    hit_row_plan: _OverlapRowPlan | None = overlap_plan.hit_row_plan
    miss_row_plan: _OverlapRowPlan | None = overlap_plan.miss_row_plan
    flash_hit_controls: dict[str, Any] | None = None
    flash_miss_controls: dict[str, Any] | None = None
    flash_control_tensors_ready_event: torch.cuda.Event | None = None
    compact_stage: dict[str, Any] | None = None
    compact_h2d_done_event: torch.cuda.Event | None = None
    compact_slot_fill_scheduled = False
    next_clock = int(plan_result.get("gpu_clock", int(kv_cache.get("gpu_clock", 0) or 0)))

    def _commit_planned_slot_state() -> bool:
        if _kv_cache_commit_incremental_tensor_slot_plan_state(
            kv_cache, plan_result, next_clock, frame_seqlen=frame_seqlen, total_tokens=total_tokens
        ):
            return True
        return _kv_cache_commit_tensor_slot_plan_state(kv_cache, plan_result, next_clock)

    def _ensure_flash_miss_controls() -> bool:
        nonlocal flash_miss_controls
        if flash_miss_controls is None and miss_row_plan is not None:
            miss_control_stream = _shared_cuda_stream(
                device=gpu_k.device, name="overlap_flash_miss_control_h2d"
            )
            if miss_control_stream is not None:
                with torch.cuda.stream(miss_control_stream):
                    flash_miss_controls = _prepare_compact_slot_plan_controls(
                        kv_cache,
                        device=gpu_k.device,
                        row_plan=miss_row_plan,
                        B=B,
                        num_heads=num_heads,
                        frame_seqlen=frame_seqlen,
                        workspace_name="miss",
                    )
                if isinstance(flash_miss_controls, dict):
                    miss_ready_event = torch.cuda.Event()
                    miss_ready_event.record(miss_control_stream)
                    flash_miss_controls["ready_event"] = miss_ready_event
            else:
                flash_miss_controls = _prepare_compact_slot_plan_controls(
                    kv_cache,
                    device=gpu_k.device,
                    row_plan=miss_row_plan,
                    B=B,
                    num_heads=num_heads,
                    frame_seqlen=frame_seqlen,
                    workspace_name="miss",
                )
        return isinstance(flash_miss_controls, dict)

    if can_flash_split_attention:
        if hit_row_plan is None or miss_row_plan is None:
            can_flash_split_attention = False
        elif hit_row_plan.item_count <= 0 or miss_row_plan.item_count <= 0:
            can_flash_split_attention = False
        else:
            hit_control_stream = _shared_cuda_stream(
                device=gpu_k.device, name="overlap_flash_hit_control_h2d"
            )
            if hit_control_stream is not None:
                with torch.cuda.stream(hit_control_stream):
                    flash_hit_controls = _prepare_compact_slot_plan_controls(
                        kv_cache,
                        device=gpu_k.device,
                        row_plan=hit_row_plan,
                        B=B,
                        num_heads=num_heads,
                        frame_seqlen=frame_seqlen,
                        workspace_name="hit",
                    )
                if flash_hit_controls is None:
                    can_flash_split_attention = False
                else:
                    flash_control_tensors_ready_event = torch.cuda.Event()
                    flash_control_tensors_ready_event.record(hit_control_stream)
                    flash_hit_controls["ready_event"] = flash_control_tensors_ready_event
            else:
                flash_hit_controls = _prepare_compact_slot_plan_controls(
                    kv_cache,
                    device=gpu_k.device,
                    row_plan=hit_row_plan,
                    B=B,
                    num_heads=num_heads,
                    frame_seqlen=frame_seqlen,
                    workspace_name="hit",
                )
                if flash_hit_controls is None:
                    can_flash_split_attention = False
                elif torch.cuda.is_available():
                    flash_control_tensors_ready_event = torch.cuda.Event()
                    flash_control_tensors_ready_event.record(
                        torch.cuda.current_stream(gpu_k.device)
                    )
    if not can_flash_split_attention or miss_row_plan is None:
        can_compact_onload = False
    if (
        can_compact_onload
        and hit_row_plan is not None
        and (miss_row_plan is not None)
        and (flash_hit_controls is not None)
    ):
        compact_onload_stream = _shared_cuda_stream(
            device=gpu_k.device, name="overlap_compact_onload"
        )
        compact_miss_rope_stream = _shared_cuda_stream(
            device=gpu_k.device, name="overlap_compact_miss_rope"
        )
        compact_h2d_submitted = False
        hit_materialized = None
        compact_stage = _build_compact_miss_stage(
            miss_row_plan=miss_row_plan,
            compact_scatter_plan=plan_result.get("compact_scatter_plan", None),
            fill_count=overlap_plan.fill_count,
            total_tokens=total_tokens,
            B=B,
            D=D,
            frame_seqlen=frame_seqlen,
            device=gpu_k.device,
            dtype=gpu_k.dtype,
        )
        miss_controls_ready = _ensure_flash_miss_controls() if compact_stage is not None else False
        serial_compact = not _overlap_attn_enabled_by_env() or False
        rope_producer_event = None
        if compact_stage is not None and miss_controls_ready and True:
            rope_producer_event = torch.cuda.Event()
            rope_producer_event.record(torch.cuda.current_stream(gpu_k.device))

        def _submit_compact_miss() -> None:
            nonlocal compact_h2d_submitted, compact_h2d_done_event
            if compact_stage is None or compact_onload_stream is None or (not miss_controls_ready):
                return
            compact_h2d_submitted = _submit_compact_miss_h2d(
                kv_cache,
                stage=compact_stage,
                cpu_k=cpu_k,
                cpu_v=cpu_v,
                frame_seqlen=frame_seqlen,
                stream=compact_onload_stream,
            )
            compact_h2d_done_event = compact_stage.get("copy_done", None)
            if compact_h2d_submitted and compact_miss_rope_stream is not None:
                _submit_compact_miss_lf_rope(
                    kv_cache,
                    stage=compact_stage,
                    controls=flash_miss_controls,
                    rope_grid_sizes=rope_grid_sizes,
                    rope_freqs=rope_freqs,
                    stream=compact_miss_rope_stream,
                    producer_event=rope_producer_event,
                    lf_k_start_frame=int(lf_k_start_frame),
                )
            if not compact_h2d_submitted:
                compact_h2d_done_event = None

        if serial_compact:
            _submit_compact_miss()
        if serial_compact and compact_h2d_done_event is not None and torch.cuda.is_available():
            torch.cuda.current_stream(gpu_k.device).wait_event(compact_h2d_done_event)
            miss_ready = compact_stage.get("attention_k_ready", None)
            if miss_ready is not None:
                torch.cuda.current_stream(gpu_k.device).wait_event(miss_ready)
        hit_materialized = _materialize_compact_slot_plan(
            kv_cache,
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            row_plan=hit_row_plan,
            B=B,
            D=D,
            total_tokens=total_tokens,
            num_heads=num_heads,
            gpu_cache_frames=gpu_cache_frames,
            frame_seqlen=frame_seqlen,
            live_k=live_k,
            live_v=live_v,
            live_start_index=live_start_index,
            controls=flash_hit_controls,
        )
        if hit_materialized is not None:
            hit_part = _run_compact_slot_plan_flash_attention_part_from_materialized(
                kv_cache,
                label_prefix="overlap_flash_hit",
                q_group=q_group,
                row_plan=hit_row_plan,
                materialized_part=hit_materialized,
                B=B,
                D=D,
                num_heads=num_heads,
                out_dtype=out_dtype,
                kv_sparse_config=kv_sparse_config,
                selected_frames=int(frame_indices.shape[1]),
                fast_lf_rope=True,
                rope_grid_sizes=rope_grid_sizes,
                rope_freqs=rope_freqs,
                lf_k_start_frame=int(lf_k_start_frame),
            )
            if hit_part is not None:
                if not serial_compact:
                    _submit_compact_miss()
                if compact_h2d_submitted and compact_h2d_done_event is not None:
                    miss_part = (
                        _run_compact_miss_attention_part(
                            kv_cache,
                            stage=compact_stage,
                            controls=flash_miss_controls,
                            q_group=q_group,
                            D=D,
                            out_dtype=out_dtype or q_group.dtype,
                            kv_sparse_config=kv_sparse_config,
                            selected_frames=int(frame_indices.shape[1]),
                            fast_lf_rope=True,
                            rope_grid_sizes=rope_grid_sizes,
                            rope_freqs=rope_freqs,
                            lf_k_start_frame=int(lf_k_start_frame),
                        )
                        if miss_controls_ready and isinstance(flash_miss_controls, dict)
                        else None
                    )
                    if miss_part is not None:
                        miss_read_done = torch.cuda.Event()
                        miss_read_done.record(torch.cuda.current_stream(gpu_k.device))
                        compact_slot_fill_scheduled = _schedule_compact_miss_slot_fill(
                            kv_cache,
                            stage=compact_stage,
                            gpu_k=gpu_k,
                            gpu_v=gpu_v,
                            frame_seqlen=frame_seqlen,
                            gpu_cache_frames=gpu_cache_frames,
                            source_ready_event=miss_read_done,
                        )
                        if compact_slot_fill_scheduled:
                            queued = _kv_cache_queue_compact_stage_slots_commit(
                                kv_cache,
                                compact_stage,
                                frame_seqlen=frame_seqlen,
                                total_tokens=total_tokens,
                                ready_event=compact_stage.get("slot_fill_done", None),
                                device=gpu_k.device,
                            )
                            if queued:
                                compact_stage["_slot_state_committed"] = True
                        compact_out = _merge_attention_partials(
                            hit_part[0],
                            hit_part[1],
                            miss_part[0],
                            miss_part[1],
                            out_dtype=out_dtype or q_group.dtype,
                        )
                        if compact_slot_fill_scheduled:
                            committed = bool(compact_stage.get("_slot_state_committed", False))
                            if not committed:
                                committed = _kv_cache_queue_compact_stage_slots_commit(
                                    kv_cache,
                                    compact_stage,
                                    frame_seqlen=frame_seqlen,
                                    total_tokens=total_tokens,
                                    ready_event=compact_stage.get("slot_fill_done", None),
                                    device=gpu_k.device,
                                )
                            if not committed:
                                committed = _kv_cache_commit_compact_stage_slots(
                                    kv_cache,
                                    compact_stage,
                                    frame_seqlen=frame_seqlen,
                                    total_tokens=total_tokens,
                                    device_wait_event=compact_stage.get("slot_fill_done", None),
                                )
                            if committed:
                                compact_stage["_slot_state_committed"] = True
                        flush_deferred_clean_kv_offloads(kv_cache)
                        return compact_out
                    compact_h2d_done_event = None
        else:
            compact_h2d_done_event = None
    onload_stream = _shared_cuda_stream(device=gpu_k.device, name="overlap_onload")
    assert onload_stream is not None
    if compact_h2d_done_event is not None:
        onload_stream.wait_event(compact_h2d_done_event)
    if flash_control_tensors_ready_event is not None:
        onload_stream.wait_event(flash_control_tensors_ready_event)
    onload_result: bool | None = None
    onload_done_event: torch.cuda.Event | None = None

    def _submit_onload() -> bool:
        nonlocal onload_done_event
        with torch.cuda.device(gpu_k.device):
            torch.cuda.set_device(gpu_k.device)
            copied = _kv_cache_copy_frame_tensor_plan_from_cpu_to_slots(
                kv_cache,
                frame_indices_cpu=frame_indices_cpu,
                slot_plan_cpu=slot_plan_cpu,
                copy_plan_cpu=overlap_plan.copy_plan_cpu,
                expected_copies=overlap_plan.fill_count,
                frame_seqlen=frame_seqlen,
                stream=onload_stream,
            )
            if copied:
                with torch.cuda.stream(onload_stream):
                    onload_done_event = torch.cuda.Event()
                    onload_done_event.record(onload_stream)
            return copied

    def _finish_onload() -> bool:
        nonlocal onload_result
        if onload_result is not None:
            return onload_result
        onload_result = True
        return onload_result

    if use_materialized_flash:
        if overlap_plan.fill_count > 0 and onload_result is None:
            onload_result = _submit_onload()
            if not onload_result:
                return None
        if not _finish_onload():
            return None
        if onload_done_event is not None:
            torch.cuda.current_stream(gpu_k.device).wait_event(onload_done_event)
        flash_out = _run_materialized_slot_plan_flash_attention(
            kv_cache,
            q_group=q_group,
            gpu_k=gpu_k,
            gpu_v=gpu_v,
            frame_indices_cpu=frame_indices_cpu,
            slot_plan_cpu=slot_plan_cpu,
            B=B,
            D=D,
            total_tokens=total_tokens,
            num_heads=num_heads,
            gpu_cache_frames=gpu_cache_frames,
            frame_seqlen=frame_seqlen,
            live_k=live_k,
            live_v=live_v,
            live_start_index=live_start_index,
            frame_indices_device=frame_indices if frame_indices.device == gpu_k.device else None,
            slot_indices_device=None,
            out_dtype=out_dtype,
            kv_sparse_config=kv_sparse_config,
            fast_lf_rope=True,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            layer_idx=layer_idx,
            num_frame_per_block=num_frame_per_block,
            sink_tokens=sink_tokens,
            lf_k_start_frame=int(lf_k_start_frame),
        )
        if flash_out is not None:
            slot_state_committed = _commit_planned_slot_state()
            if not slot_state_committed:
                return None
            return flash_out
    print("Error: flash failed")
    return None


def _overlapped_attention_from_gpu_slots_if_possible(
    kv_cache: dict[str, Any],
    q_group: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_seqlen: int,
    *,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_start_index: int | None = None,
    out_dtype: torch.dtype | None = None,
    slot_metadata_ready: bool = False,
    kv_sparse_config: KVSparseConfig | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
    rope_freqs: torch.Tensor | None = None,
    layer_idx: int | None = None,
    num_frame_per_block: int = 1,
    sink_tokens: int = 0,
    staged_frame_indices: dict[str, Any] | None = None,
    lf_k_start_frame: int = 0,
) -> torch.Tensor | None:
    return _overlapped_attention_from_gpu_slots_if_possible_impl(
        kv_cache,
        q_group,
        frame_indices,
        frame_seqlen,
        live_k=live_k,
        live_v=live_v,
        live_start_index=live_start_index,
        out_dtype=out_dtype,
        slot_metadata_ready=slot_metadata_ready,
        kv_sparse_config=kv_sparse_config,
        fast_lf_rope=True,
        rope_grid_sizes=rope_grid_sizes,
        rope_freqs=rope_freqs,
        layer_idx=layer_idx,
        num_frame_per_block=num_frame_per_block,
        sink_tokens=sink_tokens,
        staged_frame_indices=staged_frame_indices,
        lf_k_start_frame=int(lf_k_start_frame),
    )


def _cached_compulsory_indices_device(
    kv_cache: dict[str, Any],
    compulsory_frames: set[int] | list[int] | tuple[int, ...],
    n_frames: int,
    device: torch.device,
) -> torch.Tensor:
    values = tuple(sorted({int(v) for v in compulsory_frames if 0 <= int(v) < int(n_frames)}))
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, int(n_frames), values)
    cache = kv_cache.setdefault("_kv_sparse_compulsory_indices_device_cache", {})
    tensor = _COMPULSORY_INDICES_DEVICE_CACHE.get(key, None)
    if tensor is None and isinstance(cache, dict):
        tensor = cache.get(key, None)
    if (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and (tensor.dtype == torch.long)
        and (int(tensor.numel()) == len(values))
    ):
        ready = _COMPULSORY_INDICES_DEVICE_READY.get(key)
        if ready is not None and torch.cuda.is_available():
            current_stream = torch.cuda.current_stream(device)
            if int(current_stream.cuda_stream) != int(ready[1]):
                current_stream.wait_event(ready[0])
        return tensor
    tensor = torch.empty((len(values),), device=device, dtype=torch.long)
    if values:
        cpu_cache = kv_cache.setdefault("_kv_sparse_compulsory_indices_cpu_cache", {})
        cpu_tensor = cpu_cache.get(values, None) if isinstance(cpu_cache, dict) else None
        if (
            not isinstance(cpu_tensor, torch.Tensor)
            or cpu_tensor.device.type != "cpu"
            or cpu_tensor.dtype != torch.long
            or (int(cpu_tensor.numel()) != len(values))
        ):
            cpu_tensor = torch.tensor(values, device="cpu", dtype=torch.long)
            if _can_pin_cpu_memory(device):
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except Exception:
                    pass
            if isinstance(cpu_cache, dict):
                cpu_cache[values] = cpu_tensor
        tensor.copy_(
            cpu_tensor, non_blocking=bool(getattr(cpu_tensor, "is_pinned", lambda: False)())
        )
    if torch.cuda.is_available() and device.type == "cuda":
        producer_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        _COMPULSORY_INDICES_DEVICE_READY[key] = (ready_event, int(producer_stream.cuda_stream))
    if isinstance(cache, dict):
        cache[key] = tensor
    _COMPULSORY_INDICES_DEVICE_CACHE[key] = tensor
    return tensor


def _select_kv_frame_indices_max_kv_frames_batched_fused(
    kv_cache: dict[str, Any],
    cfg: KVSparseConfig,
    roped_query: torch.Tensor,
    frame_k_index_window: torch.Tensor | None,
    *,
    n_frames: int,
    frame_seqlen: int,
    max_kv_frames: int,
    compulsory_frames: set[int],
) -> torch.Tensor | None:
    if frame_k_index_window is None or not isinstance(frame_k_index_window, torch.Tensor):
        return None
    if roped_query.device.type != "cuda" or frame_k_index_window.device != roped_query.device:
        return None
    num_heads = int(roped_query.shape[2])
    if num_heads <= 0:
        return None
    head_indices = _all_head_indices_device(num_heads, roped_query.device)
    if int(max_kv_frames) < 0 or int(max_kv_frames) > 256:
        return None
    if int(n_frames) <= 0 or int(frame_k_index_window.shape[1]) < int(n_frames):
        return None
    compulsory_indices = _cached_compulsory_indices_device(
        kv_cache, compulsory_frames, n_frames, roped_query.device
    )
    try:
        selected = select_pooled_qk_topk(
            q=roped_query,
            frame_k=frame_k_index_window[:, : int(n_frames)],
            head_indices=head_indices,
            compulsory_indices=compulsory_indices,
            frame_seqlen=int(frame_seqlen),
            max_kv_frames=int(max_kv_frames),
        )
    except Exception as exc:
        if str(os.getenv("HORIZONKV_FUSED_SELECTION_VERBOSE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[horizonkv] fused selection fallback after failure: {exc}")
        return None
    if not isinstance(selected, torch.Tensor) or selected.device != roped_query.device:
        return None
    if selected.ndim != 2 or int(selected.shape[0]) != num_heads:
        return None
    return selected


def _sparse_kv_self_attention_cpu_offload(
    *,
    roped_query: torch.Tensor,
    selection_query: torch.Tensor | None = None,
    kv_cache: dict[str, Any],
    frame_seqlen: int,
    kv_start: int,
    sink_tokens: int,
    current_start: int,
    current_end: int,
    local_end_index: int,
    num_frame_per_block: int,
    layer_idx: int,
    current_timestep: Any = None,
    kv_sparse_config: KVSparseConfig,
    out_dtype: torch.dtype,
    rope_freqs: torch.Tensor | None = None,
    next_kv_cache: dict[str, Any] | None = None,
    next_selection_query: torch.Tensor | None = None,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_frame_k_mean: torch.Tensor | None = None,
    live_frame_k_max: torch.Tensor | None = None,
    live_frame_k_min: torch.Tensor | None = None,
    live_start_index: int | None = None,
    fast_lf_rope: bool = False,
    rope_grid_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    _kv_cache_wait_pending_slot_metadata_commit(kv_cache)
    B, Lq, H, D = roped_query.shape
    cfg = kv_sparse_config
    score_query = selection_query if selection_query is not None else roped_query
    Lk = int(local_end_index)
    n_frames = max(1, (Lk + frame_seqlen - 1) // frame_seqlen)
    sink_frames = _sink_frame_indices(Lk, kv_start, frame_seqlen, sink_tokens)
    forced_frames = _mandatory_kv_frame_indices(
        Lk, frame_seqlen, kv_start, local_end_index, current_start, current_end, num_frame_per_block
    )
    forced_frames.update(
        _preceding_kv_frame_indices(
            Lk,
            frame_seqlen,
            kv_start,
            local_end_index,
            current_start,
            current_end,
            num_frame_per_block,
            None,
        )
    )
    forced_frames.update(
        _recent_kv_frame_indices(Lk, frame_seqlen, getattr(cfg, "mandatory_recent_kv_frames", None))
    )
    compulsory_frames = set(sink_frames) | set(forced_frames)
    selected_frame_idx: torch.Tensor | None = None
    adaptive_metadata = None
    frame_k_index_window = _overlay_live_frame_stats(
        kv_cache.get("frame_k_mean", None),
        live_frame_k_mean,
        live_start_index=live_start_index,
        frame_seqlen=frame_seqlen,
        n_frames=n_frames,
    )
    shared_max_kv_frames = cfg.max_kv_frames
    use_batched_selection = Lk % frame_seqlen == 0 and shared_max_kv_frames is not None
    if use_batched_selection and selected_frame_idx is None:
        if os.getenv("HORIZONKV_ADAPTIVE_CONFIG"):
            from flagscale.inference.core.diffusion.horizonkv.algorithm.rasa import (
                select_adaptive_frames_if_supported,
            )

            adaptive_metadata = select_adaptive_frames_if_supported(
                kv_cache=kv_cache,
                cfg=cfg,
                q=score_query,
                frame_means=frame_k_index_window,
                compulsory_frames=compulsory_frames,
                frame_seqlen=frame_seqlen,
                n_frames=n_frames,
                live_start_index=live_start_index,
                live_k=live_k,
            )
        if adaptive_metadata is not None:
            cfg = adaptive_metadata["cfg"]
            selected_frame_idx = adaptive_metadata["selected"]
        else:
            selected_frame_idx = _select_kv_frame_indices_max_kv_frames_batched_fused(
                kv_cache,
                cfg,
                score_query,
                frame_k_index_window,
                n_frames=n_frames,
                frame_seqlen=frame_seqlen,
                max_kv_frames=int(shared_max_kv_frames),
                compulsory_frames=compulsory_frames,
            )
    use_batched_max_kv_frames = Lk % frame_seqlen == 0 and shared_max_kv_frames is not None
    if use_batched_max_kv_frames:
        assert selected_frame_idx is not None
        staged_frame_indices = None
        if (
            _overlap_attn_enabled_by_env()
            and live_start_index is not None
            and (
                int(live_start_index) > int(kv_cache.get("gpu_cache_frames", 0) or 0) * frame_seqlen
            )
        ):
            if adaptive_metadata is not None:
                staged_frame_indices = adaptive_metadata["staged"]
            else:
                staged_frame_indices = _stage_overlap_frame_indices_to_cpu_async(
                    kv_cache, selected_frame_idx
                )
        q_for_attention = roped_query
        if rope_grid_sizes is None or rope_freqs is None:
            raise ValueError("fast LF-RoPE requires rope_grid_sizes and rope_freqs")
        selected_frames = int(selected_frame_idx.shape[1])
        query_frames = max(1, int(math.ceil(Lq / float(frame_seqlen))))
        current_start_frame = int(current_start) // int(frame_seqlen)
        lf_k_start_frame = max(0, current_start_frame - max(0, selected_frames - query_frames))
        q_for_attention = compacted_rope_apply(
            roped_query, rope_grid_sizes, rope_freqs, start_frame=current_start_frame
        ).to(dtype=live_v.dtype if isinstance(live_v, torch.Tensor) else roped_query.dtype)
        q_attn = q_for_attention.permute(0, 2, 1, 3).reshape(B * H, Lq, 1, D)
        overlap_result = None
        overlap_result = _overlapped_attention_from_gpu_slots_if_possible(
            kv_cache,
            q_attn,
            selected_frame_idx,
            frame_seqlen,
            live_k=live_k,
            live_v=live_v,
            live_start_index=live_start_index,
            out_dtype=out_dtype,
            slot_metadata_ready=True,
            kv_sparse_config=cfg,
            fast_lf_rope=True,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            lf_k_start_frame=lf_k_start_frame,
            layer_idx=layer_idx,
            num_frame_per_block=num_frame_per_block,
            sink_tokens=sink_tokens,
            staged_frame_indices=staged_frame_indices,
        )
        if overlap_result is not None:
            return overlap_result.reshape(B, H, Lq, D).permute(0, 2, 1, 3)
        raise RuntimeError(
            "HorizonKV materialized attention failed; check the sparse and transfer extensions"
        )
