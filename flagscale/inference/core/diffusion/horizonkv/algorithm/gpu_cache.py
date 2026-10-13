# Modified for the HorizonKV inference artifact; see NOTICE.
"""GPU rolling-window adapter for the fast LF-RoPE all-hit backend."""

from __future__ import annotations

from typing import Any

import torch

from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    _mandatory_kv_frame_indices,
    _nvtx_range,
    _preceding_kv_frame_indices,
    _recent_kv_frame_indices,
    _run_materialized_slot_plan_flash_attention,
    _select_kv_frame_indices_max_kv_frames_batched_fused,
    _sink_frame_indices,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig
from flagscale.inference.core.diffusion.horizonkv.algorithm.retrieval import (
    _per_frame_simple_mean_cached_k,
    _select_kv_frame_indices_weighted_random_batched,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_kernels import lightforcing_rope_apply


def uses_fast_lf_rope_local(cfg: KVSparseConfig | None) -> bool:
    return bool(
        cfg is not None
        and cfg.enabled
        and not cfg.cpu_offload_kv
        and cfg.selector == "longlive"
        and cfg.rope_semantics == "horizonkv_compacted"
        and cfg.granularity == "kv_frame"
    )


def build_local_kv_cache_state(
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    frame_capacity: int,
    frame_seqlen: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if frame_capacity <= 0 or frame_seqlen <= 0:
        raise ValueError("Local LF-RoPE requires a positive, frame-aligned cache size")
    # Match the full-history materializer's head-major slot layout exactly.
    shape = (batch_size, num_heads * frame_capacity * frame_seqlen, 1, head_dim)
    return {
        "cache_mode": "lf_rope_local",
        "cpu_offload_kv": False,
        "gpu_cache_frames": frame_capacity,
        "frame_seqlen": frame_seqlen,
        "num_heads": num_heads,
        "gpu_k": torch.empty(shape, dtype=dtype, device=device),
        "gpu_v": torch.empty(shape, dtype=dtype, device=device),
        "frame_k_mean": torch.empty(
            (batch_size, frame_capacity, num_heads, head_dim), dtype=dtype, device=device
        ),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    }


def _local_window_slots(
    kv_cache: dict[str, Any], *, end_frame: int, sink_frames: int
) -> torch.Tensor:
    capacity = kv_cache["gpu_cache_frames"]
    if not 0 <= sink_frames < capacity:
        raise ValueError("Local LF-RoPE sink must be smaller than the GPU window")
    n_frames = min(end_frame, capacity)
    tail_capacity = capacity - sink_frames
    # Logical frame order is [sink, oldest visible history, ..., live].
    # Only this small index/mean array is reordered; historical KV never moves.
    shift = max(0, end_frame - capacity) % tail_capacity
    key = (n_frames, sink_frames, shift)
    cached = kv_cache.get("_local_window_slots")
    if cached is not None and cached[0] == key:
        return cached[1]
    indices = torch.arange(n_frames, device=kv_cache["gpu_k"].device)
    slots = torch.where(
        indices < sink_frames,
        indices,
        sink_frames + (indices - sink_frames + shift) % tail_capacity,
    )
    kv_cache["_local_window_slots"] = (key, slots)
    return slots


def _local_frame_means(
    kv_cache: dict[str, Any],
    slots: torch.Tensor,
    live_frame_k_mean: torch.Tensor,
) -> torch.Tensor:
    means = kv_cache["frame_k_mean"].index_select(1, slots)
    means[:, -live_frame_k_mean.shape[1]:] = live_frame_k_mean
    return means


def _commit_local_kv(
    kv_cache: dict[str, Any],
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    frame_k_mean: torch.Tensor,
    start_frame: int,
    sink_frames: int,
) -> None:
    capacity = kv_cache["gpu_cache_frames"]
    frame_seqlen = kv_cache["frame_seqlen"]
    batch, _, heads, dim = key.shape
    count = frame_k_mean.shape[1]
    key_frames = key.reshape(batch, count, frame_seqlen, heads, dim).permute(0, 3, 1, 2, 4)
    value_frames = value.reshape(batch, count, frame_seqlen, heads, dim).permute(0, 3, 1, 2, 4)
    gpu_k = kv_cache["gpu_k"].view(batch, heads, capacity, frame_seqlen, dim)
    gpu_v = kv_cache["gpu_v"].view_as(gpu_k)
    offset = 0
    while offset < count:
        frame = start_frame + offset
        if frame < sink_frames:
            slot = frame
            length = min(count - offset, sink_frames - frame)
        else:
            slot = sink_frames + (frame - sink_frames) % (capacity - sink_frames)
            length = min(count - offset, capacity - slot)
        gpu_k[:, :, slot:slot + length].copy_(key_frames[:, :, offset:offset + length])
        gpu_v[:, :, slot:slot + length].copy_(value_frames[:, :, offset:offset + length])
        kv_cache["frame_k_mean"][:, slot:slot + length].copy_(
            frame_k_mean[:, offset:offset + length]
        )
        offset += length


@torch.no_grad()
def local_lf_rope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_cache: dict[str, Any],
    cfg: KVSparseConfig,
    current_start: int,
    sink_frames: int,
    num_frame_per_block: int,
    layer_idx: int,
    rope_grid_sizes: torch.Tensor,
    rope_freqs: torch.Tensor,
    write_clean_kv: bool,
) -> torch.Tensor:
    batch, query_tokens, heads, dim = q.shape
    frame_seqlen = kv_cache["frame_seqlen"]
    capacity = kv_cache["gpu_cache_frames"]
    if current_start < 0 or current_start % frame_seqlen or query_tokens % frame_seqlen:
        raise ValueError("Local LF-RoPE requires complete frames")
    if cfg.rule != "max_kv_frames" or cfg.frame_selection_score != "pooled_qk":
        raise ValueError("Local fast LF-RoPE requires max_kv_frames with pooled_qk scores")
    query_frames = query_tokens // frame_seqlen
    current_end = current_start + query_tokens
    if query_frames <= 0 or sink_frames + query_frames > capacity:
        raise ValueError("Local LF-RoPE window must fit the sink and current query block")
    if current_start != int(kv_cache["global_end_index"].item()):
        raise ValueError("Local fast LF-RoPE expects sequential clean-cache updates")

    with _nvtx_range(kv_cache, "horizonkv.local_frame_means"):
        new_frame_k_mean = k.reshape(
            batch, query_frames, frame_seqlen, heads, dim
        ).mean(dim=2).to(v.dtype)
        slots = _local_window_slots(
            kv_cache, end_frame=current_end // frame_seqlen, sink_frames=sink_frames
        )
        frame_k_mean = _local_frame_means(kv_cache, slots, new_frame_k_mean)
    n_frames = int(slots.numel())
    local_end = n_frames * frame_seqlen
    live_start = local_end - query_tokens
    sink_tokens = sink_frames * frame_seqlen
    sink = _sink_frame_indices(local_end, 0, frame_seqlen, sink_tokens)
    forced = _mandatory_kv_frame_indices(
        local_end, frame_seqlen, 0, local_end,
        current_start, current_end, num_frame_per_block,
    )
    forced.update(_preceding_kv_frame_indices(
        local_end, frame_seqlen, 0, local_end,
        current_start, current_end, num_frame_per_block,
        cfg.mandatory_preceding_kv_frames,
    ))
    forced.update(_recent_kv_frame_indices(
        local_end, frame_seqlen, cfg.mandatory_recent_kv_frames,
    ))

    with _nvtx_range(kv_cache, "horizonkv.local_frame_select"):
        if cfg.frame_selection_strategy == "weighted_random":
            scores = _per_frame_simple_mean_cached_k(q.to(v.dtype), frame_k_mean, frame_seqlen)
            selected = _select_kv_frame_indices_weighted_random_batched(
                scores, int(cfg.max_kv_frames), n_frames, sink, forced
            )
        else:
            selected = _select_kv_frame_indices_max_kv_frames_batched_fused(
                kv_cache, cfg, q.to(v.dtype), frame_k_mean,
                n_frames=n_frames,
                frame_seqlen=frame_seqlen,
                max_kv_frames=int(cfg.max_kv_frames),
                compulsory_frames=sink | forced,
            )
        if selected is None:
            raise RuntimeError("Local LF-RoPE fused frame selector is unavailable")

    with _nvtx_range(kv_cache, "horizonkv.local_all_hit"):
        slot_indices = slots[selected]
        slot_indices = slot_indices.masked_fill(selected >= n_frames - query_frames, -2)
        current_start_frame = int(current_start) // int(frame_seqlen)
        lf_k_start_frame = max(
            0,
            current_start_frame - max(0, int(selected.shape[1]) - query_frames),
        )
        q_rope = lightforcing_rope_apply(
            q, rope_grid_sizes, rope_freqs,
            start_frame=current_start_frame,
        ).to(v.dtype)
        q_rows = q_rope.permute(0, 2, 1, 3).reshape(batch * heads, query_tokens, 1, dim)
        out = _run_materialized_slot_plan_flash_attention(
            kv_cache,
            q_group=q_rows,
            gpu_k=kv_cache["gpu_k"],
            gpu_v=kv_cache["gpu_v"],
            frame_indices_cpu=None,
            slot_plan_cpu=None,
            B=batch,
            D=dim,
            total_tokens=local_end,
            num_heads=heads,
            gpu_cache_frames=capacity,
            frame_seqlen=frame_seqlen,
            live_k=k,
            live_v=v,
            live_start_index=live_start,
            frame_indices_device=selected,
            slot_indices_device=slot_indices,
            out_dtype=q.dtype,
            kv_sparse_config=cfg,
            fast_lf_rope=True,
            rope_grid_sizes=rope_grid_sizes,
            rope_freqs=rope_freqs,
            lf_k_start_frame=lf_k_start_frame,
            layer_idx=layer_idx,
            num_frame_per_block=num_frame_per_block,
            sink_tokens=sink_tokens,
        )
        if out is None:
            raise RuntimeError("Local LF-RoPE all-hit attention backend is unavailable")
    if write_clean_kv:
        with _nvtx_range(kv_cache, "horizonkv.local_clean_commit"):
            _commit_local_kv(
                kv_cache, key=k, value=v, frame_k_mean=new_frame_k_mean,
                start_frame=current_start // frame_seqlen, sink_frames=sink_frames,
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end)
    return out.reshape(batch, heads, query_tokens, dim).permute(0, 2, 1, 3)
