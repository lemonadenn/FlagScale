# Modified for the HorizonKV inference artifact; see NOTICE.
from __future__ import annotations
from typing import Any
import torch
from .config import KVSparseConfig
from .oae import _sparse_kv_self_attention_cpu_offload


def sparse_kv_self_attention(
    roped_query: torch.Tensor,
    k_window: torch.Tensor | None,
    v_window: torch.Tensor | None,
    *,
    frame_k_index_window: torch.Tensor | None = None,
    frame_k_min_index_window: torch.Tensor | None = None,
    frame_k_max_index_window: torch.Tensor | None = None,
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
    rope_grid_sizes: torch.Tensor | None = None,
    selection_query: torch.Tensor | None = None,
    next_selection_query: torch.Tensor | None = None,
    kv_cache: dict[str, Any] | None = None,
    next_kv_cache: dict[str, Any] | None = None,
    live_k: torch.Tensor | None = None,
    live_v: torch.Tensor | None = None,
    live_frame_k_mean: torch.Tensor | None = None,
    live_frame_k_max: torch.Tensor | None = None,
    live_frame_k_min: torch.Tensor | None = None,
    live_start_index: int | None = None,
    prefer_lf_triton: bool = False,
    fast_lf_rope: bool = False,
) -> torch.Tensor:
    cfg = kv_sparse_config
    if kv_cache is None:
        raise ValueError("HorizonKV requires the full host-history KV cache")
    score_query = roped_query
    return _sparse_kv_self_attention_cpu_offload(
        roped_query=roped_query,
        selection_query=score_query,
        kv_cache=kv_cache,
        frame_seqlen=frame_seqlen,
        kv_start=kv_start,
        sink_tokens=sink_tokens,
        current_start=current_start,
        current_end=current_end,
        local_end_index=local_end_index,
        num_frame_per_block=num_frame_per_block,
        layer_idx=layer_idx,
        current_timestep=current_timestep,
        kv_sparse_config=cfg,
        out_dtype=out_dtype,
        rope_freqs=rope_freqs,
        next_kv_cache=next_kv_cache,
        next_selection_query=next_selection_query,
        live_k=live_k,
        live_v=live_v,
        live_frame_k_mean=live_frame_k_mean,
        live_frame_k_max=live_frame_k_max,
        live_frame_k_min=live_frame_k_min,
        live_start_index=live_start_index,
        fast_lf_rope=True,
        rope_grid_sizes=rope_grid_sizes,
    )
