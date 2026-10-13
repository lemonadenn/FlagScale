# Modified for the HorizonKV inference artifact; see NOTICE.
"""Per-head frame retrieval and 64x64 block-sparse attention.

Frames are selected before block sparsification. HorizonKV uses full host
history and OAE; HorizonKV-GPU uses a rolling GPU window. See
docs/implementation.md for selection, compulsory-frame, and RoPE semantics."""

from __future__ import annotations

import math
from typing import Any

import torch

from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    _mandatory_kv_frame_indices,
    _preceding_kv_frame_indices,
    _recent_kv_frame_indices,
    _sink_frame_indices,
    _sparse_kv_self_attention_cpu_offload,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_backend import (
    batch_blocksparse_attention,
    rows_blocksparse_attention,
    stage2_blocksparse_attention,
    stage2_blocksparse_kernel_supported,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import (
    KVSparseConfig,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.rope import reindex_compacted_qk_rope
from flagscale.inference.core.diffusion.horizonkv.algorithm.retrieval import (
    _apply_rule_to_mass_1d,
    _avg_pool_tokens_along_seq,
    _block_mask_from_selected_linear_indices_batched,
    _block_mask_from_selected_linear_indices,
    _block_density_is_dense,
    _effective_block_density_for_selected_frames,
    _extra_selected_indices,
    _forced_attn_block_linear_indices,
    _forced_attn_block_linear_indices_device,
    _frame_importance_scores,
    _head_is_dense,
    _key_token_indices_from_frames,
    _mandatory_key_positions_pooled_from_frames,
    _per_block_attn_mass_batched,
    _per_block_pooled_qk_scores_batched,
    _prepare_qk_for_importance,
    _select_candidate_blocks_by_density_batched,
    _select_density_block_mask_batched,
    _select_candidate_blocks_by_frame_density,
    _select_candidate_blocks_by_frame_density_weighted,
    _select_candidate_blocks_by_frame_recall,
    _select_kv_frame_indices_max_kv_frames_batched,
    _select_kv_frame_indices_weighted_random_batched,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.baseline_selector import calculate_chunk_sparsities
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_kernels import (
    lightforcing_attention,
    lightforcing_attention_with_block_mask,
    lightforcing_rope_apply,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.visualization import (
    _dump_selection_visualization,
    _profile_frame_selection_record,
    _profile_stage2_record,
    _profile_now,
    _should_visualize_mask,
)
from flagscale.inference.core.diffusion.horizonkv.wan.modules.attention import attention


def _key_token_indices_from_frame_indices_batched(
    frame_indices: torch.Tensor,
    Lk: int,
    frame_seqlen: int,
) -> torch.Tensor:
    head_count, n_selected = frame_indices.shape
    if head_count <= 0 or n_selected <= 0 or Lk <= 0 or frame_seqlen <= 0:
        return torch.empty((head_count, 0), device=frame_indices.device, dtype=torch.long)
    max_frame = max(0, (int(Lk) - 1) // int(frame_seqlen))
    safe_frames = frame_indices.to(torch.long).clamp(0, max_frame)
    offsets = torch.arange(frame_seqlen, device=frame_indices.device, dtype=torch.long)
    token_indices = safe_frames.unsqueeze(-1) * int(frame_seqlen) + offsets
    return token_indices.reshape(head_count, n_selected * int(frame_seqlen)).clamp_max(Lk - 1)


def _compact_kv_from_frame_indices_batched(
    k_window: torch.Tensor,
    v_window: torch.Tensor,
    head_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, Lk, _, D = k_window.shape
    head_count = int(head_indices.numel())
    if head_count <= 0 or Lk <= 0 or frame_seqlen <= 0:
        empty = torch.empty((0, 0, 1, D), device=k_window.device, dtype=k_window.dtype)
        return empty, empty
    if Lk % frame_seqlen != 0:
        raise ValueError("frame packing requires Lk to be divisible by frame_seqlen")
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)
    if frame_indices.ndim != 2:
        raise ValueError("frame_indices must be 1D or 2D")
    if int(frame_indices.shape[0]) == 1 and head_count > 1:
        frame_indices = frame_indices.expand(head_count, -1)
    elif int(frame_indices.shape[0]) != head_count:
        raise ValueError("frame_indices head dimension does not match head_indices")
    if int(frame_indices.shape[1]) <= 0:
        empty = torch.empty((0, 0, 1, D), device=k_window.device, dtype=k_window.dtype)
        return empty, empty

    head_indices = head_indices.to(device=k_window.device, dtype=torch.long)
    token_indices = _key_token_indices_from_frame_indices_batched(
        frame_indices.to(device=k_window.device),
        Lk,
        frame_seqlen,
    )
    token_count = int(token_indices.shape[1])
    head_index = head_indices.view(head_count, 1)
    k_compact = k_window[:, token_indices, head_index, :].contiguous().reshape(
        B * head_count, token_count, 1, D
    )
    v_compact = v_window[:, token_indices, head_index, :].contiguous().reshape(
        B * head_count, token_count, 1, D
    )
    return k_compact, v_compact


def _pooled_key_block_mask_from_frames(
    Lk_p: int,
    n_frames: int,
    frame_seqlen: int,
    k_pool: int,
    selected_frames: set[int],
    device: torch.device,
) -> torch.Tensor:
    if Lk_p <= 0 or n_frames <= 0 or not selected_frames:
        return torch.zeros(Lk_p, device=device, dtype=torch.bool)
    frame_mask = torch.zeros(n_frames, device=device, dtype=torch.bool)
    frame_indices = torch.tensor(sorted(selected_frames), device=device, dtype=torch.long)
    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < n_frames)]
    if frame_indices.numel() > 0:
        frame_mask[frame_indices] = True
    pooled_indices = torch.arange(Lk_p, device=device, dtype=torch.long)
    pooled_frames = ((pooled_indices * k_pool) // frame_seqlen).clamp_max(n_frames - 1)
    return frame_mask[pooled_frames]


def _frame_selection_plan_key(
    cfg: KVSparseConfig,
    *,
    current_start: int,
    current_end: int,
    local_end_index: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    n_frames: int,
) -> tuple[Any, ...]:
    return (
        id(cfg),
        str(cfg.granularity),
        int(current_start),
        int(current_end),
        int(local_end_index),
        int(frame_seqlen),
        int(num_frame_per_block),
        int(n_frames),
    )


def _normalize_frame_sequence(frames: Any, n_frames: int) -> tuple[int, ...]:
    if frames is None:
        return ()
    return tuple(sorted({int(v) for v in frames if 0 <= int(v) < int(n_frames)}))


def _get_frame_selection_plan(
    kv_cache: dict[str, Any],
    key: tuple[Any, ...],
) -> dict[str, Any] | None:
    plans = kv_cache.get("kv_sparse_selection_plans", None)
    if not isinstance(plans, dict):
        return None
    plan = plans.get(key)
    return plan if isinstance(plan, dict) else None


def _store_frame_selection_plan(
    kv_cache: dict[str, Any],
    key: tuple[Any, ...],
    *,
    selected_by_head: dict[int, list[int] | tuple[int, ...]],
    n_frames: int,
    Lk: int,
    granularity: str,
) -> None:
    plans = kv_cache.setdefault("kv_sparse_selection_plans", {})
    plans[key] = {
        "selected_by_head": {
            int(head): _normalize_frame_sequence(frames, n_frames)
            for head, frames in selected_by_head.items()
        },
        "n_frames": int(n_frames),
        "Lk": int(Lk),
        "granularity": str(granularity),
    }


def _selection_plan_head_frames(
    plan: dict[str, Any] | None,
    heads: torch.Tensor | list[int] | tuple[int, ...],
    n_frames: int,
) -> dict[int, list[int]] | None:
    if not isinstance(plan, dict):
        return None
    selected_by_head = plan.get("selected_by_head", None)
    if not isinstance(selected_by_head, dict):
        return None
    head_values = heads.detach().cpu().tolist() if isinstance(heads, torch.Tensor) else heads
    result: dict[int, list[int]] = {}
    for head in head_values:
        frames = selected_by_head.get(int(head), None)
        if frames is None:
            return None
        result[int(head)] = list(_normalize_frame_sequence(frames, n_frames))
    return result


def _require_stage2_blocksparse_backend(reason: str) -> None:
    raise RuntimeError(
        "[horizonkv] stage2 block sparsity requires Block-Sparse-Attention: "
        f"{reason}"
    )


def _stage2_rows_blocksparse_attention_if_possible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_q: int,
    block_k: int,
    out_dtype: torch.dtype,
    prefer_lf_triton: bool = False,
) -> torch.Tensor | None:
    """Run the selected block mask through the row backend or supported direct kernels."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    B, Lq, H, D = q.shape
    if B != 1 or tuple(k.shape) != tuple(v.shape) or int(k.shape[2]) != H:
        return None
    n_bq = int(math.ceil(Lq / float(block_q)))
    n_bk = int(math.ceil(int(k.shape[1]) / float(block_k)))
    if tuple(block_mask.shape) != (H, n_bq, n_bk):
        return None

    # Consume the selected mask directly in Triton when supported; otherwise use CUDA.
    if prefer_lf_triton:
        triton_out = lightforcing_attention_with_block_mask(
            q,
            k,
            v,
            block_mask,
            block_q=int(block_q),
            block_k=int(block_k),
            softmax_scale=1.0 / math.sqrt(float(D)),
        )
        if isinstance(triton_out, torch.Tensor):
            return triton_out.to(out_dtype)

    # Try the native [B,L,H,D] entry point before per-head layout conversion.
    if prefer_lf_triton:
        fast_batched = batch_blocksparse_attention(
            q=q,
            k=k,
            v=v,
            block_mask=block_mask,
            block_q=int(block_q),
            block_k=int(block_k),
            softmax_scale=1.0 / math.sqrt(float(D)),
            return_lse=False,
        )
        if isinstance(fast_batched, torch.Tensor):
            return fast_batched.to(out_dtype)

    # convert_blockmask_row_reverse() in the vendored wrapper produces the
    # selected key-block ids in descending order followed by -1 padding.
    key_cols = torch.arange(n_bk, device=q.device, dtype=torch.int32).view(1, 1, n_bk)
    packed_row_mask = torch.where(
        block_mask,
        key_cols,
        torch.full((), -1, device=q.device, dtype=torch.int32),
    )
    packed_row_mask = torch.sort(packed_row_mask, dim=-1, descending=True).values.contiguous()

    q_rows = q.permute(0, 2, 1, 3).reshape(B * H, Lq, 1, D)
    k_rows = k.permute(0, 2, 1, 3).reshape(B * H, int(k.shape[1]), 1, D)
    v_rows = v.permute(0, 2, 1, 3).reshape(B * H, int(v.shape[1]), 1, D)
    try:
        out_rows = rows_blocksparse_attention(
            q=q_rows,
            k=k_rows,
            v=v_rows,
            row_blockmask=packed_row_mask,
            block_m=int(block_q),
            block_n=int(block_k),
            softmax_scale=1.0 / math.sqrt(float(D)),
            return_lse=False,
        )
    except Exception:
        return None
    if not isinstance(out_rows, torch.Tensor):
        return None
    return (
        out_rows.reshape(B, H, Lq, D)
        .permute(0, 2, 1, 3)
        .contiguous()
        .to(out_dtype)
    )


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
    B, Lq, H, D = roped_query.shape
    device = roped_query.device
    cfg = kv_sparse_config
    chunk_k = max(256, cfg.chunk_k)
    use_cpu_offload = bool(kv_cache is not None and kv_cache.get("cpu_offload_kv", False))
    fast_lf_rope = bool(
        fast_lf_rope
        or getattr(cfg, "rope_semantics", "absolute")
        == "horizonkv_compacted"
    )
    if use_cpu_offload and getattr(cfg, "selector", "longlive") == "lightforcing":
        raise RuntimeError(
            "selector=lightforcing is implemented for GPU-window "
            "path only; disable CPU KV offload for this baseline"
        )
    reindex_rope = bool(getattr(cfg, "reindex_rope_positions", False))
    if reindex_rope and rope_freqs is None:
        raise ValueError("reindex_rope_positions requires rope_freqs")
    if reindex_rope and cfg.granularity != "kv_frame":
        raise ValueError("reindex_rope_positions currently supports granularity=kv_frame only")
    score_unroped = bool(getattr(cfg, "score_with_unroped_qk", False))
    if score_unroped and cfg.granularity != "kv_frame":
        raise ValueError("score_with_unroped_qk currently supports granularity=kv_frame only")
    score_query = (
        selection_query
        if (score_unroped or (fast_lf_rope and not use_cpu_offload))
        and selection_query is not None
        else roped_query
    )

    if use_cpu_offload:
        if (
            fast_lf_rope
            or getattr(cfg, "rope_semantics", "absolute")
            == "horizonkv_compacted"
        ):
            # Keep K raw in caches; apply compacted RoPE after selection and slot planning.
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
            kv_sparse_config=kv_sparse_config,
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
        )

    Lk = k_window.shape[1]
    assert v_window.shape == k_window.shape
    # Map compulsory frames into the packed [sink, history, current] window.
    packed_window = (
        getattr(cfg, "rope_semantics", "absolute")
        in {
            "horizonkv_compacted",
        }
    )
    selection_kv_start = 0 if packed_window else kv_start
    selection_sink_tokens = min(int(sink_tokens), int(Lk)) if packed_window else sink_tokens
    selection_local_end_index = int(Lk) if packed_window else local_end_index
    selection_current_end = int(Lk) if packed_window else current_end
    selection_current_start = (
        max(0, int(Lk) - int(Lq)) if packed_window else current_start
    )
    if not cfg.layer_has_sparse(layer_idx, H):
        return attention(roped_query, k_window, v_window)

    n_frames = max(1, (Lk + frame_seqlen - 1) // frame_seqlen)
    sink_frames = _sink_frame_indices(
        Lk, selection_kv_start, frame_seqlen, selection_sink_tokens
    )
    if packed_window:
        # Map global chunk boundaries with kv_start - sink_tokens after rollover.
        mandatory_kv_start = int(kv_start) - int(selection_sink_tokens)
        # Use the packed-cache end so mandatory-frame mapping retains rollover displacement.
        mandatory_local_end = int(local_end_index)
        mandatory_current_start = int(current_start)
        mandatory_current_end = int(current_end)
    else:
        mandatory_kv_start = selection_kv_start
        mandatory_local_end = selection_local_end_index
        mandatory_current_start = selection_current_start
        mandatory_current_end = selection_current_end
    forced_frames = _mandatory_kv_frame_indices(
        Lk,
        frame_seqlen,
        mandatory_kv_start,
        mandatory_local_end,
        mandatory_current_start,
        mandatory_current_end,
        num_frame_per_block,
    )
    forced_frames.update(
        _preceding_kv_frame_indices(
            Lk,
            frame_seqlen,
            selection_kv_start,
            selection_local_end_index,
            selection_current_start,
            selection_current_end,
            num_frame_per_block,
            getattr(cfg, "mandatory_preceding_kv_frames", None),
        )
    )
    forced_frames.update(
        _recent_kv_frame_indices(
            Lk,
            frame_seqlen,
            getattr(cfg, "mandatory_recent_kv_frames", None),
        )
    )
    if (
        cfg.granularity == "kv_frame"
        and cfg.rule in ("max_kv_frames", "topk_plus_threshold")
        and cfg.all_heads_max_kv_frames_at_least(layer_idx, range(H), n_frames)
        and not (cfg.visualize_attn_mask and cfg.visualize_output_dir)
    ):
        return attention(roped_query, k_window, v_window)

    sink_empty: set[int] = set()
    compulsory_frames = set(sink_frames) | set(forced_frames)
    selection_key = _frame_selection_plan_key(
        cfg,
        current_start=selection_current_start,
        current_end=selection_current_end,
        local_end_index=selection_local_end_index,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=num_frame_per_block,
        n_frames=n_frames,
    )
    selection_plan = (
        _get_frame_selection_plan(kv_cache, selection_key)
        if kv_cache is not None and bool(getattr(cfg, "reuse_frame_selection", False))
        else None
    )

    # Run the LightForcing selector and kernel on the visible KV window.
    if cfg.selector == "lightforcing":
        if cfg.granularity not in ("attn_block", "kv_frame_then_attn_block"):
            raise ValueError("selector=lightforcing requires granularity=attn_block")
        block_q = int(cfg.attn_block_q)
        block_k = int(cfg.attn_block_k)
        schedule = calculate_chunk_sparsities(
            int(getattr(cfg, "lightforcing_num_output_frames", 120)),
            int(num_frame_per_block),
            int(cfg.lightforcing_local_attn_size),
            cfg.lightforcing_sparsity,
            cfg.lightforcing_sparsity_base,
        )
        query_chunk_frames = max(1, int(Lq // max(1, int(frame_seqlen))))
        chunk_id = int(current_start // max(1, query_chunk_frames * int(frame_seqlen)))
        sparsity = (
            schedule[min(chunk_id, len(schedule) - 1)]
            if schedule else float(cfg.lightforcing_sparsity)
        )
        return lightforcing_attention(
            roped_query,
            k_window,
            v_window,
            sparsity=float(sparsity),
            chunk_id=chunk_id,
            block_q=block_q,
            block_k=block_k,
            frame_seq=int(frame_seqlen),
            keep_frames=int(cfg.lightforcing_keep_frames),
            keep_sink=int(cfg.lightforcing_keep_sink),
            keep_near=int(cfg.lightforcing_keep_near),
            require_sparse=True,
        ).to(out_dtype)

    if cfg.granularity == "attn_block":
        if (
            prefer_lf_triton
            and cfg.rule == "density"
            and all(cfg.get_density(layer_idx, h) >= 1.0 for h in range(H))
        ):
            # LF-RoPE's sp00 row is the dense reference; skip pooling and LUT
            # construction entirely when every head requests density=1.
            return attention(roped_query, k_window, v_window).to(out_dtype)
        bq_sz = int(cfg.attn_block_q)
        bk_sz = int(cfg.attn_block_k)
        q_pool = bq_sz
        k_pool = bk_sz
        q_i, k_i, _ = _prepare_qk_for_importance(roped_query, k_window, q_pool, k_pool)
        bq_sp, bk_sp = 1, 1
        mand_k_p = _mandatory_key_positions_pooled_from_frames(
            Lk=Lk,
            kv_start=selection_kv_start,
            frame_seqlen=frame_seqlen,
            sink_tokens=selection_sink_tokens,
            forced_frames=forced_frames,
            k_pool=k_pool,
        )
        Lq_p, Lk_p = q_i.shape[1], k_i.shape[1]
        n_bq_p = (Lq_p + bq_sp - 1) // bq_sp
        n_bk_p = (Lk_p + bk_sp - 1) // bk_sp
        n_blocks = n_bq_p * n_bk_p
        q_blocks_per_frame = max(1, int(math.ceil(frame_seqlen / max(1, q_pool))))
        k_blocks_per_frame = max(1, int(math.ceil(frame_seqlen / max(1, k_pool))))
        query_frame_count = max(1, int(math.ceil(Lq / frame_seqlen)))
        bm_all = _per_block_attn_mass_batched(q_i, k_i, chunk_k, bq_sp, bk_sp)

        # Batch uniform density heads; retain per-head fallback for other configurations.
        fast_density_path = (
            cfg.rule == "density"
            and not (cfg.visualize_attn_mask and cfg.visualize_output_dir)
            and not cfg.profile_stage2
            and all(not _head_is_dense(cfg, layer_idx, h) for h in range(H))
            and all(
                cfg.get_block_compute_backend(layer_idx, h) == "block_sparse_attention"
                for h in range(H)
            )
        )
        if fast_density_path:
            density_by_head = [cfg.get_density(layer_idx, h) for h in range(H)]
            forced_lin_device = _forced_attn_block_linear_indices_device(
                n_bq_p,
                n_bk_p,
                mand_k_p,
                device,
            )
            block_mask_group = _select_density_block_mask_batched(
                bm_all,
                n_bq=n_bq_p,
                n_bk=n_bk_p,
                densities=density_by_head,
                compulsory_linear=forced_lin_device,
                forced_outside_budget=cfg.forced_outside_budget,
                compulsory_is_unique=True,
            )
            fast_out = _stage2_rows_blocksparse_attention_if_possible(
                q=roped_query,
                k=k_window,
                v=v_window,
                block_mask=block_mask_group,
                block_q=bq_sz,
                block_k=bk_sz,
                out_dtype=out_dtype,
                prefer_lf_triton=prefer_lf_triton,
            )
            if fast_out is not None:
                return fast_out

        # Build Python sets only for visualization and fallback paths.
        forced_lin = _forced_attn_block_linear_indices(
            Lq_p,
            Lk_p,
            bq_sp,
            bk_sp,
            mand_k_p,
        )

        out = torch.empty((B, Lq, H, D), device=device, dtype=out_dtype)
        dense_heads: list[int] = []
        sparse_heads_by_backend: dict[str, list[int]] = {}
        selected_lins_by_head: dict[int, list[int]] = {}
        block_masks_by_head: dict[int, torch.Tensor] = {}

        for h in range(H):
            if _head_is_dense(cfg, layer_idx, h):
                dense_heads.append(int(h))
                continue

            sel_lin = _apply_rule_to_mass_1d(
                cfg,
                layer_idx,
                h,
                bm_all[h],
                n_blocks,
                sink_empty,
                forced_lin,
                cfg.forced_outside_budget,
            )
            selected_lins_by_head[int(h)] = [int(v) for v in sel_lin]
            block_mask = _block_mask_from_selected_linear_indices(
                n_bq_p,
                n_bk_p,
                sel_lin,
                device,
            )
            block_masks_by_head[int(h)] = block_mask
            if not sel_lin:
                out[:, :, h : h + 1, :] = 0
                continue
            backend = cfg.get_block_compute_backend(layer_idx, h)
            sparse_heads_by_backend.setdefault(backend, []).append(int(h))

        if dense_heads:
            dense_idx = torch.tensor(dense_heads, device=device, dtype=torch.long)
            out[:, :, dense_idx, :] = attention(
                roped_query.index_select(2, dense_idx),
                k_window.index_select(2, dense_idx),
                v_window.index_select(2, dense_idx),
            ).to(out_dtype)

        for backend, heads in sparse_heads_by_backend.items():
            if backend != "block_sparse_attention":
                _require_stage2_blocksparse_backend(
                    f"unexpected backend '{backend}' for attn_block granularity"
                )
            head_idx = torch.tensor(heads, device=device, dtype=torch.long)
            q_group = roped_query.index_select(2, head_idx)
            k_group = k_window.index_select(2, head_idx)
            v_group = v_window.index_select(2, head_idx)
            block_mask_group = torch.stack(
                [block_masks_by_head[int(h)] for h in heads],
                dim=0,
            )

            ok, reason = stage2_blocksparse_kernel_supported(
                q=q_group,
                k=k_group,
                block_q=bq_sz,
                block_k=bk_sz,
            )
            if not ok:
                _require_stage2_blocksparse_backend(
                    f"layer={layer_idx} heads={heads} unsupported_shape: {reason}"
                )
            try:
                out[:, :, head_idx, :] = stage2_blocksparse_attention(
                    q_group,
                    k_group,
                    v_group,
                    block_mask=block_mask_group,
                    block_q=bq_sz,
                    block_k=bk_sz,
                ).to(out_dtype)
            except Exception as exc:
                _require_stage2_blocksparse_backend(
                    f"layer={layer_idx} heads={heads} kernel_runtime_error={type(exc).__name__}: {exc}"
                )

        for h in range(H):
            if not _should_visualize_mask(cfg, layer_idx, h, current_timestep):
                continue
            if _head_is_dense(cfg, layer_idx, h):
                block_mask = torch.ones((n_bq_p, n_bk_p), device=device, dtype=torch.bool)
                metadata = {
                    "is_dense_head": True,
                    "mask_view": "query_blocks x key_blocks",
                    "x_axis": "key_frame",
                    "y_axis": "query_frame",
                    "x_major_count": int(n_frames),
                    "y_major_count": int(query_frame_count),
                    "x_units_per_major": int(k_blocks_per_frame),
                    "y_units_per_major": int(q_blocks_per_frame),
                    "pixels_per_unit": 4,
                    "query_block_count": int(n_bq_p),
                    "key_block_count": int(n_bk_p),
                    "attn_block_q": int(bq_sz),
                    "attn_block_k": int(bk_sz),
                    "q_pool": int(q_pool),
                    "k_pool": int(k_pool),
                    "Lq": int(Lq),
                    "Lk": int(Lk),
                    "frame_seqlen": int(frame_seqlen),
                    "selected_item_count": int(n_bq_p * n_bk_p),
                }
            else:
                sel_lin = selected_lins_by_head.get(int(h), [])
                block_mask = block_masks_by_head.get(
                    int(h),
                    torch.zeros((n_bq_p, n_bk_p), device=device, dtype=torch.bool),
                )
                block_pairs = [[int(v // n_bk_p), int(v % n_bk_p)] for v in sel_lin]
                metadata = {
                    "is_dense_head": False,
                    "mask_view": "query_blocks x key_blocks",
                    "x_axis": "key_frame",
                    "y_axis": "query_frame",
                    "x_major_count": int(n_frames),
                    "y_major_count": int(query_frame_count),
                    "x_units_per_major": int(k_blocks_per_frame),
                    "y_units_per_major": int(q_blocks_per_frame),
                    "pixels_per_unit": 4,
                    "query_block_count": int(n_bq_p),
                    "key_block_count": int(n_bk_p),
                    "attn_block_q": int(bq_sz),
                    "attn_block_k": int(bk_sz),
                    "q_pool": int(q_pool),
                    "k_pool": int(k_pool),
                    "Lq": int(Lq),
                    "Lk": int(Lk),
                    "frame_seqlen": int(frame_seqlen),
                    "requested_density": float(cfg.get_density(layer_idx, h)),
                    "requested_max_kv_frames": cfg.get_max_kv_frames(layer_idx, h),
                    "forced_linear_indices": sorted(int(v) for v in forced_lin),
                    "selected_linear_indices": [int(v) for v in sel_lin],
                    "selected_additional_linear_indices": _extra_selected_indices(sel_lin, forced_lin),
                    "forced_linear_count": int(len(forced_lin)),
                    "selected_linear_count": int(len(sel_lin)),
                    "selected_block_indices": block_pairs,
                    "selected_item_count": int(len(sel_lin)),
                }
            _dump_selection_visualization(
                cfg,
                granularity="attn_block",
                layer_idx=layer_idx,
                head_idx=h,
                timestep=current_timestep,
                current_start=selection_current_start,
                current_end=selection_current_end,
                kv_start=selection_kv_start,
                mask_2d=block_mask,
                metadata=metadata,
            )
        return out

    profile_score_t0 = _profile_now(cfg, device) if cfg.profile_frame_selection else 0.0
    fm_all = _frame_importance_scores(
        cfg,
        score_query,
        k_window,
        frame_seqlen,
        chunk_k,
        frame_k_index_window=frame_k_index_window,
        frame_k_min_index_window=frame_k_min_index_window,
        frame_k_max_index_window=frame_k_max_index_window,
    )
    profile_score_ms = 0.0
    if cfg.profile_frame_selection:
        profile_score_ms = (_profile_now(cfg, device) - profile_score_t0) * 1000.0

    chunk_query_frames = max(1, int(num_frame_per_block))

    if cfg.granularity == "kv_frame":
        dense_mask = torch.tensor(
            [_head_is_dense(cfg, layer_idx, h) for h in range(H)],
            device=device,
            dtype=torch.bool,
        )
        out = torch.empty((B, Lq, H, D), device=device, dtype=out_dtype)
        need_visualization = cfg.visualize_attn_mask and cfg.visualize_output_dir

        dense_heads = dense_mask.nonzero(as_tuple=False).flatten()
        if dense_heads.numel() > 0:
            out[:, :, dense_heads, :] = attention(
                roped_query[:, :, dense_heads, :],
                k_window[:, :, dense_heads, :],
                v_window[:, :, dense_heads, :],
            ).to(out_dtype)

        sparse_heads = (~dense_mask).nonzero(as_tuple=False).flatten()
        if sparse_heads.numel() == 0:
            return out
        cached_selected_by_head = _selection_plan_head_frames(
            selection_plan,
            sparse_heads,
            n_frames,
        )
        shared_max_kv_frames = cfg.shared_max_kv_frames(layer_idx, sparse_heads.tolist())

        use_batched_max_kv_frames = (
            cfg.rule == "max_kv_frames"
            and not need_visualization
            and Lk % frame_seqlen == 0
            and (cached_selected_by_head is not None or shared_max_kv_frames is not None)
        )
        if use_batched_max_kv_frames:
            selected_frame_idx: torch.Tensor | None = None
            profile_rule_t0 = _profile_now(cfg, device) if cfg.profile_frame_selection else 0.0
            if cached_selected_by_head is not None:
                cached_rows = [
                    [int(v) for v in cached_selected_by_head[int(h)]]
                    for h in sparse_heads.tolist()
                ]
                cached_lens = {len(row) for row in cached_rows}
                if len(cached_lens) == 1:
                    selected_frame_idx = torch.tensor(cached_rows, device=device, dtype=torch.long)
                else:
                    use_batched_max_kv_frames = False
            else:
                score_rows = fm_all.index_select(0, sparse_heads)
                if cfg.frame_selection_strategy == "weighted_random":
                    selected_frame_idx = _select_kv_frame_indices_weighted_random_batched(
                        score_rows,
                        int(shared_max_kv_frames),
                        n_frames,
                        sink_frames,
                        forced_frames,
                    )
                else:
                    selected_frame_idx = _select_kv_frame_indices_max_kv_frames_batched(
                        score_rows,
                        int(shared_max_kv_frames),
                        n_frames,
                        sink_frames,
                        forced_frames,
                        cfg.forced_outside_budget,
                    )
                if kv_cache is not None and bool(getattr(cfg, "reuse_frame_selection", False)):
                    _store_frame_selection_plan(
                        kv_cache,
                        selection_key,
                        selected_by_head={
                            int(h): [int(v) for v in selected_frame_idx[row].detach().cpu().tolist()]
                            for row, h in enumerate(sparse_heads.tolist())
                        },
                        n_frames=n_frames,
                        Lk=Lk,
                        granularity=cfg.granularity,
                    )
            profile_rule_ms = 0.0
            if cfg.profile_frame_selection:
                profile_rule_ms = (_profile_now(cfg, device) - profile_rule_t0) * 1000.0

            if use_batched_max_kv_frames and selected_frame_idx is not None:
                if selected_frame_idx.shape[1] == 0:
                    out[:, :, sparse_heads, :] = 0
                else:
                    head_count = int(sparse_heads.numel())
                    q_group = roped_query.index_select(2, sparse_heads)
                    selected_frame_idx_for_attn = (
                        torch.sort(selected_frame_idx, dim=1).values
                        if reindex_rope
                        else selected_frame_idx
                    )
                    k_compact, v_compact = _compact_kv_from_frame_indices_batched(
                        k_window,
                        v_window,
                        sparse_heads,
                        selected_frame_idx_for_attn,
                        frame_seqlen,
                    )
                    if reindex_rope:
                        q_group, k_compact = reindex_compacted_qk_rope(
                            q_group,
                            k_compact,
                            rope_freqs,
                            selected_frame_idx_for_attn,
                            frame_seqlen=frame_seqlen,
                            kv_start=selection_kv_start,
                            local_end_index=selection_local_end_index,
                            current_start=selection_current_start,
                            current_end=selection_current_end,
                            sink_tokens=selection_sink_tokens,
                        )
                    q_group = q_group.permute(0, 2, 1, 3).reshape(B * head_count, Lq, 1, D)

                    out_sparse = attention(q_group, k_compact, v_compact).to(out_dtype)
                    out_sparse = out_sparse.reshape(B, head_count, Lq, D).permute(0, 2, 1, 3)
                    out[:, :, sparse_heads, :] = out_sparse

                if cfg.profile_frame_selection:
                    _profile_frame_selection_record(
                        cfg,
                        layer_idx=layer_idx,
                        timestep=current_timestep,
                        Lq=Lq,
                        Lk=Lk,
                        n_frames=n_frames,
                        score_ms=profile_score_ms,
                        rule_ms=profile_rule_ms,
                        selected_counts=[int(selected_frame_idx.shape[1])] * int(sparse_heads.numel()),
                    )
                return out

        selected_indices_per_sparse_head: dict[int, list[int]] = {}
        selected_mask = (
            torch.zeros(sparse_heads.numel(), n_frames, device=device, dtype=torch.bool)
            if need_visualization
            else None
        )
        frame_groups: dict[tuple[int, ...], list[int]] = {}
        profile_rule_t0 = _profile_now(cfg, device) if cfg.profile_frame_selection else 0.0
        profile_selected_counts: list[int] = []
        selected_by_head_for_plan: dict[int, list[int]] = {}
        for row, h in enumerate(sparse_heads.tolist()):
            if cached_selected_by_head is not None:
                frames_h = [int(v) for v in cached_selected_by_head[int(h)]]
            else:
                if cfg.frame_selection_strategy == "weighted_random":
                    frames_h = _select_kv_frame_indices_weighted_random_batched(
                        fm_all[h : h + 1],
                        int(cfg.get_max_kv_frames(layer_idx, h) or 0),
                        n_frames,
                        sink_frames,
                        forced_frames,
                    )[0].tolist()
                else:
                    frames_h = _apply_rule_to_mass_1d(
                        cfg,
                        layer_idx,
                        h,
                        fm_all[h],
                        n_frames,
                        sink_frames,
                        forced_frames,
                        cfg.forced_outside_budget,
                    )
                selected_by_head_for_plan[int(h)] = [int(v) for v in frames_h]
            if cfg.profile_frame_selection:
                profile_selected_counts.append(len(frames_h))
            if not frames_h:
                out[:, :, h : h + 1, :] = 0
                continue
            selected_indices_per_sparse_head[h] = [int(v) for v in frames_h]
            if selected_mask is not None:
                idx = torch.tensor(frames_h, device=device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < n_frames)]
                if idx.numel() > 0:
                    selected_mask[row, idx] = True
            frame_groups.setdefault(tuple(int(v) for v in frames_h), []).append(int(h))

        if (
            cached_selected_by_head is None
            and kv_cache is not None
            and bool(getattr(cfg, "reuse_frame_selection", False))
        ):
            _store_frame_selection_plan(
                kv_cache,
                selection_key,
                selected_by_head=selected_by_head_for_plan,
                n_frames=n_frames,
                Lk=Lk,
                granularity=cfg.granularity,
            )

        profile_rule_ms = 0.0
        if cfg.profile_frame_selection:
            profile_rule_ms = (_profile_now(cfg, device) - profile_rule_t0) * 1000.0

        for frames_key, heads in frame_groups.items():
            head_idx = torch.tensor(heads, device=device, dtype=torch.long)
            q_group = roped_query.index_select(2, head_idx)
            selected_frame_idx = torch.tensor(frames_key, device=device, dtype=torch.long)
            if selected_frame_idx.numel() == 0 or (
                selected_frame_idx.numel() >= n_frames and not reindex_rope
            ):
                out[:, :, head_idx, :] = attention(
                    q_group,
                    k_window.index_select(2, head_idx),
                    v_window.index_select(2, head_idx),
                ).to(out_dtype)
                continue

            if Lk % frame_seqlen == 0:
                selected_frame_idx_for_attn = (
                    torch.sort(selected_frame_idx, dim=0).values
                    if reindex_rope
                    else selected_frame_idx
                )
                k_compact, v_compact = _compact_kv_from_frame_indices_batched(
                    k_window,
                    v_window,
                    head_idx,
                    selected_frame_idx_for_attn,
                    frame_seqlen,
                )
                if reindex_rope:
                    q_group, k_compact = reindex_compacted_qk_rope(
                        q_group,
                        k_compact,
                        rope_freqs,
                        selected_frame_idx_for_attn,
                        frame_seqlen=frame_seqlen,
                        kv_start=selection_kv_start,
                        local_end_index=selection_local_end_index,
                        current_start=selection_current_start,
                        current_end=selection_current_end,
                        sink_tokens=selection_sink_tokens,
                    )
                q_attn = q_group.permute(0, 2, 1, 3).reshape(B * len(heads), Lq, 1, D)
                out_group = attention(q_attn, k_compact, v_compact).to(out_dtype)
                out_group = out_group.reshape(B, len(heads), Lq, D).permute(0, 2, 1, 3)
            else:
                if reindex_rope:
                    raise ValueError("reindex_rope_positions requires frame-packed KV windows")
                selected_token_idx, _ = _key_token_indices_from_frames(
                    Lk, n_frames, frame_seqlen, set(frames_key), device
                )
                if selected_token_idx.numel() * H <= Lk * len(heads):
                    k_compact = k_window.index_select(1, selected_token_idx).index_select(2, head_idx)
                    v_compact = v_window.index_select(1, selected_token_idx).index_select(2, head_idx)
                else:
                    k_compact = k_window.index_select(2, head_idx).index_select(1, selected_token_idx)
                    v_compact = v_window.index_select(2, head_idx).index_select(1, selected_token_idx)
                out_group = attention(
                    q_group,
                    k_compact,
                    v_compact,
                ).to(out_dtype)
            out[:, :, head_idx, :] = out_group

        if cfg.profile_frame_selection:
            _profile_frame_selection_record(
                cfg,
                layer_idx=layer_idx,
                timestep=current_timestep,
                Lq=Lq,
                Lk=Lk,
                n_frames=n_frames,
                score_ms=profile_score_ms,
                rule_ms=profile_rule_ms,
                selected_counts=profile_selected_counts,
            )

        if need_visualization:
            sparse_head_to_row = {int(h): int(row) for row, h in enumerate(sparse_heads.tolist())}
            for h in range(H):
                if not _should_visualize_mask(cfg, layer_idx, h, current_timestep):
                    continue
                if int(h) in sparse_head_to_row:
                    frame_row = selected_mask[sparse_head_to_row[int(h)]].clone()
                    selected_frames = selected_indices_per_sparse_head.get(int(h), [])
                    is_dense_head = False
                else:
                    frame_row = torch.ones(n_frames, device=device, dtype=torch.bool)
                    selected_frames = list(range(n_frames))
                    is_dense_head = True
                frame_mask = frame_row.unsqueeze(0).expand(chunk_query_frames, -1).contiguous()
                selected_extra_frames = _extra_selected_indices(selected_frames, compulsory_frames)
                _dump_selection_visualization(
                    cfg,
                    granularity="kv_frame",
                    layer_idx=layer_idx,
                    head_idx=h,
                    timestep=current_timestep,
                    current_start=selection_current_start,
                    current_end=selection_current_end,
                    kv_start=selection_kv_start,
                    mask_2d=frame_mask,
                    metadata={
                        "is_dense_head": is_dense_head,
                        "mask_view": "query_frames_in_chunk x key_frames",
                        "x_axis": "key_frame",
                        "y_axis": "query_frame_in_chunk",
                        "x_major_count": int(n_frames),
                        "y_major_count": int(chunk_query_frames),
                        "x_units_per_major": 1,
                        "y_units_per_major": 1,
                        "pixels_per_unit": 18,
                        "frame_selection_score": cfg.frame_selection_score,
                        "frame_seqlen": int(frame_seqlen),
                        "frame_count": int(n_frames),
                        "chunk_query_frame_count": int(chunk_query_frames),
                        "Lq": int(Lq),
                        "Lk": int(Lk),
                        "requested_density": float(cfg.get_density(layer_idx, h)),
                        "requested_max_kv_frames": cfg.get_max_kv_frames(layer_idx, h),
                        "sink_frames": sorted(int(v) for v in sink_frames),
                        "forced_frames": sorted(int(v) for v in forced_frames),
                        "compulsory_frames": sorted(int(v) for v in compulsory_frames),
                        "selected_frames": [int(v) for v in selected_frames],
                        "selected_additional_frames": selected_extra_frames,
                        "sink_frame_count": int(len(sink_frames)),
                        "forced_frame_count": int(len(forced_frames)),
                        "compulsory_frame_count": int(len(compulsory_frames)),
                        "remaining_frame_count": int(max(0, n_frames - len(compulsory_frames))),
                        "selected_frame_count": int(len(selected_frames)),
                        "selected_additional_count": int(len(selected_extra_frames)),
                        "selected_item_count": int(len(selected_frames)),
                    },
                )
        return out

    if cfg.granularity != "kv_frame_then_attn_block":
        raise ValueError(f"unsupported granularity: {cfg.granularity}")

    bq_sz = int(cfg.attn_block_q)
    bk_sz = int(cfg.attn_block_k)
    q_pool = bq_sz
    k_pool = bk_sz
    # Raw Q supplies geometry here; apply RoPE after frame selection.
    q_i_all = _avg_pool_tokens_along_seq(roped_query, q_pool) if q_pool > 1 else roped_query
    Lq_p = q_i_all.shape[1]
    q_blocks_per_frame = max(1, int(math.ceil(frame_seqlen / max(1, q_pool))))
    k_blocks_per_frame = max(1, int(math.ceil(frame_seqlen / max(1, k_pool))))
    query_frame_count = max(1, int(math.ceil(Lq / frame_seqlen)))

    out = torch.empty((B, Lq, H, D), device=device, dtype=out_dtype)
    need_visualization = cfg.visualize_attn_mask and cfg.visualize_output_dir
    frame_sparse_heads = [h for h in range(H) if not _head_is_dense(cfg, layer_idx, h)]
    cached_selected_by_head = (
        _selection_plan_head_frames(selection_plan, frame_sparse_heads, n_frames)
        if frame_sparse_heads
        else {}
    )
    selected_by_head_for_plan: dict[int, list[int]] = {}
    head_infos: dict[int, dict[str, Any]] = {}
    grouped_heads: dict[tuple[Any, ...], list[int]] = {}

    for h in range(H):
        frame_dense = _head_is_dense(cfg, layer_idx, h)
        block_dense = _block_density_is_dense(cfg, layer_idx, h)
        if frame_dense:
            selected_frames = list(range(n_frames))
        elif cached_selected_by_head is not None:
            selected_frames = [int(v) for v in cached_selected_by_head[int(h)]]
        else:
            selected_frames = _apply_rule_to_mass_1d(
                cfg,
                layer_idx,
                h,
                fm_all[h],
                n_frames,
                sink_frames,
                forced_frames,
                cfg.forced_outside_budget,
            )
        selected_frames = [int(v) for v in selected_frames]
        selected_by_head_for_plan[int(h)] = selected_frames
        selected_frame_set = {int(v) for v in selected_frames if 0 <= int(v) < n_frames}
        dense_frame_set = (
            set(compulsory_frames) & selected_frame_set if cfg.block_force_dense_compulsory_frames else set()
        )
        sparse_frame_set = selected_frame_set - dense_frame_set
        head_infos[int(h)] = {
            "frame_dense": bool(frame_dense),
            "block_dense": bool(block_dense),
            "selected_frames": selected_frames,
            "selected_frame_set": selected_frame_set,
            "dense_frame_set": dense_frame_set,
            "sparse_frame_set": sparse_frame_set,
        }
        if not selected_frame_set:
            out[:, :, h : h + 1, :] = attention(
                roped_query[:, :, h : h + 1, :],
                k_window[:, :, h : h + 1, :],
                v_window[:, :, h : h + 1, :],
            ).to(out_dtype)
            continue
        group_key = (
            bool(frame_dense),
            bool(block_dense),
            len(selected_frames),
            cfg.get_max_kv_frames(layer_idx, h),
            cfg.get_block_rule(layer_idx, h),
            round(float(cfg.get_block_density(layer_idx, h)), 8),
            None if cfg.get_block_recall(layer_idx, h) is None else round(float(cfg.get_block_recall(layer_idx, h)), 8),
            cfg.get_block_density_allocation(layer_idx, h),
            cfg.get_block_compute_backend(layer_idx, h),
            bool(cfg.block_force_dense_compulsory_frames),
        )
        grouped_heads.setdefault(group_key, []).append(int(h))

    if (
        cached_selected_by_head is None
        and kv_cache is not None
        and bool(getattr(cfg, "reuse_frame_selection", False))
    ):
        _store_frame_selection_plan(
            kv_cache,
            selection_key,
            selected_by_head=selected_by_head_for_plan,
            n_frames=n_frames,
            Lk=Lk,
            granularity=cfg.granularity,
        )

    if Lk % frame_seqlen != 0:
        raise ValueError("two-level grouped path currently requires Lk to be divisible by frame_seqlen")

    n_bq_p = int(Lq_p)
    for group_key, heads in grouped_heads.items():
        frame_dense = bool(group_key[0])
        block_dense = bool(group_key[1])
        head_count = int(len(heads))
        head_idx = torch.tensor(heads, device=device, dtype=torch.long)
        q_group = roped_query.index_select(2, head_idx)
        selected_frame_rows = [head_infos[int(h)]["selected_frames"] for h in heads]
        selected_frame_idx = torch.tensor(
            selected_frame_rows,
            device=device,
            dtype=torch.long,
        )

        stage2_compact_ms = 0.0
        stage2_block_mass_ms = 0.0
        stage2_block_select_ms = 0.0
        stage2_mask_build_ms = 0.0
        stage2_kernel_ms = 0.0

        stage2_compact_t0 = _profile_now(cfg, device) if cfg.profile_stage2 else 0.0
        k_compact_rows, v_compact_rows = _compact_kv_from_frame_indices_batched(
            k_window,
            v_window,
            head_idx,
            selected_frame_idx,
            frame_seqlen,
        )
        if cfg.profile_stage2:
            stage2_compact_ms = (_profile_now(cfg, device) - stage2_compact_t0) * 1000.0

        token_count = int(k_compact_rows.shape[1])
        k_compact = (
            k_compact_rows.view(B, head_count, token_count, 1, D)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, token_count, head_count, D)
            .contiguous()
        )
        v_compact = (
            v_compact_rows.view(B, head_count, token_count, 1, D)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, token_count, head_count, D)
            .contiguous()
        )

        compact_frame_count = int(selected_frame_idx.shape[1])
        if fast_lf_rope and not use_cpu_offload:
            if rope_grid_sizes is None or rope_freqs is None:
                raise ValueError(
                    "Local fast LF-RoPE requires rope_grid_sizes and rope_freqs"
                )
            query_frames = max(1, int(math.ceil(Lq / float(frame_seqlen))))
            current_start_frame = int(current_start) // int(frame_seqlen)
            lf_k_start_frame = max(
                0,
                current_start_frame - max(0, compact_frame_count - query_frames),
            )
            q_group = lightforcing_rope_apply(
                q_group,
                rope_grid_sizes,
                rope_freqs,
                start_frame=current_start_frame,
            ).type_as(v_window)
        if fast_lf_rope and not use_cpu_offload:
            # Apply compacted RoPE to selected frames, matching the offload materializer.
            grid_sizes_compact = rope_grid_sizes.detach().clone()
            grid_sizes_compact[:, 0] = int(compact_frame_count)
            k_compact = lightforcing_rope_apply(
                k_compact,
                grid_sizes_compact,
                rope_freqs,
                start_frame=lf_k_start_frame,
            ).type_as(v_compact)

        Lk_compact = int(k_compact.shape[1])
        Lk_p = int(math.ceil(Lk_compact / float(max(1, k_pool))))
        n_bk_p = int(Lk_p)
        pooled_key_frames = (
            (torch.arange(n_bk_p, device=device, dtype=torch.long) * k_pool) // frame_seqlen
        ).clamp_max(max(0, compact_frame_count - 1))

        if frame_dense and block_dense:
            out[:, :, head_idx, :] = attention(q_group, k_compact, v_compact).to(out_dtype)
            block_mask_group = torch.ones((head_count, n_bq_p, n_bk_p), device=device, dtype=torch.bool)
            selected_lins_by_head = {
                int(h): torch.empty(0, device=device, dtype=torch.long) for h in heads
            }
            dense_frame_sets_compact_by_head = {
                int(h): set(range(compact_frame_count)) for h in heads
            }
            block_budget_by_head = {int(h): {} for h in heads}
            backend_name = "dense"
        elif block_dense:
            out[:, :, head_idx, :] = attention(q_group, k_compact, v_compact).to(out_dtype)
            block_mask_group = torch.ones((head_count, n_bq_p, n_bk_p), device=device, dtype=torch.bool)
            selected_lins_by_head = {
                int(h): torch.empty(0, device=device, dtype=torch.long) for h in heads
            }
            dense_frame_sets_compact_by_head = {int(h): set() for h in heads}
            block_budget_by_head = {int(h): {} for h in heads}
            backend_name = "dense"
        else:
            q_i_group = (
                _avg_pool_tokens_along_seq(q_group, q_pool)
                if q_pool > 1
                else q_group
            )
            k_i_group = _avg_pool_tokens_along_seq(k_compact, k_pool) if k_pool > 1 else k_compact
            stage2_block_mass_t0 = _profile_now(cfg, device) if cfg.profile_stage2 else 0.0
            if fast_lf_rope and not use_cpu_offload:
                # Rank blocks with the same post-RoPE pooled-QK scores as the offload path.
                bm_group = _per_block_pooled_qk_scores_batched(
                    q_i_group,
                    k_i_group,
                )
            else:
                bm_group = _per_block_attn_mass_batched(
                    q_i_group,
                    k_i_group,
                    chunk_k,
                    1,
                    1,
                )
            if cfg.profile_stage2:
                stage2_block_mass_ms = (_profile_now(cfg, device) - stage2_block_mass_t0) * 1000.0

            block_rule = str(group_key[4])
            block_density = float(group_key[5])
            block_recall = group_key[6]
            block_density_allocation = str(group_key[7])
            block_backend = str(group_key[8])
            effective_block_density = _effective_block_density_for_selected_frames(
                cfg,
                layer_idx,
                int(heads[0]),
                block_density,
                compact_frame_count,
            )
            selected_lins_by_head: dict[int, torch.Tensor] = {}
            dense_frame_sets_compact_by_head: dict[int, set[int]] = {}
            block_budget_by_head: dict[int, dict[int, int]] = {}

            stage2_block_select_t0 = _profile_now(cfg, device) if cfg.profile_stage2 else 0.0
            selected_frame_set_compact = set(range(compact_frame_count))
            dense_frame_masks = torch.zeros((head_count, compact_frame_count), device=device, dtype=torch.bool)
            candidate_k_mask_group = torch.ones((head_count, n_bk_p), device=device, dtype=torch.bool)
            dense_k_mask_group = torch.zeros((head_count, n_bk_p), device=device, dtype=torch.bool)
            selected_frame_scores_group = (
                fm_all.index_select(0, head_idx).gather(1, selected_frame_idx).float()
                if fm_all is not None
                else None
            )

            can_use_grouped_global_density = (
                block_rule != "recall"
                and block_density_allocation == "global"
                and not cfg.block_force_dense_compulsory_frames
            )

            for row, h in enumerate(heads):
                compact_source_frames = [int(v) for v in head_infos[int(h)]["selected_frames"]]
                compact_frame_index = {int(f): i for i, f in enumerate(compact_source_frames)}
                dense_frame_set_compact = {
                    compact_frame_index[int(f)]
                    for f in head_infos[int(h)]["dense_frame_set"]
                    if int(f) in compact_frame_index
                }
                dense_frame_sets_compact_by_head[int(h)] = dense_frame_set_compact
                if dense_frame_set_compact:
                    dense_idx = torch.tensor(
                        sorted(dense_frame_set_compact),
                        device=device,
                        dtype=torch.long,
                    )
                    dense_frame_masks[row, dense_idx] = True
                    dense_k = _pooled_key_block_mask_from_frames(
                        Lk_p,
                        compact_frame_count,
                        frame_seqlen,
                        k_pool,
                        dense_frame_set_compact,
                        device,
                    )
                    if dense_k.any():
                        dense_k_mask_group[row] = dense_k
                        if cfg.block_force_dense_compulsory_frames:
                            candidate_k_mask_group[row] &= ~dense_k
                if can_use_grouped_global_density:
                    block_budget_by_head[int(h)] = {}

            use_fast_local_per_q_density = (
                fast_lf_rope
                and not use_cpu_offload
                and block_rule != "recall"
                and not cfg.block_force_dense_compulsory_frames
            )

            if use_fast_local_per_q_density:
                # Assign each Q block its own K-block quota, matching the offload selector.
                keep_ratio = max(0.0, min(1.0, float(effective_block_density)))
                keep_blocks = max(
                    1,
                    min(n_bk_p, int(math.ceil(keep_ratio * float(n_bk_p)))),
                )
                score_3d = bm_group.view(head_count, n_bq_p, n_bk_p)
                selected_cols = torch.topk(
                    score_3d,
                    k=keep_blocks,
                    dim=2,
                    largest=True,
                    sorted=False,
                ).indices
                block_mask_group = torch.zeros(
                    (head_count, n_bq_p, n_bk_p),
                    device=device,
                    dtype=torch.bool,
                )
                block_mask_group.scatter_(2, selected_cols, True)
                q_block_ids = torch.arange(
                    n_bq_p,
                    device=device,
                    dtype=torch.long,
                ).view(1, n_bq_p, 1)
                selected_lin_group = (
                    q_block_ids * n_bk_p + selected_cols
                ).reshape(head_count, -1)
                for row, h in enumerate(heads):
                    selected_lins_by_head[int(h)] = selected_lin_group[row]
            elif can_use_grouped_global_density:
                selected_lin_group = _select_candidate_blocks_by_density_batched(
                    bm_group,
                    n_bq_p,
                    n_bk_p,
                    candidate_k_mask_group,
                    effective_block_density,
                )
                block_mask_group = _block_mask_from_selected_linear_indices_batched(
                    n_bq_p,
                    n_bk_p,
                    selected_lin_group,
                    device,
                )
                if dense_k_mask_group.any():
                    block_mask_group |= dense_k_mask_group.unsqueeze(1)
                for row, h in enumerate(heads):
                    selected_lin = selected_lin_group[row][selected_lin_group[row] >= 0]
                    selected_lins_by_head[int(h)] = selected_lin
            else:
                block_masks_by_head: list[torch.Tensor] = []
                for row, h in enumerate(heads):
                    dense_frame_set_compact = dense_frame_sets_compact_by_head[int(h)]
                    sparse_frame_set_compact = selected_frame_set_compact - dense_frame_set_compact
                    candidate_frame_set_compact = (
                        sparse_frame_set_compact
                        if cfg.block_force_dense_compulsory_frames
                        else selected_frame_set_compact
                    )
                    preselected_frame_set_compact = (
                        dense_frame_set_compact if cfg.block_force_dense_compulsory_frames else None
                    )
                    block_budget_by_compact_frame: dict[int, int] = {}
                    selected_frame_scores_compact = (
                        selected_frame_scores_group[row]
                        if selected_frame_scores_group is not None
                        else None
                    )
                    if block_rule == "recall":
                        if block_recall is None:
                            raise ValueError("block_target_recall required when block_rule=recall")
                        raw_selected_lin = _select_candidate_blocks_by_frame_recall(
                            bm_group[row],
                            n_bq_p,
                            n_bk_p,
                            candidate_frame_set_compact,
                            pooled_key_frames,
                            float(block_recall),
                            denominator_frames=selected_frame_set_compact,
                            preselected_frames=preselected_frame_set_compact,
                        )
                    else:
                        if (
                            block_density_allocation == "frame_score_weighted"
                            and selected_frame_scores_compact is not None
                        ):
                            raw_selected_lin, block_budget_by_compact_frame = (
                                _select_candidate_blocks_by_frame_density_weighted(
                                    bm_group[row],
                                    n_bq_p,
                                    n_bk_p,
                                    candidate_frame_set_compact,
                                    pooled_key_frames,
                                    effective_block_density,
                                    selected_frame_scores_compact,
                                )
                            )
                        else:
                            raw_selected_lin = _select_candidate_blocks_by_frame_density(
                                bm_group[row],
                                n_bq_p,
                                n_bk_p,
                                candidate_frame_set_compact,
                                pooled_key_frames,
                                effective_block_density,
                            )
                    selected_lin = torch.unique(raw_selected_lin.to(dtype=torch.long), sorted=False)
                    selected_lins_by_head[int(h)] = selected_lin
                    block_budget_by_head[int(h)] = block_budget_by_compact_frame
                    block_mask_2d = _block_mask_from_selected_linear_indices(
                        n_bq_p,
                        n_bk_p,
                        selected_lin,
                        device,
                    )
                    if dense_frame_set_compact:
                        dense_k = dense_k_mask_group[row]
                        if dense_k.any():
                            block_mask_2d[:, dense_k] = True
                    block_masks_by_head.append(block_mask_2d)
                block_mask_group = torch.stack(block_masks_by_head, dim=0)
            if cfg.profile_stage2:
                stage2_block_select_ms = (_profile_now(cfg, device) - stage2_block_select_t0) * 1000.0

            if block_backend != "block_sparse_attention":
                _require_stage2_blocksparse_backend(
                    f"unexpected backend '{block_backend}' for kv_frame_then_attn_block granularity"
                )
            ok, reason = stage2_blocksparse_kernel_supported(
                q=q_group,
                k=k_compact,
                block_q=bq_sz,
                block_k=bk_sz,
            )
            if not ok:
                _require_stage2_blocksparse_backend(
                    f"layer={layer_idx} heads={heads} unsupported_shape: {reason}"
                )
            try:
                stage2_kernel_t0 = _profile_now(cfg, device) if cfg.profile_stage2 else 0.0
                out[:, :, head_idx, :] = stage2_blocksparse_attention(
                    q_group,
                    k_compact,
                    v_compact,
                    block_mask=block_mask_group,
                    block_q=bq_sz,
                    block_k=bk_sz,
                ).to(out_dtype)
                if cfg.profile_stage2:
                    stage2_kernel_ms = (_profile_now(cfg, device) - stage2_kernel_t0) * 1000.0
            except Exception as exc:
                _require_stage2_blocksparse_backend(
                    f"layer={layer_idx} heads={heads} kernel_runtime_error={type(exc).__name__}: {exc}"
                )
            backend_name = "block_sparse_attention"

        for row, h in enumerate(heads):
            block_mask = block_mask_group[row]
            _profile_stage2_record(
                cfg,
                layer_idx=layer_idx,
                head_idx=int(h),
                timestep=current_timestep,
                compact_ms=stage2_compact_ms / max(1, head_count),
                block_mass_ms=stage2_block_mass_ms / max(1, head_count),
                block_select_ms=stage2_block_select_ms / max(1, head_count),
                mask_build_ms=stage2_mask_build_ms / max(1, head_count),
                kernel_ms=stage2_kernel_ms / max(1, head_count),
                selected_frames=len(head_infos[int(h)]["selected_frame_set"]),
                selected_blocks=int(block_mask.sum().item()),
                backend=backend_name,
            )

            if not need_visualization or not _should_visualize_mask(cfg, layer_idx, int(h), current_timestep):
                continue

            compact_source_frames = [int(v) for v in head_infos[int(h)]["selected_frames"]]
            selected_frame_scores_compact = (
                fm_all[int(h)].index_select(0, selected_frame_idx[row]).float()
                if fm_all is not None
                else None
            )
            forced_lin_for_head: set[int] = set()
            selected_lin = selected_lins_by_head[int(h)]
            _dump_selection_visualization(
                cfg,
                granularity="kv_frame_then_attn_block",
                layer_idx=layer_idx,
                head_idx=int(h),
                timestep=current_timestep,
                current_start=current_start,
                current_end=current_end,
                kv_start=kv_start,
                mask_2d=block_mask,
                metadata={
                    "is_dense_head": bool(frame_dense and block_dense),
                    "mask_view": "query_blocks x key_blocks",
                    "x_axis": "compacted_key_frame",
                    "y_axis": "query_frame",
                    "x_major_count": int(compact_frame_count),
                    "y_major_count": int(query_frame_count),
                    "x_units_per_major": int(k_blocks_per_frame),
                    "y_units_per_major": int(q_blocks_per_frame),
                    "pixels_per_unit": 4,
                    "frame_selection_score": cfg.frame_selection_score,
                    "query_block_count": int(n_bq_p),
                    "key_block_count": int(n_bk_p),
                    "attn_block_q": int(bq_sz),
                    "attn_block_k": int(bk_sz),
                    "q_pool": int(q_pool),
                    "k_pool": int(k_pool),
                    "Lq": int(Lq),
                    "Lk": int(Lk_compact),
                    "original_Lk": int(Lk),
                    "frame_seqlen": int(frame_seqlen),
                    "key_space": "compacted_selected_frames",
                    "requested_density": float(cfg.get_density(layer_idx, int(h))),
                    "requested_max_kv_frames": cfg.get_max_kv_frames(layer_idx, int(h)),
                    "block_rule": cfg.get_block_rule(layer_idx, int(h)),
                    "block_density_allocation": cfg.get_block_density_allocation(layer_idx, int(h)),
                    "requested_block_density": float(cfg.get_block_density(layer_idx, int(h))),
                    "requested_block_recall": cfg.get_block_recall(layer_idx, int(h)),
                    "sink_frames": sorted(int(v) for v in sink_frames),
                    "forced_frames": sorted(int(v) for v in forced_frames),
                    "compulsory_frames": sorted(int(v) for v in compulsory_frames),
                    "dense_frames": sorted(int(v) for v in head_infos[int(h)]["dense_frame_set"]),
                    "compact_dense_frames": sorted(int(v) for v in dense_frame_sets_compact_by_head[int(h)]),
                    "selected_frames": sorted(int(v) for v in head_infos[int(h)]["selected_frame_set"]),
                    "compact_source_frames": compact_source_frames,
                    "compact_source_frame_scores": (
                        [float(v) for v in selected_frame_scores_compact.detach().cpu().tolist()]
                        if selected_frame_scores_compact is not None
                        else []
                    ),
                    "block_budget_by_compact_frame": {
                        int(k): int(v) for k, v in block_budget_by_head[int(h)].items()
                    },
                    "selected_sparse_frames": sorted(int(v) for v in head_infos[int(h)]["sparse_frame_set"]),
                    "compact_selected_sparse_frames": sorted(
                        int(v)
                        for v in (set(range(compact_frame_count)) - dense_frame_sets_compact_by_head[int(h)])
                    ),
                    "selected_additional_frames": _extra_selected_indices(
                        sorted(int(v) for v in head_infos[int(h)]["selected_frame_set"]),
                        compulsory_frames,
                    ),
                    "forced_linear_indices": sorted(int(v) for v in forced_lin_for_head),
                    "selected_linear_indices": [int(v) for v in selected_lin.detach().cpu().tolist()],
                    "selected_additional_linear_indices": _extra_selected_indices(
                        [int(v) for v in selected_lin.detach().cpu().tolist()],
                        forced_lin_for_head,
                    ),
                    "selected_item_count": int(block_mask.sum().item()),
                },
            )

    return out
