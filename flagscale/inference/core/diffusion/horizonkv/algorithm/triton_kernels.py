# Modified for the HorizonKV inference artifact; see NOTICE.
"""Block selection and attention merging kernels for HorizonKV."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch


_BLOCKSPARSE_BACKEND_TRACE_SEEN: set[str] = set()
_BLOCKSPARSE_SELECTOR_TRACE_SEEN: set[str] = set()
_GLOBAL_SELECTION_STATS_TRACE_SEEN: set[tuple[int, int, int, int]] = set()
_GLOBAL_UNIFORM_VARLEN_CONTROLS: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

try:
    from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_backend import (
        rows_blocksparse_attention as _rows_blocksparse_attention,
        rows_blocksparse_attention_varlen as _rows_blocksparse_attention_varlen,
    )
except Exception:  # pragma: no cover - optional compiled CUDA backend
    _rows_blocksparse_attention = None
    _rows_blocksparse_attention_varlen = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:  # pragma: no cover - optional runtime dependency
    _flash_attn_func = None
    _flash_attn_varlen_func = None

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )


@contextmanager
def _nvtx_range(label: str):
    """Emit an optional NVTX range without affecting the default hot path."""
    enabled = _truthy_env("HORIZONKV_NVTX", "0") and torch.cuda.is_available()
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(str(label))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _trace_blocksparse_backend(
    backend: str,
    *,
    q: torch.Tensor,
    Lk: int,
    block_m: int,
    block_n: int,
    bucket_count: int = 0,
) -> None:
    if not _truthy_env("HORIZONKV_BLOCKSPARSE_BACKEND_TRACE", "0"):
        return
    key = str(backend)
    if key in _BLOCKSPARSE_BACKEND_TRACE_SEEN:
        return
    _BLOCKSPARSE_BACKEND_TRACE_SEEN.add(key)
    print(
        "[horizonkv] blocksparse backend="
        f"{backend} rows={int(q.shape[0])} Lq={int(q.shape[1])} Lk={int(Lk)} "
        f"D={int(q.shape[-1])} block={int(block_m)}x{int(block_n)} "
        f"buckets={int(bucket_count)}"
    )


def _trace_blocksparse_selector(
    selector: str,
    *,
    q: torch.Tensor,
    Lk: int,
    block_m: int,
    block_n: int,
) -> None:
    if not _truthy_env("HORIZONKV_BLOCKSPARSE_BACKEND_TRACE", "0"):
        return
    key = str(selector)
    if key in _BLOCKSPARSE_SELECTOR_TRACE_SEEN:
        return
    _BLOCKSPARSE_SELECTOR_TRACE_SEEN.add(key)
    print(
        "[horizonkv] blocksparse selector="
        f"{key} rows={int(q.shape[0])} Lq={int(q.shape[1])} Lk={int(Lk)} "
        f"block={int(block_m)}x{int(block_n)}"
    )


def _uniform_varlen_controls(
    device: torch.device,
    rows: int,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached all-uniform varlen controls for a materialized K tensor."""
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, int(rows), int(token_count))
    cached = _GLOBAL_UNIFORM_VARLEN_CONTROLS.get(key)
    if cached is not None:
        return cached
    k_lens = torch.full(
        (int(rows),),
        int(token_count),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = (
        torch.arange(int(rows) + 1, dtype=torch.int32, device=device)
        * int(token_count)
    ).contiguous()
    _GLOBAL_UNIFORM_VARLEN_CONTROLS[key] = (k_lens, cu_seqlens_k)
    return k_lens, cu_seqlens_k


if triton is not None and tl is not None:


    @triton.jit
    def _merge_attention_parts_kernel(
        out_a,
        m_a,
        l_a,
        out_b,
        m_b,
        l_b,
        out,
        total_elements: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ) -> None:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        token_linear = offsets // head_dim

        ma = tl.load(m_a + token_linear, mask=mask, other=-3.4028234663852886e38)
        mb = tl.load(m_b + token_linear, mask=mask, other=-3.4028234663852886e38)
        la = tl.load(l_a + token_linear, mask=mask, other=0.0)
        lb = tl.load(l_b + token_linear, mask=mask, other=0.0)
        m = tl.maximum(ma, mb)

        wa = tl.where(la > 0.0, tl.exp(ma - m) * la, 0.0)
        wb = tl.where(lb > 0.0, tl.exp(mb - m) * lb, 0.0)
        denom = wa + wb

        va = tl.load(out_a + offsets, mask=mask, other=0.0).to(tl.float32)
        vb = tl.load(out_b + offsets, mask=mask, other=0.0).to(tl.float32)
        result = (va * wa + vb * wb) / tl.maximum(denom, 1.0e-20)
        tl.store(out + offsets, result, mask=mask)


    @triton.jit
    def _merge_attention_lse_parts_kernel(
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        total_elements: tl.constexpr,
        Lq: tl.constexpr,
        q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ) -> None:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        token_linear = offsets // head_dim
        token_in_row = token_linear % (Lq * q_heads)
        lq = token_in_row // q_heads
        qh = token_in_row - lq * q_heads
        row = token_linear // (Lq * q_heads)
        lse_offset = (row * q_heads + qh) * Lq + lq

        la = tl.load(lse_a + lse_offset, mask=mask, other=-3.4028234663852886e38)
        lb = tl.load(lse_b + lse_offset, mask=mask, other=-3.4028234663852886e38)
        m = tl.maximum(la, lb)
        both_invalid = m < -3.0e38
        safe_m = tl.where(both_invalid, 0.0, m)
        wa = tl.where(la < -3.0e38, 0.0, tl.exp(la - safe_m))
        wb = tl.where(lb < -3.0e38, 0.0, tl.exp(lb - safe_m))
        denom = wa + wb

        va = tl.load(out_a + offsets, mask=mask, other=0.0).to(tl.float32)
        vb = tl.load(out_b + offsets, mask=mask, other=0.0).to(tl.float32)
        result = (va * wa + vb * wb) / tl.maximum(denom, 1.0e-20)
        result = tl.where(both_invalid, 0.0, result)
        tl.store(out + offsets, result, mask=mask)


    @triton.jit
    def _blocksparse_attention_padded_kernel(
        q,
        k,
        v,
        block_cols,
        k_lens,
        cu_seqlens_k,
        out,
        lse_out,
        Lq: tl.constexpr,
        Lk,
        head_dim: tl.constexpr,
        n_q_blocks,
        max_k_blocks,
        keep_ratio,
        log2_scale,
        ln2,
        RETURN_LSE: tl.constexpr,
        PACKED_VARLEN_K: tl.constexpr,
        BLOCK_COLS_VALID: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        q_block = tl.program_id(0)
        row = tl.program_id(1)

        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        neg_inf = -3.4028234663852886e38
        if PACKED_VARLEN_K:
            row_k_len = tl.load(k_lens + row)
            row_k_start = tl.load(cu_seqlens_k + row)
            valid_k_blocks = (row_k_len + BLOCK_N - 1) // BLOCK_N
            row_keep = tl.ceil(valid_k_blocks.to(tl.float32) * keep_ratio).to(tl.int32)
            row_keep = tl.maximum(1, tl.minimum(valid_k_blocks, row_keep))
        else:
            row_k_len = Lk
            row_k_start = row * Lk
            row_keep = max_k_blocks

        q_offsets = (row * Lq + offs_m[:, None]) * head_dim + offs_d[None, :]
        if EVEN_M and EVEN_D:
            q_tile = tl.load(q + q_offsets)
        elif EVEN_D:
            q_tile = tl.load(q + q_offsets, mask=offs_m[:, None] < Lq, other=0.0)
        elif EVEN_M:
            q_tile = tl.load(q + q_offsets, mask=d_mask[None, :], other=0.0)
        else:
            q_tile = tl.load(
                q + q_offsets,
                mask=(offs_m[:, None] < Lq) & d_mask[None, :],
                other=0.0,
            )
        # Match FlashAttention's forward path: scale Q once while it is kept
        # resident instead of scaling every QK tile in the K-block loop.
        q_tile = (q_tile * log2_scale).to(k.dtype.element_ty)

        m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        cols_base = (row * n_q_blocks + q_block) * max_k_blocks
        for col_rank in range(0, row_keep):
            block_col = tl.load(block_cols + cols_base + col_rank)
            if BLOCK_COLS_VALID:
                # Selector-owned lists have a valid prefix followed by -1 padding.
                block_valid = True
                safe_block_col = block_col
            else:
                block_valid = block_col >= 0
                safe_block_col = tl.maximum(block_col, 0)
            key_pos = safe_block_col * BLOCK_N + offs_n
            key_valid = block_valid
            if PACKED_VARLEN_K or not EVEN_N:
                key_valid = key_valid & (key_pos < row_k_len)

            k_offsets = (row_k_start + key_pos[None, :]) * head_dim + offs_d[:, None]
            v_offsets = (row_k_start + key_pos[:, None]) * head_dim + offs_d[None, :]
            if EVEN_N and EVEN_D and BLOCK_COLS_VALID:
                k_tile = tl.load(k + k_offsets)
                v_tile = tl.load(v + v_offsets)
            elif EVEN_D:
                k_tile = tl.load(k + k_offsets, mask=key_valid[None, :], other=0.0)
                v_tile = tl.load(v + v_offsets, mask=key_valid[:, None], other=0.0)
            elif EVEN_N and BLOCK_COLS_VALID:
                k_tile = tl.load(
                    k + k_offsets,
                    mask=d_mask[:, None],
                    other=0.0,
                )
                v_tile = tl.load(
                    v + v_offsets,
                    mask=d_mask[None, :],
                    other=0.0,
                )
            else:
                k_mask = d_mask[:, None] & key_valid[None, :]
                v_mask = key_valid[:, None] & d_mask[None, :]
                k_tile = tl.load(k + k_offsets, mask=k_mask, other=0.0)
                v_tile = tl.load(v + v_offsets, mask=v_mask, other=0.0)

            qk = tl.dot(q_tile, k_tile, out_dtype=tl.float32)
            if EVEN_M and EVEN_N and BLOCK_COLS_VALID:
                pass
            elif EVEN_M:
                qk = tl.where(key_valid[None, :], qk, neg_inf)
            elif EVEN_N:
                qk = tl.where(
                    (offs_m[:, None] < Lq) & key_valid[None, :],
                    qk,
                    neg_inf,
                )
            else:
                qk = tl.where((offs_m[:, None] < Lq) & key_valid[None, :], qk, neg_inf)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            acc *= alpha[:, None]
            acc += tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        denom = tl.maximum(l_i[:, None], 1.0e-20)
        result = acc / denom
        result = tl.where(l_i[:, None] > 0.0, result, 0.0)
        out_offsets = (row * Lq + offs_m[:, None]) * head_dim + offs_d[None, :]
        if EVEN_M and EVEN_D:
            tl.store(out + out_offsets, result)
        elif EVEN_D:
            tl.store(out + out_offsets, result, mask=offs_m[:, None] < Lq)
        elif EVEN_M:
            tl.store(out + out_offsets, result, mask=d_mask[None, :])
        else:
            out_mask = (offs_m[:, None] < Lq) & d_mask[None, :]
            tl.store(out + out_offsets, result, mask=out_mask)

        if RETURN_LSE:
            lse = (m_i + tl.math.log2(l_i)) * ln2
            lse = tl.where(l_i > 0.0, lse, neg_inf)
            tl.store(lse_out + row * Lq + offs_m, lse, mask=offs_m < Lq)


    @triton.jit
    def _block_cols_to_descending_rowmask_kernel(
        block_cols,
        row_blockmask,
        n_k_blocks,
        max_k_blocks,
        BLOCK_KEEP: tl.constexpr,
        BLOCK_MASK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        keep_offsets = tl.arange(0, BLOCK_KEEP)
        cols = tl.load(
            block_cols + row * max_k_blocks + keep_offsets,
            mask=keep_offsets < max_k_blocks,
            other=-1,
        )
        cols = tl.sort(cols, descending=True)
        mask_base = row * n_k_blocks
        tl.store(
            row_blockmask + mask_base + keep_offsets,
            cols,
            mask=keep_offsets < max_k_blocks,
        )

        fill_offsets = max_k_blocks + tl.arange(0, BLOCK_MASK)
        tl.store(
            row_blockmask + mask_base + fill_offsets,
            -1,
            mask=fill_offsets < n_k_blocks,
        )


    @triton.jit
    def _mean_pool_qk_blocks_kernel(
        q,
        k,
        k_lens,
        cu_seqlens_k,
        q_out,
        k_out,
        Lq: tl.constexpr,
        Lk,
        head_dim: tl.constexpr,
        n_q_blocks,
        n_k_blocks,
        PACKED_VARLEN_K: tl.constexpr,
        Q_BLOCK: tl.constexpr,
        K_BLOCK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pool_block = tl.program_id(0)
        row = tl.program_id(1)
        d_block = tl.program_id(2)
        is_q = pool_block < n_q_blocks
        local_block = tl.where(is_q, pool_block, pool_block - n_q_blocks)
        block_size = tl.where(is_q, Q_BLOCK, K_BLOCK)
        if PACKED_VARLEN_K:
            row_k_len = tl.load(k_lens + row)
            row_k_start = tl.load(cu_seqlens_k + row)
        else:
            row_k_len = Lk
            row_k_start = row * Lk
        seqlen = tl.where(is_q, Lq, row_k_len)

        offs_t = tl.arange(0, BLOCK_T)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        token_pos = local_block * block_size + offs_t
        token_valid = token_pos < seqlen
        d_valid = offs_d < head_dim

        q_offsets = (row * Lq + token_pos[:, None]) * head_dim + offs_d[None, :]
        k_offsets = (row_k_start + token_pos[:, None]) * head_dim + offs_d[None, :]
        q_values = tl.load(
            q + q_offsets,
            mask=is_q & token_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        k_values = tl.load(
            k + k_offsets,
            mask=(~is_q) & token_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        values = q_values.to(tl.float32) + k_values.to(tl.float32)
        count = tl.minimum(block_size, seqlen - local_block * block_size)
        pooled = tl.sum(values, axis=0) / tl.maximum(count, 1)

        q_out_offsets = (row * n_q_blocks + local_block) * head_dim + offs_d
        k_out_offsets = (row * n_k_blocks + local_block) * head_dim + offs_d
        tl.store(q_out + q_out_offsets, pooled, mask=is_q & d_valid)
        tl.store(k_out + k_out_offsets, pooled, mask=(~is_q) & d_valid)


    @triton.jit
    def _pooled_qk_topk_kernel(
        q_pool,
        k_pool,
        k_lens,
        block_cols,
        head_dim: tl.constexpr,
        n_q_blocks,
        n_k_blocks,
        keep,
        scale: tl.constexpr,
        KEEP_RATIO: tl.constexpr,
        K_BLOCK: tl.constexpr,
        PACKED_VARLEN_K: tl.constexpr,
        ROUND_SCORES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        q_block = tl.program_id(0)
        row = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        scores = tl.zeros((BLOCK_K,), tl.float32)
        q_base = (row * n_q_blocks + q_block) * head_dim
        k_base = row * n_k_blocks * head_dim
        if PACKED_VARLEN_K:
            row_k_len = tl.load(k_lens + row)
            valid_k_blocks = (row_k_len + K_BLOCK - 1) // K_BLOCK
            row_keep = tl.ceil(valid_k_blocks.to(tl.float32) * KEEP_RATIO).to(tl.int32)
            row_keep = tl.maximum(1, tl.minimum(valid_k_blocks, row_keep))
        else:
            valid_k_blocks = n_k_blocks
            row_keep = keep

        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_valid = offs_d < head_dim
            q_values = tl.load(
                q_pool + q_base + offs_d,
                mask=d_valid,
                other=0.0,
            ).to(tl.float32)
            k_offsets = k_base + offs_k[:, None] * head_dim + offs_d[None, :]
            k_values = tl.load(
                k_pool + k_offsets,
                mask=(offs_k[:, None] < n_k_blocks) & d_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(k_values * q_values[None, :], axis=1)

        scores *= scale
        if ROUND_SCORES:
            scores = scores.to(q_pool.dtype.element_ty).to(tl.float32)
        scores = tl.where(offs_k < valid_k_blocks, scores, -float("inf"))

        score_bits = scores.to(tl.uint32, bitcast=True)
        sign = score_bits >> 31
        ordered = score_bits ^ tl.where(sign != 0, 0xFFFFFFFF, 0x80000000)
        packed = (ordered.to(tl.uint64) << 32) | offs_k.to(tl.uint64)
        packed = tl.sort(packed, descending=True)
        selected = (packed & 0xFFFFFFFF).to(tl.int32)
        selected = tl.where(offs_k < row_keep, selected, -1)
        tl.store(
            block_cols + (row * n_q_blocks + q_block) * keep + offs_k,
            selected,
            mask=offs_k < keep,
        )


    @triton.jit
    def _global_pack_selected_block_cols_kernel(
        selected,
        block_cols,
        counts,
        n_q_blocks,
        n_k_blocks,
        keep_total,
        max_k_blocks,
        BLOCK_ITEMS: tl.constexpr,
    ) -> None:
        """Pack flattened q_block*n_k_blocks+k_block indices using atomic row counters.

        Output order is unspecified until the row-sort kernel runs."""
        item_block = tl.program_id(0)
        row = tl.program_id(1)
        offs = item_block * BLOCK_ITEMS + tl.arange(0, BLOCK_ITEMS)
        valid_item = (row >= 0) & (offs < keep_total)
        selected_offsets = row * keep_total + offs
        flat_index = tl.load(selected + selected_offsets, mask=valid_item, other=0)
        q_block = flat_index // n_k_blocks
        k_block = flat_index - q_block * n_k_blocks
        valid_item = valid_item & (q_block < n_q_blocks) & (k_block < n_k_blocks)

        # Keep pointer arithmetic valid for masked lanes.  The store remains
        # masked, so invalid top-k padding never changes the counters/output.
        safe_q_block = tl.where(valid_item, q_block, 0)
        count_ptr = counts + row * n_q_blocks + safe_q_block
        rank = tl.atomic_add(count_ptr, 1, mask=valid_item, sem="relaxed")
        out_offsets = (
            (row * n_q_blocks + safe_q_block) * max_k_blocks + rank
        )
        tl.store(
            block_cols + out_offsets,
            k_block.to(tl.int32),
            mask=valid_item & (rank < max_k_blocks),
        )


    @triton.jit
    def _global_sort_selected_block_cols_kernel(
        block_cols,
        counts,
        n_q_blocks,
        n_k_blocks,
        max_k_blocks,
        SORT_DESCENDING: tl.constexpr,
        BLOCK_SORT: tl.constexpr,
    ) -> None:
        """Sort selected columns and pad with -1; CUDA row masks require descending IDs."""
        row_q_block = tl.program_id(0)
        row = row_q_block // n_q_blocks
        q_block = row_q_block - row * n_q_blocks
        offs = tl.arange(0, BLOCK_SORT)
        base = (row * n_q_blocks + q_block) * max_k_blocks
        cols = tl.load(
            block_cols + base + offs,
            mask=offs < max_k_blocks,
            other=-1,
        )
        row_count = tl.load(counts + row * n_q_blocks + q_block)
        if SORT_DESCENDING:
            # -1 is the padding sentinel for the native CUDA row-mask format;
            # using it before the sort keeps valid columns at the prefix.
            cols = tl.where(offs < row_count, cols, -1)
            cols = tl.sort(cols, descending=True)
        else:
            # Ascending compact-column compatibility format uses n_k_blocks as
            # a temporary sentinel so valid columns remain at the prefix.
            cols = tl.where(offs < row_count, cols, n_k_blocks)
            cols = tl.sort(cols, descending=False)
            cols = tl.where(cols < n_k_blocks, cols, -1)
        tl.store(
            block_cols + base + offs,
            cols,
            mask=offs < max_k_blocks,
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _triton_blocksparse_block_sizes(head_dim: int) -> tuple[int, int]:
    head_dim = int(head_dim)
    if head_dim <= 64:
        return 64, 128
    if head_dim <= 128:
        return 128, 128
    return 64, 128


def _triton_blocksparse_launch_params(
    *,
    head_dim: int,
    block_m: int,
    block_n: int,
) -> tuple[int, int]:
    head_dim = int(head_dim)
    block_m = int(block_m)
    block_n = int(block_n)
    if head_dim <= 64:
        return 4, 2
    if head_dim <= 128:
        return (8, 2) if block_n >= 128 else (4, 2)
    return (8, 1) if block_m <= 64 and block_n >= 128 else (8, 1)


def _blocksparse_backend_name() -> str:
    return str(
        os.getenv("HORIZONKV_TRITON_BLOCKSPARSE_BACKEND", "cuda_flash")
    ).strip().lower()


def _blocksparse_flash_backend_required() -> bool:
    return _truthy_env("HORIZONKV_REQUIRE_FLASH_BACKEND", "1")


def _blocksparse_flash_backend_enabled() -> bool:
    backend = _blocksparse_backend_name()
    if backend in ("triton", "triton_only"):
        return False
    if backend in ("flash", "flash_varlen", "paged_flash"):
        return True
    return _truthy_env("HORIZONKV_TRITON_BLOCKSPARSE_FLASH_VARLEN", "0")


def _blocksparse_cuda_backend_enabled() -> bool:
    backend = _blocksparse_backend_name()
    return backend in (
        "cuda",
        "cuda_flash",
        "flash_cuda",
        "block_sparse_attention",
    )


def _blocksparse_attention_cuda_flash(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_cols: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if _rows_blocksparse_attention is None:
        return None
    rows, Lq, _, _ = q.shape
    Lk = int(k.shape[1])
    n_q_blocks = (int(Lq) + int(block_m) - 1) // int(block_m)
    n_k_blocks = (int(Lk) + int(block_n) - 1) // int(block_n)
    max_k_blocks = int(block_cols.shape[2])
    if (
        int(block_m) not in (64, 128)
        or int(block_n) not in (64, 128)
        or max_k_blocks <= 0
        or max_k_blocks > n_k_blocks
    ):
        return None
    direct_global_rowmask = (
        _truthy_env("HORIZONKV_TRITON_GLOBAL_DIRECT_ROWMASK", "1")
        and _truthy_env("HORIZONKV_TRITON_GLOBAL_BLOCK_SELECTION", "0")
        and int(max_k_blocks) == int(n_k_blocks)
        and block_cols.dtype == torch.int32
        and block_cols.is_contiguous()
    )
    if direct_global_rowmask:
        # Reuse native descending row masks to avoid another conversion kernel.
        row_blockmask = block_cols
    else:
        row_blockmask = torch.empty(
            (int(rows), n_q_blocks, n_k_blocks),
            device=q.device,
            dtype=torch.int32,
        )
        block_keep = triton.next_power_of_2(max_k_blocks)
        block_mask = triton.next_power_of_2(n_k_blocks)
        _block_cols_to_descending_rowmask_kernel[(int(rows) * n_q_blocks,)](
            block_cols,
            row_blockmask,
            int(n_k_blocks),
            int(max_k_blocks),
            BLOCK_KEEP=block_keep,
            BLOCK_MASK=block_mask,
            num_warps=4,
            num_stages=1,
        )
    return _rows_blocksparse_attention(
        q=q,
        k=k,
        v=v,
        row_blockmask=row_blockmask,
        block_m=int(block_m),
        block_n=int(block_n),
        softmax_scale=float(softmax_scale),
        return_lse=bool(return_lse),
    )


def _blocksparse_attention_cuda_flash_varlen(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_cols: torch.Tensor,
    k_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if (
        _rows_blocksparse_attention_varlen is None
        or not _truthy_env("HORIZONKV_BLOCKSPARSE_VARLEN_CUDA", "1")
    ):
        return None
    rows, Lq, _, _ = q.shape
    n_q_blocks = (int(Lq) + int(block_m) - 1) // int(block_m)
    n_k_blocks = (int(max_seqlen_k) + int(block_n) - 1) // int(block_n)
    max_k_blocks = int(block_cols.shape[2])
    if (
        int(block_m) not in (64, 128)
        or int(block_n) not in (64, 128)
        or max_k_blocks <= 0
        or max_k_blocks > n_k_blocks
        or int(k_lens.numel()) != int(rows)
    ):
        return None
    direct_global_rowmask = (
        _truthy_env("HORIZONKV_TRITON_GLOBAL_DIRECT_ROWMASK", "1")
        and _truthy_env("HORIZONKV_TRITON_GLOBAL_BLOCK_SELECTION", "0")
        and int(max_k_blocks) == int(n_k_blocks)
        and block_cols.dtype == torch.int32
        and block_cols.is_contiguous()
    )
    if direct_global_rowmask:
        row_blockmask = block_cols
    else:
        row_blockmask = torch.empty(
            (int(rows), n_q_blocks, n_k_blocks),
            device=q.device,
            dtype=torch.int32,
        )
        block_keep = triton.next_power_of_2(max_k_blocks)
        block_mask = triton.next_power_of_2(n_k_blocks)
        _block_cols_to_descending_rowmask_kernel[(int(rows) * n_q_blocks,)](
            block_cols,
            row_blockmask,
            int(n_k_blocks),
            int(max_k_blocks),
            BLOCK_KEEP=block_keep,
            BLOCK_MASK=block_mask,
            num_warps=4,
            num_stages=1,
        )
    return _rows_blocksparse_attention_varlen(
        q=q,
        k=k,
        v=v,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=int(max_seqlen_k),
        row_blockmask=row_blockmask,
        block_m=int(block_m),
        block_n=int(block_n),
        softmax_scale=float(softmax_scale),
        return_lse=bool(return_lse),
    )


def _dense_flash_attention_rows(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if _flash_attn_func is None:
        return None
    try:
        result = _flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            return_attn_probs=bool(return_lse),
        )
        if not return_lse:
            return result
        if not (
            isinstance(result, tuple)
            and len(result) >= 2
            and isinstance(result[0], torch.Tensor)
            and isinstance(result[1], torch.Tensor)
        ):
            return None
        return result[0], result[1]
    except Exception:
        return None


def _blocksparse_attention_paged_flash(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_cols: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
) -> torch.Tensor | None:
    if _flash_attn_varlen_func is None:
        return None
    rows, Lq, _, head_dim = q.shape
    Lk = int(k.shape[1])
    block_m = int(block_m)
    block_n = int(block_n)
    if block_m <= 0 or block_n <= 0:
        return None
    if block_n % 256 != 0:
        return None
    if int(Lq) % block_m != 0 or int(Lk) % block_n != 0:
        return None
    n_q_blocks = int(Lq) // block_m
    n_k_blocks = int(Lk) // block_n
    if (
        block_cols.ndim != 3
        or int(block_cols.shape[0]) != int(rows)
        or int(block_cols.shape[1]) != n_q_blocks
    ):
        return None
    keep = int(block_cols.shape[2])
    if keep <= 0 or keep >= n_k_blocks:
        return None

    try:
        block_cols_i32 = block_cols.to(device=q.device, dtype=torch.int32).contiguous()
        batch_blocks = int(rows) * n_q_blocks
        q_flat = q.reshape(int(rows), n_q_blocks, block_m, 1, int(head_dim))
        q_flat = q_flat.reshape(batch_blocks * block_m, 1, int(head_dim))
        k_pages = k.reshape(int(rows), n_k_blocks, block_n, 1, int(head_dim))
        k_pages = k_pages.reshape(int(rows) * n_k_blocks, block_n, 1, int(head_dim))
        v_pages = v.reshape(int(rows), n_k_blocks, block_n, 1, int(head_dim))
        v_pages = v_pages.reshape(int(rows) * n_k_blocks, block_n, 1, int(head_dim))

        row_offsets = torch.arange(
            int(rows),
            device=q.device,
            dtype=torch.int32,
        ).view(int(rows), 1, 1)
        block_table = (block_cols_i32 + row_offsets * n_k_blocks).reshape(
            batch_blocks,
            keep,
        ).contiguous()
        seq = torch.arange(batch_blocks + 1, device=q.device, dtype=torch.int32)
        cu_seqlens_q = seq * block_m
        cu_seqlens_k = seq * (keep * block_n)
        out = _flash_attn_varlen_func(
            q_flat,
            k_pages,
            v_pages,
            cu_seqlens_q,
            cu_seqlens_k,
            block_m,
            keep * block_n,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            block_table=block_table,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(int(rows), n_q_blocks, block_m, 1, int(head_dim)).reshape(
            int(rows),
            int(Lq),
            1,
            int(head_dim),
        )
    except Exception as exc:
        if _truthy_env("HORIZONKV_BLOCKSPARSE_VERBOSE", "0"):
            print(f"[horizonkv] flash-varlen blocksparse fallback after failure: {exc}")
        return None


def _mean_pool_attention_blocks(
    x: torch.Tensor,
    block: int,
    *,
    force_fp32: bool = False,
) -> torch.Tensor:
    rows, seqlen, one_head, head_dim = x.shape
    if one_head != 1 or rows <= 0 or seqlen <= 0 or head_dim <= 0:
        raise ValueError("expected x to have shape [rows, seqlen, 1, head_dim]")
    block = max(1, int(block))
    n_blocks = (int(seqlen) + block - 1) // block
    xf = x.squeeze(2)
    if force_fp32 and xf.dtype != torch.float32:
        xf = xf.to(dtype=torch.float32)
    pad = n_blocks * block - int(seqlen)
    if pad > 0:
        xf = torch.nn.functional.pad(xf, (0, 0, 0, pad), value=0.0)
    pooled = xf.reshape(rows, n_blocks, block, head_dim).sum(dim=2)
    if pad > 0:
        counts = torch.full((n_blocks,), block, device=x.device, dtype=pooled.dtype)
        tail = int(seqlen) - (n_blocks - 1) * block
        counts[-1] = max(1, tail)
        pooled = pooled / counts.view(1, n_blocks, 1)
    else:
        pooled = pooled / float(block)
    return pooled.contiguous()


def _mean_pool_qk_attention_blocks_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    block_m: int,
    block_n: int,
    *,
    force_fp32: bool = False,
    k_lens: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or tl is None:
        return None
    rows, Lq, one_head, head_dim = q.shape
    packed_varlen_k = k.ndim == 3
    if (
        one_head != 1
        or k.ndim not in (3, 4)
        or (not packed_varlen_k and int(k.shape[0]) != int(rows))
        or int(k.shape[-2]) != 1
        or int(k.shape[-1]) != int(head_dim)
        or q.dtype != k.dtype
        or q.device != k.device
        or not (q.is_cuda and q.is_contiguous() and k.is_contiguous())
    ):
        return None
    block_m = max(1, int(block_m))
    block_n = max(1, int(block_n))
    if block_m > 128 or block_n > 128 or int(head_dim) > 256:
        return None
    if packed_varlen_k:
        if (
            not isinstance(k_lens, torch.Tensor)
            or not isinstance(cu_seqlens_k, torch.Tensor)
            or k_lens.device != q.device
            or cu_seqlens_k.device != q.device
            or k_lens.dtype != torch.int32
            or cu_seqlens_k.dtype != torch.int32
            or not k_lens.is_contiguous()
            or not cu_seqlens_k.is_contiguous()
            or int(k_lens.numel()) != int(rows)
            or int(cu_seqlens_k.numel()) != int(rows) + 1
            or int(max_seqlen_k or 0) <= 0
        ):
            return None
        Lk = int(max_seqlen_k)
    else:
        Lk = int(k.shape[1])
    n_q_blocks = (int(Lq) + block_m - 1) // block_m
    n_k_blocks = (Lk + block_n - 1) // block_n
    out_dtype = torch.float32 if force_fp32 else q.dtype
    q_pool = torch.empty(
        (int(rows), n_q_blocks, int(head_dim)),
        device=q.device,
        dtype=out_dtype,
    )
    k_pool = torch.empty(
        (int(rows), n_k_blocks, int(head_dim)),
        device=q.device,
        dtype=out_dtype,
    )
    block_t = triton.next_power_of_2(max(block_m, block_n))
    block_d = 64 if int(head_dim) > 32 else 32
    grid = (n_q_blocks + n_k_blocks, int(rows), triton.cdiv(int(head_dim), block_d))
    _mean_pool_qk_blocks_kernel[grid](
        q,
        k,
        k_lens if packed_varlen_k else q,
        cu_seqlens_k if packed_varlen_k else q,
        q_pool,
        k_pool,
        int(Lq),
        int(Lk),
        int(head_dim),
        int(n_q_blocks),
        int(n_k_blocks),
        PACKED_VARLEN_K=packed_varlen_k,
        Q_BLOCK=block_m,
        K_BLOCK=block_n,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return q_pool, k_pool


def _select_pooled_block_cols_triton(
    q_pool: torch.Tensor,
    k_pool: torch.Tensor,
    *,
    keep: int,
    scale: float,
    round_scores: bool,
    keep_ratio: float,
    block_n: int,
    k_lens: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if triton is None or tl is None:
        return None
    if (
        q_pool.ndim != 3
        or k_pool.ndim != 3
        or int(q_pool.shape[0]) != int(k_pool.shape[0])
        or int(q_pool.shape[2]) != int(k_pool.shape[2])
        or q_pool.device != k_pool.device
        or q_pool.dtype != k_pool.dtype
        or not (q_pool.is_cuda and q_pool.is_contiguous() and k_pool.is_contiguous())
    ):
        return None
    rows, n_q_blocks, head_dim = q_pool.shape
    n_k_blocks = int(k_pool.shape[1])
    packed_varlen_k = isinstance(k_lens, torch.Tensor)
    keep = int(keep)
    if (
        int(rows) <= 0
        or int(n_q_blocks) <= 0
        or int(head_dim) <= 0
        or int(head_dim) > 256
        or n_k_blocks <= 1
        # Keep the uniform selector's existing BMM/top-k fallback unchanged.
        or n_k_blocks > (512 if packed_varlen_k else 256)
        or keep <= 0
        or keep >= n_k_blocks
    ):
        return None
    if packed_varlen_k and (
        k_lens.device != q_pool.device
        or k_lens.dtype != torch.int32
        or not k_lens.is_contiguous()
        or int(k_lens.numel()) != int(rows)
    ):
        return None
    block_cols = torch.empty(
        (int(rows), int(n_q_blocks), keep),
        device=q_pool.device,
        dtype=torch.int32,
    )
    block_k = triton.next_power_of_2(n_k_blocks)
    block_d = 16 if int(head_dim) >= 64 else 8
    _pooled_qk_topk_kernel[(int(n_q_blocks), int(rows))](
        q_pool,
        k_pool,
        k_lens if packed_varlen_k else q_pool,
        block_cols,
        int(head_dim),
        int(n_q_blocks),
        int(n_k_blocks),
        int(keep),
        float(scale),
        KEEP_RATIO=float(keep_ratio),
        K_BLOCK=int(block_n),
        PACKED_VARLEN_K=packed_varlen_k,
        ROUND_SCORES=bool(round_scores),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return block_cols


def _pack_global_selected_block_cols_triton(
    selected: torch.Tensor,
    *,
    rows: int,
    n_q_blocks: int,
    n_k_blocks: int,
    sort_cols: bool = True,
    sort_descending: bool = False,
    fixed_width: bool | None = None,
) -> tuple[torch.Tensor, int] | None:
    """Pack [rows,keep_total] global top-k indices into per-Q-block lists.

    Atomic counters pack entries; row sorting supplies the requested column order."""
    if triton is None or tl is None:
        return None
    if not _truthy_env("HORIZONKV_TRITON_GLOBAL_PACK", "1"):
        return None
    if (
        selected.ndim != 2
        or int(selected.shape[0]) != int(rows)
        or int(selected.shape[1]) <= 0
        or not selected.is_cuda
        or not selected.is_contiguous()
        or selected.dtype not in (torch.int32, torch.int64)
        or int(n_q_blocks) <= 0
        or int(n_k_blocks) <= 1
        or int(n_k_blocks) > 256
    ):
        return None

    keep_total = int(selected.shape[1])
    # Full-density allocations use the dense fallback.
    total_candidates = int(n_q_blocks) * int(n_k_blocks)
    if keep_total >= total_candidates:
        return None

    # Full-width output avoids synchronizing the data-dependent maximum row count.
    counts = torch.zeros(
        (int(rows), int(n_q_blocks)),
        device=selected.device,
        dtype=torch.int32,
    )
    block_cols = torch.full(
        (int(rows), int(n_q_blocks), int(n_k_blocks)),
        device=selected.device,
        dtype=torch.int32,
        fill_value=-1,
    )
    if fixed_width is None:
        fixed_width = (
            _truthy_env("HORIZONKV_TRITON_GLOBAL_FIXED_WIDTH", "1")
            and _blocksparse_cuda_backend_enabled()
        )
    try:
        block_items = 128 if keep_total >= 128 else triton.next_power_of_2(keep_total)
        _global_pack_selected_block_cols_kernel[
            (triton.cdiv(keep_total, block_items), int(rows))
        ](
            selected,
            block_cols,
            counts,
            int(n_q_blocks),
            int(n_k_blocks),
            int(keep_total),
            int(n_k_blocks),
            BLOCK_ITEMS=block_items,
            num_warps=4,
            num_stages=1,
        )

        if sort_cols:
            block_sort = triton.next_power_of_2(int(n_k_blocks))
            _global_sort_selected_block_cols_kernel[(int(rows) * int(n_q_blocks),)](
                block_cols,
                counts,
                int(n_q_blocks),
                int(n_k_blocks),
                int(n_k_blocks),
                SORT_DESCENDING=bool(sort_descending),
                BLOCK_SORT=block_sort,
                num_warps=4,
                num_stages=1,
            )
        if bool(fixed_width):
            # CUDA row masks are full [q_block, k_block] rectangles and use
            # -1 padding, so no dynamic width is needed downstream.
            max_keep = int(n_k_blocks)
        else:
            # Compatibility mode for callers that explicitly request the
            # compactest rectangular block_cols representation.
            max_keep = int(counts.amax().item())
    except Exception as exc:
        if _truthy_env("HORIZONKV_BLOCKSPARSE_VERBOSE", "0"):
            print(f"[horizonkv] global Triton pack fallback after failure: {exc}")
        return None
    if max_keep <= 0:
        return None
    stats_key = (int(rows), int(n_q_blocks), int(n_k_blocks), int(keep_total))
    if (
        _truthy_env("HORIZONKV_GLOBAL_SELECTION_STATS", "0")
        and stats_key not in _GLOBAL_SELECTION_STATS_TRACE_SEEN
    ):
        _GLOBAL_SELECTION_STATS_TRACE_SEEN.add(stats_key)
        # Diagnostics intentionally synchronize once per geometry.  They stay
        # completely off the normal benchmark path unless explicitly enabled.
        counts_f = counts.to(dtype=torch.float32)
        mean_count = float(counts_f.mean().item())
        max_count = float(counts_f.max().item())
        print(
            "[horizonkv] global_selection_counts "
            f"rows={int(rows)} q_blocks={int(n_q_blocks)} k_blocks={int(n_k_blocks)} "
            f"budget={int(keep_total)} min={float(counts_f.min().item()):.2f} "
            f"mean={mean_count:.2f} max={max_count:.2f} "
            f"std={float(counts_f.std(unbiased=False).item()):.2f} "
            f"max_over_mean={(max_count / max(mean_count, 1.0)):.3f}"
        )
    return block_cols[..., :max_keep].contiguous(), max_keep


@torch.no_grad()
def _select_pooled_block_cols(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    keep_ratio: float,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
    sort_by_block_id: bool = False,
    k_lens: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor | None:
    rows, Lq, _, head_dim = q.shape
    packed_varlen_k = k.ndim == 3
    Lk = int(max_seqlen_k or 0) if packed_varlen_k else int(k.shape[1])
    block_m = max(1, int(block_m))
    block_n = max(1, int(block_n))
    n_q_blocks = (int(Lq) + block_m - 1) // block_m
    n_k_blocks = (int(Lk) + block_n - 1) // block_n
    if rows <= 0 or n_q_blocks <= 0 or n_k_blocks <= 1:
        return None
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    keep = max(1, min(n_k_blocks, int(math.ceil(keep_ratio * float(n_k_blocks)))))
    # Compute the global budget before rejecting per-row quotas that round to dense.
    global_selection = _truthy_env("HORIZONKV_TRITON_GLOBAL_BLOCK_SELECTION", "0")
    if keep >= n_k_blocks and not global_selection:
        return None

    use_fp32 = _truthy_env("HORIZONKV_TRITON_SELECTOR_FP32", "0")
    pooled_qk = None
    if _truthy_env("HORIZONKV_TRITON_FUSED_QK_POOL", "1"):
        pooled_qk = _mean_pool_qk_attention_blocks_triton(
            q,
            k,
            block_m,
            block_n,
            force_fp32=use_fp32,
            k_lens=k_lens,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=Lk,
        )
    if pooled_qk is None:
        if packed_varlen_k:
            return None
        q_pool = _mean_pool_attention_blocks(q, block_m, force_fp32=use_fp32)
        k_pool = _mean_pool_attention_blocks(k, block_n, force_fp32=use_fp32)
    else:
        q_pool, k_pool = pooled_qk
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(float(head_dim))
    block_cols = None
    if global_selection:
        # Global allocation ranks the flattened Q/K block map and pads per-Q lists with -1.
        if packed_varlen_k:
            # Global compact selection requires uniform materialized rows.
            _trace_blocksparse_selector(
                "global_packed_varlen_fallback",
                q=q,
                Lk=Lk,
                block_m=block_m,
                block_n=block_n,
            )
            return None
        with _nvtx_range("horizonkv.global_score_bmm"):
            scores = torch.bmm(q_pool, k_pool.transpose(1, 2)) * scale
        total_candidates = int(n_q_blocks) * int(n_k_blocks)
        keep_total = max(
            1,
            min(
                total_candidates,
                int(math.ceil(float(keep_ratio) * float(total_candidates))),
            ),
        )
        selected = None
        if keep_total < total_candidates:
            flat_scores = scores.reshape(int(rows), total_candidates)
            with _nvtx_range("horizonkv.global_topk"):
                selected = torch.topk(
                    flat_scores,
                    k=keep_total,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
            direct_rowmask = (
                _truthy_env("HORIZONKV_TRITON_GLOBAL_DIRECT_ROWMASK", "1")
                and _blocksparse_cuda_backend_enabled()
                and _blocksparse_flash_backend_required()
                and not bool(sort_by_block_id)
            )
            with _nvtx_range("horizonkv.global_pack"):
                packed_global = _pack_global_selected_block_cols_triton(
                    selected,
                    rows=int(rows),
                    n_q_blocks=int(n_q_blocks),
                    n_k_blocks=int(n_k_blocks),
                    # The direct CUDA row-mask path needs descending columns;
                    # the normal compact path leaves ordering to its backend.
                    sort_cols=bool(direct_rowmask),
                    sort_descending=bool(direct_rowmask),
                    fixed_width=bool(
                        direct_rowmask
                        or (
                            _truthy_env("HORIZONKV_TRITON_GLOBAL_FIXED_WIDTH", "1")
                            and _blocksparse_cuda_backend_enabled()
                        )
                    ),
                )
            if packed_global is not None:
                block_cols, _ = packed_global
                _trace_blocksparse_selector(
                    "global_triton_pack",
                    q=q,
                    Lk=Lk,
                    block_m=block_m,
                    block_n=block_n,
                )
                if direct_rowmask:
                    _trace_blocksparse_selector(
                        "global_direct_rowmask",
                        q=q,
                        Lk=Lk,
                        block_m=block_m,
                        block_n=block_n,
                    )

        # Use PyTorch for unsupported shapes, full density, or Triton compilation failure.
        if block_cols is None:
            selected_mask = torch.zeros(
                (int(rows), total_candidates),
                device=scores.device,
                dtype=torch.bool,
            )
            if keep_total >= total_candidates:
                selected_mask.fill_(True)
            else:
                selected_mask.scatter_(1, selected, True)
            selected_mask = selected_mask.view(
                int(rows), int(n_q_blocks), int(n_k_blocks)
            )
            per_q_counts = selected_mask.sum(dim=-1)
            max_keep = int(per_q_counts.max().item())
            if max_keep <= 0:
                return None
            block_ids = torch.arange(
                int(n_k_blocks), device=scores.device, dtype=torch.int32
            ).view(1, 1, int(n_k_blocks))
            invalid_sentinel = torch.full_like(block_ids, int(n_k_blocks))
            sorted_ids = torch.sort(
                torch.where(selected_mask, block_ids, invalid_sentinel),
                dim=-1,
            ).values[..., :max_keep]
            block_cols = torch.where(
                sorted_ids < int(n_k_blocks),
                sorted_ids,
                torch.full_like(sorted_ids, -1),
            )
    elif _truthy_env("HORIZONKV_TRITON_FUSED_SELECTOR_TOPK", "1"):
        block_cols = _select_pooled_block_cols_triton(
            q_pool,
            k_pool,
            keep=keep,
            scale=scale,
            round_scores=not use_fp32,
            keep_ratio=keep_ratio,
            block_n=block_n,
            k_lens=k_lens if packed_varlen_k else None,
        )
    if block_cols is None:
        if packed_varlen_k:
            return None
        scores = torch.bmm(q_pool, k_pool.transpose(1, 2)) * scale
        block_cols = torch.topk(scores, k=keep, dim=-1, largest=True, sorted=False).indices
    if bool(sort_by_block_id):
        invalid_sentinel = torch.full_like(block_cols, n_k_blocks)
        block_cols = torch.where(block_cols >= 0, block_cols, invalid_sentinel)
        block_cols = torch.sort(block_cols, dim=-1).values
        block_cols = torch.where(block_cols < n_k_blocks, block_cols, -1)
    return block_cols.to(dtype=torch.int32).contiguous()


def blocksparse_attention_pooled_triton(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float,
    block_m: int = 64,
    block_n: int = 64,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    block_cols: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    force_sparse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Forward-only block-sparse attention for materialized per-head rows.

    The block layout is selected from average-pooled Q/K scores.  A sparsity
    of 0.5 keeps roughly half of the K blocks for every Q block.
    """
    if triton is None or tl is None:
        return None
    if not _truthy_env("HORIZONKV_TRITON_BLOCKSPARSE_ATTN", "1"):
        return None
    if q.dtype not in (torch.float16, torch.bfloat16):
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return None
    if q.device != k.device or v.device != q.device:
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return None
    if q.ndim != 4 or k.ndim not in (3, 4) or v.ndim != k.ndim:
        return None
    if tuple(k.shape) != tuple(v.shape):
        return None
    packed_varlen_k = k.ndim == 3
    if (
        (not packed_varlen_k and int(q.shape[0]) != int(k.shape[0]))
        or int(q.shape[2]) != 1
        or int(k.shape[-2]) != 1
    ):
        return None
    if int(q.shape[-1]) != int(k.shape[-1]):
        return None

    rows, Lq, _, head_dim = q.shape
    if packed_varlen_k:
        if (
            not isinstance(k_lens, torch.Tensor)
            or not isinstance(cu_seqlens_k, torch.Tensor)
            or k_lens.device != q.device
            or cu_seqlens_k.device != q.device
            or k_lens.dtype != torch.int32
            or cu_seqlens_k.dtype != torch.int32
            or not k_lens.is_contiguous()
            or not cu_seqlens_k.is_contiguous()
            or int(k_lens.numel()) != int(rows)
            or int(cu_seqlens_k.numel()) != int(rows) + 1
            or int(max_seqlen_k or 0) <= 0
        ):
            return None
        Lk = int(max_seqlen_k)
    else:
        Lk = int(k.shape[1])
    if rows <= 0 or Lq <= 0 or Lk <= 0 or head_dim <= 0 or head_dim > 256:
        return None

    sparsity = max(0.0, min(1.0, float(sparsity)))
    if sparsity <= 0.0:
        if packed_varlen_k:
            return None
        return _dense_flash_attention_rows(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
        )
    keep_ratio = 1.0 - sparsity
    if keep_ratio >= 1.0:
        return _dense_flash_attention_rows(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
        )

    if (
        not force_sparse
        and not packed_varlen_k
        and _truthy_env("HORIZONKV_TRITON_BLOCKSPARSE_DENSE_FALLBACK", "1")
    ):
        backend = _blocksparse_backend_name()
        if backend in ("triton", "triton_only"):
            default_min_sparsity = 0.30
        elif backend in ("cuda", "cuda_flash", "flash_cuda", "block_sparse_attention"):
            default_min_sparsity = 0.20
        else:
            default_min_sparsity = 0.20
        min_sparse_sparsity = max(
            0.0,
            min(
                1.0,
                _env_float(
                    "HORIZONKV_TRITON_BLOCKSPARSE_MIN_SPARSITY",
                    default_min_sparsity,
                ),
            ),
        )
        min_dense_score_elems = _env_int(
            "HORIZONKV_TRITON_BLOCKSPARSE_MIN_DENSE_SCORE_ELEMS",
            64_000_000,
        )
        dense_score_elems = int(rows) * int(Lq) * int(Lk)
        if block_cols is None and (
            sparsity < min_sparse_sparsity or dense_score_elems < min_dense_score_elems
        ):
            dense_out = _dense_flash_attention_rows(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
            )
            if dense_out is not None:
                return dense_out

    auto_block_m, auto_block_n = _triton_blocksparse_block_sizes(int(head_dim))
    if block_cols is None and _truthy_env("HORIZONKV_TRITON_BLOCKSPARSE_FLASH_BLOCKS", "1"):
        block_m, block_n = auto_block_m, auto_block_n
    global_block_selection = _truthy_env(
        "HORIZONKV_TRITON_GLOBAL_BLOCK_SELECTION", "0"
    )
    # Only selector-owned prefixes permit the fast path; global/user masks need validation.
    generated_block_cols = block_cols is None and not global_block_selection
    if block_cols is None:
        block_m = _env_int("HORIZONKV_TRITON_BLOCKSPARSE_BLOCK_M", int(block_m))
        block_n = _env_int("HORIZONKV_TRITON_BLOCKSPARSE_BLOCK_N", int(block_n))
    else:
        block_m = int(block_m)
        block_n = int(block_n)
    if block_m not in (16, 32, 64, 128) or block_n not in (16, 32, 64, 128, 256):
        return None
    if global_block_selection:
        _trace_blocksparse_selector(
            "global",
            q=q,
            Lk=Lk,
            block_m=block_m,
            block_n=block_n,
        )

    if (
        not global_block_selection
        and not packed_varlen_k
        and _blocksparse_flash_backend_enabled()
        and not return_lse
    ):
        flash_cols = None
        flash_block_n = 0
        if block_cols is None:
            flash_block_n = _env_int("HORIZONKV_TRITON_BLOCKSPARSE_FLASH_PAGE_SIZE", 256)
            if flash_block_n > 0 and flash_block_n % 256 == 0:
                flash_cols = _select_pooled_block_cols(
                    q,
                    k,
                    keep_ratio=keep_ratio,
                    block_m=block_m,
                    block_n=flash_block_n,
                    softmax_scale=softmax_scale,
                    sort_by_block_id=_truthy_env(
                        "HORIZONKV_TRITON_BLOCKSPARSE_SORT_COLS",
                        "0",
                    ),
                )
        elif block_n % 256 == 0:
            flash_block_n = block_n
            flash_cols = block_cols
        if flash_cols is not None:
            flash_out = _blocksparse_attention_paged_flash(
                q=q,
                k=k,
                v=v,
                block_cols=flash_cols,
                block_m=block_m,
                block_n=flash_block_n,
                softmax_scale=softmax_scale,
            )
            if isinstance(flash_out, torch.Tensor):
                _trace_blocksparse_backend(
                    "paged_flash",
                    q=q,
                    Lk=Lk,
                    block_m=block_m,
                    block_n=flash_block_n,
                )
                return flash_out

    if block_n not in (16, 32, 64, 128):
        return None
    if Lk <= block_n:
        return None

    if block_cols is None:
        block_cols = _select_pooled_block_cols(
            q,
            k,
            keep_ratio=keep_ratio,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=softmax_scale,
            sort_by_block_id=_truthy_env(
                "HORIZONKV_TRITON_BLOCKSPARSE_SORT_COLS",
                "0",
            ),
            k_lens=k_lens if packed_varlen_k else None,
            cu_seqlens_k=cu_seqlens_k if packed_varlen_k else None,
            max_seqlen_k=Lk,
        )
    elif (
        not isinstance(block_cols, torch.Tensor)
        or block_cols.device != q.device
        or block_cols.ndim != 3
        or int(block_cols.shape[0]) != rows
    ):
        return None
    else:
        block_cols = block_cols.to(device=q.device, dtype=torch.int32).contiguous()
    if block_cols is None:
        return None

    n_q_blocks = int(block_cols.shape[1])
    max_k_blocks = int(block_cols.shape[2])
    expected_n_q_blocks = (Lq + block_m - 1) // block_m
    if n_q_blocks != expected_n_q_blocks:
        return None
    n_k_blocks = (Lk + block_n - 1) // block_n
    if n_q_blocks <= 0 or max_k_blocks <= 0 or max_k_blocks > n_k_blocks:
        return None

    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(float(head_dim))
    # Repack uniform global rows for the varlen kernel, with padded CUDA fallback.
    use_uniform_varlen_global = (
        global_block_selection
        and not packed_varlen_k
        and _truthy_env("HORIZONKV_GLOBAL_VARLEN_BACKEND", "1")
    )
    backend_is_varlen = packed_varlen_k
    if _blocksparse_cuda_backend_enabled():
        try:
            if packed_varlen_k:
                cuda_out = _blocksparse_attention_cuda_flash_varlen(
                    q=q,
                    k=k,
                    v=v,
                    block_cols=block_cols,
                    k_lens=k_lens,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_k=Lk,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=scale,
                    return_lse=return_lse,
                )
            elif use_uniform_varlen_global:
                k_varlen = k.reshape(int(rows) * int(Lk), 1, int(head_dim))
                v_varlen = v.reshape(int(rows) * int(Lk), 1, int(head_dim))
                uniform_k_lens, uniform_cu_seqlens_k = _uniform_varlen_controls(
                    q.device,
                    int(rows),
                    int(Lk),
                )
                try:
                    cuda_out = _blocksparse_attention_cuda_flash_varlen(
                        q=q,
                        k=k_varlen,
                        v=v_varlen,
                        block_cols=block_cols,
                        k_lens=uniform_k_lens,
                        cu_seqlens_k=uniform_cu_seqlens_k,
                        max_seqlen_k=Lk,
                        block_m=block_m,
                        block_n=block_n,
                        softmax_scale=scale,
                        return_lse=return_lse,
                    )
                    backend_is_varlen = isinstance(cuda_out, torch.Tensor) or (
                        isinstance(cuda_out, tuple)
                        and len(cuda_out) == 2
                        and isinstance(cuda_out[0], torch.Tensor)
                    )
                except Exception:
                    cuda_out = None
                if cuda_out is None:
                    cuda_out = _blocksparse_attention_cuda_flash(
                        q=q,
                        k=k,
                        v=v,
                        block_cols=block_cols,
                        block_m=block_m,
                        block_n=block_n,
                        softmax_scale=scale,
                        return_lse=return_lse,
                    )
                    backend_is_varlen = False
            else:
                cuda_out = _blocksparse_attention_cuda_flash(
                    q=q,
                    k=k,
                    v=v,
                    block_cols=block_cols,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=scale,
                    return_lse=return_lse,
                )
            if return_lse and (
                isinstance(cuda_out, tuple)
                and len(cuda_out) == 2
                and isinstance(cuda_out[0], torch.Tensor)
                and isinstance(cuda_out[1], torch.Tensor)
            ):
                _trace_blocksparse_backend(
                    "cuda_flash_varlen" if backend_is_varlen else "cuda_flash",
                    q=q,
                    Lk=Lk,
                    block_m=block_m,
                    block_n=block_n,
                )
                return cuda_out
            if not return_lse and isinstance(cuda_out, torch.Tensor):
                _trace_blocksparse_backend(
                    "cuda_flash_varlen" if backend_is_varlen else "cuda_flash",
                    q=q,
                    Lk=Lk,
                    block_m=block_m,
                    block_n=block_n,
                )
                return cuda_out
        except Exception as exc:
            if _truthy_env("HORIZONKV_BLOCKSPARSE_VERBOSE", "0"):
                print(f"[horizonkv] CUDA blocksparse fallback after failure: {exc}")

        if _blocksparse_flash_backend_required():
            raise RuntimeError(
                "FlashAttention block-sparse backend was requested but did not produce "
                "an output. Build the bundled HorizonKV third_party/Block-Sparse-Attention "
                "environment, or explicitly set HORIZONKV_REQUIRE_FLASH_BACKEND=0 "
                "to allow the Triton fallback."
            )

    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        return None
    out = torch.empty_like(q)
    lse = torch.empty((rows, Lq), device=q.device, dtype=torch.float32) if return_lse else out
    log2_scale = scale * math.log2(math.e)
    num_warps_default, num_stages_default = _triton_blocksparse_launch_params(
        head_dim=int(head_dim),
        block_m=block_m,
        block_n=block_n,
    )
    num_warps = _env_int("HORIZONKV_TRITON_BLOCKSPARSE_NUM_WARPS", num_warps_default)
    num_stages = _env_int("HORIZONKV_TRITON_BLOCKSPARSE_NUM_STAGES", num_stages_default)
    _trace_blocksparse_backend(
        "triton_packed_dynamic" if packed_varlen_k else "triton",
        q=q,
        Lk=Lk,
        block_m=block_m,
        block_n=block_n,
    )
    _blocksparse_attention_padded_kernel[(n_q_blocks, rows)](
        q,
        k,
        v,
        block_cols,
        k_lens if packed_varlen_k else q,
        cu_seqlens_k if packed_varlen_k else q,
        out,
        lse,
        int(Lq),
        int(Lk),
        int(head_dim),
        int(n_q_blocks),
        int(max_k_blocks),
        float(keep_ratio),
        log2_scale,
        math.log(2.0),
        RETURN_LSE=bool(return_lse),
        PACKED_VARLEN_K=packed_varlen_k,
        BLOCK_COLS_VALID=generated_block_cols,
        EVEN_M=int(Lq) % block_m == 0,
        EVEN_N=(not packed_varlen_k) and int(Lk) % block_n == 0,
        EVEN_D=int(head_dim) == int(block_d),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if return_lse:
        return out, lse.view(rows, 1, Lq)
    return out


def merge_attention_parts_triton(
    out_a: torch.Tensor,
    m_a: torch.Tensor,
    l_a: torch.Tensor,
    out_b: torch.Tensor,
    m_b: torch.Tensor,
    l_b: torch.Tensor,
) -> torch.Tensor | None:
    if triton is None or tl is None:
        return None
    if out_a.shape != out_b.shape or m_a.shape != m_b.shape or l_a.shape != l_b.shape:
        return None
    if out_a.ndim != 4 or out_a.shape[2] != 1:
        return None
    if m_a.ndim != 2 or tuple(m_a.shape) != (int(out_a.shape[0]), int(out_a.shape[1])):
        return None
    tensors = (out_a, out_b, m_a, m_b, l_a, l_b)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return None
    if m_a.dtype != torch.float32 or m_b.dtype != torch.float32 or l_a.dtype != torch.float32 or l_b.dtype != torch.float32:
        return None
    if out_a.dtype != out_b.dtype:
        return None

    out = torch.empty_like(out_a)
    total_elements = int(out.numel())
    if total_elements <= 0:
        return out
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _merge_attention_parts_kernel[grid](
        out_a,
        m_a,
        l_a,
        out_b,
        m_b,
        l_b,
        out,
        total_elements,
        int(out_a.shape[-1]),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def merge_attention_lse_parts_triton(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor | None:
    if triton is None or tl is None:
        return None
    if out_a.shape != out_b.shape or lse_a.shape != lse_b.shape:
        return None
    if out_a.ndim != 4:
        return None
    rows = int(out_a.shape[0])
    Lq = int(out_a.shape[1])
    q_heads = int(out_a.shape[2])
    head_dim = int(out_a.shape[3])
    if tuple(lse_a.shape) != (rows, q_heads, Lq):
        return None
    tensors = (out_a, out_b, lse_a, lse_b)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return None
    if out_a.dtype != out_b.dtype:
        return None
    if lse_a.dtype != torch.float32 or lse_b.dtype != torch.float32:
        return None

    out = torch.empty_like(out_a)
    total_elements = int(out.numel())
    if total_elements <= 0:
        return out
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size),)
    _merge_attention_lse_parts_kernel[grid](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        total_elements,
        Lq,
        q_heads,
        head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


__all__ = [
    "blocksparse_attention_pooled_triton",
    "merge_attention_lse_parts_triton",
    "merge_attention_parts_triton",
]
