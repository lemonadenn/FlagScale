# Modified for the HorizonKV inference artifact; see NOTICE.
"""Pooled-QK block selection and LSE merging; attention uses the CUDA backend."""
from __future__ import annotations
import math
import os
import torch

_BLOCKSPARSE_BACKEND_TRACE_SEEN: set[str] = set()
_BLOCKSPARSE_SELECTOR_TRACE_SEEN: set[str] = set()
_GLOBAL_SELECTION_STATS_TRACE_SEEN: set[tuple[int, int, int, int]] = set()
_GLOBAL_UNIFORM_VARLEN_CONTROLS: dict[
    tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_backend import (
    rows_blocksparse_attention as _rows_blocksparse_attention,
    rows_blocksparse_attention_varlen as _rows_blocksparse_attention_varlen,
)
from flash_attn import flash_attn_func as _flash_attn_func
import triton
import triton.language as tl


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


if triton is not None and tl is not None:

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
        both_invalid = m < -3e38
        safe_m = tl.where(both_invalid, 0.0, m)
        wa = tl.where(la < -3e38, 0.0, tl.exp(la - safe_m))
        wb = tl.where(lb < -3e38, 0.0, tl.exp(lb - safe_m))
        denom = wa + wb
        va = tl.load(out_a + offsets, mask=mask, other=0.0).to(tl.float32)
        vb = tl.load(out_b + offsets, mask=mask, other=0.0).to(tl.float32)
        result = (va * wa + vb * wb) / tl.maximum(denom, 1e-20)
        result = tl.where(both_invalid, 0.0, result)
        tl.store(out + offsets, result, mask=mask)

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
        tl.store(row_blockmask + mask_base + keep_offsets, cols, mask=keep_offsets < max_k_blocks)
        fill_offsets = max_k_blocks + tl.arange(0, BLOCK_MASK)
        tl.store(row_blockmask + mask_base + fill_offsets, -1, mask=fill_offsets < n_k_blocks)

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
            q + q_offsets, mask=is_q & token_valid[:, None] & d_valid[None, :], other=0.0
        )
        k_values = tl.load(
            k + k_offsets, mask=~is_q & token_valid[:, None] & d_valid[None, :], other=0.0
        )
        values = q_values.to(tl.float32) + k_values.to(tl.float32)
        count = tl.minimum(block_size, seqlen - local_block * block_size)
        pooled = tl.sum(values, axis=0) / tl.maximum(count, 1)
        q_out_offsets = (row * n_q_blocks + local_block) * head_dim + offs_d
        k_out_offsets = (row * n_k_blocks + local_block) * head_dim + offs_d
        tl.store(q_out + q_out_offsets, pooled, mask=is_q & d_valid)
        tl.store(k_out + k_out_offsets, pooled, mask=~is_q & d_valid)

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
            q_values = tl.load(q_pool + q_base + offs_d, mask=d_valid, other=0.0).to(tl.float32)
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
        scores = tl.where(offs_k < valid_k_blocks, scores, -1e309)
        score_bits = scores.to(tl.uint32, bitcast=True)
        sign = score_bits >> 31
        ordered = score_bits ^ tl.where(sign != 0, 4294967295, 2147483648)
        packed = ordered.to(tl.uint64) << 32 | offs_k.to(tl.uint64)
        packed = tl.sort(packed, descending=True)
        selected = (packed & 4294967295).to(tl.int32)
        selected = tl.where(offs_k < row_keep, selected, -1)
        tl.store(
            block_cols + (row * n_q_blocks + q_block) * keep + offs_k, selected, mask=offs_k < keep
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
    rows, Lq, _, _ = q.shape
    Lk = int(k.shape[1])
    n_q_blocks = (int(Lq) + int(block_m) - 1) // int(block_m)
    n_k_blocks = (int(Lk) + int(block_n) - 1) // int(block_n)
    max_k_blocks = int(block_cols.shape[2])
    row_blockmask = torch.empty(
        (int(rows), n_q_blocks, n_k_blocks), device=q.device, dtype=torch.int32
    )
    block_keep = triton.next_power_of_2(max_k_blocks)
    block_mask = triton.next_power_of_2(n_k_blocks)
    _block_cols_to_descending_rowmask_kernel[int(rows) * n_q_blocks,](
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
    rows, Lq, _, _ = q.shape
    n_q_blocks = (int(Lq) + int(block_m) - 1) // int(block_m)
    n_k_blocks = (int(max_seqlen_k) + int(block_n) - 1) // int(block_n)
    max_k_blocks = int(block_cols.shape[2])
    row_blockmask = torch.empty(
        (int(rows), n_q_blocks, n_k_blocks), device=q.device, dtype=torch.int32
    )
    block_keep = triton.next_power_of_2(max_k_blocks)
    block_mask = triton.next_power_of_2(n_k_blocks)
    _block_cols_to_descending_rowmask_kernel[int(rows) * n_q_blocks,](
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
        return (result[0], result[1])
    except Exception:
        return None


def _mean_pool_attention_blocks(
    x: torch.Tensor, block: int, *, force_fp32: bool = False
) -> torch.Tensor:
    rows, seqlen, one_head, head_dim = x.shape
    if one_head != 1 or rows <= 0 or seqlen <= 0 or (head_dim <= 0):
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
        or (int(k.shape[-2]) != 1)
        or (int(k.shape[-1]) != int(head_dim))
        or (q.dtype != k.dtype)
        or (q.device != k.device)
        or (not (q.is_cuda and q.is_contiguous() and k.is_contiguous()))
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
            or (cu_seqlens_k.device != q.device)
            or (k_lens.dtype != torch.int32)
            or (cu_seqlens_k.dtype != torch.int32)
            or (not k_lens.is_contiguous())
            or (not cu_seqlens_k.is_contiguous())
            or (int(k_lens.numel()) != int(rows))
            or (int(cu_seqlens_k.numel()) != int(rows) + 1)
            or (int(max_seqlen_k or 0) <= 0)
        ):
            return None
        Lk = int(max_seqlen_k)
    else:
        Lk = int(k.shape[1])
    n_q_blocks = (int(Lq) + block_m - 1) // block_m
    n_k_blocks = (Lk + block_n - 1) // block_n
    out_dtype = torch.float32 if force_fp32 else q.dtype
    q_pool = torch.empty((int(rows), n_q_blocks, int(head_dim)), device=q.device, dtype=out_dtype)
    k_pool = torch.empty((int(rows), n_k_blocks, int(head_dim)), device=q.device, dtype=out_dtype)
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
    return (q_pool, k_pool)


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
        or (int(q_pool.shape[2]) != int(k_pool.shape[2]))
        or (q_pool.device != k_pool.device)
        or (q_pool.dtype != k_pool.dtype)
        or (not (q_pool.is_cuda and q_pool.is_contiguous() and k_pool.is_contiguous()))
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
        or (int(head_dim) > 256)
        or (n_k_blocks <= 1)
        or (n_k_blocks > (512 if packed_varlen_k else 256))
        or (keep <= 0)
        or (keep >= n_k_blocks)
    ):
        return None
    if packed_varlen_k and (
        k_lens.device != q_pool.device
        or k_lens.dtype != torch.int32
        or (not k_lens.is_contiguous())
        or (int(k_lens.numel()) != int(rows))
    ):
        return None
    block_cols = torch.empty(
        (int(rows), int(n_q_blocks), keep), device=q_pool.device, dtype=torch.int32
    )
    block_k = triton.next_power_of_2(n_k_blocks)
    block_d = 16 if int(head_dim) >= 64 else 8
    _pooled_qk_topk_kernel[int(n_q_blocks), int(rows)](
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
    if keep >= n_k_blocks:
        return None
    pooled_qk = None
    pooled_qk = _mean_pool_qk_attention_blocks_triton(
        q,
        k,
        block_m,
        block_n,
        force_fp32=False,
        k_lens=k_lens,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=Lk,
    )
    if pooled_qk is None:
        if packed_varlen_k:
            return None
        q_pool = _mean_pool_attention_blocks(q, block_m, force_fp32=False)
        k_pool = _mean_pool_attention_blocks(k, block_n, force_fp32=False)
    else:
        q_pool, k_pool = pooled_qk
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(float(head_dim))
    block_cols = None
    block_cols = _select_pooled_block_cols_triton(
        q_pool,
        k_pool,
        keep=keep,
        scale=scale,
        round_scores=True,
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
    rows, Lq, _, head_dim = q.shape
    packed_varlen_k = k.ndim == 3
    Lk = int(max_seqlen_k) if packed_varlen_k else int(k.shape[1])
    if sparsity <= 0 and (not packed_varlen_k):
        return _dense_flash_attention_rows(
            q=q, k=k, v=v, softmax_scale=softmax_scale, return_lse=return_lse
        )
    if Lk <= block_n:
        return None
    if block_cols is None:
        block_cols = _select_pooled_block_cols(
            q,
            k,
            keep_ratio=1 - sparsity,
            block_m=block_m,
            block_n=block_n,
            softmax_scale=softmax_scale,
            k_lens=k_lens if packed_varlen_k else None,
            cu_seqlens_k=cu_seqlens_k if packed_varlen_k else None,
            max_seqlen_k=Lk,
        )
    if block_cols is None:
        return None
    scale = softmax_scale if softmax_scale is not None else head_dim ** (-0.5)
    if packed_varlen_k:
        return _blocksparse_attention_cuda_flash_varlen(
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
    return _blocksparse_attention_cuda_flash(
        q=q,
        k=k,
        v=v,
        block_cols=block_cols,
        block_m=block_m,
        block_n=block_n,
        softmax_scale=scale,
        return_lse=return_lse,
    )


def merge_attention_lse_parts_triton(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> torch.Tensor | None:
    rows = int(out_a.shape[0])
    Lq = int(out_a.shape[1])
    q_heads = int(out_a.shape[2])
    head_dim = int(out_a.shape[3])
    tensors = (out_a, out_b, lse_a, lse_b)
    out = torch.empty_like(out_a)
    total_elements = int(out.numel())
    if total_elements <= 0:
        return out
    block_size = 256
    grid = (triton.cdiv(total_elements, 256),)
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
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out
