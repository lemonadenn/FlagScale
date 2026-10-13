# Modified for the HorizonKV inference artifact; see NOTICE.
"""Optional stage-2 block-sparse backend using Block-Sparse-Attention."""

from __future__ import annotations

import math
import os

import torch


_IMPORT_ERROR: Exception | None = None
_BLOCK_SPARSE_ATTN_FUNC = None
_BLOCK_SPARSE_ATTN_FORWARD = None
_ROW_METADATA_CACHE: dict[
    tuple[int, int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_VARLEN_ROW_METADATA_CACHE: dict[
    tuple[int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

# Trace each backend once per process.
_BACKEND_TRACE_SEEN: set[str] = set()

try:
    from block_sparse_attn import block_sparse_attn_func as _BLOCK_SPARSE_ATTN_FUNC
except Exception as exc:  # pragma: no cover - optional dependency path
    _IMPORT_ERROR = exc
    _BLOCK_SPARSE_ATTN_FUNC = None

try:
    from block_sparse_attn.block_sparse_attn_interface import (
        _wrapped_block_sparse_attn_forward as _BLOCK_SPARSE_ATTN_FORWARD,
    )
except Exception:  # pragma: no cover - optional private forward entry point
    _BLOCK_SPARSE_ATTN_FORWARD = None


def blocksparse_backend_is_available() -> bool:
    return _BLOCK_SPARSE_ATTN_FUNC is not None


def rows_blocksparse_backend_is_available() -> bool:
    return _BLOCK_SPARSE_ATTN_FORWARD is not None


def blocksparse_backend_error() -> str | None:
    if _IMPORT_ERROR is None:
        return None
    return f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"


def _trace_backend_once(
    label: str,
    *,
    q: torch.Tensor,
    lk: int,
    block_q: int,
    block_k: int,
) -> None:
    raw = str(os.getenv("HORIZONKV_BLOCKSPARSE_BACKEND_TRACE", "0"))
    if raw.strip().lower() not in {"1", "true", "yes", "on", "y"}:
        return
    if label in _BACKEND_TRACE_SEEN:
        return
    _BACKEND_TRACE_SEEN.add(label)
    print(
        "[horizonkv] blocksparse backend="
        f"{label} rows={int(q.shape[0])} Lq={int(q.shape[1])} "
        f"Lk={int(lk)} H={int(q.shape[2])} D={int(q.shape[3])} "
        f"block={int(block_q)}x{int(block_k)}",
        flush=True,
    )


def stage2_blocksparse_kernel_supported(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    block_q: int,
    block_k: int,
) -> tuple[bool, str | None]:
    if _BLOCK_SPARSE_ATTN_FUNC is None:
        return False, blocksparse_backend_error() or "block_sparse_attn import failed"
    if q.device.type != "cuda" or k.device.type != "cuda":
        return False, "block_sparse_attention backend requires CUDA tensors"
    if q.shape[0] != 1 or k.shape[0] != 1:
        return False, "block_sparse_attention backend currently only supports batch_size=1 here"
    if block_q != block_k:
        return False, "block_sparse_attention backend is currently wired for square blocks only"
    if block_q != 64:
        return (
            False,
            f"HorizonKV requires 64x64 blocks, got {block_q}x{block_k}",
        )
    return True, None


def _reshape_for_varlen(x: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, nheads, dim = x.shape
    assert bsz == 1
    return x.reshape(seqlen, nheads, dim).contiguous()


def _rows_blocksparse_metadata(
    device: torch.device,
    *,
    rows: int,
    lq: int,
    lk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    key = (device_index, int(rows), int(lq), int(lk))
    cached = _ROW_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    cu_q = torch.tensor([0, int(lq)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, int(lk)], device=device, dtype=torch.int32)
    head_mask_type = torch.arange(1, int(rows) + 1, device=device, dtype=torch.int32)
    streaming_info = torch.zeros(int(rows) * 2, device=device, dtype=torch.int32)
    result = (cu_q, cu_k, head_mask_type, streaming_info)
    _ROW_METADATA_CACHE[key] = result
    return result


@torch.no_grad()
def rows_blocksparse_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_blockmask: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run the vendored FlashAttention-style CUDA kernel on per-head rows."""
    if _BLOCK_SPARSE_ATTN_FORWARD is None:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if tuple(k.shape) != tuple(v.shape):
        return None
    if (
        q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    ):
        return None
    rows, lq, one_head, head_dim = q.shape
    lk = int(k.shape[1])
    if (
        int(one_head) != 1
        or int(k.shape[0]) != int(rows)
        or int(k.shape[2]) != 1
        or int(k.shape[-1]) != int(head_dim)
    ):
        return None
    n_q_blocks = math.ceil(int(lq) / int(block_m))
    n_k_blocks = math.ceil(int(lk) / int(block_n))
    if (
        row_blockmask.device != q.device
        or row_blockmask.dtype != torch.int32
        or not row_blockmask.is_contiguous()
        or tuple(row_blockmask.shape) != (int(rows), n_q_blocks, n_k_blocks)
    ):
        return None

    cu_q, cu_k, head_mask_type, streaming_info = _rows_blocksparse_metadata(
        q.device,
        rows=int(rows),
        lq=int(lq),
        lk=int(lk),
    )
    q_rows = q.squeeze(2).permute(1, 0, 2)
    k_rows = k.squeeze(2).permute(1, 0, 2)
    v_rows = v.squeeze(2).permute(1, 0, 2)
    forward_result = _BLOCK_SPARSE_ATTN_FORWARD(
        q_rows,
        k_rows,
        v_rows,
        cu_q,
        cu_k,
        int(block_m),
        int(block_n),
        head_mask_type,
        streaming_info,
        row_blockmask.unsqueeze(0),
        int(lq),
        int(lk),
        0.0,
        float(softmax_scale),
        False,
        False,
        False,
        -1,
        -1,
    )
    out = forward_result[0]
    # The inverse head-row permutation preserves a contiguous view.
    out = out.permute(1, 0, 2).unsqueeze(2).contiguous()
    _trace_backend_once(
        "cuda_flash_rows",
        q=q,
        lk=lk,
        block_q=int(block_m),
        block_k=int(block_n),
    )
    if not return_lse:
        return out

    softmax_lse = forward_result[1]
    if (
        not isinstance(softmax_lse, torch.Tensor)
        or softmax_lse.ndim != 3
        or int(softmax_lse.shape[0]) != 1
        or int(softmax_lse.shape[1]) != int(rows)
        or int(softmax_lse.shape[2]) < int(lq)
    ):
        return None
    return out, softmax_lse[0, :, : int(lq)].unsqueeze(1).contiguous()


@torch.no_grad()
def batch_blocksparse_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_q: int,
    block_k: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run block-sparse attention directly on [B,L,H,D], avoiding per-head copies."""
    if _BLOCK_SPARSE_ATTN_FORWARD is None:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if tuple(k.shape) != tuple(v.shape):
        return None
    if q.device.type != "cuda" or q.device != k.device or q.device != v.device:
        return None
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return None
    B, Lq, H, D = (int(dim) for dim in q.shape)
    if B != 1 or int(k.shape[2]) != H or int(k.shape[3]) != D:
        return None
    block_q = int(block_q)
    block_k = int(block_k)
    if block_q <= 0 or block_k <= 0:
        return None
    n_bq = math.ceil(Lq / block_q)
    Lk = int(k.shape[1])
    n_bk = math.ceil(Lk / block_k)
    if tuple(block_mask.shape) == (n_bq, n_bk):
        block_mask = block_mask.unsqueeze(0).expand(H, -1, -1)
    if tuple(block_mask.shape) != (H, n_bq, n_bk):
        return None
    if block_mask.device != q.device:
        return None

    # The extension expects descending key-block IDs padded with -1.
    block_mask_u8 = block_mask.to(dtype=torch.uint8)
    nonzero_val, nonzero_idx = block_mask_u8.sort(
        dim=-1, stable=True, descending=False
    )
    row_blockmask = torch.where(
        nonzero_val != 0,
        nonzero_idx,
        torch.full_like(nonzero_idx, -1),
    )
    row_blockmask = torch.flip(row_blockmask, dims=[-1]).to(dtype=torch.int32)
    row_blockmask = row_blockmask.unsqueeze(0).contiguous()

    cu_q, cu_k, head_mask_type, streaming_info = _rows_blocksparse_metadata(
        q.device,
        rows=H,
        lq=Lq,
        lk=Lk,
    )
    # Remove the singleton batch dimension without a layout copy.
    q_unpad = q.squeeze(0)
    k_unpad = k.squeeze(0)
    v_unpad = v.squeeze(0)
    try:
        forward_result = _BLOCK_SPARSE_ATTN_FORWARD(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_q,
            cu_k,
            block_q,
            block_k,
            head_mask_type,
            streaming_info,
            row_blockmask,
            Lq,
            Lk,
            0.0,
            float(softmax_scale),
            False,
            False,
            False,
            -1,
            -1,
        )
    except Exception:
        return None
    if not isinstance(forward_result, tuple) or len(forward_result) < 2:
        return None
    out_unpad = forward_result[0]
    if not isinstance(out_unpad, torch.Tensor) or tuple(out_unpad.shape) != (Lq, H, D):
        return None
    out = out_unpad.unsqueeze(0).contiguous()
    _trace_backend_once(
        "cuda_flash_batched",
        q=q,
        lk=Lk,
        block_q=block_q,
        block_k=block_k,
    )
    if not return_lse:
        return out
    softmax_lse = forward_result[1]
    if (
        not isinstance(softmax_lse, torch.Tensor)
        or softmax_lse.ndim != 3
        or int(softmax_lse.shape[0]) != 1
        or int(softmax_lse.shape[1]) != H
        or int(softmax_lse.shape[2]) < Lq
    ):
        return None
    return out, softmax_lse[:, :, :Lq].contiguous()


def _varlen_rows_blocksparse_metadata(
    device: torch.device,
    *,
    rows: int,
    lq: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    key = (device_index, int(rows), int(lq))
    cached = _VARLEN_ROW_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    cu_q = torch.arange(
        0,
        (int(rows) + 1) * int(lq),
        step=int(lq),
        device=device,
        dtype=torch.int32,
    )
    head_mask_type = torch.ones(1, device=device, dtype=torch.int32)
    streaming_info = torch.zeros(2, device=device, dtype=torch.int32)
    result = (cu_q, head_mask_type, streaming_info)
    _VARLEN_ROW_METADATA_CACHE[key] = result
    return result


@torch.no_grad()
def rows_blocksparse_attention_varlen(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    row_blockmask: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run packed ragged K/V as one blocksparse sequence per query row."""
    if _BLOCK_SPARSE_ATTN_FORWARD is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or v.ndim != 3 or tuple(k.shape) != tuple(v.shape):
        return None
    if (
        q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    ):
        return None
    rows, lq, one_head, head_dim = q.shape
    max_seqlen_k = int(max_seqlen_k)
    if (
        int(one_head) != 1
        or int(k.shape[1]) != 1
        or int(k.shape[2]) != int(head_dim)
        or max_seqlen_k <= 0
        or cu_seqlens_k.device != q.device
        or cu_seqlens_k.dtype != torch.int32
        or not cu_seqlens_k.is_contiguous()
        or int(cu_seqlens_k.numel()) != int(rows) + 1
    ):
        return None
    n_q_blocks = math.ceil(int(lq) / int(block_m))
    n_k_blocks = math.ceil(max_seqlen_k / int(block_n))
    if (
        row_blockmask.device != q.device
        or row_blockmask.dtype != torch.int32
        or not row_blockmask.is_contiguous()
        or tuple(row_blockmask.shape) != (int(rows), n_q_blocks, n_k_blocks)
    ):
        return None

    cu_q, head_mask_type, streaming_info = _varlen_rows_blocksparse_metadata(
        q.device,
        rows=int(rows),
        lq=int(lq),
    )
    q_packed = q.reshape(int(rows) * int(lq), 1, int(head_dim))
    forward_result = _BLOCK_SPARSE_ATTN_FORWARD(
        q_packed,
        k,
        v,
        cu_q,
        cu_seqlens_k,
        int(block_m),
        int(block_n),
        head_mask_type,
        streaming_info,
        row_blockmask.unsqueeze(1),
        int(lq),
        max_seqlen_k,
        0.0,
        float(softmax_scale),
        False,
        False,
        False,
        -1,
        -1,
    )
    out = forward_result[0].view(int(rows), int(lq), 1, int(head_dim)).contiguous()
    if not return_lse:
        return out

    softmax_lse = forward_result[1]
    if (
        not isinstance(softmax_lse, torch.Tensor)
        or softmax_lse.ndim != 3
        or int(softmax_lse.shape[0]) != int(rows)
        or int(softmax_lse.shape[1]) != 1
        or int(softmax_lse.shape[2]) < int(lq)
    ):
        return None
    return out, softmax_lse[:, :, : int(lq)].contiguous()


def _dense_block_mask_from_token_mask(
    token_mask_2d: torch.Tensor,
    *,
    block_q: int,
    block_k: int,
) -> torch.Tensor:
    if token_mask_2d.dtype != torch.bool:
        token_mask_2d = token_mask_2d.to(dtype=torch.bool)
    lq, lk = token_mask_2d.shape
    q_pad = (block_q - (lq % block_q)) % block_q
    k_pad = (block_k - (lk % block_k)) % block_k
    if q_pad or k_pad:
        token_mask_2d = torch.nn.functional.pad(token_mask_2d, (0, k_pad, 0, q_pad), value=False)
    n_bq = token_mask_2d.shape[0] // block_q
    n_bk = token_mask_2d.shape[1] // block_k
    return (
        token_mask_2d.view(n_bq, block_q, n_bk, block_k)
        .any(dim=3)
        .any(dim=1)
        .contiguous()
    )


def stage2_blocksparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_mask_2d: torch.Tensor | None = None,
    *,
    block_mask: torch.Tensor | None = None,
    block_q: int = 64,
    block_k: int = 64,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    ok, reason = stage2_blocksparse_kernel_supported(q=q, k=k, block_q=block_q, block_k=block_k)
    if not ok:
        raise RuntimeError(reason or "block_sparse_attention backend is not available")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(q.shape[-1]))

    q_unpad = _reshape_for_varlen(q)
    k_unpad = _reshape_for_varlen(k)
    v_unpad = _reshape_for_varlen(v)
    cu_q = torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor([0, k.shape[1]], dtype=torch.int32, device=k.device)
    nheads = q.shape[2]
    head_mask_type = torch.ones((nheads,), dtype=torch.int32, device=q.device)
    if block_mask is not None:
        if block_mask.dtype != torch.bool:
            block_mask = block_mask.to(dtype=torch.bool)
        if block_mask.ndim == 2:
            block_mask = block_mask.unsqueeze(0).unsqueeze(0)
        elif block_mask.ndim == 3:
            block_mask = block_mask.unsqueeze(0)
        if block_mask.ndim != 4:
            raise ValueError("block_mask must have shape [n_bq, n_bk], [H, n_bq, n_bk], or [B, H, n_bq, n_bk]")
        if block_mask.shape[0] != 1:
            raise ValueError("block_sparse_attention backend currently only supports batch_size=1 here")
        if block_mask.shape[1] == 1 and nheads > 1:
            block_mask = block_mask.expand(1, nheads, -1, -1)
        elif block_mask.shape[1] != nheads:
            raise ValueError(
                f"block_mask head dimension mismatch: got {block_mask.shape[1]}, expected {nheads}"
            )
        base_blockmask = block_mask.to(dtype=torch.int32).contiguous()
    else:
        if token_mask_2d is None:
            raise ValueError("either token_mask_2d or block_mask must be provided")
        base_blockmask = _dense_block_mask_from_token_mask(
            token_mask_2d[0, 0],
            block_q=block_q,
            block_k=block_k,
        ).to(dtype=torch.int32).unsqueeze(0).expand(1, nheads, -1, -1).contiguous()

    out = _BLOCK_SPARSE_ATTN_FUNC(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_q,
        cu_k,
        head_mask_type,
        None,
        base_blockmask,
        q.shape[1],
        k.shape[1],
        0.0,
        deterministic=False,
        softmax_scale=softmax_scale,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
        m_block_dim=block_q,
        n_block_dim=block_k,
    )
    return out.view(1, q.shape[1], nheads, q.shape[-1]).contiguous()


__all__ = [
    "batch_blocksparse_attention",
    "blocksparse_backend_error",
    "blocksparse_backend_is_available",
    "rows_blocksparse_attention",
    "rows_blocksparse_attention_varlen",
    "rows_blocksparse_backend_is_available",
    "stage2_blocksparse_attention",
    "stage2_blocksparse_kernel_supported",
]
