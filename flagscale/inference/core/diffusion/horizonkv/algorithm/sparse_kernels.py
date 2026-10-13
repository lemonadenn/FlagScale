# Modified for the HorizonKV inference artifact; see NOTICE.
"""LightForcing sparse attention and compacted RoPE kernels."""

from __future__ import annotations

import os
import warnings

import torch

try:  # pragma: no cover - exercised only on CUDA environments
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def _device_sm(device=None):
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


DEVICE_SM = _device_sm()
_LF_BACKEND_TRACE_SEEN: set[str] = set()

# Cache device geometry to avoid repeated small pageable H2D transfers.
_LF_GRID_DEVICE_CACHE: dict[tuple[str, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
_LF_GRID_CPU_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
_LF_GRID_DEVICE_READY: dict[
    tuple[str, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.cuda.Event, int],
] = {}


def _lf_grid_sizes_on_device(
    grid_sizes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached contiguous CUDA copy of the grid geometry."""
    if not isinstance(grid_sizes, torch.Tensor):
        raise TypeError("grid_sizes must be a tensor")
    device = torch.device(device)
    if grid_sizes.device == device and grid_sizes.dtype == torch.int64 and grid_sizes.is_contiguous():
        return grid_sizes

    # Use value-based caching only for CPU metadata to avoid device synchronization.
    if grid_sizes.device.type != "cpu":
        return grid_sizes.to(device=device, dtype=torch.int64).contiguous()
    values = tuple(int(v) for v in grid_sizes.detach().reshape(-1).tolist())
    shape = tuple(int(v) for v in grid_sizes.shape)
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, shape, values)
    cached = _LF_GRID_DEVICE_CACHE.get(key)
    if (
        isinstance(cached, torch.Tensor)
        and cached.device == device
        and cached.dtype == torch.int64
        and tuple(cached.shape) == tuple(grid_sizes.shape)
    ):
        ready = _LF_GRID_DEVICE_READY.get(key)
        if ready is not None and torch.cuda.is_available():
            current_stream = torch.cuda.current_stream(device)
            producer_stream = int(ready[1])
            if int(current_stream.cuda_stream) != producer_stream:
                current_stream.wait_event(ready[0])
        return cached
    source = grid_sizes
    # Pin the first geometry transfer as well.
    if torch.cuda.is_available():
        cpu_key = (shape, values)
        cached_cpu = _LF_GRID_CPU_CACHE.get(cpu_key)
        if (
            isinstance(cached_cpu, torch.Tensor)
            and cached_cpu.device.type == "cpu"
            and cached_cpu.dtype == torch.int64
            and tuple(cached_cpu.shape) == shape
        ):
            source = cached_cpu
        else:
            try:
                if not source.is_pinned():
                    source = source.pin_memory()
            except Exception:
                source = grid_sizes
            if source.device.type == "cpu" and source.dtype == torch.int64 and source.is_pinned():
                _LF_GRID_CPU_CACHE[cpu_key] = source
    uploaded = source.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    _LF_GRID_DEVICE_CACHE[key] = uploaded
    if torch.cuda.is_available() and device.type == "cuda":
        producer_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        _LF_GRID_DEVICE_READY[key] = (ready_event, int(producer_stream.cuda_stream))
    return uploaded


def _trace_lf_backend_once(
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
    if label in _LF_BACKEND_TRACE_SEEN:
        return
    _LF_BACKEND_TRACE_SEEN.add(label)
    print(
        "[horizonkv] lf_rope attention backend="
        f"{label} B={int(q.shape[0])} Lq={int(q.shape[1])} "
        f"Lk={int(lk)} "
        f"H={int(q.shape[2])} D={int(q.shape[3])} "
        f"block={int(block_q)}x{int(block_k)}",
        flush=True,
    )


def _dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float):
    # Keep the fallback independent of ``wan`` package initialization so CPU
    # selector tests can run on machines without a CUDA driver.
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=softmax_scale,
    ).transpose(1, 2).contiguous()


def _rope_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0):
    """Reference implementation matching Wan's causal rotary layout."""
    n, c = x.size(2), x.size(3) // 2
    split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = int(f * h * w)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            split_freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]], dim=0))
    return torch.stack(output).type_as(x)


if triton is not None:

    @triton.jit
    def _causal_rope_apply_kernel(
        OUT, X, FREQS_REAL, GRID,
        NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SEQ_STRIDE: tl.constexpr, T_DIM: tl.constexpr,
        H_DIM: tl.constexpr, START_FRAME: tl.constexpr,
        CONJUGATE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        HEAD_MAJOR: tl.constexpr = False, SELECTED_FRAMES: tl.constexpr = 0,
    ):
        row_idx = tl.program_id(0)
        batch_idx = row_idx // (SEQ_STRIDE * NUM_HEADS)
        seq_head_idx = row_idx - batch_idx * SEQ_STRIDE * NUM_HEADS
        token_idx = seq_head_idx % SEQ_STRIDE if HEAD_MAJOR else seq_head_idx // NUM_HEADS
        if SELECTED_FRAMES > 0:
            f = SELECTED_FRAMES
        else:
            f = tl.load(GRID + batch_idx * 3)
        h = tl.load(GRID + batch_idx * 3 + 1)
        w = tl.load(GRID + batch_idx * 3 + 2)
        valid_seq_len = f * h * w
        half_head_dim: tl.constexpr = HEAD_DIM // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < half_head_dim
        x_row = X + row_idx * HEAD_DIM
        out_row = OUT + row_idx * HEAD_DIM
        x_real = tl.load(x_row + col_offsets * 2, mask=mask, other=0.0)
        x_imag = tl.load(x_row + col_offsets * 2 + 1, mask=mask, other=0.0)
        is_valid = token_idx < valid_seq_len
        hw = h * w
        t = token_idx // hw
        rem = token_idx - t * hw
        y = rem // w
        z = rem - y * w
        freq_row = tl.where(
            col_offsets < T_DIM,
            START_FRAME + t,
            tl.where(col_offsets < T_DIM + H_DIM, y, z),
        )
        freq_offset = (freq_row * half_head_dim + col_offsets) * 2
        load_mask = mask & is_valid
        cos_vals = tl.load(FREQS_REAL + freq_offset, mask=load_mask, other=1.0)
        sin_vals = tl.load(FREQS_REAL + freq_offset + 1, mask=load_mask, other=0.0)
        if CONJUGATE:
            sin_vals = -sin_vals
        xr = x_real.to(tl.float32)
        xi = x_imag.to(tl.float32)
        rr = xr * cos_vals.to(tl.float32) - xi * sin_vals.to(tl.float32)
        ri = xr * sin_vals.to(tl.float32) + xi * cos_vals.to(tl.float32)
        tl.store(out_row + col_offsets * 2, tl.where(is_valid, rr, xr), mask=mask)
        tl.store(out_row + col_offsets * 2 + 1, tl.where(is_valid, ri, xi), mask=mask)


def lightforcing_rope_apply(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0
) -> torch.Tensor:
    """LightForcing's inference RoPE kernel, with a reference fallback."""
    if triton is None or not x.is_cuda or x.shape[-1] % 2:
        return _rope_reference(x, grid_sizes, freqs, start_frame=start_frame)
    b, seq_len, heads, head_dim = x.shape
    c = head_dim // 2
    t_dim = c - 2 * (c // 3)
    h_dim = c // 3
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device))
    block_size = triton.next_power_of_2(c)
    _causal_rope_apply_kernel[(b * seq_len * heads,)](
        out, x_contig, freqs_real, grid, heads, head_dim, seq_len,
        t_dim, h_dim, int(start_frame), False, BLOCK_SIZE=block_size,
        num_warps=1, num_stages=2,
    )
    return out.type_as(x)


def _rope_apply_packed_rows_reference(
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    row_lens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int,
    inplace: bool,
) -> torch.Tensor:
    """Reference RoPE for hit/miss rows using ordinals in the full selected window."""
    if x.ndim == 4:
        rows, max_tokens, heads, head_dim = (int(v) for v in x.shape)
        packed = False
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is only valid for packed row tensors")
    elif x.ndim == 3:
        total_tokens, heads, head_dim = (int(v) for v in x.shape)
        rows = int(frame_positions.shape[0])
        packed = True
        if cu_seqlens is None:
            raise ValueError("packed row RoPE requires cu_seqlens")
    else:
        raise ValueError("packed row RoPE expects a 3D or 4D tensor")
    if heads <= 0 or head_dim <= 0 or head_dim % 2:
        return x
    if frame_positions.ndim != 2 or int(frame_positions.shape[0]) != rows:
        raise ValueError("frame_positions shape does not match compact rows")
    if row_lens.ndim != 1 or int(row_lens.numel()) != rows:
        raise ValueError("row_lens shape does not match compact rows")
    if int(grid_sizes.ndim) != 2 or int(grid_sizes.shape[1]) < 3:
        raise ValueError("grid_sizes must have shape [B,3]")

    # All rows share the spatial grid.
    height = int(grid_sizes[0, 1].item())
    width = int(grid_sizes[0, 2].item())
    if height <= 0 or width <= 0:
        return x
    frame_seqlen = int(frame_seqlen)
    if frame_seqlen <= 0:
        return x
    c = head_dim // 2
    t_dim = c - 2 * (c // 3)
    h_dim = c // 3
    split_freqs = freqs.split([t_dim, h_dim, h_dim], dim=1)
    out = x if inplace else x.clone()

    def apply_one(row: int, src: torch.Tensor, dst: torch.Tensor) -> None:
        valid_tokens = int(row_lens[row].item())
        valid_tokens = max(0, min(valid_tokens, int(src.shape[0])))
        if valid_tokens <= 0:
            return
        frame_count = (valid_tokens + frame_seqlen - 1) // frame_seqlen
        positions = frame_positions[row, :frame_count].to(device=src.device, dtype=torch.long)
        token_idx = torch.arange(valid_tokens, device=src.device, dtype=torch.long)
        frame_idx = token_idx // frame_seqlen
        temporal = positions.index_select(0, frame_idx) + int(start_frame)
        in_frame = token_idx - frame_idx * frame_seqlen
        y = in_frame // width
        z = in_frame - y * width
        freq_rows = torch.cat(
            [
                split_freqs[0].index_select(0, temporal),
                split_freqs[1].index_select(0, y),
                split_freqs[2].index_select(0, z),
            ],
            dim=1,
        )
        values = torch.view_as_complex(
            src[:valid_tokens].to(torch.float64).reshape(valid_tokens, heads, -1, 2)
        )
        values = torch.view_as_real(values * freq_rows.view(valid_tokens, 1, -1)).flatten(2)
        dst[:valid_tokens].copy_(values.type_as(dst))

    if packed:
        cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        for row in range(rows):
            start = int(cu_cpu[row].item())
            end = int(cu_cpu[row + 1].item())
            apply_one(row, x[start:end], out[start:end])
    else:
        for row in range(rows):
            apply_one(row, x[row], out[row])
    return out.type_as(x)


if triton is not None:

    @triton.jit
    def _packed_rows_rope_apply_kernel(
        OUT,
        X,
        FRAME_POSITIONS,
        ROW_LENS,
        CU_SEQLENS,
        GRID,
        FREQS_REAL,
        MAX_TOKENS: tl.constexpr,
        MAX_FRAMES: tl.constexpr,
        FRAME_SEQLEN: tl.constexpr,
        START_FRAME: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        T_DIM: tl.constexpr,
        H_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row = pid // MAX_TOKENS
        token = pid - row * MAX_TOKENS
        row_len = tl.load(ROW_LENS + row).to(tl.int32)
        valid = token < row_len
        if PACKED:
            token_offset = tl.load(CU_SEQLENS + row).to(tl.int64) + token
        else:
            token_offset = row * MAX_TOKENS + token

        half_dim: tl.constexpr = HEAD_DIM // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < half_dim
        x_ptr = X + token_offset * HEAD_DIM
        out_ptr = OUT + token_offset * HEAD_DIM
        # Skip invalid packed rows; padded rows retain their original values.
        if PACKED:
            data_mask = col_mask & valid
        else:
            data_mask = col_mask
        x_real = tl.load(x_ptr + col_offsets * 2, mask=data_mask, other=0.0)
        x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=data_mask, other=0.0)

        frame_idx = token // FRAME_SEQLEN
        frame_pos = tl.load(
            FRAME_POSITIONS + row * MAX_FRAMES + frame_idx,
            mask=valid & (frame_idx < MAX_FRAMES),
            other=0,
        ).to(tl.int64) + START_FRAME
        in_frame = token - frame_idx * FRAME_SEQLEN
        width = tl.load(GRID + 2).to(tl.int32)
        y = in_frame // width
        z = in_frame - y * width
        freq_row = tl.where(
            col_offsets < T_DIM,
            frame_pos,
            tl.where(col_offsets < T_DIM + H_DIM, y, z),
        )
        freq_offset = (freq_row * half_dim + col_offsets) * 2
        load_mask = col_mask & valid
        cos_vals = tl.load(FREQS_REAL + freq_offset, mask=load_mask, other=1.0)
        sin_vals = tl.load(FREQS_REAL + freq_offset + 1, mask=load_mask, other=0.0)
        xr = x_real.to(tl.float32)
        xi = x_imag.to(tl.float32)
        rr = xr * cos_vals.to(tl.float32) - xi * sin_vals.to(tl.float32)
        ri = xr * sin_vals.to(tl.float32) + xi * cos_vals.to(tl.float32)
        tl.store(
            out_ptr + col_offsets * 2,
            tl.where(valid, rr, xr),
            mask=data_mask,
        )
        tl.store(
            out_ptr + col_offsets * 2 + 1,
            tl.where(valid, ri, xi),
            mask=data_mask,
        )


def lightforcing_rope_apply_packed_rows(
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    row_lens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    *,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    inplace: bool = False,
) -> torch.Tensor | None:
    """Apply RoPE using full-window ordinals, preserving padded or packed layout.

    The input stays raw unless inplace=True; miss KV must stay raw until slot fill."""
    if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4):
        return None
    if not isinstance(frame_positions, torch.Tensor) or frame_positions.ndim != 2:
        return None
    if not isinstance(row_lens, torch.Tensor) or row_lens.ndim != 1:
        return None
    if not isinstance(grid_sizes, torch.Tensor) or not isinstance(freqs, torch.Tensor):
        return None
    if x.shape[-1] % 2 or int(x.shape[-2]) != 1:
        return None
    rows = int(x.shape[0]) if x.ndim == 4 else int(row_lens.numel())
    if int(frame_positions.shape[0]) != rows or int(row_lens.numel()) != rows:
        return None
    if x.ndim == 3 and (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or int(cu_seqlens.numel()) != rows + 1
    ):
        return None
    if int(frame_seqlen) <= 0:
        return None

    # Pin CPU row-plan metadata before asynchronous H2D.
    if frame_positions.device.type == "cpu" and torch.cuda.is_available():
        try:
            if not frame_positions.is_pinned():
                frame_positions = frame_positions.pin_memory()
        except Exception:
            pass
    frame_positions = frame_positions.to(
        device=x.device,
        dtype=torch.long,
        non_blocking=True,
    ).contiguous()
    row_lens = row_lens.to(device=x.device, dtype=torch.int32).contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()

    # Launch by plan width; ROW_LENS masks padding without device-side reductions.
    max_frames = int(frame_positions.shape[1])
    max_tokens = (
        int(x.shape[1])
        if x.ndim == 4
        else max(1, max_frames * int(frame_seqlen))
    )
    if max_frames <= 0 or max_tokens <= 0:
        return None

    # Hits may be rotated in place; misses stay raw until asynchronous slot fill.
    inplace = bool(inplace and x.is_contiguous())

    if triton is None or not x.is_cuda:
        return _rope_apply_packed_rows_reference(
            x,
            frame_positions,
            row_lens,
            cu_seqlens,
            int(frame_seqlen),
            grid_sizes,
            freqs,
            int(start_frame),
            inplace,
        )

    head_dim = int(x.shape[-1])
    half_dim = head_dim // 2
    t_dim = half_dim - 2 * (half_dim // 3)
    h_dim = half_dim // 3
    block_size = triton.next_power_of_2(half_dim)
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device)).contiguous()
    out = x if inplace else torch.empty_like(x)
    _packed_rows_rope_apply_kernel[(rows * max_tokens,)](
        out,
        x.contiguous(),
        frame_positions,
        row_lens,
        cu_seqlens if cu_seqlens is not None else frame_positions,
        grid,
        freqs_real,
        MAX_TOKENS=max_tokens,
        MAX_FRAMES=max_frames,
        FRAME_SEQLEN=int(frame_seqlen),
        START_FRAME=int(start_frame),
        HEAD_DIM=head_dim,
        T_DIM=t_dim,
        H_DIM=h_dim,
        BLOCK_SIZE=block_size,
        PACKED=x.ndim == 3,
        num_warps=1,
        num_stages=2,
    )
    return out.type_as(x)


def lightforcing_rope_apply_compact(
    x: torch.Tensor,
    *,
    num_heads: int,
    selected_frames: int,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor | None:
    """Apply RoPE to uniform [B*H,T,1,D] rows, anchored at start_frame."""
    if (
        triton is None or not x.is_cuda or not x.is_contiguous()
        or x.ndim != 4 or int(x.shape[2]) != 1 or int(x.shape[-1]) % 2
        or int(num_heads) <= 0 or int(x.shape[0]) % int(num_heads)
        or int(selected_frames) <= 0 or int(frame_seqlen) <= 0
        or int(x.shape[1]) != int(selected_frames) * int(frame_seqlen)
    ):
        return None
    rows, tokens, _, head_dim = x.shape
    half_dim = head_dim // 2
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device)).contiguous()
    out = torch.empty_like(x)
    _causal_rope_apply_kernel[(rows * tokens,)](
        out, x, freqs_real, grid, int(num_heads), head_dim, tokens,
        half_dim - 2 * (half_dim // 3), half_dim // 3, int(start_frame), False,
        BLOCK_SIZE=triton.next_power_of_2(half_dim),
        HEAD_MAJOR=True,
        SELECTED_FRAMES=int(selected_frames),
        num_warps=1,
        num_stages=2,
    )
    return out


if triton is not None:

    @triton.jit
    def _block_mean_kernel(X, XM, H, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr):
        idx_l = tl.program_id(0)
        idx_bh = tl.program_id(1)
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, D)
        x_base = idx_b * L * H * D + idx_h * D
        x = tl.load(
            X + x_base + offs_l[:, None] * (H * D) + offs_d[None, :],
            mask=offs_l[:, None] < L,
            other=0.0,
        ).to(tl.float32)
        blocks = (L + BLOCK_L - 1) // BLOCK_L
        out_base = idx_b * blocks * H * D + idx_l * H * D + idx_h * D
        nx = tl.minimum(BLOCK_L, L - idx_l * BLOCK_L)
        tl.store(XM + out_base + offs_d, (tl.sum(x, axis=0) / nx).to(XM.dtype.element_ty), mask=offs_d < D)


    def _mean_pool_blhd(x: torch.Tensor, block: int) -> torch.Tensor:
        b, length, h, d = x.shape
        blocks = (length + block - 1) // block
        out = torch.empty((b, blocks, h, d), device=x.device, dtype=x.dtype)
        _block_mean_kernel[(blocks, b * h)](
            x.contiguous(), out, h, length, d, block,
            num_warps=4, num_stages=3,
        )
        return out


    @triton.jit
    def _selected_block_score_kernel(
        QP, KP, KEEP, SCORES,
        H: tl.constexpr, Q_BLOCKS: tl.constexpr, K_BLOCKS: tl.constexpr,
        D: tl.constexpr, KEEP_FRAMES: tl.constexpr, KEEP_OFFSET: tl.constexpr,
        KEEP_SINK: tl.constexpr, KEEP_NEAR: tl.constexpr,
        FRAME_SEQ: tl.constexpr, F_PAST: tl.constexpr, PAST_TOKENS: tl.constexpr,
        K_TOKENS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        idx_k = tl.program_id(0)
        idx_bhq = tl.program_id(1).to(tl.int64)
        idx_q = idx_bhq % Q_BLOCKS
        idx_bh = idx_bhq // Q_BLOCKS
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        offs_k = idx_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        block_start = offs_k * BLOCK_N
        block_end = tl.minimum(block_start + BLOCK_N, K_TOKENS)
        first_frame = block_start // FRAME_SEQ
        last_frame = (block_end - 1) // FRAME_SEQ
        is_tail = block_end > PAST_TOKENS
        keep_base = (idx_bh * Q_BLOCKS + idx_q) * KEEP_FRAMES
        is_sink = (first_frame < KEEP_SINK) | (last_frame < KEEP_SINK)
        is_near = (last_frame >= F_PAST - KEEP_NEAR) & (first_frame < F_PAST)
        is_keep = is_tail | is_sink | is_near
        for i in tl.static_range(0, KEEP_FRAMES):
            frame = tl.load(KEEP + keep_base + i).to(tl.int64) + KEEP_OFFSET
            is_keep = is_keep | (first_frame == frame) | (last_frame == frame)
        is_keep = is_keep & (offs_k < K_BLOCKS)
        q_base = idx_b * Q_BLOCKS * H * D + idx_q * H * D + idx_h * D
        q = tl.load(QP + q_base + offs_d, mask=offs_d < D, other=0.0)
        k_base = idx_b * K_BLOCKS * H * D + offs_k[:, None] * H * D + idx_h * D
        kval = tl.load(KP + k_base + offs_d[None, :], mask=is_keep[:, None] & (offs_d[None, :] < D), other=0.0)
        score = tl.sum(kval * q[None, :], axis=1)
        tl.store(
            SCORES + (idx_bh * Q_BLOCKS + idx_q) * K_BLOCKS + offs_k,
            tl.where(is_keep, score, -float("inf")),
            mask=offs_k < K_BLOCKS,
        )


    def _score_selected_blocks(
        pooled_q, pooled_k, keep_idx, frame_seq, f_past, k_tokens,
        block_n, keep_offset=0, keep_sink=0, keep_near=0,
    ):
        b, q_blocks, h, d = pooled_q.shape
        k_blocks = pooled_k.shape[1]
        scores = torch.empty((b, h, q_blocks, k_blocks), device=pooled_q.device, dtype=pooled_q.dtype)
        _selected_block_score_kernel[(triton.cdiv(k_blocks, 64), b * h * q_blocks)](
            pooled_q, pooled_k, keep_idx, scores, h, q_blocks, k_blocks, d,
            keep_idx.shape[-1], keep_offset, keep_sink, keep_near, frame_seq,
            f_past, f_past * frame_seq, k_tokens, block_n, 64,
            triton.next_power_of_2(d), num_warps=4, num_stages=3,
        )
        return scores


def _select_middle_frames(pooled_q, k, frame_seq, f_past, keep_frames, keep_sink, keep_near):
    b, q_blocks, h, _ = pooled_q.shape
    middle_start = keep_sink
    middle_end = f_past - keep_near
    keep_middle = keep_frames - keep_sink - keep_near
    if keep_middle <= 0 or middle_end <= middle_start:
        return torch.empty((b, h, q_blocks, 0), device=pooled_q.device, dtype=torch.long)
    historical_tokens = f_past * frame_seq
    frame_k = k[:, :historical_tokens].reshape(b, f_past, frame_seq, h, -1).mean(dim=2)
    scores = pooled_q.transpose(1, 2) @ frame_k[:, middle_start:middle_end].permute(0, 2, 3, 1)
    take = min(keep_middle, middle_end - middle_start)
    return torch.topk(scores, take, dim=-1, largest=True, sorted=False).indices + middle_start


def _dense_map(b, lq, h, lk, block_q, block_k, device):
    nq = (lq + block_q - 1) // block_q
    nk = (lk + block_k - 1) // block_k
    lut = torch.arange(nk, device=device, dtype=torch.long).view(1, 1, 1, nk).expand(b, h, nq, nk).contiguous()
    return torch.ones((b, h, nq, nk), device=device, dtype=torch.int8), lut, nk


def lightforcing_block_map(q, k, topk_ratio, *, block_q=64, block_k=64, frame_seq=1560, keep_frames=3, keep_sink=1, keep_near=1):
    """Return the same block LUT semantics as LightForcing's SM80 kernel."""
    if q.shape[1] % frame_seq or (k.shape[1] - q.shape[1]) % frame_seq:
        raise ValueError("LightForcing sparse attention requires complete frame-packed Q/K")
    if triton is None or DEVICE_SM not in (80, 120):
        return None
    pooled_q = _mean_pool_blhd(q, block_q)
    pooled_k = _mean_pool_blhd(k, block_k)
    past_frames = (k.shape[1] - q.shape[1]) // frame_seq
    if past_frames == 0:
        # Native LightForcing deliberately executes the first generated chunk
        # with a dense block map, even when later chunks are sparse.
        return _dense_map(
            q.shape[0], q.shape[1], q.shape[2], k.shape[1],
            block_q, block_k, q.device,
        )
    if past_frames <= keep_frames:
        # The released implementation uses one-stage selection here.  The
        # full score matmul is intentionally retained for exact semantics.
        scores = pooled_q.transpose(1, 2) @ pooled_k.permute(0, 2, 3, 1)
    else:
        keep_idx = _select_middle_frames(pooled_q, k, frame_seq, past_frames, keep_frames, keep_sink, keep_near)
        scores = _score_selected_blocks(
            pooled_q, pooled_k, keep_idx, frame_seq, past_frames, k.shape[1],
            block_k, keep_sink=keep_sink, keep_near=keep_near,
        )
    k_blocks = scores.shape[-1]
    topk = max(1, min(k_blocks, int(float(topk_ratio) * k_blocks)))
    lut = torch.topk(scores, topk, dim=-1, largest=True, sorted=False).indices.contiguous()
    block_map = torch.zeros_like(scores, dtype=torch.int8)
    block_map.scatter_(-1, lut, 1)
    return block_map, lut, topk


if triton is not None:

    @triton.jit
    def _attn_forward_kernel(
        Q, K, V, qk_scale: tl.constexpr, topk: tl.constexpr, LUT, OS,
        H: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr,
        M_BLOCKS: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        idx_m = tl.program_id(0).to(tl.int64)
        idx_bh = tl.program_id(1).to(tl.int64)
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        HD: tl.constexpr = H * D
        q_offset = idx_b * LQ * HD + idx_h * D
        kv_offset = idx_b * LK * HD + idx_h * D
        lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
        offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        q_ptr = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        out_ptr = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        q = tl.load(q_ptr, mask=offs_m[:, None] < LQ, other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        out = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for block_idx in tl.range(topk):
            idx_n = tl.load(LUT + lut_offset + block_idx).to(tl.int64)
            k_start = idx_n * BLOCK_N
            k_mask = (k_start + offs_n) < LK
            k_ptr = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            v_ptr = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            kval = tl.load(k_ptr, mask=k_mask[None, :], other=0.0)
            qk = tl.dot(q, kval) * (qk_scale * 1.4426950408889634)
            qk = tl.where(k_mask[None, :], qk, -float("inf"))
            vval = tl.load(v_ptr, mask=k_mask[:, None], other=0.0)
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(m_i, local_m)
            p = tl.math.exp2(qk - new_m[:, None])
            alpha = tl.math.exp2(m_i - new_m)
            out = out * alpha[:, None] + tl.dot(p.to(vval.dtype), vval)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = new_m
        out = out / l_i[:, None]
        tl.store(out_ptr, out.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)


    @triton.jit
    def _attn_forward_blockmask_kernel(
        Q, K, V, qk_scale: tl.constexpr,
        LUT, ROW_LENS, OS,
        H: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr,
        M_BLOCKS: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        TOPK_MAX: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Attention over per-row block lists; ROW_LENS excludes padding from softmax."""
        idx_m = tl.program_id(0).to(tl.int64)
        idx_bh = tl.program_id(1).to(tl.int64)
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        HD: tl.constexpr = H * D
        q_offset = idx_b * LQ * HD + idx_h * D
        kv_offset = idx_b * LK * HD + idx_h * D
        lut_offset = (idx_bh * M_BLOCKS + idx_m) * TOPK_MAX
        row_len = tl.load(ROW_LENS + idx_bh * M_BLOCKS + idx_m).to(tl.int32)
        offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q_ptr = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        out_ptr = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        q = tl.load(
            q_ptr,
            mask=(offs_m[:, None] < LQ) & d_mask[None, :],
            other=0.0,
        )
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        out = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for block_idx in tl.range(TOPK_MAX):
            valid_block = block_idx < row_len
            idx_n = tl.load(LUT + lut_offset + block_idx, mask=valid_block, other=0).to(tl.int64)
            k_start = idx_n * BLOCK_N
            k_mask = valid_block & ((k_start + offs_n) < LK)
            k_ptr = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            v_ptr = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            kval = tl.load(
                k_ptr,
                mask=k_mask[None, :] & d_mask[:, None],
                other=0.0,
            )
            # Scale after matmul to preserve the reference BF16/FP16 rounding.
            qk = tl.dot(q, kval, out_dtype=tl.float32) * (
                qk_scale * 1.4426950408889634
            )
            qk = tl.where(k_mask[None, :], qk, -float("inf"))
            vval = tl.load(
                v_ptr,
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(m_i, local_m)
            finite_new = new_m > -float("inf")
            safe_new_m = tl.where(finite_new, new_m, 0.0)
            p = tl.where(
                k_mask[None, :],
                tl.math.exp2(qk - safe_new_m[:, None]),
                0.0,
            )
            alpha = tl.where(
                finite_new,
                tl.math.exp2(m_i - safe_new_m),
                0.0,
            )
            out = out * alpha[:, None] + tl.dot(
                p.to(vval.dtype), vval, out_dtype=tl.float32
            )
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = new_m

        has_mass = l_i > 0.0
        out = tl.where(has_mass[:, None], out / tl.maximum(l_i, 1e-20)[:, None], 0.0)
        tl.store(
            out_ptr,
            out.to(OS.type.element_ty),
            mask=(offs_m[:, None] < LQ) & d_mask[None, :],
        )


def lightforcing_attention(
    q, k, v, *, sparsity, chunk_id, block_q=64, block_k=64,
    frame_seq=1560, keep_frames=3, keep_sink=1, keep_near=1,
    softmax_scale=None, require_sparse=False,
):
    """Forward-only LightForcing attention with a dense SDPA fallback."""
    if softmax_scale is None:
        softmax_scale = float(q.shape[-1]) ** -0.5
    ratio = 1.0 - float(sparsity)
    sparse_runtime_ok = triton is not None and DEVICE_SM in (80, 120) and q.is_cuda
    if not sparse_runtime_ok:
        if require_sparse and q.is_cuda:
            raise RuntimeError(
                f"LightForcing Triton sparse attention requires SM80/SM120; detected SM{DEVICE_SM}"
            )
        return _dense_attention(q, k, v, softmax_scale)
    result = lightforcing_block_map(
        q, k, ratio, block_q=block_q, block_k=block_k, frame_seq=frame_seq,
        keep_frames=keep_frames, keep_sink=keep_sink, keep_near=keep_near,
    )
    if result is None:
        if require_sparse:
            raise RuntimeError("LightForcing block-map kernel is unavailable on this CUDA runtime")
        return _dense_attention(q, k, v, softmax_scale)
    _, lut, topk = result
    out = torch.empty_like(q)
    _attn_forward_kernel[
        (triton.cdiv(q.shape[1], block_q), q.shape[0] * q.shape[2])
    ](
        q.contiguous(), k.contiguous(), v.contiguous(), float(softmax_scale), topk,
        lut, out, q.shape[2], q.shape[1], k.shape[1],
        triton.cdiv(q.shape[1], block_q), q.shape[-1], block_q, block_k,
        num_warps=4, num_stages=3,
    )
    return out


def lightforcing_attention_with_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    block_q: int = 64,
    block_k: int = 64,
    softmax_scale: float | None = None,
) -> torch.Tensor | None:
    """Run Triton attention on the supplied mask; return None for CUDA fallback."""
    if str(os.getenv("HORIZONKV_LF_ROPE_TRITON", "1")).strip().lower() in {
        "0", "false", "no", "off"
    }:
        return None
    if triton is None or DEVICE_SM not in (80, 120) or not q.is_cuda:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if q.shape[0] != 1 or tuple(k.shape) != tuple(v.shape):
        return None
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return None
    block_q = int(block_q)
    block_k = int(block_k)
    B, LQ, H, D = (int(x) for x in q.shape)
    LK = int(k.shape[1])
    if block_q <= 0 or block_k <= 0 or D <= 0 or LQ <= 0 or LK <= 0 or H <= 0:
        return None
    n_q = (LQ + block_q - 1) // block_q
    n_k = (LK + block_k - 1) // block_k
    if block_mask.ndim == 2:
        block_mask = block_mask.unsqueeze(0).expand(H, -1, -1)
    if tuple(block_mask.shape) != (H, n_q, n_k) or block_mask.device != q.device:
        return None
    if block_mask.dtype != torch.bool:
        block_mask = block_mask.to(dtype=torch.bool)

    # Sort block IDs per row without materializing a token mask.
    selected_values, selected_indices = block_mask.to(dtype=torch.uint8).sort(
        dim=-1, descending=True, stable=True
    )
    row_lens = selected_values.sum(dim=-1, dtype=torch.int32)
    max_row_len = int(row_lens.max().item())
    if max_row_len <= 0:
        return torch.zeros_like(q)
    topk_max = max_row_len
    lut = selected_indices[..., :topk_max].contiguous()
    row_lens = row_lens.unsqueeze(0).contiguous()
    if softmax_scale is None:
        softmax_scale = float(D) ** -0.5
    out = torch.empty_like(q)
    # Triton attention warp count is configurable independently of the CUDA backend.
    default_warps = 8 if D >= 128 and block_q <= 64 and block_k <= 64 else 4
    triton_warps = int(
        os.getenv("HORIZONKV_LF_ROPE_TRITON_WARPS", str(default_warps))
        or default_warps
    )
    if triton_warps not in (1, 2, 4, 8):
        triton_warps = 4
    triton_stages = int(
        os.getenv("HORIZONKV_LF_ROPE_TRITON_STAGES", "2") or 2
    )
    if triton_stages not in (1, 2, 3, 4, 5):
        triton_stages = 2
    _trace_lf_backend_once(
        "triton_blockmask",
        q=q,
        lk=LK,
        block_q=block_q,
        block_k=block_k,
    )
    try:
        _attn_forward_blockmask_kernel[(n_q, B * H)](
            q,
            k,
            v,
            float(softmax_scale),
            lut,
            row_lens,
            out,
            H,
            LQ,
            LK,
            n_q,
            D,
            block_q,
            block_k,
            topk_max,
            triton.next_power_of_2(D),
            num_warps=triton_warps,
            num_stages=triton_stages,
        )
    except Exception as exc:  # pragma: no cover - depends on CUDA/Triton
        if str(os.getenv("HORIZONKV_BLOCKSPARSE_VERBOSE", "0")).strip().lower() in {
            "1", "true", "yes", "on", "y"
        }:
            warnings.warn(
                f"LF-RoPE Triton block-mask attention failed; using CUDA fallback: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    return out


__all__ = [
    "DEVICE_SM", "lightforcing_rope_apply", "lightforcing_rope_apply_packed_rows",
    "lightforcing_block_map",
    "lightforcing_attention", "lightforcing_attention_with_block_mask",
]
