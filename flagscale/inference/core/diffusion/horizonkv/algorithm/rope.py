# Modified for the HorizonKV inference artifact; see NOTICE.
from __future__ import annotations

import math

import torch


def _temporal_complex_dim(head_dim: int) -> int:
    c = int(head_dim) // 2
    return c - 2 * (c // 3)


def _frame_positions_to_token_positions(
    frame_positions: torch.Tensor,
    frame_seqlen: int,
    token_count: int,
) -> torch.Tensor:
    if frame_positions.ndim != 2:
        raise ValueError("frame_positions must be [rows, frames]")
    if int(frame_seqlen) <= 0:
        raise ValueError("frame_seqlen must be positive")
    token_positions = frame_positions.to(dtype=torch.long).repeat_interleave(
        int(frame_seqlen),
        dim=1,
    )
    return token_positions[:, : int(token_count)]


def _normalize_token_positions(
    positions: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    pos = positions.to(device=device, dtype=torch.long)
    if pos.ndim == 1:
        if int(pos.shape[0]) != int(seq_len):
            raise ValueError(f"position length mismatch: got {pos.shape[0]}, expected {seq_len}")
        return pos.view(1, seq_len, 1).expand(batch, -1, heads)
    if pos.ndim != 2:
        raise ValueError("positions must be [seq], [heads, seq], or [batch, seq] for H=1")
    if int(pos.shape[0]) == int(heads) and int(pos.shape[1]) == int(seq_len):
        return pos.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1)
    if int(heads) == 1 and int(pos.shape[0]) == int(batch) and int(pos.shape[1]) == int(seq_len):
        return pos.view(batch, seq_len, 1)
    raise ValueError(
        "position shape mismatch: expected [seq], [heads, seq], "
        f"or [batch, seq] for H=1; got {tuple(pos.shape)}"
    )


def rerotate_temporal_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
) -> torch.Tensor:
    """Move only the temporal RoPE component from old_positions to new_positions."""
    if x.ndim != 4:
        raise ValueError("x must have shape [B, L, H, D]")
    B, L, H, D = x.shape
    if D % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    temporal_dim = _temporal_complex_dim(int(D))
    if temporal_dim <= 0:
        return x

    device = x.device
    old_pos = _normalize_token_positions(old_positions, batch=B, seq_len=L, heads=H, device=device)
    new_pos = _normalize_token_positions(new_positions, batch=B, seq_len=L, heads=H, device=device)
    max_pos = int(torch.maximum(old_pos.max(), new_pos.max()).item()) if old_pos.numel() else -1
    if max_pos >= int(freqs.shape[0]):
        raise ValueError(
            f"RoPE reindex position {max_pos} exceeds frequency table length {int(freqs.shape[0])}"
        )

    temporal_freqs = freqs.to(device=device)[:, :temporal_dim]
    flat_old = old_pos.reshape(-1)
    flat_new = new_pos.reshape(-1)
    old_freq = temporal_freqs.index_select(0, flat_old).view(B, L, H, temporal_dim)
    new_freq = temporal_freqs.index_select(0, flat_new).view(B, L, H, temporal_dim)
    ratio = new_freq * old_freq.conj()

    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(B, L, H, -1, 2))
    out = x_complex.clone()
    out[..., :temporal_dim] = out[..., :temporal_dim] * ratio.to(dtype=out.dtype)
    return torch.view_as_real(out).flatten(3).type_as(x)


def selected_frame_global_positions(
    frame_indices: torch.Tensor,
    *,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_end: int,
    sink_tokens: int = 0,
) -> torch.Tensor:
    if frame_indices.ndim != 2:
        raise ValueError("frame_indices must be [heads, frames]")
    local_frame_tokens = int(kv_start) + frame_indices.to(dtype=torch.long) * int(frame_seqlen)
    global_tail_tokens = int(current_end) - int(local_end_index) + local_frame_tokens
    if int(sink_tokens) > 0:
        global_tokens = torch.where(
            local_frame_tokens < int(sink_tokens),
            local_frame_tokens,
            global_tail_tokens,
        )
    else:
        global_tokens = global_tail_tokens
    return torch.div(global_tokens, int(frame_seqlen), rounding_mode="floor")


def compacted_rope_positions(
    frame_indices: torch.Tensor,
    *,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    query_token_count: int,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return old K frames, new K frames, and per-head new Q start frames."""
    old_k_frames = selected_frame_global_positions(
        frame_indices,
        frame_seqlen=frame_seqlen,
        kv_start=kv_start,
        local_end_index=local_end_index,
        current_end=current_end,
        sink_tokens=sink_tokens,
    )
    query_frame_count = max(1, int(math.ceil(int(query_token_count) / float(frame_seqlen))))
    current_start_frame = int(current_start) // int(frame_seqlen)
    current_end_frame = current_start_frame + query_frame_count

    is_current = (old_k_frames >= current_start_frame) & (old_k_frames < current_end_frame)
    is_history = ~is_current
    history_rank = torch.cumsum(is_history.to(dtype=torch.long), dim=1) - 1
    history_count = is_history.to(dtype=torch.long).sum(dim=1)
    current_offset = old_k_frames - current_start_frame
    new_k_frames = torch.where(
        is_current,
        history_count.unsqueeze(1) + current_offset,
        history_rank,
    )
    return old_k_frames, new_k_frames.clamp_min(0), history_count


def reindex_compacted_qk_rope(
    q: torch.Tensor,
    k_compact: torch.Tensor,
    freqs: torch.Tensor,
    frame_indices: torch.Tensor,
    *,
    frame_seqlen: int,
    kv_start: int,
    local_end_index: int,
    current_start: int,
    current_end: int,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode selected compacted KV frames and current Q as one compressed timeline."""
    if q.ndim != 4 or k_compact.ndim != 4:
        raise ValueError("q and k_compact must both have shape [B, L, H, D]")
    B, Lq, H, D = q.shape
    rows, Lk, one, Dk = k_compact.shape
    if one != 1 or Dk != D:
        raise ValueError("k_compact must have shape [B * H, Lk, 1, D]")
    if rows != B * H:
        raise ValueError(f"k_compact row mismatch: got {rows}, expected {B * H}")
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)
    if frame_indices.shape[0] == 1 and H > 1:
        frame_indices = frame_indices.expand(H, -1)
    if int(frame_indices.shape[0]) != int(H):
        raise ValueError(
            f"frame_indices head dimension mismatch: got {frame_indices.shape[0]}, expected {H}"
        )

    frame_indices = torch.sort(frame_indices.to(device=q.device, dtype=torch.long), dim=1).values
    old_k_frames, new_k_frames, q_new_start = compacted_rope_positions(
        frame_indices,
        frame_seqlen=frame_seqlen,
        kv_start=kv_start,
        local_end_index=local_end_index,
        current_start=current_start,
        current_end=current_end,
        query_token_count=Lq,
        sink_tokens=sink_tokens,
    )

    q_frame_offsets = torch.div(
        torch.arange(Lq, device=q.device, dtype=torch.long),
        int(frame_seqlen),
        rounding_mode="floor",
    )
    q_old_positions = int(current_start) // int(frame_seqlen) + q_frame_offsets
    q_new_positions = q_new_start.unsqueeze(1) + q_frame_offsets.unsqueeze(0)
    q_reindexed = rerotate_temporal_rope(q, freqs, q_old_positions, q_new_positions)

    old_rows = old_k_frames.unsqueeze(0).expand(B, -1, -1).reshape(B * H, -1)
    new_rows = new_k_frames.unsqueeze(0).expand(B, -1, -1).reshape(B * H, -1)
    k_old_positions = _frame_positions_to_token_positions(old_rows, frame_seqlen, Lk)
    k_new_positions = _frame_positions_to_token_positions(new_rows, frame_seqlen, Lk)
    k_reindexed = rerotate_temporal_rope(k_compact, freqs, k_old_positions, k_new_positions)
    return q_reindexed, k_reindexed
