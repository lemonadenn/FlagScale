# Modified for the HorizonKV inference artifact; see NOTICE.
"""LightForcing hierarchical block selection and chunk sparsity budgets."""

from __future__ import annotations

import math

import torch


def calculate_chunk_sparsities(
    num_output_frames: int,
    num_frame_per_block: int,
    local_attn_size: int = 21,
    sparsity: float = 0.85,
    sparsity_base: float | None = None,
) -> list[float]:
    """Port LightForcing's FLOP-balanced sparsity schedule."""
    target = float(sparsity)
    base = target if sparsity_base is None else float(sparsity_base)
    num_output_frames = max(0, int(num_output_frames))
    num_frame_per_block = max(1, int(num_frame_per_block))
    local_attn_size = int(local_attn_size)
    chunk_frames = range(
        2 * num_frame_per_block,
        num_output_frames + 1,
        num_frame_per_block,
    )
    kv_lengths = [
        count if local_attn_size == -1 else min(count, local_attn_size)
        for count in chunk_frames
    ]
    alphas = [1.0 / math.sqrt(max(1, count)) for count in chunk_frames]
    target_flops = sum((1.0 - target) * n for n in kv_lengths)
    base_flops = sum((1.0 - base) * n for n in kv_lengths)
    weighted = sum(a * n for a, n in zip(alphas, kv_lengths))
    if weighted <= 0.0:
        return [base] * len(kv_lengths)
    beta = (target_flops - base_flops) / weighted
    # The original implementation emits a dense entry for the first chunk.
    return [0.0] + [base - alpha * beta for alpha in alphas]


def _mean_pool(x: torch.Tensor, block: int) -> torch.Tensor:
    if block <= 1:
        return x
    b, length, h, d = x.shape
    n = (length + block - 1) // block
    padded = n * block - length
    if padded:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded))
    return x.reshape(b, n, block, h, d).mean(dim=2)


def _frame_ids(num_blocks: int, block_k: int, frame_seq: int, device: torch.device) -> torch.Tensor:
    if num_blocks <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    positions = torch.arange(num_blocks, device=device, dtype=torch.long) * int(block_k)
    return torch.div(positions, max(1, int(frame_seq)), rounding_mode="floor")


def _frame_means(
    pooled_k: torch.Tensor,
    block_k: int,
    frame_seq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frame ids and mean K vectors, preserving tail blocks."""
    b, blocks, h, d = pooled_k.shape
    ids = _frame_ids(blocks, block_k, frame_seq, pooled_k.device)
    n_frames = int(ids.max().item()) + 1 if ids.numel() else 0
    out = torch.zeros((b, n_frames, h, d), device=pooled_k.device, dtype=pooled_k.dtype)
    counts = torch.zeros((n_frames,), device=pooled_k.device, dtype=pooled_k.dtype)
    if n_frames:
        out.index_add_(1, ids, pooled_k)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=pooled_k.dtype))
        out = out / counts.clamp_min(1).view(1, -1, 1, 1)
    return ids, out


@torch.no_grad()
def select_lightforcing_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    topk_ratio: float,
    block_q: int = 64,
    block_k: int = 64,
    frame_seq: int = 1560,
    keep_frames: int = 3,
    keep_sink: int = 0,
    keep_near: int = 1,
) -> torch.Tensor:
    """Build a ``[B,H,Q_blocks,K_blocks]`` boolean LightForcing mask.

    ``q`` is the current denoising chunk and ``k`` is the complete causal KV
    window (history + current chunk).  The first-stage frame selector is used
    only once enough historical frames exist; otherwise all blocks are scored
    directly, matching LightForcing's one-stage path.
    """
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k must be [B,L,H,D] with matching batch/head dimensions")
    if block_q <= 0 or block_k <= 0 or frame_seq <= 0:
        raise ValueError("block_q, block_k and frame_seq must be positive")
    b, lq, h, _ = q.shape
    lk = int(k.shape[1])
    qp = _mean_pool(q, int(block_q))
    kp = _mean_pool(k, int(block_k))
    q_blocks = int(qp.shape[1])
    k_blocks = int(kp.shape[1])
    if k_blocks == 0 or q_blocks == 0:
        return torch.zeros((b, h, q_blocks, k_blocks), device=q.device, dtype=torch.bool)

    # Dot products are deliberately accumulated in fp32, while the returned
    # mask stays compact and can be passed directly to the existing backend.
    scores = torch.matmul(
        qp.float().transpose(1, 2),
        kp.float().permute(0, 2, 3, 1),
    )
    ratio = max(0.0, min(1.0, float(topk_ratio)))
    topk = max(1, min(k_blocks, int(ratio * k_blocks)))

    past_tokens = max(0, lk - int(lq))
    # Include any partial tail frame.
    past_frames = past_tokens // int(frame_seq)
    if past_tokens % int(frame_seq):
        past_frames = (past_tokens + int(frame_seq) - 1) // int(frame_seq)
    use_hierarchical = past_frames > int(keep_frames)

    if use_hierarchical and keep_frames > 0:
        frame_block_ids, frame_k = _frame_means(kp, int(block_k), int(frame_seq))
        n_frames = int(frame_k.shape[1])
        past_frames = min(max(0, int(past_frames)), n_frames)
        sink = max(0, min(int(keep_sink), int(keep_frames)))
        near = max(0, min(int(keep_near), int(keep_frames) - sink))
        middle_start = min(sink, past_frames)
        middle_end = max(middle_start, past_frames - near)
        keep_middle = max(0, int(keep_frames) - sink - near)

        selected_middle = torch.empty((b, h, q_blocks, 0), device=q.device, dtype=torch.long)
        middle_count = middle_end - middle_start
        if keep_middle > 0 and middle_count > 0:
            middle_k = frame_k[:, middle_start:middle_end]
            frame_scores = torch.matmul(
                qp.float().transpose(1, 2),
                middle_k.float().permute(0, 2, 3, 1),
            )
            take = min(keep_middle, middle_count)
            selected_middle = torch.topk(
                frame_scores, k=take, dim=-1, largest=True, sorted=False
            ).indices + middle_start

        allowed_frames = torch.zeros((b, h, q_blocks, n_frames), device=q.device, dtype=torch.bool)
        if sink:
            allowed_frames[..., : min(sink, n_frames)] = True
        if near:
            near_start = max(0, past_frames - near)
            allowed_frames[..., near_start:past_frames] = True
        if selected_middle.numel():
            allowed_frames.scatter_(-1, selected_middle, True)
        # All current query frames are always visible; this also handles a
        # partial tail frame and is the causal equivalent of LightForcing's
        # ``is_tail`` path.
        allowed_frames |= (torch.arange(n_frames, device=q.device).view(1, 1, 1, -1) >= past_frames)
        key_allowed = allowed_frames.gather(
            -1,
            frame_block_ids.view(1, 1, 1, -1).expand(b, h, q_blocks, -1),
        )
    else:
        key_allowed = torch.ones((b, h, q_blocks, k_blocks), device=q.device, dtype=torch.bool)

    scores = scores.masked_fill(~key_allowed, -torch.inf)
    # If a pathological configuration filters every key block, restore the
    # current chunk rather than producing NaNs in the sparse kernel.
    no_valid = ~key_allowed.any(dim=-1, keepdim=True)
    if no_valid.any():
        current_start = max(0, past_tokens // int(block_k))
        fallback = torch.arange(k_blocks, device=q.device).view(1, 1, 1, -1) >= current_start
        key_allowed = torch.where(no_valid, fallback, key_allowed)
        scores = scores.masked_fill(~key_allowed, -torch.inf)

    # A rectangular mask is required by the backend. Select a fixed k and pad
    # with the best valid blocks when a row has fewer valid candidates.
    selected = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros((b, h, q_blocks, k_blocks), device=q.device, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    mask &= key_allowed
    # Keep every query row executable even when hierarchical filtering leaves
    # fewer than the requested rectangular top-k budget.
    empty_rows = ~mask.any(dim=-1, keepdim=True)
    if empty_rows.any():
        mask |= empty_rows & key_allowed
    return mask


__all__ = ["calculate_chunk_sparsities", "select_lightforcing_block_mask"]
