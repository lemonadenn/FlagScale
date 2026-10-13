# Modified for the HorizonKV inference artifact; see NOTICE.
from __future__ import annotations

import math

import torch

from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig

MAX_FULL_ATTN_ELEMENTS = 16 * 1024 * 1024
_FORCED_ATTN_BLOCK_DEVICE_CACHE: dict[
    tuple[str, int | None, int, int, tuple[int, ...]], torch.Tensor
] = {}


def _avg_pool_tokens_along_seq(x: torch.Tensor, pool: int) -> torch.Tensor:
    if pool <= 1:
        return x
    B, L, H, D = x.shape
    n_full = L // pool
    rest = L - n_full * pool
    parts: list[torch.Tensor] = []
    if n_full > 0:
        parts.append(
            x[:, : n_full * pool]
            .contiguous()
            .reshape(B, n_full, pool, H, D)
            .mean(dim=2)
        )
    if rest > 0:
        parts.append(x[:, n_full * pool :].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1)


def _minmax_pool_tokens_along_seq(x: torch.Tensor, pool: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pool <= 1:
        return x, x
    B, L, H, D = x.shape
    n_full = L // pool
    rest = L - n_full * pool
    max_parts: list[torch.Tensor] = []
    min_parts: list[torch.Tensor] = []
    if n_full > 0:
        view = x[:, : n_full * pool].contiguous().reshape(B, n_full, pool, H, D)
        max_parts.append(view.max(dim=2).values)
        min_parts.append(view.min(dim=2).values)
    if rest > 0:
        tail = x[:, n_full * pool :]
        max_parts.append(tail.max(dim=1, keepdim=True).values)
        min_parts.append(tail.min(dim=1, keepdim=True).values)
    return torch.cat(max_parts, dim=1), torch.cat(min_parts, dim=1)


def _prepare_qk_for_importance(
    q: torch.Tensor,
    k: torch.Tensor,
    q_pool: int,
    k_pool: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lk_orig = k.shape[1]
    if k_pool > 1:
        k = _avg_pool_tokens_along_seq(k, k_pool)
    if q_pool > 1:
        q = _avg_pool_tokens_along_seq(q, q_pool)
    return q, k, lk_orig


def _can_use_full_pooled_attention(B: int, Hs: int, Lq: int, Lk: int) -> bool:
    return B * Hs * Lq * Lk <= MAX_FULL_ATTN_ELEMENTS


@torch.no_grad()
def _qk_logits_chunk(qf: torch.Tensor, k_chunk: torch.Tensor, scale: float) -> torch.Tensor:
    B, Lq, Hs, D = qf.shape
    W = k_chunk.shape[1]
    qv = qf.permute(0, 2, 1, 3).contiguous().reshape(B * Hs, Lq, D)
    kv = k_chunk.permute(0, 2, 1, 3).contiguous().reshape(B * Hs, W, D)
    logits = torch.bmm(qv, kv.transpose(-2, -1))
    return logits.view(B, Hs, Lq, W) * scale


def _scatter_pooled_contrib_to_frames(
    frame_mass: torch.Tensor,
    contrib: torch.Tensor,
    j_start: int,
    k_pool: int,
    lk_orig: int,
    frame_seqlen: int,
    n_frames: int,
) -> None:
    Hs, W = contrib.shape
    if W == 0 or n_frames <= 0:
        return
    device = frame_mass.device

    js = torch.arange(W, device=device, dtype=torch.long) + j_start
    t0 = js * k_pool
    t1 = torch.minimum(
        (js + 1) * k_pool,
        torch.tensor(lk_orig, device=device, dtype=torch.long),
    )
    valid = t0 < t1
    if not valid.any():
        return

    t0 = t0[valid]
    t1 = t1[valid]
    contrib = contrib[:, valid]

    f_start = (t0 // frame_seqlen).clamp_max(n_frames - 1)
    f_end = ((t1 - 1) // frame_seqlen + 1).clamp_max(n_frames)
    n_covered = f_end - f_start
    valid = n_covered > 0
    if not valid.any():
        return

    f_start = f_start[valid]
    n_covered = n_covered[valid]
    contrib = contrib[:, valid]

    max_covered = int(n_covered.max().item())
    offsets = torch.arange(max_covered, device=device, dtype=torch.long)
    covered = offsets.unsqueeze(0) < n_covered.unsqueeze(1)
    frame_idx = f_start.unsqueeze(1) + offsets.unsqueeze(0)
    weights = covered.to(dtype=contrib.dtype) / n_covered.to(dtype=contrib.dtype).unsqueeze(1)
    scatter_vals = contrib.unsqueeze(-1) * weights.unsqueeze(0)
    scatter_idx = frame_idx.unsqueeze(0).expand(Hs, -1, -1)
    frame_mass.scatter_add_(1, scatter_idx.reshape(Hs, -1), scatter_vals.reshape(Hs, -1))


@torch.no_grad()
def _per_frame_attn_mass_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    frame_seqlen: int,
    chunk_k: int,
    *,
    lk_orig: int | None = None,
    k_pool: int = 1,
) -> torch.Tensor:
    B, Lq, Hs, D = q.shape
    Lk = k.shape[1]
    if lk_orig is None:
        lk_orig = Lk
    device = q.device
    scale = 1.0 / (D**0.5)
    n_frames = max(1, (lk_orig + frame_seqlen - 1) // frame_seqlen)
    frame_mass = torch.zeros(Hs, n_frames, device=device, dtype=torch.float32)

    qf = q.float()
    kf = k.float()

    if _can_use_full_pooled_attention(B, Hs, Lq, Lk):
        logits = _qk_logits_chunk(qf, kf, scale)
        contrib = torch.softmax(logits, dim=-1).sum(dim=(0, 2))
        if k_pool <= 1:
            t = torch.arange(Lk, device=device, dtype=torch.long)
            f_bins = (t // frame_seqlen).clamp_max(n_frames - 1)
            idx = f_bins.unsqueeze(0).expand(Hs, -1)
            frame_mass.scatter_add_(1, idx, contrib)
        else:
            _scatter_pooled_contrib_to_frames(
                frame_mass, contrib, 0, k_pool, lk_orig, frame_seqlen, n_frames
            )
        return frame_mass

    logits_max = torch.full((B, Hs, Lq, 1), -float("inf"), device=device, dtype=torch.float32)
    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        logits_max = torch.maximum(logits_max, logits.max(dim=-1, keepdim=True).values)
        k0 = k1

    denom = torch.zeros(B, Hs, Lq, device=device, dtype=torch.float32)
    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        denom += torch.exp(logits - logits_max).sum(dim=-1)
        k0 = k1
    denom = denom.clamp_min(1e-8)

    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        p = torch.exp(logits - logits_max) / denom.unsqueeze(-1)
        contrib = p.sum(dim=(0, 2))

        if k_pool <= 1:
            t = torch.arange(k0, k1, device=device, dtype=torch.long)
            f_bins = (t // frame_seqlen).clamp_max(n_frames - 1)
            idx = f_bins.unsqueeze(0).expand(Hs, -1)
            frame_mass.scatter_add_(1, idx, contrib)
        else:
            _scatter_pooled_contrib_to_frames(
                frame_mass, contrib, k0, k_pool, lk_orig, frame_seqlen, n_frames
            )
        k0 = k1

    return frame_mass


@torch.no_grad()
def _per_block_attn_mass_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    chunk_k: int,
    bq_size: int,
    bk_size: int,
) -> torch.Tensor:
    B, Lq, Hs, D = q.shape
    Lk = k.shape[1]
    device = q.device
    scale = 1.0 / (D**0.5)
    n_bq = (Lq + bq_size - 1) // bq_size
    n_bk = (Lk + bk_size - 1) // bk_size
    n_blocks = n_bq * n_bk
    block_mass = torch.zeros(Hs, n_blocks, device=device, dtype=torch.float32)

    qf = q.float()
    kf = k.float()

    if _can_use_full_pooled_attention(B, Hs, Lq, Lk):
        logits = _qk_logits_chunk(qf, kf, scale)
        ps = torch.softmax(logits, dim=-1).sum(dim=0)
        q_blk = (torch.arange(Lq, device=device, dtype=torch.long) // bq_size).clamp_max(n_bq - 1)
        kk = torch.arange(Lk, device=device, dtype=torch.long)
        k_blk = (kk // bk_size).clamp_max(n_bk - 1)
        lin = (q_blk.unsqueeze(1) * n_bk + k_blk.unsqueeze(0)).flatten().long()
        block_mass.scatter_add_(1, lin.unsqueeze(0).expand(Hs, -1), ps.reshape(Hs, -1))
        return block_mass

    logits_max = torch.full((B, Hs, Lq, 1), -float("inf"), device=device, dtype=torch.float32)
    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        logits_max = torch.maximum(logits_max, logits.max(dim=-1, keepdim=True).values)
        k0 = k1

    denom = torch.zeros(B, Hs, Lq, device=device, dtype=torch.float32)
    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        denom += torch.exp(logits - logits_max).sum(dim=-1)
        k0 = k1

    denom = denom.clamp_min(1e-8)
    q_blk = (torch.arange(Lq, device=device, dtype=torch.long) // bq_size).clamp_max(n_bq - 1)

    k0 = 0
    while k0 < Lk:
        k1 = min(k0 + chunk_k, Lk)
        logits = _qk_logits_chunk(qf, kf[:, k0:k1, :, :], scale)
        p = torch.exp(logits - logits_max) / denom.unsqueeze(-1)
        ps = p.sum(dim=0)
        kk = torch.arange(k0, k1, device=device, dtype=torch.long)
        k_blk = (kk // bk_size).clamp_max(n_bk - 1)
        lin = (q_blk.unsqueeze(1) * n_bk + k_blk.unsqueeze(0)).flatten().long()
        block_mass.scatter_add_(1, lin.unsqueeze(0).expand(Hs, -1), ps.reshape(Hs, -1))
        k0 = k1

    return block_mass


@torch.no_grad()
def _per_block_pooled_qk_scores_batched(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Return pooled-QK dot-product scores for each Q/K block pair.

    ``q`` and ``k`` are expected to already be pooled to one token per
    attention block, with layout ``[B, blocks, heads, D]``.  The result uses
    the same ``[heads, q_blocks * k_blocks]`` layout as
    :func:`_per_block_attn_mass_batched`, but deliberately keeps the raw
    pooled dot product rather than converting it to softmax mass.  This is the
    selector used by the fast CPU-offload overlap path.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("pooled q/k must have shape [B, blocks, heads, D]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "pooled q/k batch, head, and head-dimension shapes must match"
        )
    B, _, Hs, D = q.shape
    if B <= 0 or Hs <= 0 or D <= 0 or q.shape[1] <= 0 or k.shape[1] <= 0:
        return torch.empty(
            (max(0, int(Hs)), 0),
            device=q.device,
            dtype=torch.float32,
        )
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf)
    scores = scores * (1.0 / math.sqrt(float(D)))
    # The normal inference path uses batch_size=1.  Summing here preserves
    # the existing block-mass helper's batch reduction for completeness.
    return scores.sum(dim=0).reshape(int(Hs), -1)


def _mandatory_key_token_indices(
    Lk: int,
    kv_start: int,
    frame_seqlen: int,
    sink_tokens: int,
    forced_frames: set[int],
) -> set[int]:
    out: set[int] = set()
    sink_len = min(Lk, max(0, int(sink_tokens) - int(kv_start)))
    if sink_len > 0:
        out.update(range(sink_len))
    for f in forced_frames:
        t0 = int(f) * frame_seqlen
        t1 = min((int(f) + 1) * frame_seqlen, Lk)
        if t0 < t1:
            out.update(range(t0, t1))
    return out


def _mandatory_key_positions_pooled(
    mand_tok_orig: set[int],
    Lk: int,
    k_pool: int,
) -> set[int]:
    if not mand_tok_orig:
        return set()
    Lk_p = (Lk + k_pool - 1) // k_pool
    tok = torch.tensor(list(mand_tok_orig), dtype=torch.long)
    tok = tok[(tok >= 0) & (tok < Lk)]
    if tok.numel() == 0:
        return set()
    pooled = (tok // k_pool).clamp_max(Lk_p - 1).unique()
    return set(pooled.tolist())


def _mandatory_key_positions_pooled_from_frames(
    *,
    Lk: int,
    kv_start: int,
    frame_seqlen: int,
    sink_tokens: int,
    forced_frames: set[int],
    k_pool: int,
) -> set[int]:
    """Compute compulsory pooled-K positions directly from half-open frame spans."""
    Lk = max(0, int(Lk))
    k_pool = max(1, int(k_pool))
    frame_seqlen = max(1, int(frame_seqlen))
    if Lk <= 0:
        return set()

    pooled: set[int] = set()

    def add_span(start: int, end: int) -> None:
        start = max(0, min(int(start), Lk))
        end = max(start, min(int(end), Lk))
        if start >= end:
            return
        first = start // k_pool
        last = (end - 1) // k_pool
        pooled.update(range(first, last + 1))

    sink_len = min(Lk, max(0, int(sink_tokens) - int(kv_start)))
    add_span(0, sink_len)
    for frame in forced_frames:
        frame = int(frame)
        add_span(frame * frame_seqlen, (frame + 1) * frame_seqlen)
    return pooled


def _forced_attn_block_linear_indices_device(
    n_bq: int,
    n_bk: int,
    mandatory_key_blocks: set[int] | list[int] | tuple[int, ...] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the compulsory Cartesian block set directly on ``device``."""
    n_bq = max(0, int(n_bq))
    n_bk = max(0, int(n_bk))
    if n_bq <= 0 or n_bk <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    cache_key = None
    if isinstance(mandatory_key_blocks, torch.Tensor):
        key_blocks = mandatory_key_blocks.to(
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        key_blocks = key_blocks[(key_blocks >= 0) & (key_blocks < n_bk)]
        key_blocks = torch.unique(key_blocks, sorted=True)
    else:
        values = sorted(
            {
                int(value)
                for value in mandatory_key_blocks
                if 0 <= int(value) < n_bk
            }
        )
        if not values:
            return torch.empty((0,), device=device, dtype=torch.long)
        cache_key = (
            str(device.type),
            device.index,
            n_bq,
            n_bk,
            tuple(values),
        )
        cached = _FORCED_ATTN_BLOCK_DEVICE_CACHE.get(cache_key, None)
        if isinstance(cached, torch.Tensor) and cached.device == device:
            return cached
        key_blocks = torch.tensor(values, device=device, dtype=torch.long)
    if int(key_blocks.numel()) <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    query_blocks = torch.arange(n_bq, device=device, dtype=torch.long)
    result = (query_blocks[:, None] * n_bk + key_blocks[None, :]).reshape(-1)
    if cache_key is not None:
        _FORCED_ATTN_BLOCK_DEVICE_CACHE[cache_key] = result
    return result


def _forced_attn_block_linear_indices(
    Lq: int,
    Lk: int,
    bq_size: int,
    bk_size: int,
    mandatory_key_positions: set[int],
) -> set[int]:
    n_bq = (Lq + bq_size - 1) // bq_size
    n_bk = (Lk + bk_size - 1) // bk_size
    if not mandatory_key_positions:
        return set()
    pos = torch.tensor(list(mandatory_key_positions), dtype=torch.long)
    pos = pos[(pos >= 0) & (pos < Lk)]
    if pos.numel() == 0:
        return set()
    bk_needed = (pos // bk_size).clamp_max(n_bk - 1).unique()
    bq = torch.arange(n_bq, dtype=torch.long)
    lin = (bq.unsqueeze(1) * n_bk + bk_needed.unsqueeze(0)).reshape(-1)
    return set(lin.tolist())


def _block_mask_from_selected_linear_indices(
    n_bq: int,
    n_bk: int,
    selected_linear: set[int] | list[int] | tuple[int, ...] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    block_selected = torch.zeros((max(0, n_bq), max(0, n_bk)), device=device, dtype=torch.bool)
    if n_bq <= 0 or n_bk <= 0:
        return block_selected

    if isinstance(selected_linear, torch.Tensor):
        selected_lin = selected_linear.to(device=device, dtype=torch.long).reshape(-1)
    else:
        if not selected_linear:
            return block_selected
        selected_lin = torch.tensor(list(selected_linear), device=device, dtype=torch.long)
    selected_lin = selected_lin[(selected_lin >= 0) & (selected_lin < n_bq * n_bk)]
    if selected_lin.numel() == 0:
        return block_selected

    bq = torch.div(selected_lin, n_bk, rounding_mode="floor")
    bk = torch.remainder(selected_lin, n_bk)
    block_selected[bq, bk] = True
    return block_selected


def _block_mask_from_selected_linear_indices_batched(
    n_bq: int,
    n_bk: int,
    selected_linear: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if selected_linear.ndim == 1:
        selected_linear = selected_linear.unsqueeze(0)
    Hs = int(selected_linear.shape[0]) if selected_linear.ndim >= 2 else 0
    block_selected = torch.zeros(
        (max(0, Hs), max(0, n_bq), max(0, n_bk)),
        device=device,
        dtype=torch.bool,
    )
    if Hs <= 0 or n_bq <= 0 or n_bk <= 0:
        return block_selected

    selected_lin = selected_linear.to(device=device, dtype=torch.long).reshape(Hs, -1)
    valid = (selected_lin >= 0) & (selected_lin < n_bq * n_bk)
    if not valid.any():
        return block_selected

    flat_mask = block_selected.view(Hs, n_bq * n_bk)
    safe_idx = selected_lin.clamp(min=0, max=max(0, n_bq * n_bk - 1))
    flat_mask.scatter_(1, safe_idx, valid)
    return block_selected


@torch.no_grad()
def _select_density_block_mask_batched(
    block_mass_2d: torch.Tensor,
    *,
    n_bq: int,
    n_bk: int,
    densities: torch.Tensor | list[float] | tuple[float, ...],
    compulsory_linear: set[int] | list[int] | tuple[int, ...] | torch.Tensor,
    forced_outside_budget: bool = True,
    compulsory_is_unique: bool = False,
) -> torch.Tensor:
    """Select per-head [heads,n_bq,n_bk] density masks without host synchronization.

    With forced_outside_budget, retain compulsory blocks and apply density to
    remaining blocks. Otherwise use the normal density budget."""
    if block_mass_2d.ndim != 2:
        raise ValueError("block_mass_2d must have shape [heads, blocks]")
    heads = int(block_mass_2d.shape[0])
    n_bq = int(n_bq)
    n_bk = int(n_bk)
    n_blocks = n_bq * n_bk
    if int(block_mass_2d.shape[1]) != n_blocks:
        raise ValueError(
            "block_mass_2d block dimension mismatch: "
            f"got {int(block_mass_2d.shape[1])}, expected {n_blocks}"
        )
    device = block_mass_2d.device
    if heads <= 0 or n_blocks <= 0:
        return torch.zeros(
            (max(0, heads), max(0, n_bq), max(0, n_bk)),
            device=device,
            dtype=torch.bool,
        )

    if isinstance(densities, torch.Tensor):
        density_values = [float(value) for value in densities.detach().cpu().reshape(-1).tolist()]
    else:
        density_values = [float(value) for value in densities]
    if len(density_values) == 1 and heads > 1:
        density_values = density_values * heads
    if len(density_values) != heads:
        raise ValueError(
            f"densities must contain one value per head, got {len(density_values)} for {heads} heads"
        )
    density_values = [max(0.0, min(1.0, value)) for value in density_values]

    compulsory_values: list[int] | None
    if isinstance(compulsory_linear, torch.Tensor):
        compulsory = compulsory_linear.to(
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        if not compulsory_is_unique:
            compulsory = compulsory[(compulsory >= 0) & (compulsory < n_blocks)]
            compulsory = torch.unique(compulsory, sorted=True)
        compulsory_values = None
    else:
        compulsory_values = [
            int(value) for value in compulsory_linear if 0 <= int(value) < n_blocks
        ]
        compulsory_values = sorted(set(compulsory_values))
        compulsory = torch.tensor(compulsory_values, device=device, dtype=torch.long)

    selected_flat = torch.zeros((heads, n_blocks), device=device, dtype=torch.bool)
    if forced_outside_budget:
        compulsory_count = int(compulsory.numel())
        candidate_count = max(0, n_blocks - compulsory_count)
        n_take_values = [
            min(candidate_count, max(0, int(math.ceil(value * candidate_count))))
            for value in density_values
        ]
        if compulsory_count > 0:
            selected_flat[:, compulsory] = True
            candidate_scores = block_mass_2d.masked_fill(
                selected_flat,
                torch.finfo(block_mass_2d.dtype).min,
            )
        else:
            candidate_scores = block_mass_2d
    else:
        # The legacy non-forced density rule always keeps at least one block.
        n_take_values = [
            min(n_blocks, max(1, int(math.ceil(value * n_blocks))))
            for value in density_values
        ]
        candidate_scores = block_mass_2d

    n_take = torch.tensor(n_take_values, device=device, dtype=torch.long)
    max_take = max(n_take_values, default=0)
    if max_take > 0:
        chosen = torch.topk(
            candidate_scores,
            k=max_take,
            dim=1,
            largest=True,
            # Heads may have different densities. Keep the prefix ordered so
            # rank < n_take is the same set as each head's independent top-k.
            sorted=True,
        ).indices
        valid_rank = torch.arange(max_take, device=device).unsqueeze(0) < n_take.unsqueeze(1)
        chosen_flat = torch.zeros_like(selected_flat)
        chosen_flat.scatter_(1, chosen, valid_rank)
        selected_flat |= chosen_flat

    return selected_flat.view(heads, n_bq, n_bk)


def _attn_mask_from_selected_pooled_blocks(
    Lq: int,
    Lk: int,
    bq_sp: int,
    bk_sp: int,
    q_pool: int,
    k_pool: int,
    n_bk_p: int,
    selected_linear: set[int] | list[int] | tuple[int, ...] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(selected_linear, torch.Tensor):
        if selected_linear.numel() == 0:
            return torch.zeros((1, 1, Lq, Lk), device=device, dtype=torch.bool)
    elif not selected_linear:
        return torch.zeros((1, 1, Lq, Lk), device=device, dtype=torch.bool)

    Lq_p = (Lq + q_pool - 1) // q_pool
    Lk_p = (Lk + k_pool - 1) // k_pool
    n_bq_p = (Lq_p + bq_sp - 1) // bq_sp

    block_selected = _block_mask_from_selected_linear_indices(
        n_bq_p,
        n_bk_p,
        selected_linear,
        device,
    )

    q_idx = torch.arange(Lq, device=device, dtype=torch.long)
    k_idx = torch.arange(Lk, device=device, dtype=torch.long)
    q_p = (q_idx // q_pool).clamp(max=Lq_p - 1)
    k_p = (k_idx // k_pool).clamp(max=Lk_p - 1)
    q_block = (q_p // bq_sp).clamp(max=n_bq_p - 1)
    k_block = (k_p // bk_sp).clamp(max=n_bk_p - 1)

    is_allowed = block_selected[q_block.unsqueeze(1), k_block.unsqueeze(0)]
    return is_allowed.unsqueeze(0).unsqueeze(0).contiguous()


def _attn_mask_from_block_mask_batched(
    Lq: int,
    Lk: int,
    bq_sp: int,
    bk_sp: int,
    q_pool: int,
    k_pool: int,
    block_selected: torch.Tensor,
) -> torch.Tensor:
    if block_selected.ndim == 2:
        block_selected = block_selected.unsqueeze(0)
    Hs = int(block_selected.shape[0]) if block_selected.ndim >= 3 else 0
    device = block_selected.device
    if Hs <= 0:
        return torch.empty((1, 0, Lq, Lk), device=device, dtype=torch.bool)

    n_bq_p = int(block_selected.shape[1])
    n_bk_p = int(block_selected.shape[2])
    if n_bq_p <= 0 or n_bk_p <= 0:
        return torch.zeros((1, Hs, Lq, Lk), device=device, dtype=torch.bool)

    Lq_p = (Lq + q_pool - 1) // q_pool
    Lk_p = (Lk + k_pool - 1) // k_pool
    q_idx = torch.arange(Lq, device=device, dtype=torch.long)
    k_idx = torch.arange(Lk, device=device, dtype=torch.long)
    q_p = (q_idx // q_pool).clamp(max=Lq_p - 1)
    k_p = (k_idx // k_pool).clamp(max=Lk_p - 1)
    q_block = (q_p // bq_sp).clamp(max=n_bq_p - 1)
    k_block = (k_p // bk_sp).clamp(max=n_bk_p - 1)

    allowed_rows = block_selected.index_select(1, q_block)
    gather_idx = k_block.view(1, 1, Lk).expand(Hs, Lq, Lk)
    is_allowed = allowed_rows.gather(2, gather_idx)
    return is_allowed.unsqueeze(0).contiguous()


def _attn_mask_from_selected_kv_frames_batched(
    selected_mask: torch.Tensor,
    Lk: int,
    frame_seqlen: int,
) -> torch.Tensor:
    Hs, n_frames = selected_mask.shape
    device = selected_mask.device
    if Hs == 0:
        return torch.empty((1, 0, 1, Lk), device=device, dtype=torch.bool)

    k_idx = torch.arange(Lk, device=device, dtype=torch.long)
    k_frame = (k_idx // frame_seqlen).clamp_max(n_frames - 1)
    mask_k = selected_mask[:, k_frame]
    return mask_k.unsqueeze(0).unsqueeze(2)


def _select_kv_frame_indices_density(
    frame_mass_1d: torch.Tensor,
    density: float,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
    forced_outside_budget: bool = True,
) -> list[int]:
    compulsory: set[int] = set(forced_frames) | sink_frames

    if forced_outside_budget:
        n_compulsory = len(compulsory)
        max_budget_for_selection = max(0, n_frames - n_compulsory)
        n_select_target = min(max_budget_for_selection, int(math.ceil(density * max_budget_for_selection)))

        all_idx = torch.arange(n_frames, device=frame_mass_1d.device, dtype=torch.long)
        keep_mask = torch.zeros(n_frames, device=frame_mass_1d.device, dtype=torch.bool)
        if compulsory:
            comp_idx = torch.tensor(list(compulsory), device=frame_mass_1d.device, dtype=torch.long)
            comp_idx = comp_idx[(comp_idx >= 0) & (comp_idx < n_frames)]
            if comp_idx.numel() > 0:
                keep_mask[comp_idx] = True

        rem = all_idx[~keep_mask]
        selected: set[int] = set(compulsory)
        if rem.numel() > 0:
            order = rem[torch.argsort(frame_mass_1d[rem], descending=True)]
            need = max(0, n_select_target)
            if need > 0:
                selected.update(order[:need].tolist())
        return sorted(selected)

    n_keep_target = max(1, min(n_frames, int(math.ceil(density * n_frames))))
    all_idx = torch.arange(n_frames, device=frame_mass_1d.device, dtype=torch.long)
    order = all_idx[torch.argsort(frame_mass_1d, descending=True)]
    selected = set(order[:n_keep_target].tolist())
    return sorted(selected)


def _select_kv_frame_indices_recall(
    frame_mass_1d: torch.Tensor,
    target_recall: float,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
    forced_outside_budget: bool = True,
) -> list[int]:
    compulsory: set[int] = set(forced_frames) | sink_frames
    tr = max(0.0, min(1.0, float(target_recall)))
    total = float(frame_mass_1d.sum().clamp_min(1e-8).item())
    all_idx = torch.arange(n_frames, device=frame_mass_1d.device, dtype=torch.long)

    if forced_outside_budget:
        keep_mask = torch.zeros(n_frames, device=frame_mass_1d.device, dtype=torch.bool)
        selected: set[int] = set(compulsory)
        selected_mass = 0.0
        if compulsory:
            comp_idx = torch.tensor(list(compulsory), device=frame_mass_1d.device, dtype=torch.long)
            comp_idx = comp_idx[(comp_idx >= 0) & (comp_idx < n_frames)]
            if comp_idx.numel() > 0:
                keep_mask[comp_idx] = True
                selected_mass = float(frame_mass_1d[comp_idx].sum().item())

        rem = all_idx[~keep_mask]
        if rem.numel() > 0 and selected_mass / total < tr - 1e-5:
            order = rem[torch.argsort(frame_mass_1d[rem], descending=True)]
            need_mass = tr * total - selected_mass
            rem_mass = frame_mass_1d[order]
            csum = torch.cumsum(rem_mass, dim=0)
            pos = torch.searchsorted(csum, torch.tensor(need_mass, device=csum.device))
            n_take = min(order.numel(), int(pos.item()) + 1)
            if n_take > 0:
                selected.update(order[:n_take].tolist())
        return sorted(selected)

    order = all_idx[torch.argsort(frame_mass_1d, descending=True)]
    sorted_mass = frame_mass_1d[order]
    csum = torch.cumsum(sorted_mass, dim=0)
    need_mass = tr * total
    pos = torch.searchsorted(csum, torch.tensor(need_mass, device=csum.device))
    n_take = min(order.numel(), int(pos.item()) + 1)
    selected = set(order[:n_take].tolist()) if n_take > 0 else set()
    return sorted(selected)


def _select_kv_frame_indices_max_kv_frames(
    frame_mass_1d: torch.Tensor,
    max_kv_frames: int,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
    forced_outside_budget: bool = True,
) -> list[int]:
    if n_frames <= 0:
        return []

    # For rule=max_kv_frames, the budget is the final total kept frame count.
    # Mandatory current/sink frames therefore consume this budget as well.
    _ = forced_outside_budget
    cap = max(0, int(max_kv_frames))
    sink_valid: set[int] = {int(v) for v in set(sink_frames) if 0 <= int(v) < n_frames}
    forced_valid: set[int] = {int(v) for v in set(forced_frames) if 0 <= int(v) < n_frames}
    compulsory: set[int] = sink_valid | forced_valid
    if n_frames <= cap:
        return list(range(n_frames))

    all_idx = torch.arange(n_frames, device=frame_mass_1d.device, dtype=torch.long)
    keep_mask = torch.zeros(n_frames, device=frame_mass_1d.device, dtype=torch.bool)
    if compulsory:
        comp_idx = torch.tensor(sorted(compulsory), device=frame_mass_1d.device, dtype=torch.long)
        keep_mask[comp_idx] = True

    rem = all_idx[~keep_mask]
    selected: set[int] = set(compulsory)
    if cap <= 0:
        return sorted(selected)
    if len(compulsory) >= n_frames:
        return sorted(selected)
    if len(compulsory) >= cap:
        return sorted(selected)
    n_take = min(int(cap - len(compulsory)), int(rem.numel()))
    if n_take > 0:
        order = rem[torch.argsort(frame_mass_1d[rem], descending=True)]
        selected.update(order[:n_take].tolist())
    return sorted(selected)


def _select_kv_frame_indices_topk_plus_threshold(
    frame_mass_1d: torch.Tensor,
    max_kv_frames: int,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
    score_threshold_ratio: float,
    forced_outside_budget: bool = True,
) -> list[int]:
    # Keep the usual top-k budget first, then expand with near-peak frames.
    selected = set(
        _select_kv_frame_indices_max_kv_frames(
            frame_mass_1d,
            max_kv_frames,
            n_frames,
            sink_frames,
            forced_frames,
            forced_outside_budget,
        )
    )
    if n_frames <= 0 or frame_mass_1d.numel() <= 0:
        return sorted(selected)

    ratio = max(0.0, float(score_threshold_ratio))
    threshold = float(frame_mass_1d.max().item()) * ratio
    all_idx = torch.arange(n_frames, device=frame_mass_1d.device, dtype=torch.long)
    extra = all_idx[frame_mass_1d > threshold]
    if extra.numel() > 0:
        selected.update(extra.tolist())
    return sorted(selected)


def _select_kv_frame_indices_max_kv_frames_batched(
    frame_mass_2d: torch.Tensor,
    max_kv_frames: int,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
    forced_outside_budget: bool = True,
) -> torch.Tensor:
    Hs = int(frame_mass_2d.shape[0])
    device = frame_mass_2d.device
    if Hs <= 0 or n_frames <= 0:
        return torch.empty((Hs, 0), device=device, dtype=torch.long)

    _ = forced_outside_budget
    cap = max(0, int(max_kv_frames))
    sink_valid = sorted(int(v) for v in set(sink_frames) if 0 <= int(v) < n_frames)
    forced_valid = sorted(int(v) for v in set(forced_frames) if 0 <= int(v) < n_frames)
    compulsory = sorted(set(sink_valid) | set(forced_valid))
    if n_frames <= cap:
        all_frames = torch.arange(n_frames, device=device, dtype=torch.long)
        return all_frames.unsqueeze(0).expand(Hs, -1)

    if cap <= 0 or len(compulsory) >= cap or len(compulsory) >= n_frames:
        if not compulsory:
            return torch.empty((Hs, 0), device=device, dtype=torch.long)
        comp_idx = torch.tensor(compulsory, device=device, dtype=torch.long)
        return comp_idx.unsqueeze(0).expand(Hs, -1)

    if compulsory:
        comp_idx = torch.tensor(compulsory, device=device, dtype=torch.long)
    else:
        comp_idx = None

    n_remaining = int(n_frames - len(compulsory))
    n_take = min(int(cap - len(compulsory)), n_remaining)
    if n_take <= 0 or n_remaining <= 0:
        if comp_idx is None:
            return torch.empty((Hs, 0), device=device, dtype=torch.long)
        return comp_idx.unsqueeze(0).expand(Hs, -1)

    if comp_idx is not None and comp_idx.numel() > 0:
        candidate_scores = frame_mass_2d.clone()
        candidate_scores.index_fill_(1, comp_idx, -torch.inf)
    else:
        candidate_scores = frame_mass_2d

    top_frames = torch.topk(
        candidate_scores,
        k=n_take,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    if comp_idx is not None and comp_idx.numel() > 0:
        selected = torch.cat((comp_idx.unsqueeze(0).expand(Hs, -1), top_frames), dim=1)
    else:
        selected = top_frames
    return torch.sort(selected, dim=1).values


@torch.no_grad()
def _select_kv_frame_indices_weighted_random_batched(
    frame_score_2d: torch.Tensor,
    max_kv_frames: int,
    n_frames: int,
    sink_frames: set[int],
    forced_frames: set[int],
) -> torch.Tensor:
    """Select compulsory frames plus weighted-random historical frames.

    ``frame_score_2d`` contains pooled-QK logits for each head and frame.
    Sampling weights are their softmax probabilities over non-compulsory
    frames. A tiny floor only prevents numerical underflow from making a
    candidate ineligible when sampling without replacement.
    """
    if frame_score_2d.ndim != 2:
        raise ValueError("frame_score_2d must be [heads, frames]")
    Hs = int(frame_score_2d.shape[0])
    n_frames = max(0, int(n_frames))
    device = frame_score_2d.device
    if Hs <= 0 or n_frames <= 0:
        return torch.empty((Hs, 0), device=device, dtype=torch.long)
    if int(frame_score_2d.shape[1]) < n_frames:
        raise ValueError("frame_score_2d has fewer frames than n_frames")

    cap = max(0, int(max_kv_frames))
    compulsory = sorted(
        {
            int(v)
            for v in set(sink_frames) | set(forced_frames)
            if 0 <= int(v) < n_frames
        }
    )
    if n_frames <= cap:
        all_frames = torch.arange(n_frames, device=device, dtype=torch.long)
        return all_frames.unsqueeze(0).expand(Hs, -1)
    if cap <= 0 or len(compulsory) >= cap:
        comp = torch.tensor(compulsory, device=device, dtype=torch.long)
        return comp.unsqueeze(0).expand(Hs, -1) if comp.numel() else torch.empty(
            (Hs, 0), device=device, dtype=torch.long
        )

    candidate_logits = frame_score_2d[:, :n_frames].float()
    finite_limit = torch.finfo(candidate_logits.dtype).max
    candidate_logits = torch.nan_to_num(
        candidate_logits,
        nan=0.0,
        posinf=finite_limit,
        neginf=-finite_limit,
    )
    if compulsory:
        compulsory_idx = torch.tensor(compulsory, device=device, dtype=torch.long)
        candidate_logits.index_fill_(1, compulsory_idx, -torch.inf)
    weights = torch.softmax(candidate_logits, dim=1)
    # Extremely small softmax probabilities can underflow to zero in float32.
    # Keep every non-compulsory frame eligible for no-replacement sampling.
    weights.clamp_min_(torch.finfo(weights.dtype).tiny)
    if compulsory:
        weights.index_fill_(1, compulsory_idx, 0.0)
    n_take = min(cap - len(compulsory), n_frames - len(compulsory))
    sampled = torch.multinomial(weights, num_samples=n_take, replacement=False)
    if compulsory:
        comp = torch.tensor(compulsory, device=device, dtype=torch.long)
        sampled = torch.cat((comp.unsqueeze(0).expand(Hs, -1), sampled), dim=1)
    return torch.sort(sampled, dim=1).values


def _select_kv_frame_indices_density_if_low_recall(
    frame_mass_1d: torch.Tensor,
    density: float,
    n_bins: int,
    sink_frames: set[int],
    forced_frames: set[int],
    low_recall_threshold: float,
    expand_target_recall: float,
    forced_outside_budget: bool = True,
) -> list[int]:
    gate = max(0.0, min(1.0, float(low_recall_threshold)))
    tr = max(0.0, min(1.0, float(expand_target_recall)))
    total = float(frame_mass_1d.sum().clamp_min(1e-8).item())
    all_idx = torch.arange(n_bins, device=frame_mass_1d.device, dtype=torch.long)

    if forced_outside_budget:
        sel_d = _select_kv_frame_indices_density(
            frame_mass_1d, density, n_bins, sink_frames, forced_frames, True
        )
        selected: set[int] = set(sel_d)

        keep_mask = torch.zeros(n_bins, device=frame_mass_1d.device, dtype=torch.bool)
        cum = 0.0
        if selected:
            sel_idx = torch.tensor(list(selected), device=frame_mass_1d.device, dtype=torch.long)
            sel_idx = sel_idx[(sel_idx >= 0) & (sel_idx < n_bins)]
            if sel_idx.numel() > 0:
                keep_mask[sel_idx] = True
                cum = float(frame_mass_1d[sel_idx].sum().item())

        rem = all_idx[~keep_mask]
        r = cum / total
        if rem.numel() > 0 and r < gate - 1e-5 and r < tr - 1e-5:
            order = rem[torch.argsort(frame_mass_1d[rem], descending=True)]
            need_mass = tr * total - cum
            rem_mass = frame_mass_1d[order]
            csum = torch.cumsum(rem_mass, dim=0)
            pos = torch.searchsorted(csum, torch.tensor(need_mass, device=csum.device))
            n_take = min(order.numel(), int(pos.item()) + 1)
            if n_take > 0:
                selected.update(order[:n_take].tolist())
        return sorted(selected)

    n_keep_target = max(1, min(n_bins, int(math.ceil(density * n_bins))))
    order = all_idx[torch.argsort(frame_mass_1d, descending=True)]
    selected: set[int] = set(order[:n_keep_target].tolist())

    sel_idx = torch.tensor(list(selected), device=frame_mass_1d.device, dtype=torch.long)
    sel_idx = sel_idx[(sel_idx >= 0) & (sel_idx < n_bins)]
    cum = float(frame_mass_1d[sel_idx].sum().item()) if sel_idx.numel() > 0 else 0.0
    r = cum / total

    if r < gate - 1e-5:
        keep_mask = torch.zeros(n_bins, device=frame_mass_1d.device, dtype=torch.bool)
        if selected:
            keep_mask[sel_idx] = True
        rem = all_idx[~keep_mask]
        if rem.numel() > 0 and r < tr - 1e-5:
            order_rem = rem[torch.argsort(frame_mass_1d[rem], descending=True)]
            need_mass = tr * total - cum
            rem_mass = frame_mass_1d[order_rem]
            csum = torch.cumsum(rem_mass, dim=0)
            pos = torch.searchsorted(csum, torch.tensor(need_mass, device=csum.device))
            n_take = min(order_rem.numel(), int(pos.item()) + 1)
            if n_take > 0:
                selected.update(order_rem[:n_take].tolist())
    return sorted(selected)


@torch.no_grad()
def _per_frame_simple_mean(
    q: torch.Tensor,
    k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    scale = 1.0 / (D**0.5)

    if frame_seqlen > 0 and q.shape[1] % frame_seqlen == 0:
        q_mean = q.float().mean(dim=1)
    else:
        qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
        if qf.shape[1] == 0:
            return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
        q_mean = qf.mean(dim=1)
    kf = _avg_pool_tokens_along_seq(k, frame_seqlen).float()
    if kf.shape[1] == 0:
        return torch.zeros(Hs, kf.shape[1], device=q.device, dtype=torch.float32)
    importance = (torch.einsum("bhd,bjhd->bhj", q_mean, kf) * scale).mean(dim=0)
    return importance


@torch.no_grad()
def _per_frame_simple_mean_cached_k(
    q: torch.Tensor,
    frame_k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    if frame_k.shape[1] == 0:
        return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)
    if frame_seqlen > 0 and q.shape[1] % frame_seqlen == 0:
        q_mean = q.float().mean(dim=1)
    else:
        qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
        if qf.shape[1] == 0:
            return torch.zeros(Hs, frame_k.shape[1], device=q.device, dtype=torch.float32)
        q_mean = qf.mean(dim=1)
    kf = frame_k.float()
    importance = (torch.einsum("bhd,bjhd->bhj", q_mean, kf) * scale).mean(dim=0)
    return importance


@torch.no_grad()
def _per_frame_softmax_qmax(
    q: torch.Tensor,
    k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    scale = 1.0 / (D**0.5)

    qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
    kf = _avg_pool_tokens_along_seq(k, frame_seqlen).float()
    if qf.shape[1] == 0 or kf.shape[1] == 0:
        return torch.zeros(Hs, kf.shape[1], device=q.device, dtype=torch.float32)

    logits = torch.einsum("bihd,bjhd->bhij", qf, kf) * scale
    mass = torch.softmax(logits, dim=-1)
    return mass.max(dim=2).values.mean(dim=0)


@torch.no_grad()
def _per_frame_softmax_qmax_cached_k(
    q: torch.Tensor,
    frame_k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    if frame_k.shape[1] == 0:
        return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
    kf = frame_k.float()
    if qf.shape[1] == 0:
        return torch.zeros(Hs, kf.shape[1], device=q.device, dtype=torch.float32)

    logits = torch.einsum("bihd,bjhd->bhij", qf, kf) * scale
    mass = torch.softmax(logits, dim=-1)
    return mass.max(dim=2).values.mean(dim=0)


@torch.no_grad()
def _per_frame_attn_mass_cached_k(
    q: torch.Tensor,
    frame_k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    if frame_k.shape[1] == 0:
        return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
    kf = frame_k.float()
    if qf.shape[1] == 0:
        return torch.zeros(Hs, kf.shape[1], device=q.device, dtype=torch.float32)

    logits = torch.einsum("bihd,bjhd->bhij", qf, kf) * scale
    mass = torch.softmax(logits, dim=-1)
    return mass.sum(dim=2).mean(dim=0)


@torch.no_grad()
def _per_frame_q_sign_minmax_qmax(
    q: torch.Tensor,
    k: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    if k.shape[1] == 0:
        return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
    k_max, k_min = _minmax_pool_tokens_along_seq(k, frame_seqlen)
    k_max = k_max.float()
    k_min = k_min.float()
    if qf.shape[1] == 0:
        return torch.zeros(Hs, k_max.shape[1], device=q.device, dtype=torch.float32)

    q_exp = qf.unsqueeze(2)
    k_signed = torch.where(q_exp >= 0, k_max.unsqueeze(1), k_min.unsqueeze(1))
    score = (q_exp * k_signed).sum(dim=-1) * scale
    return score.permute(0, 3, 1, 2).max(dim=2).values.mean(dim=0)


@torch.no_grad()
def _per_frame_q_sign_minmax_qmax_cached_k(
    q: torch.Tensor,
    k_max: torch.Tensor,
    k_min: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    B, _, Hs, D = q.shape
    _ = B
    if k_max.shape[1] == 0 or k_min.shape[1] == 0:
        return torch.zeros(Hs, 0, device=q.device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    qf = _avg_pool_tokens_along_seq(q, frame_seqlen).float()
    if qf.shape[1] == 0:
        return torch.zeros(Hs, k_max.shape[1], device=q.device, dtype=torch.float32)

    k_max = k_max.float()
    k_min = k_min.float()
    q_exp = qf.unsqueeze(2)
    k_signed = torch.where(q_exp >= 0, k_max.unsqueeze(1), k_min.unsqueeze(1))
    score = (q_exp * k_signed).sum(dim=-1) * scale
    return score.permute(0, 3, 1, 2).max(dim=2).values.mean(dim=0)


def _frame_importance_scores(
    cfg: KVSparseConfig,
    q: torch.Tensor,
    k: torch.Tensor | None,
    frame_seqlen: int,
    chunk_k: int,
    frame_k_index_window: torch.Tensor | None = None,
    frame_k_min_index_window: torch.Tensor | None = None,
    frame_k_max_index_window: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.frame_selection_score == "pooled_qk":
        if frame_k_index_window is not None:
            return _per_frame_simple_mean_cached_k(q, frame_k_index_window, frame_seqlen)
        if k is None:
            raise ValueError("pooled_qk frame scoring requires cached frame means or raw k")
        return _per_frame_simple_mean(q, k, frame_seqlen)
    if cfg.frame_selection_score == "attn_mass_qmax":
        if frame_k_index_window is not None:
            return _per_frame_softmax_qmax_cached_k(q, frame_k_index_window, frame_seqlen)
        if k is None:
            raise ValueError("attn_mass_qmax frame scoring requires cached frame means or raw k")
        return _per_frame_softmax_qmax(q, k, frame_seqlen)
    if cfg.frame_selection_score == "q_sign_minmax_qmax":
        if frame_k_min_index_window is not None and frame_k_max_index_window is not None:
            return _per_frame_q_sign_minmax_qmax_cached_k(
                q, frame_k_max_index_window, frame_k_min_index_window, frame_seqlen
            )
        if k is None:
            raise ValueError("q_sign_minmax_qmax frame scoring requires cached frame min/max or raw k")
        return _per_frame_q_sign_minmax_qmax(q, k, frame_seqlen)
    if k is None:
        if frame_k_index_window is not None:
            return _per_frame_attn_mass_cached_k(q, frame_k_index_window, frame_seqlen)
        raise ValueError("attn_mass frame scoring requires cached frame means or raw k")
    q_pool = cfg.importance_q_pool
    k_pool = cfg.importance_k_pool
    q_i, k_i, lk_orig = _prepare_qk_for_importance(q, k, q_pool, k_pool)
    return _per_frame_attn_mass_batched(
        q_i, k_i, frame_seqlen, chunk_k, lk_orig=lk_orig, k_pool=k_pool
    )


def _key_token_mask_from_frames(
    Lk: int,
    n_frames: int,
    frame_seqlen: int,
    selected_frames: set[int],
    device: torch.device,
) -> torch.Tensor:
    if Lk <= 0 or n_frames <= 0 or not selected_frames:
        return torch.zeros(Lk, device=device, dtype=torch.bool)
    frame_mask = torch.zeros(n_frames, device=device, dtype=torch.bool)
    idx = torch.tensor(sorted(selected_frames), device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    if idx.numel() > 0:
        frame_mask[idx] = True
    k_idx = torch.arange(Lk, device=device, dtype=torch.long)
    k_frame = (k_idx // frame_seqlen).clamp_max(n_frames - 1)
    return frame_mask[k_frame]


def _key_token_indices_from_frames(
    Lk: int,
    n_frames: int,
    frame_seqlen: int,
    selected_frames: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    if Lk <= 0 or n_frames <= 0 or frame_seqlen <= 0 or not selected_frames:
        return torch.empty(0, device=device, dtype=torch.long), []

    frame_ids = [
        int(f)
        for f in sorted({int(v) for v in selected_frames})
        if 0 <= int(f) < n_frames and int(f) * frame_seqlen < Lk
    ]
    if not frame_ids:
        return torch.empty(0, device=device, dtype=torch.long), []

    parts: list[torch.Tensor] = []
    kept_frames: list[int] = []
    for frame_id in frame_ids:
        t0 = frame_id * frame_seqlen
        t1 = min((frame_id + 1) * frame_seqlen, Lk)
        if t0 >= t1:
            continue
        parts.append(torch.arange(t0, t1, device=device, dtype=torch.long))
        kept_frames.append(frame_id)

    if not parts:
        return torch.empty(0, device=device, dtype=torch.long), []
    return torch.cat(parts, dim=0), kept_frames


def _select_candidate_blocks_by_density(
    block_mass_1d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_k_mask: torch.Tensor,
    density: float,
) -> torch.Tensor:
    density = max(0.0, min(1.0, float(density)))
    if density <= 0.0 or n_bq <= 0 or n_bk <= 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    candidate_k_mask = candidate_k_mask.to(dtype=torch.bool, device=block_mass_1d.device)
    cand_k = candidate_k_mask.nonzero(as_tuple=False).flatten()
    if cand_k.numel() == 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    total_candidates = int(n_bq * cand_k.numel())
    n_take = min(total_candidates, int(math.ceil(density * total_candidates)))
    if n_take <= 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    mass_2d = block_mass_1d.view(n_bq, n_bk)
    candidate_scores = mass_2d[:, cand_k].reshape(-1)
    if n_take >= int(candidate_scores.numel()):
        chosen = torch.arange(candidate_scores.numel(), device=block_mass_1d.device, dtype=torch.long)
    else:
        chosen = torch.topk(candidate_scores, k=n_take, largest=True, sorted=False).indices
    k_count = int(cand_k.numel())
    q_idx = chosen // k_count
    k_idx = cand_k[chosen % k_count]
    return (q_idx * n_bk + k_idx).to(dtype=torch.long)


def _select_candidate_blocks_by_density_batched(
    block_mass_2d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_k_mask: torch.Tensor,
    density: float,
) -> torch.Tensor:
    density = max(0.0, min(1.0, float(density)))
    Hs = int(block_mass_2d.shape[0])
    device = block_mass_2d.device
    if density <= 0.0 or Hs <= 0 or n_bq <= 0 or n_bk <= 0:
        return torch.empty((Hs, 0), device=device, dtype=torch.long)

    if candidate_k_mask.ndim == 1:
        candidate_k_mask = candidate_k_mask.unsqueeze(0).expand(Hs, -1)
    candidate_k_mask = candidate_k_mask.to(device=device, dtype=torch.bool)
    if candidate_k_mask.shape != (Hs, n_bk):
        raise ValueError(
            f"candidate_k_mask shape mismatch: expected {(Hs, n_bk)}, got {tuple(candidate_k_mask.shape)}"
        )
    if not candidate_k_mask.any():
        return torch.empty((Hs, 0), device=device, dtype=torch.long)

    candidate_counts = candidate_k_mask.sum(dim=1, dtype=torch.long) * int(n_bq)
    if int(candidate_counts.max().item()) <= 0:
        return torch.empty((Hs, 0), device=device, dtype=torch.long)

    n_take = torch.ceil(candidate_counts.to(dtype=torch.float32) * float(density)).to(dtype=torch.long)
    n_take = torch.minimum(n_take, candidate_counts)
    max_take = int(n_take.max().item())
    if max_take <= 0:
        return torch.empty((Hs, 0), device=device, dtype=torch.long)

    if bool(candidate_k_mask.all().item()):
        chosen = torch.topk(block_mass_2d, k=max_take, dim=1, largest=True, sorted=False).indices
    else:
        same_mask = Hs == 1 or bool((candidate_k_mask == candidate_k_mask[:1]).all().item())
        if same_mask:
            cand_k = candidate_k_mask[0].nonzero(as_tuple=False).flatten()
            candidate_scores = (
                block_mass_2d.view(Hs, n_bq, n_bk)
                .index_select(2, cand_k)
                .reshape(Hs, -1)
            )
            chosen_local = torch.topk(candidate_scores, k=max_take, dim=1, largest=True, sorted=False).indices
            k_count = int(cand_k.numel())
            q_idx = chosen_local // k_count
            k_idx = cand_k[chosen_local % k_count]
            chosen = (q_idx * n_bk + k_idx).to(dtype=torch.long)
        else:
            mass_3d = block_mass_2d.view(Hs, n_bq, n_bk)
            invalid_fill = torch.finfo(mass_3d.dtype).min
            masked_scores = mass_3d.masked_fill(~candidate_k_mask.unsqueeze(1), invalid_fill).reshape(Hs, -1)
            chosen = torch.topk(masked_scores, k=max_take, dim=1, largest=True, sorted=False).indices
    rank = torch.arange(max_take, device=device, dtype=torch.long).unsqueeze(0)
    chosen = torch.where(rank < n_take.unsqueeze(1), chosen, torch.full_like(chosen, -1))
    return chosen.to(dtype=torch.long)


def _allocate_integer_budget_by_weights(
    caps: torch.Tensor,
    weights: torch.Tensor,
    total: int,
) -> torch.Tensor:
    counts = torch.zeros_like(caps)
    total = max(0, int(total))
    if total <= 0 or caps.numel() == 0:
        return counts

    remaining = total
    caps_rem = caps.clone()
    weights = weights.to(device=caps.device, dtype=torch.float32)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

    while remaining > 0:
        active = caps_rem > 0
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=False).flatten()
        active_caps = caps_rem.index_select(0, active_idx)
        active_weights = weights.index_select(0, active_idx).clone()
        if active_weights.numel() == 0:
            break
        probs = torch.softmax(active_weights, dim=0)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 1e-8:
            probs = active_caps.float()
            if float(probs.sum().item()) <= 1e-8:
                probs = torch.ones_like(probs)
            probs = probs / probs.sum()

        quota = probs * float(remaining)
        take = torch.floor(quota).to(dtype=torch.long)
        take = torch.minimum(take, active_caps)
        taken_now = int(take.sum().item())
        if taken_now > 0:
            counts[active_idx] += take
            caps_rem[active_idx] -= take
            remaining -= taken_now
            continue

        frac = quota - torch.floor(quota)
        order = torch.argsort(frac, descending=True)
        assigned = False
        for pos in order.tolist():
            if remaining <= 0:
                break
            idx = int(active_idx[pos].item())
            if int(caps_rem[idx].item()) <= 0:
                continue
            counts[idx] += 1
            caps_rem[idx] -= 1
            remaining -= 1
            assigned = True
        if not assigned:
            break

    return counts


def _select_candidate_blocks_by_recall(
    block_mass_1d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_k_mask: torch.Tensor,
    target_recall: float,
    *,
    denominator_k_mask: torch.Tensor | None = None,
    preselected_k_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target_recall = max(0.0, min(1.0, float(target_recall)))
    if target_recall <= 0.0 or n_bq <= 0 or n_bk <= 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    candidate_k_mask = candidate_k_mask.to(dtype=torch.bool, device=block_mass_1d.device)
    if preselected_k_mask is not None:
        preselected_k_mask = preselected_k_mask.to(dtype=torch.bool, device=block_mass_1d.device)
        candidate_k_mask = candidate_k_mask & ~preselected_k_mask
    cand_k = candidate_k_mask.nonzero(as_tuple=False).flatten()
    if cand_k.numel() == 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    if denominator_k_mask is None:
        denominator_k_mask = candidate_k_mask
    else:
        denominator_k_mask = denominator_k_mask.to(dtype=torch.bool, device=block_mass_1d.device)
    denom_k = denominator_k_mask.nonzero(as_tuple=False).flatten()
    if denom_k.numel() == 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    mass_2d = block_mass_1d.view(n_bq, n_bk)
    total_mass = float(mass_2d[:, denom_k].sum().item())
    if total_mass <= 0.0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    preselected_mass = 0.0
    if preselected_k_mask is not None:
        pre_k = preselected_k_mask.nonzero(as_tuple=False).flatten()
        if pre_k.numel() > 0:
            preselected_mass = float(mass_2d[:, pre_k].sum().item())

    need_mass = target_recall * total_mass - preselected_mass
    if need_mass <= 1e-8:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    candidate_scores = mass_2d[:, cand_k].reshape(-1)
    if candidate_scores.numel() == 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    order = torch.argsort(candidate_scores, descending=True)
    sorted_scores = candidate_scores[order]
    csum = torch.cumsum(sorted_scores, dim=0)
    pos = torch.searchsorted(
        csum,
        torch.tensor(need_mass, device=csum.device, dtype=csum.dtype),
    )
    n_take = min(int(order.numel()), int(pos.item()) + 1)
    if n_take <= 0:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    chosen = order[:n_take]
    k_count = int(cand_k.numel())
    q_idx = chosen // k_count
    k_idx = cand_k[chosen % k_count]
    return (q_idx * n_bk + k_idx).to(dtype=torch.long)


def _pooled_key_mask_from_frames(
    pooled_key_frames: torch.Tensor,
    frames: set[int],
    device: torch.device,
) -> torch.Tensor:
    key_frames = pooled_key_frames.to(device=device, dtype=torch.long)
    if not frames:
        return torch.zeros_like(key_frames, dtype=torch.bool, device=device)
    frame_ids = sorted({int(v) for v in frames if int(v) >= 0})
    if not frame_ids:
        return torch.zeros_like(key_frames, dtype=torch.bool, device=device)
    frame_ids_t = torch.tensor(frame_ids, device=device, dtype=torch.long)
    return (key_frames.unsqueeze(0) == frame_ids_t.unsqueeze(1)).any(dim=0)


def _select_candidate_blocks_by_frame_density(
    block_mass_1d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_frames: set[int],
    pooled_key_frames: torch.Tensor,
    density: float,
) -> torch.Tensor:
    density = max(0.0, min(1.0, float(density)))
    if density <= 0.0 or n_bq <= 0 or n_bk <= 0 or not candidate_frames:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    frame_ids = sorted({int(v) for v in candidate_frames if int(v) >= 0})
    if not frame_ids:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)

    device = block_mass_1d.device
    candidate_k_mask = _pooled_key_mask_from_frames(pooled_key_frames, set(frame_ids), device)
    return _select_candidate_blocks_by_density(
        block_mass_1d,
        n_bq,
        n_bk,
        candidate_k_mask,
        density,
    )


def _select_candidate_blocks_by_frame_density_weighted(
    block_mass_1d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_frames: set[int],
    pooled_key_frames: torch.Tensor,
    density: float,
    frame_scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, int]]:
    density = max(0.0, min(1.0, float(density)))
    if density <= 0.0 or n_bq <= 0 or n_bk <= 0 or not candidate_frames:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long), {}

    device = block_mass_1d.device
    frame_ids = sorted({int(v) for v in candidate_frames if int(v) >= 0})
    if not frame_ids:
        return torch.empty(0, device=device, dtype=torch.long), {}

    pooled_key_frames = pooled_key_frames.to(device=device, dtype=torch.long)
    frame_scores = frame_scores.to(device=device, dtype=torch.float32)
    mass_2d = block_mass_1d.view(n_bq, n_bk)

    per_frame_k_cols: list[torch.Tensor] = []
    per_frame_caps: list[int] = []
    per_frame_weights: list[float] = []
    valid_frame_ids: list[int] = []
    for frame_id in frame_ids:
        k_cols = (pooled_key_frames == int(frame_id)).nonzero(as_tuple=False).flatten()
        if k_cols.numel() == 0:
            continue
        per_frame_k_cols.append(k_cols)
        per_frame_caps.append(int(n_bq * k_cols.numel()))
        weight = 0.0
        if 0 <= int(frame_id) < int(frame_scores.numel()):
            weight = float(frame_scores[int(frame_id)].item())
        per_frame_weights.append(weight)
        valid_frame_ids.append(int(frame_id))

    if not per_frame_caps:
        return torch.empty(0, device=device, dtype=torch.long), {}

    total_candidates = int(sum(per_frame_caps))
    n_take = min(total_candidates, int(math.ceil(density * total_candidates)))
    if n_take <= 0:
        return torch.empty(0, device=device, dtype=torch.long), {frame_id: 0 for frame_id in valid_frame_ids}

    caps_t = torch.tensor(per_frame_caps, device=device, dtype=torch.long)
    weights_t = torch.tensor(per_frame_weights, device=device, dtype=torch.float32)
    take_t = _allocate_integer_budget_by_weights(caps_t, weights_t, n_take)

    selected_tensors: list[torch.Tensor] = []
    budget_by_frame: dict[int, int] = {}
    for row, frame_id in enumerate(valid_frame_ids):
        frame_take = int(take_t[row].item())
        budget_by_frame[int(frame_id)] = frame_take
        if frame_take <= 0:
            continue
        k_cols = per_frame_k_cols[row]
        frame_scores_2d = mass_2d[:, k_cols].reshape(-1)
        if frame_take >= int(frame_scores_2d.numel()):
            chosen = torch.arange(frame_scores_2d.numel(), device=device, dtype=torch.long)
        else:
            chosen = torch.topk(frame_scores_2d, k=frame_take, largest=True, sorted=False).indices
        k_count = int(k_cols.numel())
        q_idx = chosen // k_count
        k_idx = k_cols[chosen % k_count]
        selected_tensors.append((q_idx * n_bk + k_idx).to(dtype=torch.long))

    if not selected_tensors:
        return torch.empty(0, device=device, dtype=torch.long), budget_by_frame
    selected = torch.cat(selected_tensors, dim=0)
    selected = torch.unique(selected, sorted=False)
    return selected, budget_by_frame


def _select_candidate_blocks_by_frame_recall(
    block_mass_1d: torch.Tensor,
    n_bq: int,
    n_bk: int,
    candidate_frames: set[int],
    pooled_key_frames: torch.Tensor,
    target_recall: float,
    *,
    denominator_frames: set[int] | None = None,
    preselected_frames: set[int] | None = None,
) -> torch.Tensor:
    if n_bq <= 0 or n_bk <= 0 or not candidate_frames:
        return torch.empty(0, device=block_mass_1d.device, dtype=torch.long)
    device = block_mass_1d.device
    candidate_k_mask = _pooled_key_mask_from_frames(pooled_key_frames, candidate_frames, device)
    denominator_k_mask = _pooled_key_mask_from_frames(
        pooled_key_frames,
        denominator_frames if denominator_frames is not None else candidate_frames,
        device,
    )
    preselected_k_mask = (
        _pooled_key_mask_from_frames(pooled_key_frames, preselected_frames, device)
        if preselected_frames
        else None
    )
    return _select_candidate_blocks_by_recall(
        block_mass_1d,
        n_bq,
        n_bk,
        candidate_k_mask,
        target_recall,
        denominator_k_mask=denominator_k_mask,
        preselected_k_mask=preselected_k_mask,
    )


def _apply_rule_to_mass_1d(
    cfg: KVSparseConfig,
    layer_idx: int,
    head: int,
    mass_1d: torch.Tensor,
    n_bins: int,
    sink: set[int],
    forced: set[int],
    forced_outside_budget: bool = True,
) -> list[int]:
    r = cfg.rule
    if r == "density":
        d = cfg.get_density(layer_idx, head)
        return _select_kv_frame_indices_density(mass_1d, float(d), n_bins, sink, forced, forced_outside_budget)
    if r == "recall":
        if cfg.target_recall is None:
            raise ValueError("target_recall required when rule=recall")
        return _select_kv_frame_indices_recall(
            mass_1d, float(cfg.target_recall), n_bins, sink, forced, forced_outside_budget
        )
    if r == "max_kv_frames":
        if cfg.granularity not in ("kv_frame", "kv_frame_then_attn_block"):
            raise ValueError(
                "rule=max_kv_frames only supports granularity=kv_frame or kv_frame_then_attn_block"
            )
        requested_max_kv_frames = cfg.get_max_kv_frames(layer_idx, head)
        if requested_max_kv_frames is None:
            raise ValueError("max_kv_frames required when rule=max_kv_frames")
        return _select_kv_frame_indices_max_kv_frames(
            mass_1d,
            int(requested_max_kv_frames),
            n_bins,
            sink,
            forced,
            forced_outside_budget,
        )
    if r == "topk_plus_threshold":
        if cfg.granularity not in ("kv_frame", "kv_frame_then_attn_block"):
            raise ValueError(
                "rule=topk_plus_threshold only supports granularity=kv_frame or kv_frame_then_attn_block"
            )
        requested_max_kv_frames = cfg.get_max_kv_frames(layer_idx, head)
        if requested_max_kv_frames is None:
            raise ValueError("max_kv_frames required when rule=topk_plus_threshold")
        ratio = (
            float(cfg.frame_selection_score_threshold_ratio)
            if cfg.frame_selection_score_threshold_ratio is not None
            else 0.5
        )
        return _select_kv_frame_indices_topk_plus_threshold(
            mass_1d,
            int(requested_max_kv_frames),
            n_bins,
            sink,
            forced,
            ratio,
            forced_outside_budget,
        )
    if r == "density_if_low_recall":
        ex = cfg.expand_target_recall
        if ex is None:
            raise ValueError("expand_target_recall required when rule=density_if_low_recall")
        gate = float(cfg.low_recall_threshold) if cfg.low_recall_threshold is not None else 0.5
        d = cfg.get_density(layer_idx, head)
        return _select_kv_frame_indices_density_if_low_recall(
            mass_1d,
            float(d),
            n_bins,
            sink,
            forced,
            gate,
            float(ex),
            forced_outside_budget,
        )
    raise ValueError(f"unknown rule: {cfg.rule}")


def _head_is_dense(cfg: KVSparseConfig, layer_idx: int, head: int) -> bool:
    if cfg.rule == "recall":
        return cfg.target_recall is not None and float(cfg.target_recall) >= 1.0 - 1e-6
    if cfg.rule == "max_kv_frames":
        return False
    if cfg.rule == "topk_plus_threshold":
        return False
    if cfg.rule == "density_if_low_recall":
        d = cfg.get_density(layer_idx, head)
        ex = cfg.expand_target_recall
        if ex is None:
            return d >= 1.0 - 1e-6
        return d >= 1.0 - 1e-6 and float(ex) >= 1.0 - 1e-6
    return cfg.get_density(layer_idx, head) >= 1.0 - 1e-6


def _block_density_is_dense(cfg: KVSparseConfig, layer_idx: int, head: int) -> bool:
    if cfg.get_block_rule(layer_idx, head) == "recall":
        rec = cfg.get_block_recall(layer_idx, head)
        return rec is None or rec >= 1.0 - 1e-6
    return cfg.get_block_density(layer_idx, head) >= 1.0 - 1e-6


def _effective_block_density_for_selected_frames(
    cfg: KVSparseConfig,
    layer_idx: int,
    head: int,
    block_density: float,
    selected_frames: int,
) -> float:
    density = max(0.0, min(1.0, float(block_density)))
    if not bool(getattr(cfg, "block_density_scale_by_max_kv_frames", False)):
        return density
    if cfg.rule != "max_kv_frames":
        return density
    target_frames = cfg.get_max_kv_frames(layer_idx, head)
    if target_frames is None:
        return density
    selected_frames = max(0, int(selected_frames))
    target_frames = max(1, int(target_frames))
    if selected_frames <= 0 or selected_frames >= target_frames:
        return density
    return max(0.0, min(1.0, density * (float(target_frames) / float(selected_frames))))


def _extra_selected_indices(selected: list[int], compulsory: set[int]) -> list[int]:
    return sorted(int(v) for v in selected if int(v) not in compulsory)


__all__ = [
    "_apply_rule_to_mass_1d",
    "_attn_mask_from_selected_kv_frames_batched",
    "_attn_mask_from_block_mask_batched",
    "_attn_mask_from_selected_pooled_blocks",
    "_avg_pool_tokens_along_seq",
    "_block_mask_from_selected_linear_indices_batched",
    "_block_mask_from_selected_linear_indices",
    "_block_density_is_dense",
    "_effective_block_density_for_selected_frames",
    "_extra_selected_indices",
    "_forced_attn_block_linear_indices",
    "_frame_importance_scores",
    "_head_is_dense",
    "_key_token_indices_from_frames",
    "_key_token_mask_from_frames",
    "_mandatory_key_positions_pooled",
    "_mandatory_key_token_indices",
    "_per_block_attn_mass_batched",
    "_per_block_pooled_qk_scores_batched",
    "_prepare_qk_for_importance",
    "_select_candidate_blocks_by_density_batched",
    "_select_candidate_blocks_by_frame_density",
    "_select_candidate_blocks_by_frame_density_weighted",
    "_select_candidate_blocks_by_frame_recall",
    "_select_kv_frame_indices_max_kv_frames_batched",
    "_select_kv_frame_indices_weighted_random_batched",
    "_select_kv_frame_indices_topk_plus_threshold",
]
