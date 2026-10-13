# Modified for the HorizonKV inference artifact; see NOTICE.
# Derived from the included causal inference adapter; see NOTICE.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from flagscale.inference.core.diffusion.horizonkv.wan.modules.attention import attention
from flagscale.inference.core.diffusion.horizonkv.wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import copy
import os
import torch.nn as nn
import torch
import math
from flagscale.inference.core.diffusion.horizonkv.algorithm.attention_map import AttentionMapRecorder
from flagscale.inference.core.diffusion.horizonkv.algorithm.baseline_selector import calculate_chunk_sparsities
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig
from flagscale.inference.core.diffusion.horizonkv.algorithm.attention import (
    sparse_kv_self_attention,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    copy_kv_to_cpu_cache_async,
    queue_deferred_clean_kv_to_cpu_cache,
    update_gpu_kv_cache_frames,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_kernels import lightforcing_attention, lightforcing_rope_apply
from flagscale.inference.core.diffusion.horizonkv.algorithm.gpu_cache import local_lf_rope_attention


# Wan 1.3B needs flex-attention autotuning for its head geometry.
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def _normalize_debug_timestep(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        # Avoid synchronizing CUDA timesteps for optional diagnostic labels.
        if value.device.type == "cuda" and os.getenv("HORIZONKV_DEBUG_SYNC_TIMESTEP", "0") not in {
            "1", "true", "yes", "on"
        }:
            return None
        value = value.reshape(-1)[0].detach().cpu().item()
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_clean_kv_update_timestep(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool((value.detach().abs() <= 1e-6).all().item())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_clean_kv_update_timestep(v) for v in value)
    if isinstance(value, (int, float)):
        return abs(float(value)) <= 1e-6
    try:
        return abs(float(value)) <= 1e-6
    except (TypeError, ValueError):
        return False


def _mean_pool_along_frames(x: torch.Tensor, frame_seqlen: int) -> torch.Tensor:
    if frame_seqlen <= 0:
        raise ValueError("frame_seqlen must be positive")
    if x.shape[1] == 0:
        return x[:, :0]
    batch, length, heads, dim = x.shape
    n_full = length // frame_seqlen
    rest = length - n_full * frame_seqlen
    parts = []
    if n_full > 0:
        parts.append(
            x[:, : n_full * frame_seqlen]
            .contiguous()
            .reshape(batch, n_full, frame_seqlen, heads, dim)
            .mean(dim=2)
        )
    if rest > 0:
        parts.append(x[:, n_full * frame_seqlen :].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1)


def _minmax_pool_along_frames(
    x: torch.Tensor,
    frame_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_seqlen <= 0:
        raise ValueError("frame_seqlen must be positive")
    if x.shape[1] == 0:
        return x[:, :0], x[:, :0]
    batch, length, heads, dim = x.shape
    n_full = length // frame_seqlen
    rest = length - n_full * frame_seqlen
    max_parts = []
    min_parts = []
    if n_full > 0:
        view = x[:, : n_full * frame_seqlen].contiguous().reshape(
            batch, n_full, frame_seqlen, heads, dim
        )
        max_parts.append(view.max(dim=2).values)
        min_parts.append(view.min(dim=2).values)
    if rest > 0:
        tail = x[:, n_full * frame_seqlen :]
        max_parts.append(tail.max(dim=1, keepdim=True).values)
        min_parts.append(tail.min(dim=1, keepdim=True).values)
    return torch.cat(max_parts, dim=1), torch.cat(min_parts, dim=1)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 long_video_gen=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.long_video_gen = bool(long_video_gen)
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # Cache frame geometry on CPU to avoid per-layer device synchronization.
        self._cached_frame_seqlen: int | None = None
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        debug_block_index=None,
        debug_timestep=None,
        kv_sparse_config=None,
        num_frame_per_block: int = 1,
        kv_sparse_timestep=None,
        next_kv_cache=None,
        kv_cache_update: bool | None = None,
        selection_query_override: torch.Tensor | None = None,
        next_selection_query_override: torch.Tensor | None = None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # HorizonKV and LightForcing apply RoPE on the compacted visible window.
        use_compacted_window_rope = bool(self.long_video_gen) or (
            isinstance(kv_sparse_config, KVSparseConfig)
            and getattr(kv_sparse_config, "rope_semantics", "absolute")
            in {
                "horizonkv_compacted",
            }
        )
        use_sparse_layer = bool(
            isinstance(kv_sparse_config, KVSparseConfig)
            and kv_sparse_config.enabled
            and debug_block_index is not None
            and kv_sparse_config.layer_has_sparse(debug_block_index, self.num_heads)
        )

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            use_cpu_offload = bool(kv_cache.get("cpu_offload_kv", False))
            fast_lf_rope = bool(
                isinstance(kv_sparse_config, KVSparseConfig)
                and getattr(kv_sparse_config, "rope_semantics", "absolute")
                == "horizonkv_compacted"
            )
            if use_cpu_offload:
                if torch.is_grad_enabled():
                    raise RuntimeError(
                        "HorizonKV CPU KV offload supports no-grad inference only"
                    )

                frame_seqlen = kv_cache.get("frame_seqlen", None)
                if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
                    # Infer missing cache geometry once per module.
                    frame_seqlen = self._cached_frame_seqlen
                if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
                    frame_seqlen = int(math.prod(grid_sizes[0][1:].detach().cpu().tolist()))
                    self._cached_frame_seqlen = frame_seqlen
                current_start_frame = current_start // frame_seqlen
                if use_compacted_window_rope:
                    # Select over raw full-history K, then apply compacted RoPE.
                    roped_query = q
                    roped_key = None
                    cache_key = k
                else:
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
                    roped_key = causal_rope_apply(
                        k, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
                    cache_key = roped_key

                score_with_unroped_qk = bool(
                    isinstance(kv_sparse_config, KVSparseConfig)
                    and getattr(kv_sparse_config, "score_with_unroped_qk", False)
                )
                selection_query = (
                    selection_query_override
                    if selection_query_override is not None
                    else (
                        q.type_as(v)
                        if (score_with_unroped_qk or use_compacted_window_rope)
                        else None
                    )
                )
                selection_key_tensor = (
                    k.type_as(v)
                    if (score_with_unroped_qk or use_compacted_window_rope)
                    else roped_key
                )

                current_end = current_start + roped_query.shape[1]
                cache_end = cache_start + roped_query.shape[1]
                sink_tokens = self.sink_size * frame_seqlen
                local_start_index = int(cache_start)
                local_end_index = int(cache_end)
                cache_capacity = int(kv_cache["token_capacity"])
                if local_start_index < 0 or local_end_index > cache_capacity:
                    raise ValueError(
                        "CPU offloaded KV cache capacity is too small: "
                        f"span=[{local_start_index}, {local_end_index}) "
                        f"capacity={cache_capacity}"
                    )

                # Use the supplied update flag; inspecting CUDA timesteps synchronizes each layer.
                write_clean_kv = (
                    bool(kv_cache_update)
                    if kv_cache_update is not None
                    else _is_clean_kv_update_timestep(kv_sparse_timestep)
                )
                use_sparse = bool(
                    isinstance(kv_sparse_config, KVSparseConfig)
                    and kv_sparse_config.enabled
                    and debug_block_index is not None
                    and kv_sparse_config.layer_has_sparse(
                        debug_block_index, self.num_heads
                    )
                )
                if not use_sparse:
                    raise RuntimeError(
                        "CPU KV offload migration currently requires an enabled frame-sparse "
                        "configuration for every transformer layer"
                    )

                new_frame_k_mean = _mean_pool_along_frames(
                    selection_key_tensor, frame_seqlen
                ).type_as(v)
                needs_frame_minmax = bool(
                    kv_sparse_config.frame_selection_score == "q_sign_minmax_qmax"
                    and "frame_k_max" in kv_cache
                    and "frame_k_min" in kv_cache
                )
                new_frame_k_max = None
                new_frame_k_min = None
                if needs_frame_minmax:
                    new_frame_k_max, new_frame_k_min = _minmax_pool_along_frames(
                        selection_key_tensor, frame_seqlen
                    )
                    new_frame_k_max = new_frame_k_max.type_as(v)
                    new_frame_k_min = new_frame_k_min.type_as(v)

                local_end_frame = int(math.ceil(local_end_index / float(frame_seqlen)))
                local_start_frame = local_end_frame - int(new_frame_k_mean.shape[1])

                if write_clean_kv:
                    # Read current KV live; flush deferred D2H before later historical reads.
                    queued_clean_d2h = queue_deferred_clean_kv_to_cpu_cache(
                        kv_cache,
                        next_kv_cache,
                        local_start_index=local_start_index,
                        local_end_index=local_end_index,
                        key=cache_key,
                        value=v,
                    )
                    if not queued_clean_d2h:
                        copy_kv_to_cpu_cache_async(
                            kv_cache,
                            local_start_index=local_start_index,
                            local_end_index=local_end_index,
                            key=cache_key,
                            value=v,
                        )
                    kv_cache["frame_k_mean"][:, local_start_frame:local_end_frame] = (
                        new_frame_k_mean
                    )
                    if new_frame_k_max is not None:
                        kv_cache["frame_k_max"][:, local_start_frame:local_end_frame] = (
                            new_frame_k_max
                        )
                        kv_cache["frame_k_min"][:, local_start_frame:local_end_frame] = (
                            new_frame_k_min
                        )

                x = sparse_kv_self_attention(
                    roped_query,
                    None,
                    None,
                    frame_k_index_window=None,
                    frame_k_min_index_window=None,
                    frame_k_max_index_window=None,
                    frame_seqlen=frame_seqlen,
                    kv_start=0,
                    sink_tokens=sink_tokens,
                    current_start=int(current_start),
                    current_end=int(current_end),
                    local_end_index=int(local_end_index),
                    num_frame_per_block=int(num_frame_per_block),
                    layer_idx=int(debug_block_index),
                    current_timestep=kv_sparse_timestep,
                    kv_sparse_config=kv_sparse_config,
                    out_dtype=roped_query.dtype,
                    rope_freqs=freqs,
                    rope_grid_sizes=grid_sizes,
                    selection_query=selection_query,
                    kv_cache=kv_cache,
                    next_kv_cache=next_kv_cache,
                    next_selection_query=next_selection_query_override,
                    live_k=cache_key,
                    live_v=v,
                    live_frame_k_mean=None if write_clean_kv else new_frame_k_mean,
                    live_frame_k_max=None if write_clean_kv else new_frame_k_max,
                    live_frame_k_min=None if write_clean_kv else new_frame_k_min,
                    live_start_index=local_start_index,
                    fast_lf_rope=fast_lf_rope,
                )
                if x is None:
                    raise RuntimeError(
                        "CPU KV sparse overlap path returned no attention output; "
                        "check the FlashAttention block-sparse backend build and overlap flags"
                    )

                x = self.o(x.flatten(2))

                if write_clean_kv:
                    # Slots store raw K for compacted RoPE and rotated K for absolute RoPE.
                    update_gpu_kv_cache_frames(
                        kv_cache,
                        start_frame=local_start_frame,
                        end_frame=local_end_frame,
                        source_start_index=local_start_index,
                        current_start=int(current_start),
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=int(num_frame_per_block),
                        frame_k=cache_key.detach(),
                        frame_v=v.detach(),
                    )
                kv_cache["global_end_index"].fill_(int(current_end))
                kv_cache["local_end_index"].fill_(local_end_index)

                return x

            if kv_cache.get("cache_mode") == "lf_rope_local":
                if torch.is_grad_enabled():
                    raise RuntimeError("Local fast LF-RoPE supports no-grad inference only")
                if cache_start != current_start:
                    raise ValueError("Local fast LF-RoPE requires cache_start=current_start")
                x = local_lf_rope_attention(
                    q, k, v,
                    kv_cache=kv_cache,
                    cfg=kv_sparse_config,
                    current_start=int(current_start),
                    sink_frames=self.sink_size,
                    num_frame_per_block=int(num_frame_per_block),
                    layer_idx=int(debug_block_index),
                    rope_grid_sizes=grid_sizes,
                    rope_freqs=freqs,
                    write_clean_kv=(
                        bool(kv_cache_update) if kv_cache_update is not None
                        else _is_clean_kv_update_timestep(kv_sparse_timestep)
                    ),
                )
                # The local adapter commits clean KV in place. There is no
                # deferred whole-window update for the model to apply.
                return self.o(x.flatten(2)), None

            frame_seqlen = self._cached_frame_seqlen
            if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
                frame_seqlen = int(math.prod(grid_sizes[0][1:].detach().cpu().tolist()))
                self._cached_frame_seqlen = frame_seqlen
            current_start_frame = current_start // frame_seqlen
            if use_compacted_window_rope:
                # Defer both rotary operations until after the visible
                # sink/history window is assembled, as in native LightForcing.
                roped_query = None
                roped_key = None
                cache_key = k
                current_end = current_start + q.shape[1]
                num_new_tokens = q.shape[1]
            else:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                cache_key = roped_key
                current_end = current_start + roped_query.shape[1]
                num_new_tokens = roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            if roped_query is not None:
                num_new_tokens = roped_query.shape[1]

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                
                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = cache_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": cache_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }

            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = cache_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": cache_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }


            if use_compacted_window_rope:
                # Pack rolling history immediately after the sink.
                kv_start = max(
                    sink_tokens,
                    local_end_index - self.max_attention_size + sink_tokens,
                )
                grid_sizes_kv = copy.deepcopy(grid_sizes)
                grid_sizes_kv[:, 0] = (
                    sink_tokens + (local_end_index - kv_start)
                ) // frame_seqlen
                q_start_frame = int(
                    (sink_tokens + (local_end_index - kv_start) - q.shape[1])
                    // frame_seqlen
                )
                if fast_lf_rope and not use_cpu_offload and use_sparse_layer:
                    # Keep Q raw for frame selection; apply RoPE before block attention.
                    roped_query = q.type_as(v)
                else:
                    roped_query = lightforcing_rope_apply(
                        q, grid_sizes, freqs, start_frame=q_start_frame
                    ).type_as(v)
                k_window_unroped = torch.cat(
                    [temp_k[:, :sink_tokens], temp_k[:, kv_start:local_end_index]],
                    dim=1,
                )
                v_window = torch.cat(
                    [temp_v[:, :sink_tokens], temp_v[:, kv_start:local_end_index]],
                    dim=1,
                )
                k_window = lightforcing_rope_apply(
                    k_window_unroped, grid_sizes_kv, freqs, start_frame=0
                ).type_as(v)
            else:
                kv_start = max(0, local_end_index - self.max_attention_size)
                k_window = temp_k[:, kv_start:local_end_index]
                v_window = temp_v[:, kv_start:local_end_index]
            debug_step = _normalize_debug_timestep(debug_timestep)

            AttentionMapRecorder.get().record(
                attn_type="self",
                q=roped_query,
                k=k_window,
                block_index=debug_block_index,
                timestep=debug_step,
                current_start=current_start,
            )

            use_sparse = (
                kv_sparse_config is not None
                and isinstance(kv_sparse_config, KVSparseConfig)
                and kv_sparse_config.enabled
                and debug_block_index is not None
                and kv_sparse_config.layer_has_sparse(debug_block_index, self.num_heads)
            )
            # Compacted-window sparse selection is compatible with long_video_gen.
            config_compacted_window_rope = (
                isinstance(kv_sparse_config, KVSparseConfig)
                and getattr(kv_sparse_config, "rope_semantics", "absolute")
                in {
                    "horizonkv_compacted",
                }
            )
            if (
                use_sparse
                and self.long_video_gen
                and kv_sparse_config.selector != "lightforcing"
                and not config_compacted_window_rope
            ):
                raise RuntimeError(
                    "long_video_gen requires selector=lightforcing or an explicit "
                    "compacted-window LF-RoPE configuration; use regular GPU cache "
                    "semantics for absolute-RoPE selectors"
                )
            if use_sparse and self.long_video_gen and kv_sparse_config.selector == "lightforcing":
                schedule = calculate_chunk_sparsities(
                    int(kv_sparse_config.lightforcing_num_output_frames),
                    int(num_frame_per_block),
                    int(kv_sparse_config.lightforcing_local_attn_size),
                    float(kv_sparse_config.lightforcing_sparsity),
                    kv_sparse_config.lightforcing_sparsity_base,
                )
                query_chunk_frames = max(1, int(roped_query.shape[1] // frame_seqlen))
                chunk_id = int(current_start_frame // query_chunk_frames)
                sparsity = (
                    schedule[min(chunk_id, len(schedule) - 1)]
                    if schedule else float(kv_sparse_config.lightforcing_sparsity)
                )
                x = lightforcing_attention(
                    roped_query,
                    k_window,
                    v_window,
                    sparsity=float(sparsity),
                    chunk_id=chunk_id,
                    block_q=int(kv_sparse_config.attn_block_q),
                    block_k=int(kv_sparse_config.attn_block_k),
                    frame_seq=int(kv_sparse_config.lightforcing_frame_seq),
                    keep_frames=int(kv_sparse_config.lightforcing_keep_frames),
                    keep_sink=int(kv_sparse_config.lightforcing_keep_sink),
                    keep_near=int(kv_sparse_config.lightforcing_keep_near),
                    require_sparse=True,
                ).to(roped_query.dtype)
            elif use_sparse:
                selection_query_local = (
                    q.type_as(v) if fast_lf_rope and not use_cpu_offload else None
                )
                sparse_k_window = (
                    k_window_unroped
                    if (
                        fast_lf_rope
                        and not use_cpu_offload
                        and use_sparse_layer
                        and use_compacted_window_rope
                    )
                    else k_window
                )
                x = sparse_kv_self_attention(
                    roped_query,
                    sparse_k_window,
                    v_window,
                    frame_seqlen=frame_seqlen,
                    kv_start=kv_start,
                    sink_tokens=sink_tokens,
                    current_start=int(current_start),
                    current_end=int(current_end),
                    local_end_index=int(local_end_index),
                    num_frame_per_block=int(num_frame_per_block),
                    layer_idx=int(debug_block_index),
                    current_timestep=kv_sparse_timestep,
                    kv_sparse_config=kv_sparse_config,
                    out_dtype=roped_query.dtype,
                    rope_freqs=freqs,
                    rope_grid_sizes=grid_sizes,
                    selection_query=selection_query_local,
                    fast_lf_rope=fast_lf_rope and not use_cpu_offload,
                )
            else:
                x = attention(
                    roped_query,
                    k_window,
                    v_window
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 long_video_gen=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            long_video_gen=long_video_gen,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        debug_block_index=None,
        debug_timestep=None,
        kv_sparse_config=None,
        num_frame_per_block: int = 1,
        kv_sparse_timestep=None,
        next_kv_cache=None,
        kv_cache_update: bool | None = None,
        selection_query_override: torch.Tensor | None = None,
        next_selection_query_override: torch.Tensor | None = None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start,
            debug_block_index=debug_block_index,
            debug_timestep=debug_timestep,
            kv_sparse_config=kv_sparse_config,
            num_frame_per_block=num_frame_per_block,
            kv_sparse_timestep=kv_sparse_timestep,
            next_kv_cache=next_kv_cache,
            kv_cache_update=kv_cache_update,
            selection_query_override=selection_query_override,
            next_selection_query_override=next_selection_query_override,
            sink_recache_after_switch=sink_recache_after_switch,
        )

        use_cpu_offload = bool(
            isinstance(kv_cache, dict) and kv_cache.get("cpu_offload_kv", False)
        )
        if kv_cache is not None and not use_cpu_offload:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                crossattn_cache=crossattn_cache,
                debug_block_index=debug_block_index,
                debug_timestep=debug_timestep,
                debug_current_start=current_start,
            )
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 long_video_gen=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.long_video_gen = bool(long_video_gen)
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    long_video_gen=long_video_gen)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self.kv_sparse_config = None

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value


    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
        kv_cache_update: bool | None = None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        debug_timestep = _normalize_debug_timestep(t)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            kv_sparse_config=getattr(self, "kv_sparse_config", None),
            num_frame_per_block=int(getattr(self, "num_frame_per_block", 1)),
            kv_sparse_timestep=t,
            debug_timestep=debug_timestep,
            sink_recache_after_switch=sink_recache_after_switch,
            kv_cache_update=kv_cache_update,
        )
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "debug_block_index": block_index,
                        "debug_timestep": debug_timestep,
                        "selection_query_override": None,
                        "next_selection_query_override": None,
                        "next_kv_cache": (
                            kv_cache[block_index + 1]
                            if kv_cache is not None and block_index + 1 < len(kv_cache)
                            else None
                        ),
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "debug_block_index": block_index,
                        "debug_timestep": debug_timestep,
                        "selection_query_override": None,
                        "next_selection_query_override": None,
                        "next_kv_cache": (
                            kv_cache[block_index + 1]
                            if kv_cache is not None and block_index + 1 < len(kv_cache)
                            else None
                        ),
                    }
                )
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                else:
                    x = result
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError("HorizonKV provides inference only")

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
