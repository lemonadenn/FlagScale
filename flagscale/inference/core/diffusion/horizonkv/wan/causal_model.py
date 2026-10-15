# Modified for the HorizonKV inference artifact; see NOTICE.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from .baseline_causal_model import CausalWanModel as DenseCausalWanModel
from flagscale.inference.core.diffusion.horizonkv.wan.model import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
)
from torch.nn.attention.flex_attention import flex_attention
from diffusers.configuration_utils import register_to_config
import os
import torch.nn as nn
import torch
import math
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import KVSparseConfig
from flagscale.inference.core.diffusion.horizonkv.algorithm.attention import (
    sparse_kv_self_attention,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    copy_kv_to_cpu_cache_async,
    queue_deferred_clean_kv_to_cpu_cache,
    update_gpu_kv_cache_frames,
)

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def _normalize_debug_timestep(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.device.type == "cuda" and os.getenv("HORIZONKV_DEBUG_SYNC_TIMESTEP", "0") not in {
            "1",
            "true",
            "yes",
            "on",
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
        return bool((value.detach().abs() <= 1e-06).all().item())
    if isinstance(value, (list, tuple)):
        return bool(value) and all((_is_clean_kv_update_timestep(v) for v in value))
    if isinstance(value, (int, float)):
        return abs(float(value)) <= 1e-06
    try:
        return abs(float(value)) <= 1e-06
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


class CausalWanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-06,
        long_video_gen=False,
    ):
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
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        self._cached_frame_seqlen: int | None = None
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
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return (q, k, v)

        q, k, v = qkv_fn(x)
        use_compacted_window_rope = bool(self.long_video_gen) or (
            isinstance(kv_sparse_config, KVSparseConfig) and True
        )
        use_sparse_layer = bool(
            isinstance(kv_sparse_config, KVSparseConfig)
            and debug_block_index is not None
            and kv_sparse_config.layer_has_sparse(debug_block_index, self.num_heads)
        )
        use_cpu_offload = True
        fast_lf_rope = bool(isinstance(kv_sparse_config, KVSparseConfig) and True)
        if torch.is_grad_enabled():
            raise RuntimeError("HorizonKV CPU KV offload supports no-grad inference only")
        frame_seqlen = kv_cache.get("frame_seqlen", None)
        if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
            frame_seqlen = self._cached_frame_seqlen
        if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
            frame_seqlen = int(math.prod(grid_sizes[0][1:].detach().cpu().tolist()))
            self._cached_frame_seqlen = frame_seqlen
        current_start_frame = current_start // frame_seqlen
        roped_query = q
        roped_key = None
        cache_key = k
        score_with_unroped_qk = bool(isinstance(kv_sparse_config, KVSparseConfig) and False)
        selection_query = (
            selection_query_override if selection_query_override is not None else q.type_as(v)
        )
        selection_key_tensor = k.type_as(v)
        current_end = current_start + roped_query.shape[1]
        cache_end = cache_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        local_start_index = int(cache_start)
        local_end_index = int(cache_end)
        cache_capacity = int(kv_cache["token_capacity"])
        if local_start_index < 0 or local_end_index > cache_capacity:
            raise ValueError(
                f"CPU offloaded KV cache capacity is too small: span=[{local_start_index}, {local_end_index}) capacity={cache_capacity}"
            )
        write_clean_kv = (
            bool(kv_cache_update)
            if kv_cache_update is not None
            else _is_clean_kv_update_timestep(kv_sparse_timestep)
        )
        use_sparse = bool(
            isinstance(kv_sparse_config, KVSparseConfig)
            and debug_block_index is not None
            and kv_sparse_config.layer_has_sparse(debug_block_index, self.num_heads)
        )
        new_frame_k_mean = _mean_pool_along_frames(selection_key_tensor, frame_seqlen).type_as(v)
        needs_frame_minmax = False
        new_frame_k_max = None
        new_frame_k_min = None
        local_end_frame = int(math.ceil(local_end_index / float(frame_seqlen)))
        local_start_frame = local_end_frame - int(new_frame_k_mean.shape[1])
        if write_clean_kv:
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
            kv_cache["frame_k_mean"][:, local_start_frame:local_end_frame] = new_frame_k_mean
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
            live_frame_k_max=None if write_clean_kv else None,
            live_frame_k_min=None if write_clean_kv else None,
            live_start_index=local_start_index,
            fast_lf_rope=True,
        )
        if x is None:
            raise RuntimeError(
                "CPU KV sparse overlap path returned no attention output; check the FlashAttention block-sparse backend build and overlap flags"
            )
        x = self.o(x.flatten(2))
        if write_clean_kv:
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


class CausalWanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-06,
        long_video_gen=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, long_video_gen=long_video_gen
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
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
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        self_attn_result = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            ).flatten(1, 2),
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
        use_cpu_offload = bool(isinstance(kv_cache, dict) and True)
        if kv_cache is not None and (not use_cpu_offload):
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

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
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        if cache_update_info is not None:
            return (x, cache_update_info)
        else:
            return x


class CausalWanModel(DenseCausalWanModel):
    attention_block_cls = CausalWanAttentionBlock
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
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
        eps=1e-06,
        long_video_gen=False,
    ):
        self.long_video_gen = bool(long_video_gen)
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.kv_sparse_config = None

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
        x, e, e0, seq_lens, grid_sizes, context = self._prepare_inputs(
            x, t, context, seq_len, clip_fea, y
        )
        debug_timestep = _normalize_debug_timestep(t)
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=None,
            block_mask=self.block_mask,
            kv_sparse_config=getattr(self, "kv_sparse_config", None),
            num_frame_per_block=int(getattr(self, "num_frame_per_block", 1)),
            kv_sparse_timestep=t,
            debug_timestep=debug_timestep,
            sink_recache_after_switch=sink_recache_after_switch,
            kv_cache_update=kv_cache_update,
        )
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
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
            if kv_cache is not None and isinstance(result, tuple):
                x, block_cache_update_info = result
                cache_update_infos.append((block_index, block_cache_update_info))
            else:
                x = result
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)

    def _attention_block_kwargs(self):
        return {"long_video_gen": self.long_video_gen}
