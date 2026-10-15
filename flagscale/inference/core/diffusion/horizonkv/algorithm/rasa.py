# Modified for the HorizonKV inference artifact; see NOTICE.
from copy import copy
from functools import lru_cache
import json
import os
import time
import torch
from flagscale.inference.core.diffusion.horizonkv.algorithm.rasa_policy import load_policy

_STATS = {}


@lru_cache(maxsize=8)
def _settings(path):
    policy = load_policy(path)
    return dict(
        mode="on",
        bounded=dict(
            primary=dict(
                topk=policy["best_config"]["frames"],
                block_sparsity=policy["best_config"]["block_sparsity"],
            ),
            fallback=dict(
                topk=policy["min_config"]["frames"],
                block_sparsity=policy["min_config"]["block_sparsity"],
            ),
            miss_threshold=policy["switch_miss_items"] - 1,
        ),
    )


def select_adaptive_frames_if_supported(**args):
    settings = _settings(os.environ["HORIZONKV_ADAPTIVE_CONFIG"])
    return _select_bounded_frames(settings=settings, **args)


def reset_stats():
    _STATS.clear()


def snapshot_stats():
    return json.loads(json.dumps(_STATS))


def _select_bounded_frames(
    *,
    settings,
    kv_cache,
    cfg,
    q,
    frame_means,
    compulsory_frames,
    frame_seqlen,
    n_frames,
    live_start_index,
    live_k,
):
    from flagscale.inference.core.diffusion.horizonkv.algorithm import oae as c
    from flagscale.inference.core.diffusion.horizonkv.algorithm.transfer import _load_module

    spec = settings["bounded"]
    primary = spec.get("primary", {})
    fallback = spec.get("fallback", {})
    primary_topk = int(primary.get("topk", cfg.max_kv_frames))
    fallback_topk = int(fallback.get("topk", primary_topk))
    primary_sparsity = float(
        primary.get("block_sparsity", getattr(cfg, "overlap_block_sparsity", 0.5))
    )
    fallback_sparsity = float(fallback.get("block_sparsity", primary_sparsity))
    threshold = int(spec.get("miss_threshold", 0))
    if primary_topk <= 0 or fallback_topk <= 0 or threshold < 0:
        raise ValueError("Invalid bounded adaptive parameters")
    if not (0.0 <= primary_sparsity < 1.0 and 0.0 <= fallback_sparsity < 1.0):
        raise ValueError("bounded block_sparsity must be in [0, 1)")
    if (
        primary_topk != cfg.max_kv_frames
        or abs(primary_sparsity - cfg.overlap_block_sparsity) > 1e-08
    ):
        raise ValueError("bounded primary must match the generated KV config topk/sparsity")
    if (
        live_start_index is None
        or live_start_index <= int(kv_cache["gpu_cache_frames"]) * frame_seqlen
    ):
        return None
    if not (
        frame_means is not None
        and isinstance(live_k, torch.Tensor)
        and (q.dtype == torch.bfloat16)
        and (q.device.type == "cuda")
        and (not c._truthy_env("HORIZONKV_TRITON_GLOBAL_BLOCK_SELECTION", "0"))
        and (not c._truthy_env("HORIZONKV_TRITON_SELECTOR_FP32", "0"))
        and c._overlap_flash_materialize_enabled_by_env()
    ):
        raise RuntimeError(
            "RASA requires the fast pooled-QK materializer with fixed endpoint block density"
        )
    if len(compulsory_frames) > fallback_topk:
        raise ValueError("bounded fallback topk cannot cover all compulsory frames")
    module = _load_module()
    if module is None or not hasattr(module, "kv_sparse_adaptive_count_misses"):
        raise RuntimeError("bounded adaptive requires the rebuilt optional onload extension")
    selected = c._select_kv_frame_indices_max_kv_frames_batched_fused(
        kv_cache,
        cfg,
        q,
        frame_means,
        n_frames=n_frames,
        frame_seqlen=frame_seqlen,
        max_kv_frames=primary_topk,
        compulsory_frames=compulsory_frames,
    )
    if selected is None:
        raise RuntimeError("bounded primary frame selection failed")
    state = c._kv_cache_slot_state_tensors(kv_cache)
    if state is None:
        raise RuntimeError("bounded slot metadata unavailable")
    started = time.perf_counter()
    staged = c._stage_overlap_frame_indices_to_cpu_async(kv_cache, selected)
    frames_cpu = c._resolve_staged_overlap_frame_indices_cpu(staged)
    if frames_cpu is None:
        raise RuntimeError("bounded primary frame staging failed")
    live_first = int(live_start_index) // frame_seqlen
    live_end = live_first + int(live_k.shape[1]) // frame_seqlen
    misses = int(
        module.kv_sparse_adaptive_count_misses(
            frames_cpu, state[0], state[1], state[2], live_first, live_end, frame_seqlen
        )
    )
    if misses < 0:
        raise RuntimeError("bounded encountered invalid selected frame IDs")
    gate_us = (time.perf_counter() - started) * 1000000.0
    switched = misses > threshold
    applied = switched and settings["mode"] == "on"
    effective_cfg = cfg
    if applied:
        selected_fallback = c._select_kv_frame_indices_max_kv_frames_batched_fused(
            kv_cache,
            cfg,
            q,
            frame_means,
            n_frames=n_frames,
            frame_seqlen=frame_seqlen,
            max_kv_frames=fallback_topk,
            compulsory_frames=compulsory_frames,
        )
        if selected_fallback is None:
            raise RuntimeError("bounded fallback frame selection failed")
        selected = selected_fallback
        staged = c._stage_overlap_frame_indices_to_cpu_async(kv_cache, selected)
        if staged is None:
            raise RuntimeError("bounded fallback frame staging failed")
        effective_cfg = copy(cfg)
        effective_cfg.max_kv_frames = fallback_topk
        effective_cfg.overlap_block_sparsity = fallback_sparsity
        effective_cfg.block_target_density = 1 - fallback_sparsity
    stats = _STATS.setdefault(
        "bounded_" + settings.get("mode", "on"),
        dict(
            calls=0,
            switched=0,
            applied=0,
            misses=0,
            max_misses=0,
            gate_us=0.0,
            primary_topk=primary_topk,
            fallback_topk=fallback_topk,
            threshold=threshold,
        ),
    )
    stats["calls"] += 1
    stats["misses"] += misses
    stats["switched"] += int(switched)
    stats["applied"] += int(applied)
    stats["max_misses"] = max(stats["max_misses"], misses)
    stats["gate_us"] += gate_us
    return {
        "kind": "bounded",
        "selected": selected,
        "staged": staged,
        "cfg": effective_cfg,
        "misses": misses,
        "switched": switched,
        "primary_sparsity": primary_sparsity,
        "fallback_sparsity": fallback_sparsity,
    }
