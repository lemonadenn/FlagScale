# Modified for the HorizonKV inference artifact; see NOTICE.
from __future__ import annotations

import atexit
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from flagscale.inference.core.diffusion.horizonkv.algorithm.config import (
    KVSparseConfig,
    _normalize_timestep_token,
    _sanitize_token,
    _scalarize_timestep,
)


def _axis_ticks(length: int) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]
    if length <= 16:
        return list(range(length))
    tick_count = 6
    return sorted(
        {
            int(round(i * (length - 1) / max(1, tick_count - 1)))
            for i in range(tick_count)
        }
    )


def _nice_frame_tick_step(frame_count: int, max_labels: int = 12) -> int:
    if frame_count <= max_labels:
        return 1
    candidates = [1, 2, 5]
    scale = 1
    while True:
        for base in candidates:
            step = base * scale
            if math.ceil(frame_count / step) <= max_labels:
                return step
        scale *= 10


def _should_visualize_mask(
    cfg: KVSparseConfig,
    layer_idx: int,
    head_idx: int,
    timestep: Any,
) -> bool:
    if not cfg.visualize_attn_mask or not cfg.visualize_output_dir:
        return False
    if cfg.visualize_layers is not None and layer_idx not in cfg.visualize_layers:
        return False
    if cfg.visualize_heads is not None and head_idx not in cfg.visualize_heads:
        return False
    if cfg.visualize_timesteps is not None:
        return _normalize_timestep_token(timestep) in cfg.visualize_timesteps
    return True


def _profile_sync_if_needed(cfg: KVSparseConfig, device: torch.device) -> None:
    if not (cfg.profile_frame_selection or cfg.profile_stage2) or not cfg.profile_sync_cuda:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def _profile_now(cfg: KVSparseConfig, device: torch.device) -> float:
    _profile_sync_if_needed(cfg, device)
    return time.perf_counter()


def _profile_frame_selection_summary(cfg: KVSparseConfig) -> None:
    calls = int(cfg._profile_index)
    if calls <= 0:
        return
    head_calls = int(cfg._profile_head_select_calls)
    score_ms_total = float(cfg._profile_score_ms_total)
    rule_ms_total = float(cfg._profile_rule_ms_total)
    total_ms = score_ms_total + rule_ms_total
    mean_score_ms = score_ms_total / calls
    mean_rule_ms = rule_ms_total / max(1, head_calls)
    mean_total_ms = total_ms / calls
    mean_selected = float(cfg._profile_selected_frames_total) / max(1, head_calls)
    print(
        "[kv_sparse_profile_summary] "
        f"calls={calls} head_select_calls={head_calls} "
        f"score_ms_total={score_ms_total:.3f} rule_ms_total={rule_ms_total:.3f} "
        f"total_ms={total_ms:.3f} mean_score_ms={mean_score_ms:.3f} "
        f"mean_rule_ms_per_head={mean_rule_ms:.4f} "
        f"mean_total_ms_per_call={mean_total_ms:.3f} "
        f"mean_selected_frames_per_head={mean_selected:.2f}"
    )
    stage2_head_calls = int(cfg._profile_stage2_head_calls)
    if stage2_head_calls > 0:
        compact_ms = float(cfg._profile_stage2_compact_ms_total)
        block_mass_ms = float(cfg._profile_stage2_block_mass_ms_total)
        block_select_ms = float(cfg._profile_stage2_block_select_ms_total)
        mask_build_ms = float(cfg._profile_stage2_mask_build_ms_total)
        kernel_ms = float(cfg._profile_stage2_kernel_ms_total)
        total_stage2_ms = compact_ms + block_mass_ms + block_select_ms + mask_build_ms + kernel_ms
        print(
            "[kv_sparse_stage2_profile_summary] "
            f"head_calls={stage2_head_calls} total_ms={total_stage2_ms:.3f} "
            f"compact_ms_total={compact_ms:.3f} block_mass_ms_total={block_mass_ms:.3f} "
            f"block_select_ms_total={block_select_ms:.3f} mask_build_ms_total={mask_build_ms:.3f} "
            f"kernel_ms_total={kernel_ms:.3f} "
            f"mean_compact_ms_per_head={compact_ms / stage2_head_calls:.3f} "
            f"mean_block_mass_ms_per_head={block_mass_ms / stage2_head_calls:.3f} "
            f"mean_block_select_ms_per_head={block_select_ms / stage2_head_calls:.3f} "
            f"mean_mask_build_ms_per_head={mask_build_ms / stage2_head_calls:.3f} "
            f"mean_kernel_ms_per_head={kernel_ms / stage2_head_calls:.3f}"
        )


def _profile_frame_selection_record(
    cfg: KVSparseConfig,
    *,
    layer_idx: int,
    timestep: Any,
    Lq: int,
    Lk: int,
    n_frames: int,
    score_ms: float,
    rule_ms: float,
    selected_counts: list[int],
) -> None:
    if not cfg.profile_frame_selection:
        return
    cfg._profile_index += 1
    cfg._profile_score_ms_total += float(score_ms)
    cfg._profile_rule_ms_total += float(rule_ms)
    cfg._profile_head_select_calls += len(selected_counts)
    cfg._profile_selected_frames_total += sum(int(v) for v in selected_counts)
    if not cfg._profile_registered_atexit:
        atexit.register(_profile_frame_selection_summary, cfg)
        cfg._profile_registered_atexit = True

    log_every = int(cfg.profile_log_every)
    should_log = bool(cfg.profile_verbose)
    if log_every > 0 and cfg._profile_index % log_every == 0:
        should_log = True
    if not should_log:
        return

    head_calls = len(selected_counts)
    mean_selected = sum(selected_counts) / max(1, head_calls)
    print(
        "[kv_sparse_profile] "
        f"call={cfg._profile_index} layer={int(layer_idx)} "
        f"timestep={_normalize_timestep_token(timestep)} Lq={int(Lq)} Lk={int(Lk)} "
        f"frames={int(n_frames)} heads={head_calls} "
        f"score_ms={float(score_ms):.3f} rule_ms={float(rule_ms):.3f} "
        f"total_ms={float(score_ms) + float(rule_ms):.3f} "
        f"rule_ms_per_head={float(rule_ms) / max(1, head_calls):.4f} "
        f"mean_selected_frames={mean_selected:.2f}"
    )


def _profile_stage2_record(
    cfg: KVSparseConfig,
    *,
    layer_idx: int,
    head_idx: int,
    timestep: Any,
    compact_ms: float,
    block_mass_ms: float,
    block_select_ms: float,
    mask_build_ms: float,
    kernel_ms: float,
    selected_frames: int,
    selected_blocks: int,
    backend: str,
) -> None:
    if not cfg.profile_stage2:
        return
    cfg._profile_stage2_head_calls += 1
    cfg._profile_stage2_compact_ms_total += float(compact_ms)
    cfg._profile_stage2_block_mass_ms_total += float(block_mass_ms)
    cfg._profile_stage2_block_select_ms_total += float(block_select_ms)
    cfg._profile_stage2_mask_build_ms_total += float(mask_build_ms)
    cfg._profile_stage2_kernel_ms_total += float(kernel_ms)
    if not cfg._profile_registered_atexit:
        atexit.register(_profile_frame_selection_summary, cfg)
        cfg._profile_registered_atexit = True

    log_every = int(cfg.profile_log_every)
    should_log = bool(cfg.profile_verbose)
    if log_every > 0 and cfg._profile_stage2_head_calls % log_every == 0:
        should_log = True
    if not should_log:
        return

    total_ms = (
        float(compact_ms)
        + float(block_mass_ms)
        + float(block_select_ms)
        + float(mask_build_ms)
        + float(kernel_ms)
    )
    print(
        "[kv_sparse_stage2_profile] "
        f"call={cfg._profile_stage2_head_calls} layer={int(layer_idx)} head={int(head_idx)} "
        f"timestep={_normalize_timestep_token(timestep)} backend={backend} "
        f"selected_frames={int(selected_frames)} selected_blocks={int(selected_blocks)} "
        f"compact_ms={float(compact_ms):.3f} block_mass_ms={float(block_mass_ms):.3f} "
        f"block_select_ms={float(block_select_ms):.3f} mask_build_ms={float(mask_build_ms):.3f} "
        f"kernel_ms={float(kernel_ms):.3f} total_ms={total_ms:.3f}"
    )


def _save_binary_mask_image(
    mask_2d: torch.Tensor,
    output_path: Path,
    *,
    x_axis_label: str,
    y_axis_label: str,
    cell_w: int | None = None,
    cell_h: int | None = None,
    x_major_count: int | None = None,
    y_major_count: int | None = None,
    x_units_per_major: int | None = None,
    y_units_per_major: int | None = None,
    pixels_per_unit: int | None = None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("visualize_attn_mask requires Pillow to be installed") from exc

    mask_u8 = mask_2d.detach().to(device="cpu", dtype=torch.uint8).mul(255).contiguous()
    h, w = mask_u8.shape
    img = Image.fromarray(mask_u8.numpy(), mode="L").convert("RGB")

    grouped_scale_mode = (
        x_major_count is not None
        and y_major_count is not None
        and x_units_per_major is not None
        and y_units_per_major is not None
        and x_major_count > 0
        and y_major_count > 0
        and x_units_per_major > 0
        and y_units_per_major > 0
    )
    if grouped_scale_mode:
        if pixels_per_unit is None:
            pixels_per_unit = 18
        target_w = max(1, int(round(w * pixels_per_unit)))
        target_h = max(1, int(round(h * pixels_per_unit)))
        img = img.resize((target_w, target_h), resample=Image.NEAREST)
    else:
        if cell_w is None:
            cell_w = max(2, min(12, 1200 // max(1, w)))
        if cell_h is None:
            cell_h = max(2, min(24, 512 // max(1, h)))
        img = img.resize((max(1, w * cell_w), max(1, h * cell_h)), resample=Image.NEAREST)
    left_margin = 60
    right_margin = 20
    top_margin = 24
    bottom_margin = 48
    canvas = Image.new(
        "RGB",
        (img.width + left_margin + right_margin, img.height + top_margin + bottom_margin),
        color="white",
    )
    canvas.paste(img, (left_margin, top_margin))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x0 = left_margin
    y0 = top_margin
    x1 = left_margin + img.width
    y1 = top_margin + img.height
    draw.rectangle((x0, y0, x1, y1), outline="black", width=1)

    if grouped_scale_mode:
        assert pixels_per_unit is not None
        assert x_major_count is not None and y_major_count is not None
        assert x_units_per_major is not None and y_units_per_major is not None

        major_grid_color = (120, 120, 120)
        for major_idx in range(x_major_count + 1):
            unit_boundary = min(w, major_idx * x_units_per_major)
            x = x0 + int(round(unit_boundary * pixels_per_unit))
            draw.line((x, y0, x, y1), fill=major_grid_color, width=2)
        for major_idx in range(y_major_count + 1):
            unit_boundary = min(h, major_idx * y_units_per_major)
            y = y0 + int(round(unit_boundary * pixels_per_unit))
            draw.line((x0, y, x1, y), fill=major_grid_color, width=2)

        x_step = _nice_frame_tick_step(x_major_count)
        for tick in range(0, x_major_count, x_step):
            start_unit = min(w, tick * x_units_per_major)
            end_unit = min(w, start_unit + x_units_per_major)
            center_unit = 0.5 * (start_unit + end_unit)
            x = x0 + int(round(center_unit * pixels_per_unit))
            draw.line((x, y1, x, y1 + 5), fill="black", width=1)
            label = str(tick)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw / 2, y1 + 8), label, fill="black", font=font)

        y_step = _nice_frame_tick_step(y_major_count, max_labels=8)
        for tick in range(0, y_major_count, y_step):
            start_unit = min(h, tick * y_units_per_major)
            end_unit = min(h, start_unit + y_units_per_major)
            center_unit = 0.5 * (start_unit + end_unit)
            y = y0 + int(round(center_unit * pixels_per_unit))
            draw.line((x0 - 5, y, x0, y), fill="black", width=1)
            label = str(tick)
            bbox = draw.textbbox((0, 0), label, font=font)
            th = bbox[3] - bbox[1]
            tw = bbox[2] - bbox[0]
            draw.text((x0 - 8 - tw, y - th / 2), label, fill="black", font=font)
    else:
        for tick in _axis_ticks(w):
            x = x0 + int(round((tick + 0.5) * img.width / max(1, w)))
            draw.line((x, y1, x, y1 + 5), fill="black", width=1)
            label = str(tick)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw / 2, y1 + 8), label, fill="black", font=font)

        for tick in _axis_ticks(h):
            y = y0 + int(round((tick + 0.5) * img.height / max(1, h)))
            draw.line((x0 - 5, y, x0, y), fill="black", width=1)
            label = str(tick)
            bbox = draw.textbbox((0, 0), label, font=font)
            th = bbox[3] - bbox[1]
            tw = bbox[2] - bbox[0]
            draw.text((x0 - 8 - tw, y - th / 2), label, fill="black", font=font)

    draw.text((x0, 4), f"y: {y_axis_label}", fill="black", font=font)
    draw.text((x0, y1 + 28), f"x: {x_axis_label}", fill="black", font=font)
    canvas.save(output_path)


def _dump_selection_visualization(
    cfg: KVSparseConfig,
    *,
    granularity: str,
    layer_idx: int,
    head_idx: int,
    timestep: Any,
    current_start: int,
    current_end: int,
    kv_start: int,
    mask_2d: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    if not _should_visualize_mask(cfg, layer_idx, head_idx, timestep):
        return

    output_dir = Path(cfg.visualize_output_dir) / f"head{int(head_idx):02d}_layer{int(layer_idx):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestep_token = _normalize_timestep_token(timestep).replace("/", "_")
    image_index = cfg._visualization_index
    cfg._visualization_index += 1
    run_token = _sanitize_token(cfg.visualize_run_name, "run")
    config_token = _sanitize_token(cfg.source_stem, "cfg")
    stem = (
        f"{run_token}_{config_token}_{image_index:06d}"
        f"_t{timestep_token}_ctx{int(current_start):06d}_{int(current_end):06d}_{granularity}"
    )
    image_path = output_dir / f"{stem}.png"
    meta_path = output_dir / f"{stem}.json"

    mask_bool = mask_2d.detach().to(device="cpu", dtype=torch.bool)
    _save_binary_mask_image(
        mask_bool,
        image_path,
        x_axis_label=str(metadata.get("x_axis", "x")),
        y_axis_label=str(metadata.get("y_axis", "y")),
        cell_w=metadata.get("plot_cell_w"),
        cell_h=metadata.get("plot_cell_h"),
        x_major_count=metadata.get("x_major_count"),
        y_major_count=metadata.get("y_major_count"),
        x_units_per_major=metadata.get("x_units_per_major"),
        y_units_per_major=metadata.get("y_units_per_major"),
        pixels_per_unit=metadata.get("pixels_per_unit"),
    )

    payload = dict(metadata)
    if not cfg.visualize_full_indices:
        omitted_keys = []
        for key in (
            "forced_linear_indices",
            "selected_linear_indices",
            "selected_additional_linear_indices",
            "selected_block_indices",
        ):
            if key in payload:
                payload.pop(key, None)
                omitted_keys.append(key)
        if omitted_keys:
            payload["full_index_lists_omitted"] = omitted_keys
    payload.update(
        {
            "granularity": granularity,
            "layer": int(layer_idx),
            "head": int(head_idx),
            "timestep": _scalarize_timestep(timestep),
            "timestep_token": timestep_token,
            "current_start": int(current_start),
            "current_end": int(current_end),
            "kv_start": int(kv_start),
            "config_path": cfg.source_path,
            "config_stem": cfg.source_stem,
            "run_name": cfg.visualize_run_name,
            "pid": int(os.getpid()),
            "rule": cfg.rule,
            "target_density": float(cfg.target_density),
            "target_recall": cfg.target_recall,
            "max_kv_frames": cfg.max_kv_frames,
            "low_recall_threshold": cfg.low_recall_threshold,
            "expand_target_recall": cfg.expand_target_recall,
            "forced_outside_budget": bool(cfg.forced_outside_budget),
            "image_path": str(image_path),
            "mask_shape": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
            "selected_count": int(metadata.get("selected_item_count", int(mask_bool.sum().item()))),
            "selected_visual_cell_count": int(mask_bool.sum().item()),
        }
    )
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "_dump_selection_visualization",
    "_profile_frame_selection_record",
    "_profile_stage2_record",
    "_profile_now",
    "_should_visualize_mask",
]
