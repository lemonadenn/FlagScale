# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Suite selection/sharding adapted from Self-Forcing's standard-suite utilities.

Metric implementations remain in the external VBench checkout. This module
has no dependency on the Self-Forcing checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def valid_video(path: Path, frames: int, height: int, width: int, fps: int = 16) -> bool:
    """Reject truncated or wrong-duration videos before skipping/reusing them."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if float(stream.average_rate) != fps:
                return False
            count = 0
            for frame in container.decode(stream):
                if (frame.height, frame.width) != (height, width):
                    return False
                count += 1
            return count == frames
    except Exception:
        return False


ALL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_dimensions(value: str) -> list[str]:
    raw = csv_values(value)
    if not raw or raw in (["all"], ["all16"], ["full"]):
        return list(ALL_DIMENSIONS)
    unknown = sorted(set(raw) - set(ALL_DIMENSIONS))
    if unknown:
        raise SystemExit("Unsupported VBench dimension(s): " + ", ".join(unknown))
    # Canonical order makes suite ids and result tables stable.
    selected = set(raw)
    return [dimension for dimension in ALL_DIMENSIONS if dimension in selected]


def unique_prompts(entries: Iterable[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        prompt = str(entry["prompt_en"])
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def _uniform_indices(total: int, limit: int) -> list[int]:
    """Pick one midpoint from each of ``limit`` equal-width strata."""
    if limit <= 0 or limit >= total:
        return list(range(total))
    return [((2 * index + 1) * total) // (2 * limit) for index in range(limit)]


def filter_suite(
    entries: list[dict[str, Any]],
    dimensions: list[str],
    limit_per_dimension: int,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    selected = set(dimensions)
    if limit_per_dimension > 0 and selection_strategy == "uniform":
        candidate_indices = {
            dimension: [
                index
                for index, entry in enumerate(entries)
                if dimension in entry.get("dimension", [])
            ]
            for dimension in dimensions
        }
        missing = [dimension for dimension, indices in candidate_indices.items() if not indices]
        if missing:
            raise SystemExit("No official prompts found for dimension(s): " + ", ".join(missing))

        selected_dimensions_by_index: dict[int, set[str]] = {}
        for dimension, indices in candidate_indices.items():
            for position in _uniform_indices(len(indices), limit_per_dimension):
                entry_index = indices[position]
                selected_dimensions_by_index.setdefault(entry_index, set()).add(dimension)

        output: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(entries):
            chosen_dimensions = selected_dimensions_by_index.get(index)
            if not chosen_dimensions:
                continue
            entry = dict(raw_entry)
            # Only retain dimensions for which this entry was actually sampled.
            # This keeps every limited dimension at exactly N metadata records.
            entry["dimension"] = [
                dimension
                for dimension in raw_entry.get("dimension", [])
                if dimension in chosen_dimensions
            ]
            output.append(entry)
        return output

    if limit_per_dimension > 0 and selection_strategy != "head":
        raise SystemExit(f"Unsupported prompt selection strategy: {selection_strategy}")

    counts = Counter()
    output: list[dict[str, Any]] = []
    for raw_entry in entries:
        entry_dimensions = [
            dimension for dimension in raw_entry.get("dimension", []) if dimension in selected
        ]
        if not entry_dimensions:
            continue
        if limit_per_dimension > 0 and all(
            counts[dimension] >= limit_per_dimension for dimension in entry_dimensions
        ):
            continue
        entry = dict(raw_entry)
        entry["dimension"] = entry_dimensions
        output.append(entry)
        for dimension in entry_dimensions:
            if limit_per_dimension <= 0 or counts[dimension] < limit_per_dimension:
                counts[dimension] += 1

    missing = [dimension for dimension in dimensions if counts[dimension] == 0]
    if missing:
        raise SystemExit("No official prompts found for dimension(s): " + ", ".join(missing))
    return output


def suite_name(
    dimensions: list[str],
    limit_per_dimension: int,
    selection_strategy: str,
) -> str:
    if dimensions == list(ALL_DIMENSIONS) and limit_per_dimension == 0:
        return "official_all16_full"
    effective_strategy = "full" if limit_per_dimension == 0 else selection_strategy
    signature = (
        ",".join(dimensions) + f"|limit={limit_per_dimension}|selection={effective_strategy}"
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
    limit_tag = "full" if limit_per_dimension == 0 else f"n{limit_per_dimension}pd"
    strategy_tag = "" if limit_per_dimension == 0 else f"_{selection_strategy}"
    return f"official_subset_{limit_tag}{strategy_tag}_{digest}"


def prepare_suite(args: argparse.Namespace) -> None:
    full_json = Path(args.full_json).resolve()
    output_json = Path(args.output_json).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    dimensions = parse_dimensions(args.dimensions)
    if args.limit_per_dimension < 0:
        raise SystemExit("--limit-per-dimension must be nonnegative")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    entries = json.loads(full_json.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"Expected a JSON list: {full_json}")
    filtered = filter_suite(
        entries,
        dimensions,
        args.limit_per_dimension,
        args.selection_strategy,
    )
    prompts = unique_prompts(filtered)

    unsafe = [prompt for prompt in prompts if "/" in prompt or "\x00" in prompt]
    too_long = [
        prompt
        for prompt in prompts
        if len(f"{prompt}-{args.num_samples - 1}.mp4".encode("utf-8")) > 255
    ]
    if unsafe:
        raise SystemExit(f"Official prompt is unsafe as a filename: {unsafe[0]!r}")
    if too_long:
        raise SystemExit(
            "Official prompt exceeds the filesystem filename limit: " f"{too_long[0]!r}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(filtered, ensure_ascii=False, indent=2) + "\n"
    prompt_content = "".join(f"{prompt}\n" for prompt in prompts)
    existing_paths = [path for path in (output_json, prompt_file) if path.exists()]
    if existing_paths:
        if len(existing_paths) != 2:
            raise SystemExit(
                "Incomplete persisted suite; refusing to replace it: "
                + ", ".join(str(path) for path in existing_paths)
            )
        if (
            output_json.read_text(encoding="utf-8") != json_content
            or prompt_file.read_text(encoding="utf-8") != prompt_content
        ):
            raise SystemExit(
                "Persisted suite differs from the requested selection; use a new "
                f"SUITE_NAME or remove the suite directory deliberately: {output_json.parent}"
            )
        persisted = "reused"
    else:
        output_json.write_text(json_content, encoding="utf-8")
        prompt_file.write_text(prompt_content, encoding="utf-8")
        persisted = "created"

    counts = Counter(dimension for entry in filtered for dimension in entry.get("dimension", []))
    print(
        f"[vbench-standard-suite] name={suite_name(dimensions, args.limit_per_dimension, args.selection_strategy)} "
        f"selection={args.selection_strategy} persisted={persisted} "
        f"entries={len(filtered)} unique_prompts={len(prompts)} "
        f"samples={args.num_samples} expected_videos={len(prompts) * args.num_samples}"
    )
    print(f"[vbench-standard-suite] metadata_file={output_json} prompt_file={prompt_file}")
    for dimension in dimensions:
        print(f"[vbench-standard-suite] dimension={dimension} prompts={counts[dimension]}")


def read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not prompts:
        raise SystemExit(f"Prompt file is empty: {path}")
    return prompts


def prepare_eval_shards(args: argparse.Namespace) -> None:
    """Create per-GPU prompt shards for one-dimension VBench evaluation.

    Each rank receives complete samples for a disjoint set of prompts. Video
    files and precomputed split clips are symlinked, while a rank-local copy
    of the suite metadata prevents the evaluator from looking for prompts
    assigned to other ranks.
    """
    prompts = read_prompts(Path(args.prompt_file).resolve())
    suite_path = Path(args.suite_json).resolve()
    source_dir = Path(args.source_view_dir).resolve()
    output_root = Path(args.output_root).resolve()
    requested_dimension = args.dimension.strip() if args.dimension else ""
    world_size = int(args.num_shards)
    num_samples = int(args.num_samples)
    if world_size <= 0:
        raise SystemExit("--num-shards must be positive")
    if world_size > len(prompts):
        raise SystemExit(
            f"Cannot assign {len(prompts)} prompts to {world_size} shards; "
            "each rank must receive at least one prompt"
        )
    if num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    source_split = source_dir / "split_clip"
    if not source_split.is_dir():
        raise SystemExit(f"Source split cache is missing: {source_split}")
    if output_root.exists():
        raise SystemExit(f"Refusing to replace existing evaluation shards: {output_root}")

    suite_entries = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(suite_entries, list):
        raise SystemExit(f"Expected a JSON list in suite metadata: {suite_path}")
    entries_by_prompt = {str(entry["prompt_en"]): entry for entry in suite_entries}
    if requested_dimension:
        dimension_prompts = {
            str(entry["prompt_en"])
            for entry in suite_entries
            if requested_dimension in entry.get("dimension", [])
        }
        prompts = [prompt for prompt in prompts if prompt in dimension_prompts]
        if not prompts:
            raise SystemExit(f"No suite prompts found for dimension: {requested_dimension}")
    missing_metadata = [prompt for prompt in prompts if prompt not in entries_by_prompt]
    if missing_metadata:
        raise SystemExit("Suite metadata is missing prompt(s), first: " + repr(missing_metadata[0]))

    output_root.mkdir(parents=True)
    for rank in range(world_size):
        rank_prompts = prompts[rank::world_size]
        rank_dir = output_root / f"rank_{rank:02d}"
        rank_split = rank_dir / "split_clip"
        rank_dir.mkdir()
        rank_split.mkdir()
        rank_entries = [entries_by_prompt[prompt] for prompt in rank_prompts]

        for prompt in rank_prompts:
            for sample_index in range(num_samples):
                filename = f"{prompt}-{sample_index}.mp4"
                source_video = source_dir / filename
                source_clips = source_split / source_video.stem
                if not source_video.is_file():
                    raise SystemExit(f"Missing evaluation-view video: {source_video}")
                if not source_clips.is_dir():
                    raise SystemExit(f"Missing evaluation-view clips: {source_clips}")
                (rank_dir / filename).symlink_to(source_video)
                (rank_split / source_video.stem).symlink_to(source_clips, target_is_directory=True)

        (rank_dir / "VBench_full_info.json").write_text(
            json.dumps(rank_entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (rank_dir / "prompts.txt").write_text(
            "".join(f"{prompt}\n" for prompt in rank_prompts), encoding="utf-8"
        )
        (rank_dir / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "prompts": rank_prompts,
                    "num_samples": num_samples,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[vbench-standard-eval-shard] rank={rank}/{world_size} "
            f"prompts={len(rank_prompts)} input={rank_dir}"
        )
