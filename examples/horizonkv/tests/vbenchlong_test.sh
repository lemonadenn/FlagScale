#!/usr/bin/env bash
set -Eeuo pipefail
# Ctrl-\\ is commonly used by screen-recording tools. Ignore SIGQUIT in both
# this shell and the tee child so recording does not turn into a core dump.
trap '' SIGQUIT

# Reproducible VBench-Long quality comparison for a 30-second video per prompt.
# The script is intentionally self-contained so its terminal output can be
# recorded and submitted together with the run directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"
FLAG_SCALE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
SUITE_DIR="$SCRIPT_DIR"
SUITE="$SUITE_DIR/benchmarks/vbenchlong_suite.py"
REPORT="$SUITE_DIR/benchmarks/vbenchlong_report.py"

VBENCH_ROOT="${VBENCH_ROOT:-/workspace/VBench}"
SAMPLE_PYTHON_BIN="${SAMPLE_PYTHON_BIN:-$FLAG_SCALE_ROOT/.venvs/horizonkv-worker/bin/python}"
VBENCHLONG_PYTHON_BIN="${VBENCHLONG_PYTHON_BIN:-/root/miniconda3/envs/Vbenchlong/bin/python}"
SEED="${VBENCHLONG_SEED:-42}"
FPS="${VBENCHLONG_FPS:-24}"
HORIZONKV_ASSETS="${HORIZONKV_ASSETS:-$FLAG_SCALE_ROOT/models/horizonkv}"
KREA_CHECKPOINT="${KREA_CHECKPOINT:-$HORIZONKV_ASSETS/checkpoints/krea/krea-realtime-video-14b.safetensors}"
# Three 14B workers fit in the available host RAM with pageable history. Pinned
# allocations round each layer's K/V buffers to powers of two and commonly
# limit this workload to two workers on a 1 TiB host.
HORIZONKV_PIN_MEMORY="${HORIZONKV_PIN_MEMORY:-0}"
# Two concurrent 14B samplers are the validated setting for the 1 TiB host.
HORIZONKV_MAX_SAMPLING_WORKERS="${HORIZONKV_MAX_SAMPLING_WORKERS:-2}"

if [[ -n "${VBENCHLONG_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$(realpath -m -- "$VBENCHLONG_OUTPUT_ROOT")"
else
  OUTPUT_ROOT="$FLAG_SCALE_ROOT/benchmarks/vbenchlong_evaluation"
fi
mkdir -p -- "$OUTPUT_ROOT"
RUN_LOG="$OUTPUT_ROOT/run.log"
# Each invocation is a fresh screen-recording transcript; per-worker logs and
# resumable videos remain under the same output root.
: > "$RUN_LOG"
exec > >(
  trap '' SIGQUIT
  exec tee -a "$RUN_LOG"
) 2>&1

die() {
  echo "[vbenchlong] ERROR: $*" >&2
  exit 1
}

[[ -x "$SAMPLE_PYTHON_BIN" ]] || die "sample Python not found: $SAMPLE_PYTHON_BIN"
[[ -x "$VBENCHLONG_PYTHON_BIN" ]] || die "VBenchLong Python not found: $VBENCHLONG_PYTHON_BIN"
[[ -f "$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json" ]] || \
  die "VBenchLong metadata not found under $VBENCH_ROOT"
[[ -f "$KREA_CHECKPOINT" ]] || die "KREA_CHECKPOINT not found: $KREA_CHECKPOINT"

echo "[vbenchlong] output=$OUTPUT_ROOT"
echo "[vbenchlong] duration=30s fps=$FPS dimensions=all16 prompts_per_dimension=10 samples_per_prompt=1 seed=$SEED"
HORIZONKV_GPU_LIST="${HORIZONKV_GPU_LIST:-0,1}"
BASELINE_GPU_LIST="${BASELINE_GPU_LIST:-2,3,4,5,6,7}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-2,3,4,5,6,7}"
FINAL_EVAL_GPU_LIST="${FINAL_EVAL_GPU_LIST:-0,1,2,3,4,5,6,7}"
echo "[vbenchlong] horizonkv_sampling_gpus=$HORIZONKV_GPU_LIST baseline_sampling_gpus=$BASELINE_GPU_LIST evaluation_gpus=$EVAL_GPU_LIST"
echo "[vbenchlong] horizonkv_final_evaluation_gpus=$FINAL_EVAL_GPU_LIST"
echo "[vbenchlong] horizonkv_pin_memory=$HORIZONKV_PIN_MEMORY horizonkv_max_sampling_workers=$HORIZONKV_MAX_SAMPLING_WORKERS"

COMMON_ENV=(
  "OUTPUT_ROOT=$OUTPUT_ROOT"
  "VBENCH_ROOT=$VBENCH_ROOT"
  "SAMPLE_PYTHON_BIN=$SAMPLE_PYTHON_BIN"
  "VBENCHLONG_PYTHON_BIN=$VBENCHLONG_PYTHON_BIN"
  "HORIZONKV_ASSETS=$HORIZONKV_ASSETS"
  "KREA_CHECKPOINT=$KREA_CHECKPOINT"
  "HORIZONKV_PIN_MEMORY=$HORIZONKV_PIN_MEMORY"
  "HORIZONKV_MAX_SAMPLING_WORKERS=$HORIZONKV_MAX_SAMPLING_WORKERS"
  "PYTHONDONTWRITEBYTECODE=1"
  "TOKENIZERS_PARALLELISM=false"
  "DURATION_SECONDS=30"
  "OUTPUT_FPS=$FPS"
  "NUM_SAMPLES_PER_PROMPT=1"
  "PROMPT_LIMIT_PER_DIMENSION=10"
  "PROMPT_SELECTION_STRATEGY=uniform"
  "VBENCHLONG_DIMENSIONS=all16"
  "VBENCHLONG_DEV_FLAG=${VBENCHLONG_DEV_FLAG:-0}"
  "SEED=$SEED"
  "RUN_SAMPLE=1"
  "RUN_EVAL=0"
  "DRY_RUN=${DRY_RUN:-0}"
)

run_sample() {
  local cases="$1"
  local sample_gpus="$2"
  echo "[vbenchlong] sampling cases=$cases GPUs=$sample_gpus"
  env "${COMMON_ENV[@]}" \
    "RUN_EXPERIMENTS=$cases" \
    "RUN_SAMPLE=1" \
    "RUN_EVAL=0" \
    "GPU_LIST=$sample_gpus" \
    "VBENCHLONG_GPU_LIST=$EVAL_GPU_LIST" \
    "$SAMPLE_PYTHON_BIN" -u "$SUITE"
}

run_eval() {
  local cases="$1"
  local eval_gpus="$2"
  echo "[vbenchlong] evaluating cases=$cases GPUs=$eval_gpus"
  env "${COMMON_ENV[@]}" \
    "RUN_EXPERIMENTS=$cases" \
    "RUN_SAMPLE=0" \
    "RUN_EVAL=1" \
    "GPU_LIST=$eval_gpus" \
    "VBENCHLONG_GPU_LIST=$eval_gpus" \
    "$SAMPLE_PYTHON_BIN" -u "$SUITE"
}

echo
echo "[vbenchlong] launching HorizonKV sampling in parallel with baseline sampling"
run_sample "horizonkv" "$HORIZONKV_GPU_LIST" &
HORIZONKV_SAMPLE_PID=$!

baseline_status=0
if run_sample "ar_diffusion_baseline" "$BASELINE_GPU_LIST"; then
  baseline_status=0
else
  baseline_status=$?
fi
if (( baseline_status != 0 )); then
  echo "[vbenchlong] baseline sampling failed; stopping HorizonKV sampling" >&2
  kill "$HORIZONKV_SAMPLE_PID" 2>/dev/null || true
  wait "$HORIZONKV_SAMPLE_PID" 2>/dev/null || true
  exit "$baseline_status"
fi

echo
echo "[vbenchlong] baseline sampling complete; evaluating baseline while HorizonKV samples"
if run_eval "ar_diffusion_baseline" "$EVAL_GPU_LIST"; then
  baseline_status=0
else
  baseline_status=$?
  echo "[vbenchlong] baseline evaluation failed; stopping HorizonKV sampling" >&2
  kill "$HORIZONKV_SAMPLE_PID" 2>/dev/null || true
  wait "$HORIZONKV_SAMPLE_PID" 2>/dev/null || true
  exit "$baseline_status"
fi

horizonkv_status=0
wait "$HORIZONKV_SAMPLE_PID" || horizonkv_status=$?
if (( horizonkv_status != 0 )); then
  echo "[vbenchlong] HorizonKV sampling failed" >&2
  exit "$horizonkv_status"
fi

echo
echo "[vbenchlong] HorizonKV sampling complete; evaluating HorizonKV on all GPUs"
run_eval "horizonkv" "$FINAL_EVAL_GPU_LIST"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "[vbenchlong] dry run complete; no videos or scores were produced"
  echo "[vbenchlong] output=$OUTPUT_ROOT"
  exit 0
fi

if [[ "${VBENCHLONG_DEV_FLAG:-0}" == "1" ]]; then
  RESULT_PROFILE="${VBENCHLONG_PROFILE:-dev_dinov2_dreamsim}"
else
  RESULT_PROFILE="${VBENCHLONG_PROFILE:-legacy_dino_clip}"
fi
comparison_path=""
mapfile -t comparison_candidates < <(
  find "$OUTPUT_ROOT" -type f -path "*/results/$RESULT_PROFILE/comparison.json" -print
)
if (( ${#comparison_candidates[@]} == 1 )); then
  comparison_path="${comparison_candidates[0]}"
else
  die "expected one comparison.json under $OUTPUT_ROOT, found ${#comparison_candidates[@]}"
fi

echo
echo "[vbenchlong] rendering clean report"
"$VBENCHLONG_PYTHON_BIN" "$REPORT" \
  --comparison "$comparison_path" \
  --output "$OUTPUT_ROOT/vbenchlong_summary.json" \
  --markdown "$OUTPUT_ROOT/vbenchlong_summary.md"

RUN_ROOT="$(dirname -- "$(dirname -- "$(dirname -- "$comparison_path")")")"
copy_to_root() {
  local source="$1"
  local destination="$OUTPUT_ROOT/$2"
  if [[ "$source" != "$destination" ]]; then
    cp -- "$source" "$destination"
  fi
}
copy_to_root "$RUN_ROOT/suite/prompts.txt" prompts.txt
copy_to_root "$RUN_ROOT/suite/VBench_full_info.json" prompt_manifest.json
copy_to_root "${comparison_path%/comparison.json}/summary.csv" vbenchlong_summary.csv

echo
echo "[vbenchlong] complete"
echo "[vbenchlong] output=$OUTPUT_ROOT"
