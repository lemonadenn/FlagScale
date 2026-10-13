# HorizonKV video generation in FlagScale

For the distilled **Wan2.1-14B autoregressive** model, use the repository-local
HorizonKV profile with `--config-name inference`. The AR+Diffusion baseline is
available as `--config-name inference_baseline`.

FlagScale runs the native Wan2.1 autoregressive pipeline through the diffusion
inference engine. `HorizonKVTransformation` attaches the resolved HorizonKV
configuration to its causal transformer. Frame retrieval, compacted
RoPE, RASA, OAE, and the CUDA kernels are provided by the repository-local
`flagscale/inference/core/diffusion/horizonkv/` package.

The adapter preserves three latent frames per chunk, four denoising evaluations
for 1.3B (five for Krea 14B), and a separate clean-KV commit at timestep zero. Requests run
sequentially; native environment settings, asynchronous transfers, conditioning
caches, VAE state, and random-number state are isolated between calls.

## Dependencies and environment

The migrated HorizonKV implementation is repository-local under
`flagscale/inference/core/diffusion/horizonkv/`. Its modified
Block-Sparse-Attention source is bundled under that package. An upstream wheel
alone omits the required kernels.
The source uses the top-level `wan` package: do not import another Wan package
in the same worker.

The native ABI currently requires Linux, Python 3.10, PyTorch 2.5.1 + CUDA 12.4,
Triton 3.1.0, FlashAttention 2.8.3, diffusers 0.31.0, and the modified 64×64 sparse
backend. The default FlagScale CUDA environment uses a newer PyTorch ABI. Keep
the native worker separate; `experiment.cmds.before_start` selects it without
changing the controller's environment. Unsupported combinations fail before
loading model weights. NVIDIA compute capability 8.0 or newer is required.

The repository includes the HorizonKV implementation under
`flagscale/inference/core/diffusion/horizonkv`, the native model assets under
`models/horizonkv`, and the compatible worker under
`.venvs/horizonkv-worker`. If the worker must be rebuilt, provide a Python 3.10
interpreter with PyTorch 2.5.1 as the bootstrap interpreter:

```bash
export HORIZONKV_BASE_PYTHON=/path/to/python3.10-with-torch251/bin/python
export HORIZONKV_ENV=/workspace/FlagScale/.venvs/horizonkv-worker
bash examples/horizonkv/setup_env.sh
```

This creates a venv with `--system-site-packages` and adds FlagScale's worker
dependencies. The checked-in layout remains self-contained after setup; the
bootstrap interpreter is only needed when rebuilding the worker. Tests
additionally require `pytest`.

Set the source and native model paths (asset root contains `wan_models/`):

```bash
export HORIZONKV_ASSETS=/workspace/FlagScale/models/horizonkv
export HORIZONKV_CHECKPOINT=/workspace/FlagScale/models/horizonkv/checkpoints/self_forcing_dmd.pt
export HORIZONKV_GPU=0
CUDA_VISIBLE_DEVICES="$HORIZONKV_GPU" "$HORIZONKV_ENV/bin/python" \
  examples/horizonkv/check_environment.py --cuda
```

The checkpoint must contain `generator_ema`. The adapter uses strict checkpoint
loading and requires the native Wan2.1-1.3B assets (30 layers, 12 heads, head
dimension 128). No diffusers checkpoint conversion is involved. Preflight checks
the assets and loads the transfer extension; generation additionally exercises
the actual sparse kernel. Build errors can be diagnosed with
`HORIZONKV_FUSED_ONLOAD_VERBOSE=1`.

## Generate through FlagScale

Run from the FlagScale root in its normal controller environment, with the
environment variables above set:

```bash
flagscale run --config-path examples/horizonkv/conf --config-name inference --action test
```

`test` runs in the foreground. `run` uses the Runner's background execution.
The current Runner uses `backend: vllm` as its inference dispatch label; the
configured worker entrypoint is the diffusion engine, not a vLLM model.

The default example produces 24 **latent** frames, decoded into 93 pixel frames
at 16 fps. `latent_frames` must be a positive multiple of three. The adapter
currently supports one prompt per call, BF16, and a single NVIDIA GPU. Use
`generate.prompts` for sequential prompts; do not add the standard Wan example's
CFG/timestep/TaylorSeer transformations.

Modify `pipeline.from_pretrained.horizonkv` to select execution:

| Setting | Effect |
|---|---|
| `method: horizonkv` | Full host history with a bounded GPU KV cache |
| `method: horizonkv-gpu` | GPU-only rolling history of `window` latent frames |
| `method: base` | Dense nearest-neighbour sliding-window baseline |
| `method: lightforcing` | Upstream LightForcing baseline |
| `fixed: true` | Disable RASA; optional `topk` and `sparsity` choose a fixed point |
| `no_overlap: true` | Disable OAE while retaining the configured selector |
| `cache: 40` | Per-head GPU frame capacity for the host-history mode |
| `rasa: /path/to/policy.json` | Device/workload-specific calibrated policy |

The canonical upstream CLI resolver generates both the KV JSON and all native
execution flags in an engine-owned temporary directory. User environment flags
cannot accidentally override that execution profile. Explicit extension build
directory and build verbosity settings are preserved. `CUDA_VISIBLE_DEVICES`
is owned by the Runner and is never rewritten by the adapter.

`generate.generator.seed` is the actual sampling seed, consistent with the
FlagScale generator contract. The upstream CLI hashes the base seed with the
prompt; to reproduce one of its runs, supply that run's final recorded seed.
Both initial noise and intermediate denoising noise use the request Generator.

`engine.output_fps: 16` sets video encoding fps. Other diffusion configurations
retain their default of 15 fps. Programmatic callers can request
`output_type="latent"`; the ordinary video-saving entrypoint expects decoded
frames, so use the validation script for latent-only checks.

## Validation

For official-prompt sampling and VBench-Long scoring of HorizonKV 14B and the
14B original AR+Diffusion baseline, use
[`tests/vbenchlong_test.sh`](tests/vbenchlong_test.sh).
See [vbenchlong.md](vbenchlong.md) for local evaluator weights, multi-GPU
execution, resume support, screening defaults, and the complete suite command.

CPU contracts and existing transformations:

```bash
python -m pytest -q tests/unit_tests/transformations
```

Native model parity and repeated-request cleanup, using the existing model
weights and an empty output directory:

```bash
CUDA_VISIBLE_DEVICES="$HORIZONKV_GPU" "$HORIZONKV_ENV/bin/python" \
  examples/horizonkv/validate_integration.py \
  --assets-root "$HORIZONKV_ASSETS" \
  --checkpoint "$HORIZONKV_CHECKPOINT" --output outputs/horizonkv-fixed-check \
  --frames 6 --cache 12 --fixed --no-overlap --compare-native --repeat --video
```

Exercise history exceeding GPU capacity, RASA, and OAE:

```bash
CUDA_VISIBLE_DEVICES="$HORIZONKV_GPU" "$HORIZONKV_ENV/bin/python" \
  examples/horizonkv/validate_integration.py \
  --assets-root "$HORIZONKV_ASSETS" \
  --checkpoint "$HORIZONKV_CHECKPOINT" --output outputs/horizonkv-offload-check \
  --frames 48 --cache 12 --compare-native --repeat
```

The script writes `validation.json` with native configuration, versions, timings,
memory and RASA counters, plus `latents.pt` and optionally an MP4. Exact native
parity compares the same algorithm/profile, not OAE enabled versus disabled;
those paths may have different integer block quotas.

At 480×832, full-history BF16 KV costs about 0.268 GiB per latent frame across
all layers. 480 latent frames need about 128.5 GiB of host KV, excluding temporary
buffers; the reference recommendation is at least 256 GB host RAM. Forty GPU
cache frames use about 10.7 GiB for KV alone. Total model/latent/summary/VAE memory
is additional. This integration does not yet provide multi-GPU sequence
parallelism, concurrent request batching, or non-CUDA kernels.

## Efficiency comparison

The comparison launcher calls the two top-level FlagScale inference entries
directly. The entrypoint performs warmup and timed runs through
`InferenceEngine`; only `benchmark.json` and a compact comparison are written.

```bash
cd /workspace/FlagScale
export HORIZONKV_ASSETS=/workspace/FlagScale/models/horizonkv
export HORIZONKV_ENV=/workspace/FlagScale/.venvs/horizonkv-worker
export KREA_CHECKPOINT=/workspace/FlagScale/models/horizonkv/checkpoints/krea/krea-realtime-video-14b.safetensors

RUN_ID=efficiency-30s WARMUP_RUNS=1 MEASURE_RUNS=1 \
  DURATION_SECONDS=30 bash examples/horizonkv/tests/run_efficiency_comparison.sh
```

The first case uses `model_variant=krea-14b`, `method=horizonkv`,
`fixed=false`, `topk=7`, `sparsity=0.5`, `sink=1`; the native plan enforces four
mandatory recent frames and 15 GPU cache frames. Both cases use five denoising
steps. The second calls the original Krea `GenerationSession` with an expanded
window: six recomputed context latents plus three new latents (nine total), no
fixed sink, and streaming FP16 VAE. Once the history window fills, its context is
one reencoded boundary image followed by the five latest original latents.
Set `KREA_CONTEXT_FRAMES=3` to reproduce the upstream default window.
See [vbenchlong.md](vbenchlong.md) for source locations and the scoring protocol.
Set `DRY_RUN=1` on the launcher to validate paths without loading model weights.

The 14B cases also have top-level FlagScale configs, so they can be launched
through the same entrypoint convention as the other examples:

```bash
flagscale run -p examples/horizonkv/conf -n inference -a test
flagscale run -p examples/horizonkv/conf -n inference_baseline -a test
```

Override prompt, length, seed, or output location with the usual Hydra
arguments, for example `inference.generate.latent_frames=30` or
`inference.engine.results_path=/tmp/horizonkv-results`.

The efficiency launcher deliberately does not start the full VBench-Long matrix;
the generated videos can be passed to the VBench-Long evaluator separately.

### Generate 60 seconds for both 14B cases

```bash
cd /workspace/FlagScale
HORIZONKV_GPU=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  bash examples/horizonkv/tests/run_efficiency_comparison.sh
```

This runs the two cases sequentially, with one full-length warm-up and one
measured sample each. Warm-up is excluded from the reported median. Both cases
use the same prompt, seed, resolution, frame rate, and duration.

HorizonKV now keeps T5 resident and uses the native VAE path without releasing
KV before decode. Same-size requests reuse the warmed KV allocation. The 14B
profiles default to 15 GPU cache frames; `HORIZONKV_GPU_CACHE_FRAMES=12` provides
more headroom. At 832×480, each BF16 KV frame occupies 1.190 GiB across 40 layers.
The earlier 25-frame setting failed native 60-second VAE decoding with T5
resident. Export receives contiguous RGB arrays and uses the same encoder
settings for both cases; output-layout conversion remains inside generation
and end-to-end timing.

Videos and timing JSON are written to `benchmarks/efficiency_60s/<RUN_ID>/<case>/`,
with runner logs in `logs/efficiency_60s/<RUN_ID>/<case>/`. Existing runs are not
overwritten. `RUN_ID` defaults to the UTC launch timestamp.

`DRY_RUN=1 RUN_ID=review_60s bash examples/horizonkv/tests/run_efficiency_comparison.sh`
checks the launch arguments without allocating the model. CUDA availability
must be checked in the execution environment with access to `/dev/nvidia*`;
a device-hidden execution sandbox cannot establish whether the container has
GPUs. The full-history 14B case requires roughly 432 GiB of host KV for this
duration, in addition to model and working memory.

**Historical four-step/cache-12 validation:** both old cases completed a real 60-second run on an A800 80GB. Videos, previews
and metrics are in `benchmarks/efficiency_60s/generate60_gpu_20261004/`; the measured
end-to-end times were 857.593 s for HorizonKV and 487.160 s for the baseline
(one sample each, no warm-up, excludes model loading). See
[validation_report.md](validation_report.md) for timings and visual observations.
