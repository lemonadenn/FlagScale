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

`git submodule update` puts BSA **source files in the checkout** at
`flagscale/inference/core/diffusion/horizonkv/third_party/Block-Sparse-Attention`.
Its nested `csrc/cutlass` submodule supplies compilation headers. Installing BSA
then copies its Python package and compiled CUDA extension into the worker's
`site-packages`. Updating the submodule does not update an installed package;
rebuild BSA after changing its pinned commit.

The validated worker uses Linux x86_64, Python 3.10, PyTorch 2.5.1 cu124,
Triton 3.1.0, FlashAttention 2.8.3, diffusers 0.31.0 and the modified BSA fork.
Use an NVIDIA GPU with compute capability >= 8.0, a compatible NVIDIA driver,
and the **CUDA toolkit 12.4 including nvcc**. The PyTorch wheel alone does not
supply nvcc. Install Python 3.10 with venv support, Git, a C++ compiler and
CUDA toolkit before running the setup script; make `nvcc` available on PATH.
If necessary, set `CUDA_HOME=/usr/local/cuda-12.4` and prepend its `bin` to PATH.

FlagScale's default CUDA inference extra currently uses torch 2.9.0 and CUDA
12.8. This example uses an isolated worker to preserve the tested extension ABI.
It installs the same checkout's FlagScale CLI and uses the standard Runner and
diffusion engine. Avoid installing `flagscale[cuda-inference]` in this worker.
An existing FlagScale controller environment can still launch this example:
`experiment.cmds.before_start` activates the HorizonKV worker for inference.
Compatibility with torch 2.9/CUDA 12.8 has not been validated.

From a new machine, after the BSA fork's pinned commit has been published:

```bash
git clone --recurse-submodules https://github.com/lemonadenn/FlagScale.git
cd FlagScale
# Also needed after pulling a change to the submodule pointer:
git submodule update --init --recursive

export HORIZONKV_PYTHON=python3.10
export HORIZONKV_ENV="$PWD/.venvs/horizonkv-worker"
# Limit concurrent compiler jobs to fit host RAM.
export MAX_JOBS=4
bash examples/horizonkv/setup_env.sh
```

`setup_env.sh` creates a standalone venv (no inherited system site-packages),
installs the pinned PyTorch wheel and model/runtime dependencies, builds
FlashAttention, installs FlagScale editable without CUDA extras, initializes
BSA/CUTLASS, and builds the modified BSA from source. It finishes with `pip check`
and a GPU smoke check without model weights. Network access is needed for the
submodules and packages; CUDA compilation can take several minutes.
`HORIZONKV_BASE_PYTHON` remains an alias for the bootstrap Python, but no
preinstalled torch is required. An old venv created with `--system-site-packages`
is rejected: use a new `HORIZONKV_ENV` path to avoid stale inherited extensions.

After updating BSA, rebuild just that dependency:

```bash
bash examples/horizonkv/setup_blocksparse.sh
"$HORIZONKV_ENV/bin/python" examples/horizonkv/check_worker.py --cuda
```

The BSA installer always forces compilation of the modified source, bypasses
cached/upstream wheels and avoids changing torch. `check_worker.py` reports the
actual imported Python/CUDA paths, checks the variable-block interface, compares
64x64 sparse attention with unequal Q/K lengths against a torch reference, and
executes FlashAttention. It does not load model weights. If no GPU is visible,
run without `--cuda` for import/version checks; the full setup requires a GPU
for its final verification. An external BSA checkout can be selected with
`HORIZONKV_BLOCK_SPARSE_SOURCE=/path/to/modified/BSA`.

Model assets are downloaded separately; they are not included in Git or pip.
For the default Krea 14B profiles, arrange:

```text
models/horizonkv/
  wan_models/Wan2.1-T2V-1.3B/
    Wan2.1_VAE.pth
    models_t5_umt5-xxl-enc-bf16.pth
    google/umt5-xxl/             # tokenizer files
  checkpoints/krea/krea-realtime-video-14b.safetensors
```

Use the Wan2.1 VAE/T5/tokenizer assets from
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and the checkpoint
following the download instructions in
[Krea realtime video](https://github.com/krea-ai/realtime-video).
The asset directory name is shared by the 1.3B and Krea adapters; Krea's generator
comes from its own 14B checkpoint. Set the paths and validate before generation:

```bash
export HORIZONKV_ASSETS="$PWD/models/horizonkv"
export KREA_CHECKPOINT="$HORIZONKV_ASSETS/checkpoints/krea/krea-realtime-video-14b.safetensors"
export HORIZONKV_GPU=0
CUDA_VISIBLE_DEVICES="$HORIZONKV_GPU" "$HORIZONKV_ENV/bin/python" \
  examples/horizonkv/check_environment.py --cuda --model-variant krea-14b \
  --assets-root "$HORIZONKV_ASSETS" --checkpoint "$KREA_CHECKPOINT"
```

The preflight also checks model files and builds/loads HorizonKV's transfer
extension. Its first execution requires nvcc. Build errors can be diagnosed with
`HORIZONKV_FUSED_ONLOAD_VERBOSE=1`. For the Self-Forcing 1.3B profile, provide a
checkpoint containing `generator_ema` and pass `--model-variant self-forcing-1.3b`.

## Generate through FlagScale

Run from the FlagScale root in its normal controller environment, with the
environment variables above set:

```bash
"$HORIZONKV_ENV/bin/flagscale" run --config-path examples/horizonkv/conf --config-name inference --action test
```

`test` runs in the foreground. `run` uses the Runner's background execution.
The current Runner uses `backend: vllm` as its inference dispatch label; the
configured worker entrypoint is the diffusion engine, not a vLLM model.

The default Krea HorizonKV example produces 30 **latent** frames, decoded into
117 pixel frames at 24 fps. `latent_frames` must be a positive multiple of three. The adapter
currently supports one prompt per call, BF16, and a single NVIDIA GPU. Use
`generate.prompts` for sequential prompts; do not add the standard Wan example's
CFG/timestep/TaylorSeer transformations.

Modify `pipeline.from_pretrained.horizonkv` to select execution:

| Setting | Effect |
|---|---|
| `method: horizonkv` | Full host history with a bounded GPU KV cache |
| `fixed: true` | Disable RASA; optional `topk` and `sparsity` choose a fixed point |
| `no_overlap: true` | Disable OAE while retaining the configured selector |
| `cache: 40` | Per-head GPU frame capacity for the host-history mode |
| `rasa: /path/to/policy.json` | Device/workload-specific calibrated policy |

The FlagScale HorizonKV runtime generates both the KV JSON and all native
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

Check the worker without downloading weights:

```bash
"$HORIZONKV_ENV/bin/python" examples/horizonkv/check_worker.py --cuda
```

Check model assets and the transfer extension with the preflight command above.
CPU adapter contracts can be checked after installing `pytest` in the worker:

```bash
"$HORIZONKV_ENV/bin/python" -m pytest -q \
  tests/unit_tests/test_ar_diffusion_baseline.py \
  tests/unit_tests/transformations/diffusion/test_horizonkv_integration.py
```

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
