# NVIDIA Blackwell SM120 (RTX PRO 6000): vLLM kernels and integration

Inference-speed work for vLLM on one NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB), mostly with a hybrid
Qwen3.8 27B model (Gated DeltaNet plus attention layers, MTP speculative decoding). The approach is model-agnostic by
design: NVFP4 weights for every model, kernels that calibrate themselves for new layer shapes, and engine-level
changes. Dates in file names and logs are dd-mm-yyyy (September to October 2026). Apache-2.0; see `../LICENSE`,
`../NOTICE` and `../REUSE.toml`.

This release does not state which checkpoint each result used. The runs used checkpoints of the Qwen3.8 27B
architecture quantized to NVFP4 by us. Some results used a modified, non-stock checkpoint of the same architecture
whose weights are not published (the model-preparation scripts call it `checkpoint-src` and `Qwen3.8-27B-variant`).
Speed and kernel results depend on the architecture and shapes; accuracy, draft acceptance and log-prob numbers depend
on the weights and may not reproduce on the stock model.

The kernels are written in TileLang. Their source is in [`tilelang_contrib/`](tilelang_contrib/README.md): the
packages that `horizon_kernels/`, `horizon_kvfp4/` and the kernel scripts import as `tilelang.contrib.<pkg>`
(`sm120_nvfp4`, `sm120_fp8`, `horizon_ops`, `horizon_fused`, `horizon_k4`, `horizon_moe`; `horizon_ops_wip` is an alias)
and `tilelang.adaptive`. They are plain Python TileLang code with embedded CUDA strings and are meant to run on a stock
TileLang install; nothing is copied into TileLang and no TileLang patch is needed. This tree ships the production
versions of these kernels. Some modules and scripts are left out:

- the GDN decode path (`horizon_ops/gdn_decode.py`, `gdn_g3v.py`, `gdn_recover.py`, `gdn_state.py` and
  `horizon_k4/segment.py`), until a GPU re-check of one changed helper has passed; code that imports them raises
  `ImportError` until then;
- experiment scripts that need other variants of the shared kernels than the published ones, or a helper module that is
  not in any repository, and the scripts that import those scripts.

Some experiment scripts also import kernel packages generated or staged for one run (`hzgs`, `hzkv`, `stable_align`
and similar), which are not included.
`horizon_kernels/fast_start.py` (not in this release) is not published, so leave `HORIZON_FAST_START` unset. `scripts/training/trainctx.py`
imports the third-party `fla` (flash-linear-attention) and `causal_conv1d` packages when those backends are selected.
`vllm032-port/hk/horizon_kernels/gdn_round_ops/chunk_delta_h*.py` are source inputs for vLLM's vendored
flash-linear-attention (see `NOTICE`); their relative imports resolve only inside that package.
`scripts/kernels/rega/probe_up.py` imports TileLang's own test modules through `sys.path`, so it needs a TileLang source
checkout.

**How the kernels are found (bootstrap).** `tilelang_contrib/__init__.py` holds an import hook that maps exactly those
module names to its sub-folders and leaves everything else to the installed TileLang. Importing `horizon_kernels` or
`horizon_kvfp4` from this tree registers it automatically: their `__init__.py` looks for `tilelang_contrib/` next to the
package and calls `tilelang_contrib.install()` (set `HK_NO_CONTRIB_BOOTSTRAP=1` to skip that). For a stand-alone script,
or for a vLLM server that should load the plugin without importing it first, put `nvidia-sm120/` on `PYTHONPATH` and
either call `import tilelang_contrib; tilelang_contrib.install()` before the first kernel import, or write a `.pth`
file once per environment: `python -m tilelang_contrib --pth > <site-packages>/horizon_tilelang_contrib.pth`.
Scripts that put a `contrib/` folder on `tilelang.contrib.__path__` themselves work when pointed at `tilelang_contrib/`.

**TileLang version.** TileLang v0.1.15 is recommended; the minimum is upstream commit `7a5f446f` (PR #3257, the SM120
block-scaled register-operand GEMM). **Tested TileLang commit: not yet verified on stock TileLang `>= 7a5f446f`.** The
measurements in this tree were taken with these packages inside our own TileLang build (v0.1.15 based); importing and
running them through the bootstrap on an unmodified TileLang has not been tested yet. The minimum version comes from
reading the TileLang sources, not from a run. An optional TileLang change that speeds up the register-operand prefill
GEMM without changing its output is proposed upstream as PR #3286.

## What is here

| Path | Contents |
|---|---|
| [`tilelang_contrib/`](tilelang_contrib/README.md) | the TileLang kernels themselves (NVFP4/FP8 GEMMs, attention, GDN, fused ops, K4 persistent decode, MoE) and the import hook that makes them load on a stock TileLang |
| [`horizon_kernels/`](horizon_kernels/) | vLLM plugin (`vllm_plugin.py`): adaptive NVFP4 linear with per-shape kernel choice, a mode without Marlin, thread-local CUDA-graph capture and auto-calibration; dispatcher variants (`nvfp4_linear.py`), calibration (`calibration.py`), the fused FP8 MTP draft head (`draft_head.py`), GDN state and prefill work, MoE, output-head screening |
| [`horizon_kvfp4/`](horizon_kvfp4/) | FP4 KV-cache attention backend |
| [`vllm032-port/`](vllm032-port/) | the plugin ported to vLLM 0.32, with a dynamic draft-count hook, and a KV offloading spec |
| [`patches/`](patches/) | vLLM diffs: the Qwen MTP draft head (fused head, draft vocabulary) and an MTP prefix-cache tail fix |
| [`scripts/kernels/`](scripts/kernels/) | kernel races, correctness checks, calibration and microbenchmarks per track (k4 persistent decode, k5, kvfp4, gdnstate, moe, formats, ops, ...) |
| [`scripts/bench/`](scripts/bench/) | throughput, time-to-first-token, prefix-cache, long-prompt log-prob and quality benchmarks (GSM8K, MATH-500, HumanEval, MBPP, IFEval) |
| [`scripts/models/`](scripts/models/) | MTP draft-head recipe (data generation, draft vocabulary, trainer), NVFP4 and FP8 quantization, and pruning/calibration tools in `flashnext/` |
| [`scripts/training/`](scripts/training/) | LoRA context-length measurement |
| [`results/`](results/README.md) | kernel-level measurements (kernel races, microbenchmarks, ISA and tile sweeps), one folder per experiment; serving-level measurements are not included |
| [`docs/`](docs/README.md) | kernel design notes and result write-ups; [`docs/upstream/`](docs/upstream/) holds our PR and issue drafts for vLLM, flash-linear-attention and TileLang |

## Running things

- Only these shell scripts are published: `scripts/bench/*.sh`, `scripts/models/install_kernels.sh`,
  `scripts/training/trainctx*_body.sh` and `vllm032-port/scripts/install_vllm032.sh`. They set up environments,
  download models, start a standalone vLLM or run benchmarks; none of them controls a system service or reads its
  journal. The runner scripts under `scripts/kernels/`, `scripts/training/` and `scripts/models/` are not part of this
  release; where a Python file or a doc names such a `.sh` file, it refers to one of these unpublished runners. Python code tied to the original serving host (its journal, environment file or GPU
  windows) is left out too. A few published Python scripts free the GPU before a measurement; the step that stops,
  starts or queries vLLM is a marked placeholder (`# (site-specific: ...)`) to replace with your own service control
  (`scripts/models/mtp_train*.py`, `scripts/models/quant_nvfp4*.py`, `scripts/kernels/tlnext/*_race.py`).
- The fast-restart feature (`HORIZON_FAST_START`, a weight-cache daemon run as a host service) is not published; the
  plugin's switch for it stays off by default.
- The model-preparation scripts (`scripts/models/quant_nvfp4.py`, `convert_fp8.py`, `finish_nvfp4.py`) take the
  Hugging Face id of the source checkpoint as their only argument. They were written for the checkpoint used in these
  runs and also copy its extra `checkpoint_metadata.json`; a stock checkpoint has no such file, so remove it from
  their copy lists before using them with another model.
- `hz` is our measurement harness (screens, gates, paired loads); it is not published, and paths starting with `hz/`
  refer to it.
- Many scripts take the original host's GPU lock (`gpu_lock.sh`, with a `GPU_BUSY` marker file) so that only one GPU
  job runs at a time. That lock script belongs to the unpublished harness; on your own machine remove those calls or
  provide an equivalent. References to other files marked "(not in this release)" name scripts or logs that were left out.
- Benchmarks talk to vLLM on `localhost:8000` (in some places a second endpoint on `localhost:8001`).
- Paths are written as environment variables: `$HK_DATA` (models, packages, caches), `$HK_VENV` (the vLLM Python
  environment) and `$HK_WORK` (the experiment directory). Results and logs show them as `<data>`, `<venv>` and
  `<work>`. Some Python files keep them as literal strings; set them or edit the path before running. `$HK_SRC` /
  `<src>` is the directory holding the TileLang fork checkout.
- The GPU in these runs was shared through a time-slicing layer (called `gpushare` here) that hides real free memory
  and breaks some CUDA-graph capture modes; several workarounds in the plugin (thread-local capture, explicit
  memory probes) exist because of it.
- Most results come from a driver 570 / CUDA 12.8 host with forward-compatibility libraries; several FlashInfer
  paths were not usable there, which the docs note where it matters.

Research code: no API stability, no support, numbers valid only for the setup they name.
