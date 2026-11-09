> **Status:** Opened as draft PR tile-ai/tilelang#3290 (27-09-2026 11:05). This file is the local draft; the live text on GitHub is authoritative.

# Draft upstream PR 2: SM120 W4A16 NVFP4 decode GEMM (not sent)

Everything below the line is the PR text. It uses neutral names only: no company, no model name.

---

## Title

`[CUDA] Add an SM120 W4A16 NVFP4 GEMM example for decode-sized batches`

## Summary

This adds a GEMM with BF16 activations and NVFP4 weights (W4A16) for SM120, aimed at LLM decoding with M = 1 to 1024
rows. It is item 2 of the plan in the comment on #1592. It complements the W4A4 block-scaled path from #3237 and #3257.
That path quantizes activations too, while this one keeps them in BF16. In serving, the small-M decode steps are where
most tokens are generated.

- Weights are dequantized in registers, at about 2.1 instructions per weight. They come from a fragment-ordered copy
  that is prepared once from the checkpoint tensors (packed E2M1, E4M3/UE4M3 block scales, global or per-column
  scale).
- The results are deterministic and the kernels can be captured in a CUDA graph.
- On one RTX PRO 6000 Blackwell Max-Q, weighted by layer over the linear layers of a 27B-class model, the tuned path
  beats vLLM's Marlin NVFP4 W4A16 kernel by 1.2-4.6% at M = 1-32 and by 3-7% at M = 72-144. It is within ±2% of Marlin
  at most larger M, but has one 6% loss (see Benchmarks). It is 2.2-3.5x faster than a BF16 cuBLAS GEMM at M ≤ 32.

Only `examples/`, `maint/` and `testing/` change. Nothing under `tilelang/` changes, and no file overlaps the open PRs
#3081, #3099, #2171, #3284 or #3286.

## Design

**Weight layout.** `prepare_nvfp4_weight` re-lays out the weight once into blocks of 32 (n) x 32 (k). Each block is
512 bytes, ordered like the `mma.m16n8k16` B fragments of one warp:

- Lane `l` owns 16 consecutive bytes. One `ld.shared.v4` per lane per block gives a complete B fragment for two n16
  tiles x two k16 steps, so no BF16 weight tile is ever written to shared memory.
- Inside each 32-bit word the nibbles are bit-scattered. Three of the four `bf16x2` fragment registers need one shift
  and one AND each; the fourth needs three shifts and three LOP3s. E2M1 codes land in the BF16 subnormal range as
  `fp4 * 2^-126`.
- The E4M3 block scales are re-encoded as one byte `b`, so that `bf16(0x7000 | b << 4) == scale * 2^119`. This is
  exact for every finite E4M3 value, subnormals included. One PRMT plus one IMAD turn a scale word into `bf16x2` pairs.
  One `mul.rn.bf16x2` per register then gives `fp4 * scale * 2^-7` exactly, and `alpha * 2^7` is applied in the FP32
  epilogue.
- Zero scales zero their codes and negative scales flip the code signs, both at preparation time.
- The prepared copy is the same size as the checkpoint tensors: `N*K/2 + N*K/16` bytes.

**M range and kernels.** The kernels cover M = 1 to 1024 rows per launch; larger batches run as 1024-row chunks.

- The **example** (`examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py`) has two non-persistent kernels:
  - A fragment kernel: one CTA row covers M ≤ 64, with split-K on grid.y. It is used for M ≤ 96, in 64-row passes.
  - A `T.Pipelined` tile kernel for larger M.
  - Both reduce split tiles deterministically: FP32 partials, and the last CTA to arrive (atomic ticket) sums them in
    a fixed order. The tickets reset themselves.
  - `get_config` is a shape-independent heuristic.
- The **maint** module (`maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py`) adds three things:
  - A persistent stream-K kernel. `G = SMs * c` CTAs walk contiguous ranges of the (tile, k-tile) space, and a
    hand-written cp.async stage loader gives one barrier per k tile. `T.Pipelined` cannot express this loop, because
    each tile's k range is known only at run time.
  - Per-GPU tuned tables, keyed by SM count. Today there is one, for a 188-SM GPU and the layer shapes of a 27B-class
    model.
  - A stream-K heuristic for other shapes.
  - It reuses the example's layout, kernels and input checks, and has the same `nvfp4_w4a16_linear` interface.

**Why example-first.** #1592 has not yet settled whether this belongs in the library or in examples. The #2364 review
asked for small, non-persistent examples with the heavy persistent path in `maint/`, and that is how this is split.
The public surface is small, so moving it into the library later means moving modules, not redesigning them:
`prepare_nvfp4_weight` → `NVFP4W4A16Weight`, `nvfp4_w4a16_linear(x, w, config=None, out=None)`, `get_config`, and the
`FragConfig` / `TileConfig` / `StreamKConfig` named tuples. The kernels use only public APIs already present at
`7a5f446f`: `T.ptx_mma`, `TensorCoreIntrinEmitter.ldmatrix_a`, `make_mma_swizzle_layout`, `T.Kernel(prelude=)`,
`T.dynamic`, `T.Pipelined`, `T.ptx_commit_group` / `T.ptx_wait_group`, and scalar prim_func arguments with
`T.assume`.

**How it differs from `examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py`.** That example is W4A16 FP4
without block scales. It dequantizes into a shared BF16 tile and targets large M on Hopper. This PR handles NVFP4 with
E4M3 block scales (the checkpoint format), dequantizes register fragments directly, and targets decode-sized M on
SM120. It lives in `examples/gemm_sm120` next to the W4A4 block-scaled example because both target the same checkpoints
and GPU.

**Scope.** The instructions used are `mma.sync.m16n8k16` (BF16), `cp.async` and `mul.rn.bf16x2` (sm_90+). Each CTA uses
at most 99 KB of shared memory. Tests and tuning ran only on sm_120, so the example, the benchmark and the tests all
require compute capability 12.0.

## Changes

- `examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py` (869 lines): the layout, the register dequant helper
  (`mul.rn.bf16x2` inline PTX), the fragment and tile kernels, the heuristic `get_config`, and `nvfp4_w4a16_linear`.
  - Input validation: packed dtype int8/uint8; scale dtype float8_e4m3fn or UE4M3 bytes; NaN scales; shapes; x/out
    dtype, shape, device, contiguity and 16-byte alignment. A misaligned `x` is copied.
  - The per-device workspace (split-K tickets, dummy partials) is created in `prepare_nvfp4_weight`, never lazily
    inside a graph capture. A capture that would need to create it raises an error.
  - A CLI with `--verify`. The default timing backend is `cupti` (see the notes).
- `maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py` (579 lines): the stream-K kernel, the tuned table keyed by SM
  count, the stream-K heuristic (it falls back to the example's tile heuristic when the stream-K ranges would be short),
  and `nvfp4_w4a16_linear`.
- `maint/gemm/gemm_sm120/benchmark_sm120_nvfp4_w4a16_gemm.py` (237 lines): the benchmark.
  - Calls are replayed from a CUDA graph over at least 512 MB of rotated weights (more than the 128 MB L2), with a
    separator kernel whose cost is subtracted.
  - Each (shape, M) runs in 3 rounds, rotating the method order each round. The per-cell table shows the fastest
    round; the layer-weighted totals are given for both the fastest and the median rounds.
  - Baselines are BF16 cuBLAS (`F.linear`) and, when vLLM imports, Marlin (tested with vLLM 0.29.0 internals).
    `--verify` checks both our output and Marlin's against an FP32 reference.
  - Other options: `--dispatch tuned|example`, `--layer-counts`, `--json`.
- `testing/python/quantize/test_tilelang_quantize_nvfp4_w4a16_gemm.py` (350 lines): 55 tests (36 by default, 19 under
  `slow`); details below.

## Tests

The tests ran on an RTX PRO 6000 Blackwell Max-Q (sm_120, driver 570 + CUDA 13 forward compatibility), with TileLang
built from source at `7a5f446f`.

| Test | Count | What it checks | Result |
|---|---|---|---|
| `test_prepare_layout_matches_mma_b_fragments` (CPU) | 1 | Bit-exact emulation of the register dequant, placed at the PTX m16n8k16 B-fragment positions, for every finite E4M3 scale (zero, negative and subnormal) | pass |
| `test_prepare_accepts_ue4m3_bytes_and_column_alpha` (CPU) | 1 | UE4M3 byte scales and a per-column alpha | pass |
| `test_prepare_rejects_bad_inputs`, `test_linear_rejects_bad_inputs` (CPU) | 2 | TypeError / ValueError for bad dtypes (float32 scales above the E4M3 range, bf16, int32), shapes, NaN scales, alpha, `out`, non-weight arguments | pass |
| `test_configs_cover_all_rows` (CPU) | 4 | Every config from both heuristics and the tuned table, for M = 1..1024 on 10 shapes and 188 / 170 SMs, divides the shape and fits the 99 KB shared-memory limit per CTA and the 100 KB limit per SM | pass |
| `test_tuned_table_is_per_gpu` (CPU) | 1 | The tuned table is not applied on another SM count | pass |
| `test_nvfp4_w4a16_gemm` | 8 | Example and tuned paths; shapes 5120x6144, 4096x4096, 640x2304 (N % 256 ≠ 0) and 128x256; M = 1, 17, 64, 200, 1024 | pass |
| `test_nvfp4_w4a16_gemm_explicit_config` | 9 | Explicit FragConfig (including 16-row passes at M = 40), FRAG, three TileConfigs and three StreamKConfigs | pass |
| `test_nvfp4_w4a16_gemm_zero_negative_scales_chunking_and_out` | 2 | Zero and negative scales with per-column alpha; M = 1100 run as two chunks; `out=`; x not 16-byte aligned | pass |
| `test_nvfp4_w4a16_gemm_cuda_graph` | 7 | Capture (thread_local), then two replays with new inputs, bit-identical to eager calls (fragment, wrapped tile, stream-K; example and tuned) | pass |
| `test_nvfp4_w4a16_gemm_first_call_in_shared_pool_capture` | 1 | The first call on a device, captured in a graph that shares a memory pool with an earlier graph; eager call before replay, then both graphs replayed | pass |
| `test_nvfp4_w4a16_gemm_sweep` (`slow`) | 14 | Both paths, 7 shapes x 17 M values from 1 to 1024 | pass |
| `test_nvfp4_w4a16_gemm_every_tuned_entry` (`slow`) | 5 | Every tuned-table entry at the start, middle and end of its M range, with that exact config | pass |

- Every GPU check compares with an FP32 reference built from the upstream helpers `decode_packed_fp4_e2m1` and
  `decode_ue4m3_scale_bytes`. It requires:
  - a relative Frobenius error ≤ 2.5e-3 and a worst-row error ≤ 1e-2;
  - elementwise, `|y - ref| ≤ 2^-8 |ref| + 1e-3 * row_rms` on every element;
  - finite outputs, and bit-identical results on repeat calls.
- Totals: 36 passed by default (55 s with a partly warm kernel cache), 19 passed under `-m slow` (3 min 42 s), and all
  55 passed with a warm cache (17 s). The GPU tests skip on anything other than compute capability 12.0. Upstream CI
  therefore runs only the 9 CPU tests.
- The example with `--verify` reports a relative error of 1.66e-3 at M = 300 on 4096x4096 and at M = 16 on 5120x6144.
- The benchmark's `--verify` passed for both the W4A16 and the Marlin outputs.
- `ruff` 0.16.6 check and format pass, and codespell is clean.

## Benchmarks

The benchmark command was
`python -m maint.gemm.gemm_sm120.benchmark_sm120_nvfp4_w4a16_gemm --verify --layer-counts 48,16,64,64,64`, run on
27-09-2026. The setup:

- The default shapes are the linear layers of a 27B-class model: 16384x5120, 14336x5120, 5120x6144, 34816x5120 and
  5120x17408. The layer counts give the per-step weight of each shape.
- There are 32 values of M: the tuned table's breakpoints and points between them.
- Marlin was built correctly: `weight_global_scale` is the multiplier `alpha`, and its output passed `--verify`.

**Layer-weighted time per model step, W4A16 (tuned path) vs Marlin and cuBLAS.** Negative means W4A16 is faster.

| M | vs Marlin, fastest rounds | vs Marlin, median rounds | vs cuBLAS BF16 |
|---|---|---|---|
| 1 | −3.0% | −2.9% | −70.6% |
| 4 | −2.2% | −2.1% | −70.0% |
| 8 | −1.2% | −1.3% | −69.7% |
| 16 | −3.5% | −3.3% | −69.4% |
| 17 | −3.3% | −3.2% | −68.4% |
| 24 | −4.2% | −4.3% | −68.5% |
| 32 | −4.6% | −4.4% | −68.6% |
| 33 | **+1.1%** | +1.2% | −62.0% |
| 40 | +0.6% | +0.3% | −62.2% |
| 48 | −0.7% | −0.6% | −62.4% |
| 49 | **+1.0%** | +1.3% | −54.2% |
| 56 | +0.5% | +0.8% | −54.3% |
| 64 | −0.1% | +0.3% | −54.4% |
| 72 | −5.3% | −5.2% | −45.0% |
| 96 | −6.8% | −6.9% | −38.8% |
| 104 | −3.2% | −3.0% | −31.9% |
| 128 | −3.5% | −2.2% | −29.3% |
| 144 | −3.4% | −3.2% | −36.5% |
| 192 | **+1.9%** | +2.1% | −29.7% |
| 208 | +1.5% | +0.1% | −14.7% |
| 232 | −3.3% | −4.2% | −10.6% |
| 256 | **+1.3%** | +1.2% | −7.6% |
| 272 | −0.6% | −1.2% | −24.7% |
| 304 | −0.8% | −0.1% | −19.1% |
| 384 | −2.8% | −3.1% | −8.8% |
| 416 | +0.6% | −0.2% | −16.5% |
| 512 | −4.5% | −3.7% | −10.1% |
| 544 | **+6.0%** | +7.1% | −12.2% |
| 608 | −0.5% | −0.6% | −9.4% |
| 768 | −4.8% | −4.6% | −8.0% |
| 832 | **+1.9%** | +2.1% | −5.6% |
| 1024 | −4.7% | −4.5% | +1.1% |

Honest reading of the table:

- **Wins:** a clear 1-5% at M ≤ 32 (the weight-stream-bound regime) and 3-7% at M = 72-144.
- **Ties:** M = 33-64 is within about ±1%, where both kernels are near the tensor-core limit. The weakest cells are on
  34816x5120 at M = 33-64, at 0.95-0.96x Marlin.
- **Losses:** M = 192, 208, 256, 416 and 832 are 0.6-2.1% slower. **M = 544 is 6-7% slower**: its table entry covers
  513-576, and at 544 it picks a tile config that loses 10% on 34816x5120 (822 vs 744 µs).
- **cuBLAS:** at M = 1024 BF16 cuBLAS is 1.1% faster.
- **Reproducibility:** the tuned cells are in-sample. Between two runs about an hour apart on the same GPU, single totals moved by up
  to about 2-4 points at M ≥ 384.

Selected cells (µs, W4A16 vs Marlin; weight bandwidth at M = 1):

- 16384x5120: M = 1, 32.9 vs 34.0 (1.44 TB/s of weight bytes).
- 34816x5120: M = 1, 65.7 vs 67.1 (1.53 TB/s).
- 5120x17408: M = 96, 84.8 vs 97.7 (1.15x).

Across all 160 tuned cells, W4A16 is faster in 113. Per cell, Marlin time divided by W4A16 time ranges from 0.905 to
1.185.

**Untuned shapes** (4096x4096, 11008x4096, 4096x11008, 6144x4096; ratio = Marlin or cuBLAS time / W4A16 time, above 1
means W4A16 is faster):

| M band | Tuned path (maint heuristic) vs Marlin | Example path vs Marlin | Tuned path vs cuBLAS |
|---|---|---|---|
| 1-16 | 0.96-1.24 (median 1.07) | same kernels | 2.24-3.08 |
| 32-96 | 0.87-1.12 (median 0.99) | same kernels | 1.00-2.78 |
| 128-256 | 0.88-1.03 (median 0.96) | 0.68-1.03 (median 0.95) | 0.97-1.36 |
| 384-1024 | 0.75-1.07 (median 1.00) | 0.86-1.04 (median 0.94) | 0.90-1.19 |

- Untuned shapes are roughly at parity with Marlin, and below it in some cells. The worst cells are:
  - tuned path: 4096x4096 at M = 384 (0.75x);
  - example path: 4096x11008 at M = 192 (0.68x), where stream-K is 1.4x faster than the tile kernel.
- Both heuristics were fitted on 8 shapes on one GPU. Per-GPU tables are the intended way to do better.

## Notes for reviewers

- **Library placement** is the open question from #1592. If the maintainers want it in the library (for example an
  SM120 W4A16 GEMM op or intrinsics), the modules are ready to move. The split between example and maint follows the
  #2364 feedback.
- **Tuning is per GPU.** The table applies only on a GPU with 188 SMs; everything else uses the heuristic. An
  autotune hook could replace the table if preferred.
- **`tilelang.profiler.do_bench(backend="cudagraph")` misreports this call.** For example, at M = 384 on 4096x4096 it
  reports 293 µs, while `cupti` reports 76 µs and a graph-replay timer 72 µs. The cause may be the per-call
  `torch.empty` of the FP32 partials inside do_bench's graph plus its L2 flush; I have not investigated further. The
  example therefore defaults to `cupti`. I can open a separate issue with a repro.
- **Scalar grid size needs `T.assume`.** The stream-K grid size is a scalar `T.int32` argument bound to `T.Kernel(G)`.
  Without `T.assume(G > 0)`, every floor division by `G` gets sign fix-ups, and the kernel ran 1.1-1.5x slower
  (measured).
- **The kernel cache in `_kernel()` is deliberate.** A dict lookup costs 0.44 µs per call against 0.69 µs for the JIT
  factory's own cache path. That matters next to a 10-30 µs eager decode GEMM.
- **Known limits:**
  - The split-K tickets are per device, so launches on one device must not run concurrently on two streams.
  - The stream-K loader uses 32-bit offsets, so x and the packed weight must each be under 4 GiB (checked on the host).
  - N % 128 == 0 and K % 256 == 0 are required.
  - The first use of each (N, K, config) compiles JIT.
- **Overlaps:** there is a conceptual overlap with `example_dequant_gemm_bf16_fp4_hopper.py` (see Design) and with
  #2171's `examples/gemm_fp4/example_gemm_a8w4.py` (A8W4). No file overlaps an open PR. The NVFP4 reference helpers
  in `examples/dequantize_gemm/quantize/nvfp4.py` are imported, not edited, which avoids #3081 and #3099.
- **Follow-ups:**
  1. The W4A4 two-pass prefill PR comes next (item 3 of the #1592 plan): activations are quantized in two NVFP4 passes
     (value + residual) on the native `mxf4nvf4` tensor cores.
  2. Moving to the library, if requested.
  3. Tables for more GPUs, as hardware becomes available.
