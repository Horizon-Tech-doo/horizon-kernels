> **Status:** Opened as tile-ai/tilelang#3286 (27-09-2026 04:46). This file is the local draft; the live text on GitHub is authoritative.

# Draft upstream PR: SM120 block-scaled register-A GEMM (not sent)

Local branch `horizon/upstream-sm120-rega-perf` in `<src>/horizon-tilelang`, one commit
`59ab7585` on upstream `main` `7a5f446f` (#3257, 26-09-2026; `git fetch upstream` on 27-09-2026 found nothing newer).
Written 27-09-2026.

---

## Title

`[CUDA] SM120 block-scaled register-A GEMM: unroll K atoms and load scales once per atom`

## Summary

Follow-up to #3257, which added `T.gemm_blockscaled` / `T.mma_gemm_blockscaled` with A taken from a register fragment on
SM120 (`is_gemm_rs`). Two lowering issues show up as soon as the A fragment spans more than one 64-wide K atom:

1. **Runtime K-atom index into the A fragment.** `GemmMMASm120BlockScaled.lower` iterated the K atoms of a call with
   `T.serial`, so every fragment access used a runtime index. nvcc then keeps the fragment in local memory (spills on the
   MMA path) whenever it does not unroll that loop itself, which happens in pipelined and warp-specialized kernels.
   The loop is now `T.unroll(..., explicit=True)`: every access has a compile-time index.
2. **Scales re-read from shared memory for every MMA.** `TensorCoreIntrinEmitterSM120.mma()` passed shared-memory
   pointers for SFA and SFB to each `mma.sync ... block_scale` (three shared loads per MMA pair). When both scale buffers
   are in shared memory it now loads one *compact selector package* per lane and K atom, the scheme the full-tile K-major
   path already uses: lanes `t` and `t ^ 1` of a quad hold the SFA rows of MMA atom rows `2g` / `2g + 1`, quad lane `q`
   holds the SFB columns of n8 block `4g + q`, and each MMA picks the owning lane with its scale thread-id operand. That
   is `ceil(warp_rows / 2)` SFA words and `ceil(n8_blocks / 4)` SFB words per lane per atom. Row-major and chunk-K-major
   scale layouts are both handled; fragment scales keep the existing path.

The MMA sequence and the scale value each MMA consumes are unchanged, so results are bit-identical to `main`. The
full-tile K-major path (`sf_layout="blockscaled_chunk_kmajor"`, used by `examples/gemm_sm120` and the maint benchmark)
is not touched.

## Changes

- `tilelang/cuda/op/gemm/gemm_mma_sm120.py`: the register-A branch unrolls the K-atom loop in TIR.
- `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`: `mma()` dispatches to a new
  `_mma_with_compact_scale_packages()` when SFA and SFB are both shared buffers.
- `testing/python/language/test_tilelang_language_nvf4_mma_block_scale.py`:
  - four new cases in `test_nvf4_mma_block_scale_fragments_and_odd_warps`: register A with K = 256 (FullRow), 192
    (Square) and 256 transposed (FullCol), plus shared A row-major with K = 192 (the generic `mma()` path); all exact
    against the float32 reference;
  - `test_nvf4_mma_block_scale_fragment_a_unrolls_k_atoms_and_packs_scales` (FullRow, Square): the generated CUDA has
    no K-atom loop and no MMA reads `SFA_shared` / `SFB_shared` directly.
- `maint/gemm/gemm_sm120/benchmark_sm120_nvfp4_fragment_a_gemm.py`: a small pipelined register-A benchmark (A staged
  through shared memory into a `(block_M, block_K)` fragment, B and row-major scales in shared memory), with `--verify`
  (bit-exact against float32) and `--dump-source`.

## Tests

On an RTX PRO 6000 Blackwell (SM120, driver 570 + CUDA 13 forward compatibility), TileLang built from source:

| | `main` (7a5f446f) | this PR |
|---|---|---|
| `test_tilelang_language_nvf4_mma_block_scale.py` + `test_tilelang_language_gemm_blockscaled.py` | 141 passed, 4 skipped, **2 failed** (the new codegen test, as intended: it detects the old lowering) | **143 passed, 4 skipped** |
| Outputs of the existing register-A / shared-A test kernels (K = 128..256, FullRow / Square / FullCol, fragment and shared scales), `main` vs this PR | | bit-identical (6 of 6) |

`bash format.sh --files <changed files>`: ruff check and ruff format (v0.16.6) pass; no C++ changed.

## Benchmarks

Two interleaved rounds per build, `do_bench` (mean), TFLOPS. The first two rows use code paths this PR does not touch.

| Benchmark | Shape (M = N = K) | `main` | this PR | Change |
|---|---|---|---|---|
| `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` (full-tile K-major) | 4096 | 928.2 | 926.5 | −0.2% (noise) |
| same | 8192 | 1023.6 | 1027.6 | +0.4% (noise) |
| `maint/.../benchmark_sm120_nvfp4_blockscaled_gemm.py` (persistent, K-major) | 4096 | 949.7 | 941.9 | −0.8% (noise) |
| same | 8192 | 1048.3 | 1050.7 | +0.2% (noise) |
| `maint/.../benchmark_sm120_nvfp4_fragment_a_gemm.py`, block 128×128×256, FullRow | 4096 | 716.0 | 735.2 | **+2.7%** |
| same, FullRow | 8192 | 825.0 | 841.4 | **+2.0%** |
| same, Square | 8192 | 664.9 | 738.6 | **+11.1%** |
| same, FullCol | 8192 | 348.5 | 526.7 | **+51%** |

ptxas resource usage for the register-A benchmark kernels (`-O3`, sm_120a):

| Config | `main`: spill stores / loads, LDL+STL in SASS | this PR |
|---|---|---|
| FullRow, block_K 256 | 4 B / 4 B, 2 | 0 / 0, 0 |
| Square, block_K 256 | 116 B / 84 B, 42 | 0 / 0, 0 |
| FullCol, block_K 256 | 664 B / 484 B, 215 | 320 B / 160 B, 80 (register pressure of an 8-row-atom A fragment, not indexing) |

Not addressed here: with `block_K = 128` and row-major shared scales the register-A benchmark does not verify on `main`
either (same error before and after this change); it needs a separate look.

## Notes for reviewers

- Overlap: #3081 (`a_packed_words=True` register fragments for block-scaled MMA) and #3099 touch the same two files.
  This change is local to the register-A branch of `lower()` and to `mma()`; a rebase should be mechanical.
- A B-fragment ping-pong in the register-A loop (load atom `k + 1`'s B before atom `k`'s MMAs) was tried as well and
  left out: no gain on the benchmark above (FullRow/Square within 1%, FullCol −7%).
