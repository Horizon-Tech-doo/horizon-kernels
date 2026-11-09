> **Status:** Opened as issue tile-ai/tilelang#3287 + PR #3288 (27-09-2026 08:10). This file is the local draft; the live text on GitHub is authoritative.

# Draft upstream PR 3: warp-specialized release point for moved non-TMA copies (not sent)

Local branch `horizon/upstream-fix-ws-release` in a worktree of `<src>/horizon-tilelang`
(worktree `wt-wsfix`). One commit, `0f93eeb8`, on upstream `main` `7a5f446f` (#3257; still
the tip of `main` on 27-09-2026). Written 27-09-2026.
The issue draft is `issue-draft-blockk128-rega.md` in this folder. Open the issue first, then put its number in
`Fixes #`.

Verification ran 27-09-2026 07:09-08:05 (CEST) on the server, with two wheels built from pure upstream trees (`git archive`
of `7a5f446f` and of the branch commit; no fork files). GPU steps ran under the GPU lock. The server scripts and logs
are in `<work>/w4a16pr/t5/` (`res/` has the logs).

Everything below the line is the PR text.

---

## Title

`[BugFix][CUDA] Keep warp-specialized stages until moved non-TMA copies are consumed`

## Summary

Fixes #<issue>.

A `T.Pipelined` loop compiled with warp specialization can release a pipeline stage to the producer while the consumer
is still reading part of it. This produces silently wrong results. It happens when a consumer-side synchronous
global->shared copy (for example, a tile whose rows are too narrow for TMA) is moved into a TMA group's producer prefix,
and the group's release point comes before the last read of that copy's buffer.

This PR moves the group's release point after that last read. Kernels without such a moved copy generate byte-identical
CUDA (checked on 27 upstream example and test kernels for sm_90a and sm_120a, see below).

## Root cause

In `src/cuda/transform/producer_consumer_ws.cc`, `ProducerConsumerWSRewriter::BuildWSBlock`:

1. Lines 1253-1286: each TMA group's consumer release (`arrive_insert_pos[g]`) is placed after the last consumer
   access of **its own TMA destination**. With A copied into a register fragment (`A_frag[i, k] = A_shared[i, k]`),
   the A group is released right after that copy.
2. Lines 1334-1416: consumer statements in front of a group's first wait that pass `IsProducerMovableLoopPrefixStmt`
   are moved into `producer_loop_prefix_stmts[g]`. This includes synchronous global->shared copies. In the producer
   they are emitted after that group's back-pressure wait (lines 1575-1599), so they write their multi-versioned stage
   under the A group's barrier.
3. `arrive_insert_pos` is never updated for the buffers written by the moved statements. The consumer arrives on the
   A group's back-pressure barrier before it reads the moved buffer (in the GEMM or the epilogue). The producer then
   overwrites that stage for iteration `ko + num_stages` while the current one is still being read: a write-after-read
   race.

Generated CUDA before the fix (fp16 GEMM with an 8-byte-row bias tile, 2 stages, abridged; sm_90a looks the same with
`wgmma_rs`):

```cuda
// producer
mbarrier[(ko & 1) + 4].wait(...);                                    // A-group back-pressure
*(float2*)(S_shared + (ko & 1) * 256 + ...) = *(float2*)(S + ...);   // moved synchronous copy
... tma_load(A_desc, mbarrier[ko & 1], A_shared + (ko & 1) * 8192, ...);
// consumer
mbarrier[ko_1 & 1].wait(...);
A_frag = A_shared[stage];
mbarrier[(ko_1 & 1) + 4].arrive();              // releases S_shared[stage] too
... mma ...
C_frag += S_shared[(ko_1 & 1) * 256 + ...];     // still reads S_shared[stage]
```

## Fix

After the prefix-move loop, for each group, collect the shared buffers written by its moved prefix statements (the copy
destination from `ExtractProducerWriteBufferData`, plus any shared `BufferStore`). Then raise `arrive_insert_pos[g]`
to just after the last non-moved consumer statement that accesses any of them. When the TMA barriers are merged, the
single merged group takes the maximum over all groups. If a group's prefix writes nothing, or its release already comes
after the last access, nothing changes.

With the fix, the consumer arrive moves below the last read:

```diff
       A_frag = A_shared[stage];
-      mbarrier[(ko_1 & 1) + 4].arrive();
       mbarrier[(ko_1 & 1) + 2].wait(...);
       ... mma ...
       mbarrier[(ko_1 & 1) + 6].arrive();
       C_frag += S_shared[...];
+      mbarrier[(ko_1 & 1) + 4].arrive();
```

A more aggressive alternative would be to put such copies in the prefix of a group whose release already covers
their buffer, which would avoid moving any release. That is a larger change to the prefix placement, so this PR keeps
the minimal, conservative fix.

## Tests

New tests in `testing/python/transform/test_tilelang_transform_producer_consumer_ws.py`:

- `test_tiled_ws_release_covers_moved_non_tma_copy` (`requires_cuda_compute_version(9, 0)`): an fp16 GEMM with A
  through a fragment and a per-row bias tile of two float32 per row (8-byte rows, so no TMA), 2 stages. It checks the
  generated source (the last back-pressure `arrive` in the consumer follows the last `S_shared` read, which holds on
  sm_90a and sm_120a codegen with the fix and fails without it), and checks the result bit-exactly (small integers).
- `test_tiled_ws_release_covers_moved_scale_copies_sm120_blockscaled` (`requires_cuda_compute_version_eq(12, 0)`): the
  SM120 NVFP4 block-scaled GEMM with A in a fragment and row-major scales at `block_K=128` (8-byte scale rows),
  checked bit-exactly with power-of-two scales.

### Repro matrix (RTX PRO 6000 Blackwell, SM120; M = N = 256, 128x128 tiles)

`FAIL` = tens of thousands of the 65,536 outputs wrong (max abs error 200-2,300), not rounding.

| Kernel | Config | `main` | This PR |
|---|---|---|---|
| fp16, A fragment, 8-byte-row bias | 2 stages | FAIL (64,960 wrong) | pass |
| | 0 stages / WS disabled | pass | pass |
| fp16, A fragment, 16-byte-row bias (TMA) | 2 stages | pass | pass |
| block-scaled, A fragment, `block_K` 64 / 128 / 192 | 1, 2, 3 stages (9 configs) | FAIL (all 9) | pass (all 9) |
| | 0 stages / WS disabled / A in shared | pass | pass |
| block-scaled, `block_K=128`, 2 stages | K = 384 and 4096; policy Square / FullCol; 256 threads | FAIL (all 5) | pass (all 5) |
| | K = 256 (no stage reuse) | pass | pass |
| block-scaled, `block_K=256` (TMA scales) | 0, 1, 2 stages; K = 4096 | pass | pass |

Totals over the 35 configurations: 15 fail on `main` and 0 with this PR; 19 pass on both. The remaining one
(`block_K=256` with 3 stages) exceeds SM120 shared memory on both builds.

### Upstream transform tests

`test_tilelang_transform_producer_consumer_ws.py`, `..._producer_consumer_ws_persistent.py`,
`..._pipeline_barrier_ownership.py` and `..._materialize_ws_schedule.py` from this branch, on SM120:

| | `main` | This PR |
|---|---|---|
| passed | 109 | 111 |
| failed | 4 | 2 |
| skipped | 3 (need sm_90) | 3 (need sm_90) |
| `test_tiled_ws_multi_versioned_wide_linear_smem_keeps_tma` | fail: smem 131072 > SM120 limit | same |
| `test_tiled_ws_wide_linear_smem_three_stages` | fail: smem 147456 > SM120 limit | same |
| `test_tiled_ws_release_covers_moved_non_tma_copy` (new) | fail: 45,564 / 65,536 wrong, source check fails | pass |
| `test_tiled_ws_release_covers_moved_scale_copies_sm120_blockscaled` (new) | fail | pass |

The two wide-linear failures are `Failed to set the allowed dynamic shared memory size` on this GPU (SM120 allows
about 99 KiB per block), and they fail the same way without this PR. I could not run the sm_90-only tests.

### Unaffected kernels: generated CUDA is unchanged

I lowered 27 kernels from upstream examples and tests with and without the patch (`tilelang.lower`, no device
compile), for sm_90a and for sm_120a (57 sources; the three SM120-only kernels only for sm_120a), and diffed the CUDA
source:

- **Byte-identical (54 of 57):** `examples/gemm` (`example_gemm`, non-persistent 3-stage GEMM), the four WS test GEMMs
  (2 and 3 stages, dual GEMM, windowed), the wide-linear reduction, `warp_specialize/..._softpipe_stage2`,
  `gemm_splitk`, `blocksparse_gemm`, `gemm_fp8` (e4m3 and 2xAcc), `deepseek_deepgemm` fp8 2xAcc (`block_N` 128 and
  64), flash attention (MHA fwd, causal fwd, GQA fwd with 2 stages, MHA bwd and its fwd), `deepseek_mla` decode, the
  dequant GEMMs (bf16/fp4, bf16/mxfp4 with and without bias, fp16/fp4, w4a8), the SM120 block-scaled example at
  `block_K` 256 and 128 (its scales go through cp.async, not a moved synchronous copy), and the new test's kernel with
  16-byte bias rows.
- **Changed (3 of 57):** the new fp16 test kernel (sm_90a and sm_120a) and the new block-scaled test kernel (sm_120a).
  These are the only kernels in the set whose producer branch contains a synchronous shared store, i.e. a moved
  non-TMA copy. In each, the diff is one line: the A group's `mbarrier[(ko_1 & 1) + 4].arrive()` moves from after the
  fragment copy to after the last read of the moved buffer. Nothing else changes.

### Performance of affected kernels

The later release means less producer run-ahead for those kernels. RTX PRO 6000 Blackwell (SM120),
M = N = K = 4096, base and fix alternated in separate processes, 3 rounds x median of 5 `do_bench` runs:

| Kernel | `main` (wrong results) | This PR | Change |
|---|---|---|---|
| fp16, A fragment, 8-byte-row bias, 2 stages | 675.0 µs | 690.7 µs | +2.3% |
| block-scaled, A fragment, `block_K=128`, 2 stages | 202.3 µs | 208.8 µs | +3.2% |
| block-scaled, A fragment, `block_K=128`, 3 stages | 201.7 µs | 205.0 µs | +1.6% |
| block-scaled, A fragment, `block_K=64`, 3 stages | 234.1 µs | 231.6 µs | -1.1% (noise) |

The baseline numbers come from the racy code, so they are not a correct result to compare against. With
`block_K=256` the scales use TMA and the bug does not apply.

## Relation to other PRs and issues

- #3279 (open) also touches the producer-prefix path. It adds a producer-side `tvm_storage_sync` before TMA
  publication when prefix statements write shared memory. That is the forward (read-after-write) edge. It does not move
  the consumer release, so it does not fix this write-after-read race. The two changes are in different hunks and
  #3279's `producer_consumer_ws.cc` hunk applies cleanly on top of this commit.
- #2443 (open) reports wrong results with a 1-D copy placed *before* the pipelined loop (prelude path). This PR does
  not change the prelude path, so it is a separate issue.

## Checklist

- clang-format 23.1.0 (the `.pre-commit-config.yaml` pin): no changes. ruff 0.16.6 check and format: clean.
  codespell: clean.
- `maint/scripts/audit_cpp_api_style.py --include-sources src/cuda/transform/producer_consumer_ws.cc`: the same 5
  pre-existing findings before and after (3 x TLCPP004, 2 x TLCPP001); none are in the new code.
