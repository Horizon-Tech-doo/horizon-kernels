> **Status:** Posted as tile-ai/tilelang#3287 (27-09-2026 08:10). This file is the local draft; the live text on GitHub is authoritative.

# [BUG][CUDA] Warp-specialized pipeline releases a stage before a non-TMA shared copy is consumed (wrong results with a register A operand)

<!-- Local draft, 27-09-2026. Not posted. -->

### Required prerequisites

- [x] I have read the documentation <https://tilelang.com>.
- [x] I have searched the [Issue Tracker](https://github.com/tile-ai/tilelang/issues) that this hasn't already been reported. (comment there if it has.)

### What version of TileLang are you using?

0.1.14, built from source at `7a5f446f` (current `main`).

### System information

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, `sm_120a`)
- Driver 570.124.04 with the CUDA 13 forward-compatibility libraries (libcuda 580.178.04)
- CUDA toolkit 13.0 (nvcc V13.0.88), PyTorch 2.13.0+cu130, Python 3.12.3

### Problem description

A `T.Pipelined` loop (`num_stages >= 1`) produces silently wrong results when all of the following hold:

1. one operand is loaded by TMA and then first read by a copy into a register fragment (`A_frag[i, k] = A_shared[i, k]`),
2. the same loop body has a second, small global->shared `T.copy` that cannot use TMA and is not a `cp.async` copy. For example, a tile whose rows are only 8 bytes (two `uint32` / `float32` per row) is lowered to plain synchronous vector stores,
3. that small shared buffer is read later in the body than the fragment copy (by the GEMM or an epilogue).

This came up with the SM120 block-scaled GEMM (`T.gemm_blockscaled` with A in a fragment and row-major packed scales in shared memory). With `block_K = 128`, each row of the scale tile is `128 / 64 = 2` words (8 bytes), and the result is wrong for any K that needs three or more K-blocks. `block_K = 256` gives 16-byte rows, uses TMA and is exact. However, the bug is not specific to block-scaled GEMM: the plain fp16 repro below fails the same way.

The cause is in `src/cuda/transform/producer_consumer_ws.cc` (`ProducerConsumerWSRewriter::BuildWSBlock`):

- Each TMA producer group gets its consumer release point (`arrive_insert_pos`) from the last consumer access of **its own TMA destination** only (lines 1256-1286). For the A group, that access is the fragment copy, so the back-pressure `arrive` is placed right after `A_frag = A_shared`.
- Then consumer statements in front of a group's first wait that pass `IsProducerMovableLoopPrefixStmt` are moved into that group's producer prefix (lines 1334-1416). This includes the synchronous (non-TMA, non-cp.async) global->shared copies, and the moved copies are emitted after that group's back-pressure wait (line 1547). So the scale / bias copy now writes its multi-versioned stage under the **A group's** back-pressure barrier.
- `arrive_insert_pos` is not updated for the buffers the moved statements write. The consumer therefore releases the stage, and the producer overwrites the scale / bias stage for iteration `ko + num_stages` while the consumer still reads the current one in the GEMM or epilogue. This is a write-after-read race on the moved buffer.

Generated CUDA for the fp16 repro (`block_K = 64`, 2 stages, abridged):

```cuda
// producer (threads 0..127)
mbarrier[(ko & 1) + 4].wait(...);                  // A-group back-pressure
*(float2*)(S_shared + (ko & 1) * 256 + ...) = *(float2*)(S + ...);   // moved sync copy
tl::__sync_thread_partial(3, 128);
... tma_load(A_desc, mbarrier[ko & 1], A_shared + (ko & 1) * 8192, ...);
// consumer (threads 128..255)
mbarrier[ko_1 & 1].wait(...);
A_frag = A_shared[stage];                            // last access of A_shared
mbarrier[(ko_1 & 1) + 4].arrive();                 // <-- releases S_shared[stage] too early
mbarrier[(ko_1 & 1) + 2].wait(...);
mma_sync(...);                                     
mbarrier[(ko_1 & 1) + 6].arrive();
C_frag += S_shared[(ko_1 & 1) * 256 + ...];        // still reading S_shared[stage]
```

The SM120 block-scaled kernel has exactly the same shape, with `SFA_shared` / `SFB_shared` read by `sm120_mma_sync_blockscaled` after the early arrive.

### Reproducible example code

```python
import torch
import tilelang
import tilelang.language as T


def kernel(M, N, K, block_M, block_N, block_K, num_stages, bias_words):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        S: T.Tensor((M, (K // block_K) * bias_words), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_N, block_K), T.float16)
            S_shared = T.alloc_shared((block_M, bias_words), T.float32)
            A_frag = T.alloc_fragment((block_M, block_K), T.float16)
            C_frag = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_frag)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.copy(S[by * block_M, ko * bias_words], S_shared)  # 8-byte rows: no TMA
                for i, k in T.Parallel(block_M, block_K):
                    A_frag[i, k] = A_shared[i, k]
                T.gemm(A_frag, B_shared, C_frag, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_frag[i, j] += S_shared[i, 0]
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


M = N = 256
K, block_K, bias_words = 1024, 64, 2
k = tilelang.compile(kernel(M, N, K, 128, 128, block_K, 2, bias_words), out_idx=[3], target="cuda")
a = torch.randint(-2, 3, (M, K), device="cuda").half()
b = torch.randint(-2, 3, (N, K), device="cuda").half()
s = torch.randint(-64, 64, (M, (K // block_K) * bias_words), device="cuda").float()
ref = a.float() @ b.float().T + s[:, 0::bias_words].sum(1, keepdim=True)
torch.testing.assert_close(k(a, b, s), ref, rtol=0, atol=0)  # fails: 65216 / 65536 elements differ
```

All values are small integers, so the float32 reference is exact.

### Traceback

No error: the output is wrong. `AssertionError: Tensor-likes are not equal! Mismatched elements: 65216 / 65536`.

### Expected behavior

The result is bit-exact, as it is with `num_stages=0`, with `tl.disable_warp_specialized=True`, or when the small copy is TMA-eligible (`bias_words=4`).

### Failure matrix (SM120, M = N = 256, block 128x128, 128 threads unless noted)

`FAIL` means a large fraction of the 65,536 outputs is wrong (not rounding: max abs error 200-2000).

| Kernel | Variant | Result |
|---|---|---|
| fp16, A fragment, `bias_words=2` (8-byte rows) | `num_stages=2` | **FAIL** |
| | `num_stages=0` | pass |
| | `num_stages=2`, `tl.disable_warp_specialized` | pass |
| fp16, A fragment, `bias_words=4` (16-byte rows, TMA) | `num_stages=2` | pass |
| block-scaled, A fragment, row-major scales, `block_K=64` / `128` / `192` (4 / 8 / 12-byte scale rows) | `num_stages` 1, 2, 3 | **FAIL** |
| | `num_stages=0` | pass |
| | `num_stages=2`, `tl.disable_warp_specialized` | pass |
| | A in shared memory (no fragment), `num_stages=2` | pass (release follows the GEMM) |
| block-scaled, A fragment, `block_K=256` (16-byte scale rows, TMA) | `num_stages` 0, 1, 2 | pass |
| block-scaled, A fragment, `block_K=128`, `num_stages=2` | K = 256 (2 K-blocks, no stage reuse) | pass |
| | K = 384, 1024, 4096 | **FAIL** |
| | policy `FullRow` / `Square` / `FullCol`, 128 or 256 threads | **FAIL** (all) |
| | scales all equal (overwrite writes identical data) | pass |
| | scale / A copies written as `T.Parallel` loops instead of `T.copy` (no TMA, no WS) | pass |

### Suggested fix

After `producer_loop_prefix_stmts` has been computed, extend each group's `arrive_insert_pos` so that it also covers the last (not moved) consumer access of every shared buffer written by that group's moved prefix statements. When the TMA barriers are merged, take the maximum across all groups. A local patch that does this (about 45 lines, placed after the prefix-move loop in `BuildWSBlock`) makes every failing configuration in the matrix above exact: the block-scaled GEMM with `block_K` 64 / 128 / 192, stages 1-3, all three policies, 128 and 256 threads and K up to 4096, plus the fp16 repro. The consumer `arrive` for the A group now comes after the last `S_shared` / `SFA_shared` read. Configurations that already passed still pass. With and without the patch, `testing/python/transform/test_tilelang_transform_producer_consumer_ws.py`, `..._producer_consumer_ws_persistent.py`, `..._pipeline_barrier_ownership.py` and `..._materialize_ws_schedule.py` give 109 passed, 3 skipped and the same 2 failures on this GPU: `test_tiled_ws_multi_versioned_wide_linear_smem_keeps_tma` and `test_tiled_ws_wide_linear_smem_three_stages` fail with `Failed to set the allowed dynamic shared memory size` (131072 and 147456 bytes, more than SM120 allows). Lowering 27 upstream example and test kernels (GEMM pipelines, fp8 GEMMs, flash attention, MLA decode, dequant GEMMs, the SM120 block-scaled example) for sm_90a and sm_120a gives byte-identical CUDA with the patch; only kernels with a moved synchronous copy change, by one moved `arrive`. The sm_90a code for the fp16 repro has the same early `arrive` (with `wgmma_rs`), so Hopper should be affected too, although I could only run it on SM120.

<details><summary>Patch (against 7a5f446f)</summary>

```diff
diff --git a/src/cuda/transform/producer_consumer_ws.cc b/src/cuda/transform/producer_consumer_ws.cc
index 5c29da76..e7b57b9c 100644
--- a/src/cuda/transform/producer_consumer_ws.cc
+++ b/src/cuda/transform/producer_consumer_ws.cc
@@ -1415,6 +1415,56 @@ private:
       compute_cursor = wait_pos;
     }
 
+    // Statements moved into a group's producer prefix run under that group's
+    // back-pressure wait, so the group's consumer release must also cover the
+    // last consumer access of every shared buffer they write. Otherwise the
+    // producer can overwrite that stage while the consumer still reads it
+    // (e.g. a synchronous scale copy moved in front of a TMA A load whose
+    // release is placed right after A is copied to registers).
+    for (int ti = 0; ti < num_producer_groups; ++ti) {
+      std::vector<Var> prefix_written;
+      auto add_written = [&](const Var &data) {
+        for (const auto &v : prefix_written) {
+          if (v.same_as(data))
+            return;
+        }
+        prefix_written.push_back(data);
+      };
+      for (const auto &stmt : producer_loop_prefix_stmts[ti]) {
+        if (Optional<Var> w = ExtractProducerWriteBufferData(stmt)) {
+          add_written(w.value());
+        }
+        PostOrderVisit(stmt, [&](const ObjectRef &obj) {
+          if (const auto *store = obj.as<BufferStoreNode>()) {
+            if (IsSharedBuffer(store->buffer)) {
+              add_written(store->buffer->data);
+            }
+          }
+        });
+      }
+      for (const auto &data : prefix_written) {
+        for (int ci = static_cast<int>(consumer_compute_stmts.size()) - 1;
+             ci >= 0; --ci) {
+          if (moved_compute_stmts[ci]) {
+            continue;
+          }
+          BufferDataAccessInfo access = AnalyzeBufferDataAccess(
+              consumer_compute_stmts[ci], data, buffer_data_to_buffer_);
+          if (access.HasAnyAccess()) {
+            arrive_insert_pos[ti] = std::max(arrive_insert_pos[ti], ci + 1);
+            break;
+          }
+        }
+      }
+    }
+    if (can_merge_tma_barriers) {
+      int merged_arrive = arrive_insert_pos[0];
+      for (int g = 1; g < num_producer_groups; ++g) {
+        merged_arrive = std::max(merged_arrive, arrive_insert_pos[g]);
+      }
+      arrive_insert_pos[0] = merged_arrive;
+    }
+
     bool producer_needs_full_thread_extent = false;
     for (size_t i = 0;
          i < flat_stmts.size() && !producer_needs_full_thread_extent; ++i) {
```

</details>

A more conservative alternative is to not move a synchronous copy into a group's prefix when its destination is read after that group's release point.

### Additional context

The block-scaled benchmark in #3286 (`maint/gemm/gemm_sm120/benchmark_sm120_nvfp4_fragment_a_gemm.py`) uses `block_K=256` by default. With `--block-k 128` its `--verify` check fails on `main` as well, with or without that PR, and the cause is this bug.

#3279 touches the same prefix path. It adds a producer-side `tvm_storage_sync` before TMA publication when prefix statements write shared memory, which is the forward (read-after-write) edge, but it does not move the consumer release point. As far as I can tell it does not fix this issue.
