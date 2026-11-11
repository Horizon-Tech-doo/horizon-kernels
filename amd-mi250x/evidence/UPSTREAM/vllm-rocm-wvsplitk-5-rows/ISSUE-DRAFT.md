# Draft issue text for vllm-project/vllm

**Title:** [ROCm][Performance] `wvSplitK` skinny GEMM is 3 to 8x slower than hipBLASLt at 3 and 5 activation rows on gfx90a (MI250X); decode at 5 concurrent requests is 14% slower than with `VLLM_ROCM_USE_SKINNY_GEMM=0`

## Summary

On ROCm, `rocm_unquantized_gemm_impl` (`vllm/model_executor/layers/utils.py`) sends every unquantized linear layer with 1 to 5 activation rows to the `wvSplitK` kernel (`m > 8 and 0 < n <= 5`). On an MI250X (gfx90a) that kernel is faster than `torch.nn.functional.linear` (hipBLASLt) at 1 and 4 rows, but several times slower at 3 and 5 rows:

| weight N x K (bf16) | rows | dispatch (wvSplitK) ms | F.linear ms | dispatch / F.linear |
|---|---|---|---|---|
| 17408 x 5120 | 1 | 0.153 | 0.189 | 0.81 |
| 17408 x 5120 | 2 | 0.223 | 0.189 | 1.18 |
| 17408 x 5120 | 3 | 0.619 | 0.190 | **3.27** |
| 17408 x 5120 | 4 | 0.160 | 0.190 | 0.84 |
| 17408 x 5120 | 5 | 1.566 | 0.191 | **8.22** |
| 17408 x 5120 | 6, 8 | 0.191 | 0.191 | 1.00 (dispatch already uses F.linear) |
| 5120 x 8704 | 1 / 2 / 3 / 4 | 0.070 / 0.074 / 0.087 / 0.088 | 0.102 to 0.103 | 0.68 / 0.73 / 0.85 / 0.86 |
| 5120 x 8704 | 5 | 0.324 | 0.103 | **3.14** |
| 124160 x 5120 (an LM head) | 1 / 4 | 1.128 / 1.135 | 1.379 / 1.386 | 0.82 / 0.82 |
| 124160 x 5120 | 2 | 1.655 | 1.382 | 1.20 |
| 124160 x 5120 | 3 | 4.623 | 1.385 | **3.34** |
| 124160 x 5120 | 5 | 11.371 | 1.392 | **8.17** |

(200 calls between two synchronisations, median of 7 rounds, arm order rotated; relative error against an fp32 product is 1.65e-3 to 1.68e-3 for both, so results are fine, only speed differs. `wvSplitK` called directly gives the same times as the dispatch at 1 to 5 rows and raises `Unsupported N value` from 6 rows.)

## Effect on a served model

Server: `vllm serve <27B hybrid model> --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88` on two MI250X GCDs, default flags otherwise. n concurrent streaming completions with 7.7k-token prompts, greedy, 96 output tokens; decode step = streams x decode-window seconds / tokens. Two server starts per arm, order A B B A in one allocation.

| concurrent requests | default, ms per decode step | `VLLM_ROCM_USE_SKINNY_GEMM=0`, ms | default / off |
|---|---|---|---|
| 1 | 57.92, 58.58 | 69.20, 69.65 | 0.84 |
| 2 | 67.62, 68.74 | 69.83, 69.46 | 0.98 |
| 4 | 63.98, 63.31 | 69.02, 69.49 | 0.92 |
| **5** | **79.56, 79.89** | 69.33, 70.22 | **1.14** |
| 6 | 70.09, 69.38 | 69.28, 69.85 | 1.00 |
| 8 | 69.71, 70.06 | 69.95, 69.67 | 1.00 |

So the skinny path helps at 1 and 4 requests and costs 14% at exactly 5. With the default CUDA-graph capture sizes (1, 2, 4, 8, 16, ...) the transformer layers of a 5-request decode batch are padded to 8 rows and never reach `wvSplitK`; the LM head in the sampling path runs outside the graph with the true 5 rows, and 11.4 ms against 1.4 ms for a 124160 x 5120 head matches the 10 ms seen per step. We did not measure 3 concurrent requests in the server (the standalone table predicts about +3 ms per step from the head), nor a server without CUDA graphs, where every layer would hit the 5-row case (about +140 ms per step from the standalone times of this model's layers).

## Expected behaviour

The dispatch should not pick a path that is slower than the fallback it replaces. On gfx90a that means not using `wvSplitK` at 3 and 5 rows (and at 2 rows for the shapes where it loses), or fixing the kernel for those row counts.

## Workaround

`VLLM_ROCM_USE_SKINNY_GEMM=0` removes the regression at 5 (and 3) rows, at the cost of the 16% and 8% the same kernels gain at 1 and 4 concurrent requests.

## Environment

- GPU: AMD Instinct MI250X (gfx90a), `multi_processor_count` 110 per GCD; LUMI-G, HPE Cray EX, SLES 15 SP6 kernel 6.4.0 (cray_shasta_c).
- Container: vLLM ROCm image `vllm 0.30.0+rocm723`, ROCm 7.2.3, `torch 2.12.0+git6bbd260` (HIP 7.2.53211), `triton 3.7.1`, `amd-aiter 0.1.21.post2`, run with Singularity.
- Not tested: gfx942 / gfx950, other ROCm or vLLM versions, fp16 weights, shapes other than those above.

## Reproduction

`repro_wvsplitk.py` (attached, needs only vLLM and one GPU):

```
python repro_wvsplitk.py                 # the three shapes above
python repro_wvsplitk.py 4096 4096       # any other N K
```

Output of the run behind the first table: `up-wvsplitk-23626809.out`.

## Further observations

- The slow row counts are the odd ones above 1 (3 and 5), at every shape we tried, which points at how the kernel splits rows across its fixed per-block row count rather than at one shape. We did not read the kernel source.
- At 6 to 32 rows the dispatch and `F.linear` agree within 0.3% at four of five shapes, but at 5120 x 3072 the dispatch reads 0.067 ms against 0.045 ms for `F.linear` on the same tensors (1.5x) in a separate run (`extra_gm_skinny.py`, job 23625791). We did not trace that; it may be a second, smaller issue (an `aiter` tuned-GEMM entry?); not verified.
