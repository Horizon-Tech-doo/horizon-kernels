# SM120 re-measurement, 11 November 2026 (kernel level)

One NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (compute capability 12.0), host driver 570.124.04, CUDA 13.0
user libraries, PyTorch 2.13.0+cu130, vLLM 0.30.0, FlashInfer 0.6.18.post1, Triton 3.7.1, TileLang 0.1.15 (own build),
Python 3.12.3. The GPU sits behind a time-slicing layer; a co-resident process was loaded but asleep and did no work
during any cell (its request counters did not move in any of the 1,204 timed cell records). Board power limit 300 W.

## Method as run

- One process per block; arms `new`, vendor arm(s) and `ctl` (the same kernel objects as `new`, second CUDA graph) on the
  same buffers; 9 cycles per cell, arm order rotated per cycle; a forward job and a reverse-order job; all per-cycle values kept.
- Two input seeds (988 and 989), each with a forward and a reverse job: **four jobs per race**.
- Noise bound per job = max over its cells of |median(ctl)/median(new) - 1|. WIN / LOSE = above / below the bound in
  every job; everything else is "inside/mixed".
- Timing: device events around 50 CUDA-graph replays after 10 warm replays (the two prefill races: 12/4, 19/6, 3/1 and 6/2 timed/warm replays), graphs captured in thread-local mode with
  retry (0 retries were needed). Working set > 513 MiB of rotated weights, caches or state per arm.
- Correctness per cell before timing: fp64 reference (or bit-identity for the gated-delta kernels), non-finite count,
  repeat equality, graph replay after poisoning == eager.
- Energy: separate 5 s tight-loop pass per arm on a subset of cells, NVML energy counter, 20 Hz power samples. Power is
  for the whole board. Idle power (process idle, measured for 5 s at the start and end of every block) was 83-115 W per
  job mean (77-96 W at a cool block start; block-end idle records span 86-136 W); "net" subtracts the job's mean idle power.
  Most arms run at the 300 W limit, so energy ratios follow time ratios; graphics clock moved between 1,185 and 2,355 MHz
  and temperature between 47 and 86 C during the races.

## Results (ratio = vendor time / our time, higher is better for us; range over 4 jobs)

| Race | Cells | Ratio range | WIN / LOSE / inside in all 4 jobs | Largest noise bound | Correctness |
|---|---|---|---|---|---|
| W4A16 NVFP4 decode GEMM vs Marlin, 5 tuned layer shapes x 32 row counts | 160 | 0.923-1.170 (median 1.00) | 72 / 55 / 33 | 1.34% | fp64 rel L2: ours 1.63-1.70e-3 and Marlin 1.63-1.70e-3, 0 failures in 640 checks each |
| same vs BF16 cuBLAS | 160 | 0.986-3.301 (median 1.49) | 159 / 0 / 1 | 1.34% | cuBLAS on BF16-rounded weights: 2.30-5.05e-3, over the 2.5e-3 limit in 424 of 640 checks (rounding of the weights, not the GEMM) |
| W4A16 vs Marlin, 6 held-out shapes x 8 row counts | 48 | 0.746-1.156 (median 0.97) | 12 / 25 / 11 | 1.76% | ours and Marlin 1.63-1.68e-3, 0 failures in 192 checks each |
| same vs BF16 cuBLAS | 48 | 0.861-3.016 (median 1.32) | 36 / 7 / 5 | 1.76% | cuBLAS BF16: 2.29-4.97e-3, over the limit in 108 of 192 |
| Gated-delta decode, 16-bit state vs vendor CUDA kernel (fp32 state) | 6 | 1.422-2.058 (median 1.93) | 6 / 0 / 0 | 0.58% | bit-identity gates pass (fp32 and bf16 paths identical to the vendor kernel; fp16 = rounded fp32) |
| same vs vendor kernel with BF16 state | 6 | 0.995-1.385 (median 1.07) | 4 / 0 / 2 | 0.58% | same |
| Gated-delta decode, checkpoint + replay ring vs vendor kernel (fp32 state) | 6 | 0.906-2.642 (median 1.87) | 5 / 1 / 0 | 0.29% | same gates, incl. ring vs chained vendor kernel |
| same vs vendor kernel with BF16 state | 6 | 0.761-1.395 (median 1.01) | 3 / 3 / 0 | 0.29% | same |
| Fused MoE block (skinny gate, route kernel, stable alignment) vs stock routing around Marlin-MoE, synthetic layers | 8 | 1.038-2.032 (median 1.26) | 5 / 0 / 0, 3 not certified | 0.60% | accuracy rule passes in all; routing ids equal to stock in 100% of rows at 1-8 and 64 tokens; at 128 tokens 97.7% (seed 988) and 99.2% (seed 989), at 32 tokens 96.9% on seed 988 only, at 16 tokens 93.8% on seed 989 only: those three cells are not certified |
| FP8 output projection (W8A16) vs BF16 cuBLAS, 248,320 x 5,120 | 10 | 1.031-1.913 (median 1.73) | 10 / 0 / 0 | 0.39% | both arms 1.65-1.66e-3 against the dequantised product |
| FP8 decode attention vs FlashInfer XQA, 1k-128k context | 28 | 0.314-0.805 (median 0.62) | 0 / 28 / 0 | 9.82% | ours 2.15-2.35e-3 (0 of 112 over the limit); XQA 2.35-3.18e-3 (82 of 112 over) |
| 4-bit KV decode attention vs XQA on FP8 | 28 | 0.423-1.638 (median 1.04) | 14 / 14 / 0 | 0.80% | ours 2.01-2.29e-3 (0 of 112 over); XQA 1.92-2.64e-3 (48 of 112 over) |
| Short ragged prefill attention vs FlashInfer FA2 | 2 | 0.277-0.386; best variant 0.555-0.868 | 0 / 2 / 0 | 0.02% | ours 2.12-2.18e-3; FA2 2.69-2.73e-3 (8 of 8 over) |
| Prefill attention 1k-16k keys vs FA2, graph replay | 5 | 0.416-0.534; best variant 0.525-0.579 | 0 / 5 / 0 | 0.03% | ours 1.89-2.31e-3; FA2 2.20-2.82e-3 (12 of 20 over) |

Row-count detail, W4A16 vs Marlin on the tuned shapes (5 shapes per row count, seed 988, both orders): at 1, 4 and 8
rows we lose in 4 of 5 shapes (0.923-1.021); at 16, 24 and 32 rows we win in 4 of 5 and at 17 rows in 2 of 5 (0.994-1.082); at 72 and 96 rows we win
in 5 of 5 (1.034-1.157); at 256, 416, 544 and 832 rows we lose in 4 of 5 (0.935-1.014); 1,024 rows 4 wins of 5
(1.004-1.059). 4-bit KV decode: the 14 wins are one request at 32k and 128k with one query token, 8 requests from 4k with one query
token and from 32k with four, and 32 requests at every context with one query token and from 16k with four; all cells with 4 query tokens up to 4k context lose (0.423-0.952).

The FP8 decode attention noise bound is large because at 8 requests and 32k-128k context the control arm differed from
the new arm by up to 9.8% in the reverse jobs (both seeds); every cell is still a loss beyond it.

## Energy per call (vendor J / our J, gross; range over cells, orders and seeds; energy noise bound <= 3.8%)

| Race | Energy cells | Gross ratio | Net ratio (idle subtracted) | Example, our / vendor |
|---|---|---|---|---|
| W4A16 tuned vs Marlin | 10 | 0.908-1.083 | 0.918-1.069 | 10.82 / 10.88 mJ at 16384x5120, 1 row |
| W4A16 tuned vs cuBLAS BF16 | 10 | 2.58-3.36 | 2.58-3.34 | 10.82 / 34.65 mJ, same cell |
| W4A16 held-out vs Marlin | 4 | 0.855-1.161 | 0.856-1.150 | 2.97 / 3.43 mJ at 4096x4096, 1 row |
| Gated-delta 16-bit state vs vendor fp32 | 6 | 1.33-2.06 | 1.25-2.10 | 1.96 / 2.63 mJ at 1 request x 2 tokens |
| Gated-delta ring vs vendor fp32 | 6 | 1.08-2.68 | 1.12-2.64 | 2.43 / 2.64 mJ, same shape |
| MoE block vs stock | 4 | 1.02-1.51 | 1.03-1.66 | 5.80 / 8.52 mJ at 1 token |
| Output projection vs cuBLAS | 3 | 1.03-1.78 | 1.05-1.76 | 279 / 486 mJ at 1 row |
| FP8 decode attention vs XQA | 4 | 0.387-0.808 | 0.535-0.805 | 3.93 / 1.54 mJ at 1 request, 1k |
| 4-bit KV decode vs XQA | 4 | 0.917-1.390 | 0.985-1.363 | 43.6 / 60.6 mJ at 1 request, 128k |
| Short prefill vs FA2 | 2 | 0.283-0.642 | 0.303-1.26 (low-power cells; net is unreliable here) | 3.31 / 0.96 mJ |
| Long prefill vs FA2 | 1 | 0.529-0.552 | 0.583-0.627 | 367 / 200 mJ at 1,024 queries on 4,096 keys |

## Not tested

Grouped W4A16 MoE GEMM; tuned 4-bit serving kernels and cache write; caches or activations captured from a real model run; the stock
output head (top-1 agreement); stock TileLang; token rate inside a server; a GPU without the time-slicing layer; the
vendor's native FP4 GEMMs (they do not load on this driver); hardware counters.
