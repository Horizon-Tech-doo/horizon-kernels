# G1: bf16 GEMM (`gemm_bf16` through `horizon_lumi_gemm`) in a served 27B, time to first token and energy (11 Nov 2026)

Labels: [observed], [computed], [inferred]. Ratio convention: **plugin / stock, below 1 = plugin faster (or less energy).**

**Result [observed, two whole-node jobs in opposite arm order per setting, identical-code controls in every job, server alone on its card].**

| Setting | Arm | 1,419 tokens | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 |
|---|---|---|---|---|---|---|---|
| one-step prefill (`--max-num-batched-tokens 16384`) | GEMM plugin | 1.011 / 0.998 | 0.940 / 0.933 | 0.942 / 0.940 | 0.932 / 0.931 | **0.921 / 0.920** | 0.986 / 0.981 |
| one-step prefill | GEMM + prefill-attention plugins | 1.006 / 0.999 | 0.935 / 0.931 | 0.925 / 0.927 | 0.907 / 0.911 | **0.876 / 0.879** | 0.985 / 0.981 |
| vLLM defaults (1568-token chunks) | GEMM plugin, per-shape tiles | 0.985 / 0.988 | 0.973 / 0.972 | 0.968 / 0.966 | 0.968 / 0.968 | **0.969 / 0.969** | 0.994 / 0.994 |
| vLLM defaults | GEMM + prefill-attention plugins | 0.992 / 0.990 | 0.977 / 0.973 | 0.969 / 0.970 | 0.971 / 0.969 | 0.972 / 0.974 | 1.005 / 0.997 |

- **One-step prefill: our bf16 GEMM shortens time to first token by 5.8 to 8.0% for prompts of 2.9k to 14k tokens, and by 6.5 to 12.4% together with our prefill attention.** Every one of these 16 readings is outside its job's noise (at most 0.6%).
- **vLLM's defaults: 2.7 to 3.4% for 2.9k to 14k tokens and 1.2 to 1.5% at 1.4k**, all 10 readings outside noise (at most 0.6%). The attention plugin adds nothing there (P1).
- At 1,419 tokens with one-step prefill the plugin has nothing to take (the steps are 784 and 635 rows, below its 1,024-row floor). It reads +1.1% (GEMM) and +0.6% (both plugins) in job 23625074, outside noise, 5 and 3 ms; -0.2% and -0.1% in job 23625075.
- **Energy of the card per prefill cell, one-step prefill:** GEMM plugin 0.924 / 0.947 at 14,201 tokens, 0.937 / 0.943 at 7,719, 0.946 / 0.963 at 5,679; both plugins 0.887 / 0.898, 0.918 / 0.927, 0.917 / 0.931. All twelve are outside the arms' spread. At vLLM's defaults energy reads 0.94 to 0.98 for 5.7k to 14k tokens with 3 of 6 readings outside the spread, so the saving is not established there.
- Decode is not touched: the plugin returns to vLLM's path below 1,024 rows. Decode step at 1 stream 55.6 to 57.5 ms in every arm of every job; at 8 streams 68.0 to 71.2 ms (table below).
- Baseline is vLLM's default attention backend and its own dense-linear path (torch `F.linear`, hipBLASLt). Nothing here is compared against the stock `--attention-backend TRITON_ATTN` flag, which changes attention and not the GEMMs.

## What ran

- **Kernel.** `tilelang/gemm_bf16.py` 48701f59: the bf16 MFMA form (`v_mfma_f32_16x16x16bf16_1k`) of `gemm_prefetch.py`'s hand kernel (04eece68, not edited), fp32 accumulate, bf16 in and out, no casts and no weight copy. Rows form (`gemm_hand_rows`, XOR-swizzled LDS, partial waits): the row count need not be a tile multiple.
- **Plugin.** `horizon_lumi_gemm` (a `vllm.general_plugins` entry point; does nothing unless `HLG_ENABLE=1`). On ROCm every unquantized linear layer calls `utils.rocm_unquantized_gemm`, which calls a custom op that torch.compile treats as opaque. The plugin replaces that function, before the model is built, by one calling its own custom op with the same signature and fake implementation; its body runs our kernel when x is 2-D contiguous bf16 with at least `HLG_MIN` rows, the weight shape is one of the five large ones, there is no bias, no graph capture is active and the row count has a compiled kernel; otherwise vLLM's own implementation. Version 1 (aef5c3e4): one tile, 224x256. Version 2 (1e6e7173): tile and schedule per (rows, shape), or "leave to vLLM", through `HLG_FORMS`.
- **Settings.**
  - One-step: `--max-num-batched-tokens 16384` in every arm; plugin version 1; kernels for 2352, 4704, 5488, 7056, 9408, 14112 rows (the first steps of the client's prompts) x 5 shapes; `HLG_MIN` 1024.
  - vLLM defaults: plugin version 2; 1568 rows: 224x224 (gate_up), 160x256 (in_proj_qkvz and 5120 x 3072), 224x160 (qkv), down_proj left to vLLM; 784 rows: 160x256 (down), 224x160 (qkvz), 128x128 (qkv), the other two left to vLLM; `HLG_MIN` 512. Tiles from the sweep in `evidence/PROFILE/` (job 23624491).
- **Arms**, each a fresh server start: `s_rocm` stock (no plugin on the path); `g_rocm` + `horizon_lumi_gemm`; `fg_rocm` + `horizon_lumi_gemm` + `horizon_lumi_prefill` (b08c95e5, as in P1); `gv_rocm` = g with `HLG_VERIFY=2` (not a timing arm).
- **Load.** P1's client: synthetic prompts, greedy, 8 output tokens, a unique tag per prompt; one prompt at a time at 1,419 to 14,201 tokens (8 timed each) and 4 at once at 2,859. Then `pf_client.py` decode cells: 1 and 8 streams of 7.7k tokens, 96 output tokens.
- **Jobs [sacct, all COMPLETED; every arm that started has 6 timed cells of 6 and `client_done`].**

  | Job | setting | node type | arm order | elapsed s |
  |---|---|---|---|---|
  | 23625074 | one-step | whole standard-g node, server on GCDs 0 and 1 | gv s g fg fg g s | 4,319 |
  | 23625075 | one-step | whole node | fg g s s g fg | 3,633 |
  | 23625420 | defaults | whole node | gv s g fg fg g s | 4,298 |
  | 23625421 | defaults | whole node | fg g s s g fg | 3,644 |
  | 23625076 | one-step | small-g, 2 GCDs on two shared cards | gv s g g s (the first g start failed, one g repetition) | 2,990 |

  On the whole-node jobs the sampler sees 8 GCDs; only GCD 0's card draws power (about 400 W hwmon, 410 to 430 W on its counter) and the other three cards idle at 88 to 101 W in every arm, so the two GCDs are the two dies of one card and no other job shares it [observed].
- **Keep rule.** Two jobs in opposite arm order; in each, every arm twice (identical-code control); spread = (max - min) / median of an arm's two repetition medians; a reading counts when |ratio - 1| exceeds the larger spread of its two arms.

## Time to first token, s [observed; all repetitions in `tables-<job>.md`]

One-step prefill, job 23625074 / job 23625075, stock repetitions then plugin repetitions:

| Cell | 23625074 stock | GEMM | GEMM + attention | 23625075 stock | GEMM | GEMM + attention |
|---|---|---|---|---|---|---|
| 1,419 | 0.4542, 0.4546 | 0.4578, 0.4605 | 0.4559, 0.4584 | 0.4550, 0.4555 | 0.4533, 0.4557 | 0.4544, 0.4548 |
| 2,859 | 0.8818, 0.8806 | 0.8274, 0.8288 | 0.8235, 0.8250 | 0.8936, 0.8947 | 0.8332, 0.8348 | 0.8320, 0.8322 |
| 5,679 | 1.6626, 1.6598 | 1.5638, 1.5654 | 1.5364, 1.5371 | 1.6740, 1.6749 | 1.5730, 1.5746 | 1.5517, 1.5527 |
| 7,719 | 2.2398, 2.2340 | 2.0813, 2.0865 | 2.0294, 2.0294 | 2.2518, 2.2521 | 2.0981, 2.0962 | 2.0511, 2.0500 |
| 14,201 | 4.2379, 4.2328 | 3.9027, 3.9022 | 3.7063, 3.7108 | 4.2667, 4.2651 | 3.9246, 3.9225 | 3.7516, 3.7440 |
| 4 x 2,859 | 3.3076, 3.2953 | 3.2564, 3.2553 | 3.2504, 3.2505 | 3.3296, 3.3260 | 3.2663, 3.2658 | 3.2668, 3.2645 |

vLLM defaults, job 23625420 / job 23625421:

| Cell | 23625420 stock | GEMM | GEMM + attention | 23625421 stock | GEMM | GEMM + attention |
|---|---|---|---|---|---|---|
| 1,419 | 0.4460, 0.4485 | 0.4403, 0.4406 | 0.4425, 0.4447 | 0.4562, 0.4557 | 0.4499, 0.4514 | 0.4500, 0.4524 |
| 2,859 | 0.8723, 0.8731 | 0.8491, 0.8499 | 0.8519, 0.8531 | 0.8931, 0.8934 | 0.8670, 0.8693 | 0.8683, 0.8691 |
| 5,679 | 1.7285, 1.7296 | 1.6746, 1.6719 | 1.6719, 1.6796 | 1.7692, 1.7702 | 1.7069, 1.7108 | 1.7187, 1.7155 |
| 7,719 | 2.3491, 2.3498 | 2.2753, 2.2741 | 2.2751, 2.2869 | 2.4043, 2.4038 | 2.3242, 2.3279 | 2.3296, 2.3297 |
| 14,201 | 4.6245, 4.6223 | 4.4878, 4.4689 | 4.4904, 4.4931 | 4.7252, 4.7189 | 4.5803, 4.5715 | 4.5922, 4.6069 |
| 4 x 2,859 | 2.3405, 2.3336 | 2.3242, 2.3198 | 2.3579, 2.3372 | 2.3950, 2.3885 | 2.3852, 2.3715 | 2.3802, 2.3866 |

- Spread of an arm's two repetitions: at most 0.59% in the one-step jobs and 0.88% in the default jobs; under 0.3% in most cells. Readings inside the spread: the one-step 1,419-token cell in job 23625075 (GEMM arm), and at defaults the 4-stream cell (GEMM arm in job 23625421, both-plugins arm in job 23625420).
- Absolute saving at 14,201 tokens: 333 and 342 ms (GEMM, one-step), 527 and 518 ms (both plugins), 145 and 146 ms (GEMM, defaults) [computed from medians].
- **Against the estimate.** The PROFILE track computed 4.8% at 16k and 5.8% at 7.7k for one-step prefill from the profile's shares and the one-GCD kernel ratios; the server reads 7.9 to 8.0% at 14.2k and 6.8 to 6.9% at 7.7k. For vLLM's defaults it computed 3.5% at 16k; the server reads 3.1% at 14.2k. The one-step server gain is larger than computed [observed]; the profile was taken on two GCDs of two different cards, the server jobs here run both dies on one card, where the stock path may be slower relative to ours (shared 500 W cap). Not investigated [hypothesis].
- Small-g job 23625076 (two cards shared with other tenants, one job, one plugin repetition, so not part of the claim): 0.936, 0.948, 0.938, 0.926 at 2,859 to 14,201 tokens and 0.982 at 4 x 2,859; stock spread at most 0.54% above 1,419 tokens.
- One-step prefill against vLLM's default chunking, stock arms, different nodes: 4.24 and 4.27 s against 4.62 and 4.72 s at 14,201 tokens. An indication only (different jobs).

## Energy [observed: Cray `pm_counters` energy counter of the server's card, difference over each timed cell interpolated on the counter's own timestamps, nothing subtracted]

A cell is 8 requests with 0.5 s pauses (4 x 8 in the 4-stream cell); the pauses are in every arm. J per cell, stock repetitions / plugin repetitions, ratio of medians; `~` = inside the larger spread.

One-step prefill:

| Cell | 23625074 GEMM | 23625075 GEMM | 23625074 both | 23625075 both | J per 1,000 prompt tokens, stock -> GEMM -> both (23625074) |
|---|---|---|---|---|---|
| 1,419 | 0.995 ~ | 1.038 ~ | 0.939 ~ | 1.056 more | 284.0 -> 282.5 -> 266.8 |
| 2,859 | 0.940 ~ | 0.925 ~ | 0.915 less | 0.946 less | 217.5 -> 204.4 -> 198.9 |
| 5,679 | 0.946 less | 0.963 less | 0.917 less | 0.931 less | 183.2 -> 173.4 -> 168.1 |
| 7,719 | 0.937 less | 0.943 less | 0.918 less | 0.927 less | 172.6 -> 161.7 -> 158.4 |
| 14,201 | **0.924 less** | **0.947 less** | **0.887 less** | **0.898 less** | 165.5 -> 152.8 -> 146.8 |
| 4 x 2,859 | 0.985 less | 1.003 ~ | 0.981 less | 1.014 more | 167.2 -> 164.7 -> 164.1 |

vLLM defaults:

| Cell | 23625420 GEMM | 23625421 GEMM | 23625420 both | 23625421 both |
|---|---|---|---|---|
| 1,419 | 1.005 ~ | 1.019 ~ | 1.010 ~ | 1.086 ~ |
| 2,859 | 0.974 ~ | 0.955 less | 0.949 ~ | 0.969 less |
| 5,679 | 0.979 ~ | 0.960 ~ | 0.992 ~ | 0.965 less |
| 7,719 | 0.939 less | 0.950 less | 0.942 less | 0.960 ~ |
| 14,201 | 0.954 less | 0.983 ~ | 0.979 ~ | 0.956 ~ |
| 4 x 2,859 | 0.979 ~ | 1.004 ~ | 0.989 ~ | 1.013 ~ |

- The energy spread of an arm's two repetitions is 0.0 to 11.8% (largest at 1,419 tokens, where a cell is under 10 s and about 3 kJ), against under 1% for time. The energy saving follows the time saving because the card's power under load is the same in every arm: 410 to 430 W on the counter and 399 to 407 W hwmon, at the 500 W cap's level whichever kernel runs [observed].
- The 1,419-token and 4-stream readings on the "more" side (1.056, 1.014) are in one job each and not in the other; no claim either way.
- These are energies of the card for prefill-dominated requests of 8 output tokens, not energy per generated token. Node energy was not analysed.

## Decode step, ms [observed]: 8k contexts, 96 output tokens, default attention backend

| Job | cell | stock | GEMM | GEMM + attention |
|---|---|---|---|---|
| 23625074 | 1 stream | 56.56, 57.30 | 56.20, 56.88 | 55.78, 56.13 |
| 23625074 | 8 streams | 70.97, 70.53 | 69.84, 69.58 | 67.97, 68.56 |
| 23625075 | 1 stream | 56.54, 57.53 | 56.02, 56.89 | 56.23, 57.16 |
| 23625075 | 8 streams | 71.03, 71.20 | 69.22, 69.43 | 70.53, 70.71 |
| 23625420 | 1 stream | 55.91, 56.44 | 55.75, 56.35 | 56.25, 56.69 |
| 23625420 | 8 streams | 69.35, 69.49 | 68.74, 68.39 | 69.03, 68.75 |
| 23625421 | 1 stream | 55.55, 56.29 | 55.96, 56.60 | 56.08, 56.68 |
| 23625421 | 8 streams | 69.71, 69.64 | 70.26, 69.91 | 69.09, 69.27 |

No slowdown from the hook. At 8 streams the plugin arms read 0.7 to 3.5% below stock in three of four jobs and within 0.7% in the fourth; the plugins run nothing in a decode step (graph replay, 8 rows), so this is not a kernel effect and its cause is not established.

## What the plugin took [observed, counters, identical in both workers and both repetitions]

| Setting | linear calls taken | rows per call | left to vLLM: rows not compiled | left to vLLM: shape or bias | padded rows | capture | errors |
|---|---|---|---|---|---|---|---|
| one-step | 13,824 | 6,330 | 7,424 | 3,984 | 0 | 0 | 0 |
| defaults | 44,672 | 1,451 | 60,032 | 19,632 | 0 | 0 | 0 |

- One-step: 20 kernels used (2352, 5488, 7056, 14112 rows x 5 shapes; 4704 and 9408 never occurred). Not taken: 7,057 and 1,522 rows (the 4-stream cell: a prompt's rows plus decode rows of the other streams, or two prompts' remainders), 15,681 to 16,384 rows (the decode client's 8 prompts batched), remainders.
- Defaults: 7 kernels used. **More calls are left to vLLM than taken**: steps of 1,569, 1,570 and 1,571 rows (a 1,568-token chunk plus the decode rows of 1 to 3 other streams: 17,664 calls), remainders (635, 663, 1,291 to 1,293 rows), and down_proj at 1,568 rows by choice. A kernel compiled for 1,576 rows with a padded copy would take the mixed steps; not done.
- `stock_shape` is `in_proj_ba` (48 x 5120), which the plugin does not list.

## Correctness

- **Verify arms [observed]: ours against vLLM's path on the same activations and weights inside the server, first 2 calls per kernel and worker.** One-step (job 23625074): 20 kernels, 4 calls each, relative Frobenius difference 3.8e-6 to 1.2e-4, largest absolute difference 0.0625 at a reference magnitude up to 67, 0 non-finite values on either side. Defaults (job 23625420): 7 kernels, at most 6.1e-5, 0 non-finite. The differences are the size of occasional bf16 rounding ties, one to two orders of magnitude below bf16's own rounding of the result (1.66e-3 against fp32).
- **Standalone checks, one GCD, the plugin's own `run()` path (`check-tables.md`) [observed].** Error against an fp32 product equals vLLM's in every cell (1.66e-3; ratio of the two errors at most 1.0000002 in job 23624793), output repeats bitwise, shape and contiguity as vLLM's. Edge cases, all passing in all four jobs: a row count without a kernel is refused; an unlisted shape is refused; a padded call after a longer call matches vLLM's result; a non-contiguous input matches; an input holding inf and NaN gives non-finite output in exactly the rows where vLLM's does; an overflowing product gives the same non-finite pattern.
- **Standalone speed in two orders [observed].** One-step row counts (19 multiples of 784 from 1568 to 15680, 224x256 tile; forward 23624793, reversed 23626106): LEAD in 75 of 95 cells, TIE 4, LOSS 16 (all five shapes at 1568 except one tie; in_proj_qkvz at 2352 to 4704 and 7840; down_proj at 3136, 3920, 6272, 12544; three more). Default chunk with the sweep's tiles (forward 23625419, reversed 23626105): LEAD in 7 of 7 cells, 1.02 to 1.15.
- Not checked: generated text against stock (the client asks for 8 tokens), any quality benchmark, the output head (not taken by the plugin).

## What the measurements point to [computed or observed as marked]

1. **Per-(rows, shape) tiles for the one-step sizes.** Sweeps 23626594 (2352, 3136, 3920 rows) and 23626595 (6272, 12544 rows), 25 cells, one job each, a search (`evidence/PROFILE/RESULTS.md`, last section): in the 11 cells where the 224x256 tile reads below 1.02, another tile reads 1.04 to 1.14 in 10; the exception is 3920 x 8192 x 5120 (best 0.996). Examples: down_proj at 3920 rows 192x256 1.136 (224x256: 0.991); in_proj_qkvz at 3136 rows 192x256 1.139 (0.978); down_proj at 12544 rows 192x256 1.042 (0.985). The plugin's `HLG_FORMS` already takes such a table.
2. **Mixed steps at vLLM's defaults** (1,569 to 1,571 rows): 17,664 of this workload's 104,704 eligible linear calls (17%) are refused for one to three extra rows.
3. **down_proj at 1568 rows** has no leading tile (best 1.000): it is 14% of a default-chunk 16k prefill.
4. The stock `TRITON_ATTN` backend as the baseline, and energy per generated token on a mixed workload: not measured here.

## Not tested

- Stock `--attention-backend TRITON_ATTN` as the baseline. Any workload other than P1's synthetic prompts: real prompts, batches of fresh prompts, prompts above 16k, long generations.
- Output equivalence and quality. The verify arm compares one layer call at a time, not generated text.
- Row counts other than the client's; a server whose row counts are not known in advance needs kernels for all multiples of 784 (95 kernels; loading them took 3.6 s each, 5.7 minutes, in check job 23624793) or lazy loading.
- TP other than 2, one-GCD serving, other models, MI300.
- Node-level energy; energy per output token.

Files: `tables-<job>.md` (every repetition, counters, verify, power, energy), `check-tables.md`, `raw/<job>/`, `src/` (scripts and both plugin versions as run).

## Follow-up (11 Nov)

`evidence/G2/RESULTS.md` supersedes the kernel choice here: a raced tile per (rows, shape) landed in `gemm_bf16.ROWS_FORMS`, mixed steps at vLLM's defaults, the stock `TRITON_ATTN` baseline, an output-equivalence set and the server measurements repeated. The prefill-attention plugin used in the `fg_rocm` arms of this file runs the landed fp16 kernel, which was later found to drop subnormal attention weights (`evidence/P1/RESULTS.md`, addenda): its time numbers stand, its output is less accurate than vLLM's on long prompts.
