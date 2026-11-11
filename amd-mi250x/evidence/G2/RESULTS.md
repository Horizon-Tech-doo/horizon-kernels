# G2: the bf16 GEMM tile table landed, mixed steps, a second stock baseline, output equivalence, and the server measurements repeated (11 Nov 2026)

Labels: [observed], [computed], [inferred]. Ratio convention: **plugin / stock, below 1 = plugin faster (or less energy)**; kernel ratios are vLLM / ours, above 1 = ours faster.

**Result [observed; whole nodes, server alone on one card, two mirrored-order jobs per setting, every arm twice per job].**

| Setting | Arm against stock (default backend) | 1,419 tokens | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 |
|---|---|---|---|---|---|---|---|
| one-step prefill | GEMM plugin, landed table | 0.981 / 0.976 | 0.929 / 0.925 | 0.938 / 0.937 | 0.929 / 0.928 | **0.919 / 0.918** | 0.982 / 0.979 |
| one-step prefill | GEMM + prefill attention | 0.983 / 0.975 | 0.925 / 0.921 | 0.921 / 0.923 | 0.904 / 0.905 | **0.872 / 0.870** | 0.980 / 0.977 |
| one-step prefill | GEMM + prefill attention on the scaled kernel (G3) | 0.977 / 0.978 | 0.925 / 0.923 | 0.921 / 0.923 | 0.904 / 0.906 | **0.873 / 0.872** | 0.979 / 0.979 |
| vLLM defaults | GEMM plugin, landed table | 0.980 / 0.979 | 0.969 / 0.970 | 0.965 / 0.965 | 0.967 / 0.967 | **0.969 / 0.968** | 0.989 / 0.989 |
| vLLM defaults | GEMM + prefill attention incl. cached-context chunks | 0.979 / 0.976 | 0.970 / 0.967 | 0.956 / 0.953 | 0.955 / 0.953 | **0.961 / 0.958** | 0.987 / 0.988 |
| vLLM defaults | the same on the scaled kernel (G3) | 0.984 / 0.981 | 0.974 / 0.969 | 0.959 / 0.954 | 0.958 / 0.952 | **0.964 / 0.958** | 0.991 / 0.986 |

- **Time to first token, one-step prefill:** the GEMM plugin on the landed table is 6.2 to 8.2% faster from 2.9k to 14k tokens and 1.9 to 2.4% at 1.4k; with the prefill-attention plugin 7.5 to 13.0%. Every reading in the two complete jobs is outside its noise (at most 0.7%).
- **vLLM defaults:** GEMM 3.0 to 3.5% from 2.9k tokens, 2.0% at 1.4k; with cached-context prefill attention 3.0 to 4.7%. All outside noise (at most 0.7%) in the complete job; the second job lost its node after 7 of 10 arms (both stock repetitions are complete, every plugin arm has one), and its readings agree with the first to 0.3%.
- **Against the G1 jobs (one 224x256 tile):** one-step 0.940 / 0.933 -> 0.929 / 0.925 at 2.9k and 1.011 / 0.998 -> 0.981 / 0.976 at 1.4k (the 784-row kernels); within 0.4% elsewhere. Defaults: unchanged within 0.3% except the 4-stream cell, 0.994 -> 0.989 (mixed steps and 784-row kernels). Different nodes, so indicative.
- **Second baseline: stock `--attention-backend TRITON_ATTN` is slower than the default backend in prefill**, by 2.6%, 5.5%, 9.1% and **16%** at 2.9k, 5.7k, 7.7k and 14.2k tokens with one-step prefill and by 1.9%, 4.1%, 7.3% and 11% at vLLM's defaults (it is the faster one in decode, E2E). So for time to first token the default backend is the stronger stock baseline; against stock with the flag our combined arm reads 0.751 / 0.752 at 14.2k (one-step) and 0.868 / 0.863 (defaults). The GEMM gain is the same on both backends (defaults, GEMM on Triton against stock on Triton: 0.967 to 0.972 from 2.9k).
- **Card energy per prefill cell, one-step:** GEMM 0.916 / 0.925 at 14.2k, 0.943 / 0.949 at 7.7k; both plugins 0.871 / 0.885 and 0.890 / 0.917; outside the arms' spread in both jobs. At vLLM's defaults 0.966 / 0.965 (GEMM) and 0.968 / 0.958 (both) at 14.2k, outside the spread; at shorter prompts mostly inside it.
- **Output equivalence (12 fixed greedy prompts, 48 tokens, top-5 logprobs).** Stock repeats itself exactly across restarts in a job (12 of 12, logprob difference 0). Against stock in the same job: GEMM plugin 8 to 10 of 12 identical, both plugins 10 to 11, largest logprob difference of a chosen token before the first difference 0.16 to 0.37 nats. **The two stock backends differ from each other just as much**: 8 to 10 of 12 identical, 0.19 to 0.25 nats. Most first differences are at near-ties (top-2 margin 0 to 0.25 nats); each family, stock-against-stock included, has one or two where the reference margin was 1 to 2 nats. Numerics change, as expected with a different GEMM; nothing here is a quality measurement.
- **The prefill-attention plugin's accuracy problem is fixed in a new file and costs nothing end to end:** on `gqa_fa_ps.py` (weights scaled before the fp16 cast) the combined arm reads 0.873 / 0.872 at 14.2k against 0.872 / 0.870 on the landed kernel, in two mirrored jobs each.

## 1. The tile table [observed]

**Sweep (a search, one job per cell).** `gm_tiles.py`: `gemm_bf16`'s copy of `gemm_hand_rows` over 14 tiles x 5 schedules, each timed against vLLM's path in the same process with an identical-code control, 5 interleaved rounds. 105 cells: 21 row counts (784, 1568, 1600 and the 18 further multiples of 784 up to 15680) x the five large linear shapes; jobs 23624491, 23626594, 23626595, 23626776 to 23626782 (one GCD each). 36 of 70 forms build and pass the error check in every cell. Best form per cell and whether it became a candidate: `candidates.md`. 100 candidates (best form above 1 + the cell's noise, floored at 0.5%); 5 cells have none (784 x 17408 x 5120, 1568 x 5120 x 8704, 3920 x 8192 x 5120, 1600 x 5120 x 8704, 1600 x 7168 x 5120).

**Race (two opposite-order jobs, identical-code control in each).** `g2_check.py` runs every candidate through the plugin's own `run()` path against vLLM's `rocm_unquantized_gemm_impl`: forward job 23626967, reversed 23626968 (a third, forward, on dev-g: 23626969), 7 rounds. Rule, written into `g2_land.py` before the race: an entry lands when vLLM / ours exceeds 1 + noise in both jobs (noise = the cell's |stock_b / stock - 1| in that job, floored at 0.5%) and its output is finite, the right shape, bitwise repeatable and within 1.25x of vLLM's error against an fp32 product in both. **98 of 100 land; 2 tie and do not** (1600 x 5120 x 3072 at 1.004 / 1.007 and 1600 x 17408 x 5120 at 1.005 / 0.998); none loses. Per cell: `landed.md`.

**Landed: `tilelang/gemm_bf16.py` 48701f59 -> 84501f1e**, `ROWS_FORMS` (98 entries) and `gemm_rows(M, N, K)`, which returns None where the table has no entry. `gemm_prefetch.py` 04eece68 is not edited. Tiles in the table: 224x256 (22 entries), 192x256 (40, half of them on the 8-wave schedule), 160x256 (12), 224x224 (12), 128x128 (5), 224x192 (3), 224x160 (3), 256x128 (1).

**The landed bytes, checked again in two orders** (jobs 23626995 forward, 23626996 reversed; `landed-check.md`): 98 kernels, vLLM / ours **1.010 to 1.284, median 1.116**; 195 of 196 readings above 1.01; cell noise at most 0.72%. Error against the fp32 product equals vLLM's at every dispatched shape (ratio of the two errors at most 1.0000001; both 1.66e-3, the bf16 rounding of the result); every output finite, bitwise repeatable, right shape. Edge cases pass in both jobs: a row count or shape without an entry is refused, a padded call matches, a non-contiguous input matches, inf / NaN rows and an overflowing product give non-finite output exactly where vLLM's does.

Smallest of the two jobs' ratios per shape, 2352 rows and up: gate_up 1.063 to 1.153 (median 1.126); down 1.023 to 1.165 (1.092); in_proj_qkvz 1.025 to 1.132 (1.107); qkv 1.010 to 1.153 (1.109); out / o 1.048 to 1.201 (1.119). Against the single 224x256 tile of the G1 jobs at the same 93 cells: median 1.092 -> 1.112, cells below 1.02: 19 -> 1.

At vLLM's default chunk:

| rows | gate_up 17408x5120 | down 5120x8704 | in_proj_qkvz 8192x5120 | qkv 7168x5120 | out / o 5120x3072 |
|---|---|---|---|---|---|
| 784 | no entry | 1.145 / 1.148 | 1.029 / 1.047 | 1.047 / 1.060 | 1.284 / 1.284 |
| 1568 | 1.070 / 1.067 | no entry | 1.137 / 1.131 | 1.027 / 1.031 | 1.026 / 1.029 |
| 1600 (padded bucket) | no entry | no entry | 1.085 / 1.085 | no entry | no entry |
| 1571 through the padded copy | - | - | 1.104 / 1.102 | - | - |

**Mixed steps.** A 1,568-token chunk that shares its step with a few decode rows (1,569 to 1,600 rows) is now copied into a 1,600-row buffer where a 1,600-row kernel exists (`HLG_PAD_MS`, `HLG_PAD_ROWS`). 1,576 and 1,600 rows cost the same on every tile in the sweep (the tile count rounds up alike), so the bucket is 1,600. **Only in_proj_qkvz has a winning 1,600-row form**: the tiles that win at 1,568 rows are 224 rows high (1,568 = 7 x 224) and need an eighth tile row for one extra token, and the 160-row tiles tie or lose at the other four shapes. So mixed steps are taken for one of five shapes; the rest stays on vLLM's path, by the rule.

## 2. Server measurements: what ran

- **Server.** As G1: vLLM 0.30.0 ROCm image, public Qwen3.8-27B, BF16, TP=2, `--max-model-len 65536 --gpu-memory-utilization 0.88`. Whole standard-g nodes, the server on GCDs 0 and 1 (`E2E_USE=0:1`): the two dies of one card, no other job on the node.
- **Arms**, each a fresh server start, each twice per job:
  - `s_rocm` stock, default attention backend. `s_tri` stock with the stock flag `--attention-backend TRITON_ATTN` (second baseline).
  - `g_rocm`, `g_tri`: + `horizon_lumi_gemm` version 3 (dc3f1370) on `gemm_bf16.ROWS_FORMS` (84501f1e), `HLG_MIN` 512, padded bucket 1600.
  - `fg_rocm`, `fg_tri`: + `horizon_lumi_prefill` (a89a779a) on the landed `gqa_fa.py` (bb0808db), fresh steps only, buckets at every multiple of 784.
  - `cg_rocm`: as fg with `HLP_CTX=1` (also chunks with cached context, up to 9,408 total tokens). Default-chunk jobs only.
  - `fgp_rocm`, `cgp_rocm` (G3 jobs): as fg / cg with `HLP_PSHIFT=1` (prefill plugin dfe7407e on `gqa_fa_ps.py` 962f7c71, weights scaled by 2^10).
- **Load per arm.** `p1_client.py` (time to first token: one prompt at a time at 1,419 to 14,201 tokens, 8 timed each after 2 warm, and 4 at once at 2,859), then `eq_client.py` (equivalence set: 12 fixed greedy prompts, 4 short, 3 of 2.9k, 3 of 7.7k, 2 of 14.2k tokens, 48 output tokens, top-5 logprobs).
- **Jobs.** Two settings x two mirrored arm orders. One-step = `--max-num-batched-tokens 16384` in every arm.

  | Job | setting | arm order |
  |---|---|---|
  | 23626997 (g2-one-a) | one-step | s_rocm g_rocm fg_rocm s_tri fg_tri fg_tri s_tri fg_rocm g_rocm s_rocm |
  | 23626998 (g2-one-b) | one-step | fg_tri s_tri fg_rocm g_rocm s_rocm s_rocm g_rocm fg_rocm s_tri fg_tri |
  | 23626999 (g2-def-a) | vLLM defaults | s_rocm g_rocm cg_rocm s_tri g_tri g_tri s_tri cg_rocm g_rocm s_rocm |
  | 23627000 (g2-def-b; node failure after 7 arms, not rerun) | vLLM defaults | g_tri s_tri cg_rocm g_rocm s_rocm s_rocm g_rocm cg_rocm s_tri g_tri |
  | 23627538 (g3-one-a; node failure, rerun under the same id) | one-step | s_rocm fgp_rocm fgp_rocm s_rocm |
  | 23627539 (g3-one-b) | one-step | fgp_rocm s_rocm s_rocm fgp_rocm |
  | 23627540 (g3-def-a) | vLLM defaults | s_rocm cgp_rocm cgp_rocm s_rocm |
  | 23627541 (g3-def-b) | vLLM defaults | cgp_rocm s_rocm s_rocm cgp_rocm |
- **Rule.** A reading counts when |ratio - 1| exceeds the larger spread ((max - min) / median) of its two arms' repetitions in that job, in both jobs of the pair. Energy: Cray accelerator energy counter of the server's card over each timed cell, nothing subtracted.

## 3. Time to first token and energy [observed; all repetitions in `tables-<job>.md`]

Ratio in job A / job B; a bare number is outside the arms' spread, `~` is inside it; `(more)` marks an energy reading above 1 that is outside the spread.

**One-step prefill, jobs 23626997 / 23626998 (both complete, 10 arms each)**

Time to first token:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fg_rocm / s_rocm | 0.983 / 0.975 | 0.925 / 0.921 | 0.921 / 0.923 | 0.904 / 0.905 | 0.872 / 0.870 | 0.980 / 0.977 | 0.66%, 0.57% |
| fg_tri / s_rocm | 0.989 / 0.985 | 0.929 / 0.928 | 0.913 / 0.914 | 0.917 / 0.919 | 0.863 / 0.861 | 0.994 / 0.995 | 0.54%, 0.57% |
| g_rocm / s_rocm | 0.981 / 0.976 | 0.929 / 0.925 | 0.938 / 0.937 | 0.929 / 0.928 | 0.919 / 0.918 | 0.982 / 0.979 | 0.54%, 0.57% |
| s_tri / s_rocm | 1.009 / 1.003 ~ | 1.026 / 1.025 | 1.056 / 1.053 | 1.091 / 1.090 | 1.160 / 1.157 | 1.021 / 1.020 | 0.53%, 0.57% |
| fg_rocm / s_tri | 0.974 / 0.972 | 0.901 / 0.898 | 0.872 / 0.876 | 0.828 / 0.831 | 0.751 / 0.752 | 0.959 / 0.958 | 0.66%, 0.53% |
| fg_tri / s_tri | 0.980 / 0.982 | 0.905 / 0.905 | 0.865 / 0.868 | 0.841 / 0.843 | 0.744 / 0.744 | 0.974 / 0.975 | 0.54%, 0.53% |
| g_rocm / s_tri | 0.972 / 0.973 | 0.905 / 0.902 | 0.887 / 0.890 | 0.851 / 0.851 | 0.792 / 0.794 | 0.961 / 0.959 | 0.54%, 0.53% |

Card energy per prefill cell:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fg_rocm / s_rocm | 1.006 (more) / 0.995 ~ | 0.916 / 0.923 | 0.971 / 0.964 | 0.890 / 0.917 | 0.871 / 0.885 | 0.977 / 0.968 | 5.56%, 10.85% |
| fg_tri / s_rocm | 1.036 (more) / 0.985 ~ | 0.907 / 0.927 | 0.925 / 0.927 | 0.931 / 0.906 | 0.835 / 0.857 | 0.981 / 0.978 | 5.56%, 6.80% |
| g_rocm / s_rocm | 1.017 (more) / 1.008 ~ | 0.920 / 0.908 | 0.963 / 0.965 ~ | 0.943 / 0.949 | 0.916 / 0.925 | 0.945 / 0.985 ~ | 5.56%, 6.80% |
| s_tri / s_rocm | 0.988 ~ / 1.054 ~ | 0.998 ~ / 0.967 | 1.032 (more) / 1.055 (more) | 1.067 (more) / 1.060 (more) | 1.124 (more) / 1.131 (more) | 0.988 / 1.001 ~ | 5.56%, 6.80% |
| fg_rocm / s_tri | 1.018 ~ / 0.944 ~ | 0.918 / 0.954 | 0.941 / 0.914 | 0.834 / 0.865 | 0.775 / 0.782 | 0.988 ~ / 0.967 | 4.35%, 10.85% |
| fg_tri / s_tri | 1.048 (more) / 0.935 | 0.909 / 0.959 ~ | 0.897 / 0.878 | 0.873 / 0.855 | 0.743 / 0.758 | 0.992 ~ / 0.977 | 5.31%, 5.17% |
| g_rocm / s_tri | 1.029 (more) / 0.957 | 0.922 / 0.939 | 0.933 / 0.914 | 0.884 / 0.895 | 0.815 / 0.817 | 0.956 / 0.984 | 4.68%, 5.04% |

**vLLM defaults, jobs 23626999 (complete) / 23627000 (node failure after 7 arms: s_rocm x2, and g_rocm, cg_rocm, s_tri, g_tri once each; the second g_rocm was cut off and is not used. With one repetition an arm has no spread of its own: against s_rocm the verdict uses stock's spread only, against s_tri it shows `~`)**

Time to first token:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cg_rocm / s_rocm | 0.979 / 0.976 | 0.970 / 0.967 | 0.956 / 0.953 | 0.955 / 0.953 | 0.961 / 0.958 | 0.987 / 0.988 | 0.26%, 0.35% |
| g_rocm / s_rocm | 0.980 / 0.979 | 0.969 / 0.970 | 0.965 / 0.965 | 0.967 / 0.967 | 0.969 / 0.968 | 0.989 / 0.989 | 0.66%, 0.35% |
| g_tri / s_rocm | 0.986 / 0.984 | 0.989 / 0.990 | 1.007 / 1.007 | 1.039 / 1.041 | 1.075 / 1.078 | 0.987 / 0.997 ~ | 0.63%, 0.35% |
| s_tri / s_rocm | 1.003 / 1.005 | 1.018 / 1.020 | 1.041 / 1.041 | 1.073 / 1.073 | 1.108 / 1.110 | 1.000 ~ / 1.006 | 0.27%, 0.35% |
| cg_rocm / s_tri | 0.976 / 0.972 ~ | 0.954 / 0.948 ~ | 0.918 / 0.915 ~ | 0.890 / 0.888 ~ | 0.868 / 0.863 ~ | 0.987 / 0.981 ~ | 0.27%, 0.00% |
| g_rocm / s_tri | 0.977 / 0.974 ~ | 0.953 / 0.951 ~ | 0.927 / 0.927 ~ | 0.901 / 0.902 ~ | 0.874 / 0.872 ~ | 0.988 / 0.983 ~ | 0.66%, 0.00% |
| g_tri / s_tri | 0.983 / 0.980 ~ | 0.972 / 0.970 ~ | 0.967 / 0.968 ~ | 0.969 / 0.971 ~ | 0.970 / 0.971 ~ | 0.987 / 0.991 ~ | 0.63%, 0.00% |

Card energy per prefill cell:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cg_rocm / s_rocm | 0.918 ~ / 1.086 (more) | 0.916 / 0.951 | 0.968 ~ / 0.981 | 0.977 / 0.953 | 0.968 / 0.958 | 0.982 ~ / 0.989 ~ | 13.50%, 5.19% |
| g_rocm / s_rocm | 0.970 ~ / 1.115 (more) | 0.923 / 0.919 | 0.984 ~ / 0.982 | 1.009 ~ / 0.978 ~ | 0.966 / 0.965 | 0.981 / 0.965 | 13.50%, 5.19% |
| g_tri / s_rocm | 0.941 ~ / 1.021 ~ | 0.891 / 0.950 | 0.996 ~ / 1.006 ~ | 1.015 ~ / 1.033 ~ | 1.063 (more) / 1.066 (more) | 0.975 / 0.971 | 13.50%, 5.19% |
| s_tri / s_rocm | 0.947 ~ / 1.077 (more) | 0.914 / 0.973 ~ | 1.037 ~ / 0.998 ~ | 1.060 (more) / 1.023 ~ | 1.091 (more) / 1.100 (more) | 0.980 ~ / 0.999 ~ | 13.50%, 5.19% |
| cg_rocm / s_tri | 0.970 ~ / 1.009 ~ | 1.002 ~ / 0.977 ~ | 0.933 / 0.983 ~ | 0.922 / 0.931 ~ | 0.887 / 0.871 ~ | 1.002 ~ / 0.989 ~ | 11.25%, 0.00% |
| g_rocm / s_tri | 1.024 ~ / 1.036 ~ | 1.009 ~ / 0.945 ~ | 0.949 ~ / 0.984 ~ | 0.952 ~ / 0.956 ~ | 0.885 / 0.877 ~ | 1.001 ~ / 0.966 ~ | 11.25%, 0.00% |
| g_tri / s_tri | 0.994 ~ / 0.949 ~ | 0.975 ~ / 0.976 ~ | 0.961 / 1.009 ~ | 0.957 / 1.010 ~ | 0.974 / 0.969 ~ | 0.995 ~ / 0.972 ~ | 11.25%, 0.00% |

**One-step on the scaled prefill kernel, jobs 23627538 (the rerun after its node failure, complete) / 23627539**

Time to first token:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fgp_rocm / s_rocm | 0.977 / 0.978 | 0.925 / 0.923 | 0.921 / 0.923 | 0.904 / 0.906 | 0.873 / 0.872 | 0.979 / 0.979 | 0.53%, 0.79% |

Card energy per prefill cell:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fgp_rocm / s_rocm | 0.995 ~ / 0.894 ~ | 0.951 / 0.950 ~ | 0.932 / 0.953 | 0.935 / 0.918 | 0.884 / 0.890 | 0.965 ~ / 0.968 ~ | 4.43%, 12.26% |

**vLLM defaults on the scaled prefill kernel, jobs 23627540 / 23627541 (both complete)**

Time to first token:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cgp_rocm / s_rocm | 0.984 / 0.981 | 0.974 / 0.969 | 0.959 / 0.954 | 0.958 / 0.952 | 0.964 / 0.958 | 0.991 / 0.986 | 0.53%, 0.50% |

Card energy per prefill cell:

| Comparison | 1,419 | 2,859 | 5,679 | 7,719 | 14,201 | 4 x 2,859 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cgp_rocm / s_rocm | 0.970 ~ / 1.001 ~ | 1.048 ~ / 1.047 (more) | 0.956 ~ / 0.907 | 0.985 / 0.982 | 0.964 ~ / 0.982 ~ | 0.994 ~ / 0.983 ~ | 7.85%, 7.35% |

## 4. Output equivalence [observed]

`eq_client.py`: 12 fixed greedy prompts (4 short, 3 of 2.9k, 3 of 7.7k, 2 of 14.2k tokens), 48 output tokens, one request at a time, top-5 logprobs. Reference: the first stock repetition of the same job. Columns: identical token sequences of 12; largest absolute difference of the chosen token's logprob over the tokens before the first difference (nats); then every differing prompt as (kind, index, first differing token, top-2 margin of the reference at that token, how much the reference's token lost or gained there in this run). A plugin arm's two repetitions always give identical outputs, so one is listed.

**One-step, job 23626997**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_rocm-r0 | s_rocm-r0 | 8 of 12 | 0.1939 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.074), ('14.2k', 0, 14, 0.125, -0.279), ('14.2k', 1, 12, 0.125, -0.125) |
| fg_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1614 | ('short', 1, 17, 0.0, -0.062), ('14.2k', 0, 14, 0.125, -0.424) |
| s_tri-r0 | s_rocm-r0 | 10 of 12 | 0.1864 | ('short', 1, 36, 0.125, 0.045), ('14.2k', 0, 14, 0.125, -0.533) |
| fg_tri-r0 | s_rocm-r0 | 9 of 12 | 0.1760 | ('short', 1, 36, 0.125, 0.045), ('2.9k', 1, 32, 0.0, -0.055), ('14.2k', 0, 14, 0.125, -0.283) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| g_rocm-r0 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |
| fg_rocm-r0 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| fg_tri-r0 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |

**One-step, job 23626998**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| fg_tri-r0 | s_rocm-r0 | 10 of 12 | 0.1983 | ('short', 1, 17, 0.125, -0.063), ('14.2k', 0, 35, 1.75, -1.098) |
| s_tri-r0 | s_rocm-r0 | 10 of 12 | 0.2130 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.125, -0.052) |
| fg_rocm-r0 | s_rocm-r0 | 11 of 12 | 0.3686 | ('2.9k', 1, 32, 0.125, -0.07) |
| g_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1946 | ('14.2k', 0, 35, 1.75, -0.844), ('14.2k', 1, 12, 0.0, -0.065) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| fg_tri-r0 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |
| fg_rocm-r0 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| g_rocm-r0 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |

**vLLM defaults, job 23626999**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_rocm-r0 | s_rocm-r0 | 9 of 12 | 0.3298 | ('2.9k', 1, 32, 0.25, -0.123), ('7.7k', 1, 19, 0.125, -0.083), ('14.2k', 0, 35, 0.125, -0.125) |
| cg_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.3391 | ('2.9k', 1, 34, 0.0, -0.065), ('14.2k', 0, 35, 0.125, -0.755) |
| s_tri-r0 | s_rocm-r0 | 9 of 12 | 0.2153 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -0.843) |
| g_tri-r0 | s_rocm-r0 | 9 of 12 | 0.3734 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.123), ('14.2k', 0, 35, 0.125, -0.749) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| g_rocm-r0 | s_tri-r0 | 10 of 12 | 0.4068 | ('short', 1, 17, 0.0, -0.062), ('7.7k', 1, 19, 0.25, -0.127) |
| cg_rocm-r0 | s_tri-r0 | 10 of 12 | 0.2086 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.059) |
| g_tri-r0 | s_tri-r0 | 12 of 12 | 0.2868 | - |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |

**vLLM defaults, job 23627000 (partial)**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_tri-r0 | s_rocm-r1 | 11 of 12 | 0.5674 | ('short', 1, 17, 0.125, -0.063) |
| s_tri-r0 | s_rocm-r1 | 8 of 12 | 0.2493 | ('short', 1, 41, 0.0, -0.095), ('short', 3, 20, 0.125, -0.031), ('2.9k', 1, 34, 0.0, -0.065), ('7.7k', 0, 35, 2.125, -0.781) |
| g_rocm-r0 | s_rocm-r1 | 9 of 12 | 0.3298 | ('2.9k', 1, 32, 0.25, -0.123), ('7.7k', 1, 19, 0.125, -0.083), ('14.2k', 0, 35, 0.125, -0.125) |
| cg_rocm-r0 | s_rocm-r1 | 10 of 12 | 0.3391 | ('2.9k', 1, 34, 0.0, -0.065), ('14.2k', 0, 35, 0.125, -0.755) |
| s_rocm-r0 | s_rocm-r1 | 12 of 12 | 0.0000 | - |
| g_tri-r0 | s_tri-r0 | 8 of 12 | 0.1487 | ('short', 1, 17, 0.125, -0.064), ('short', 3, 20, 0.0, -0.1), ('2.9k', 1, 34, 0.125, -0.061), ('7.7k', 0, 35, 0.25, -0.162) |
| g_rocm-r0 | s_tri-r0 | 6 of 12 | 0.1825 | ('short', 1, 41, 0.125, -0.03), ('short', 3, 20, 0.0, -0.094), ('2.9k', 1, 32, 0.125, -0.057), ('7.7k', 0, 35, 0.25, -0.674), ('7.7k', 1, 19, 0.125, -0.049), ('14.2k', 0, 35, 0.25, -0.183) |
| cg_rocm-r0 | s_tri-r0 | 8 of 12 | 0.1568 | ('short', 1, 41, 0.125, -0.03), ('short', 3, 20, 0.0, -0.094), ('7.7k', 0, 35, 0.25, -0.558), ('14.2k', 0, 35, 0.25, -0.813) |

**One-step on the scaled kernel, job 23627539**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| fgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1989 | ('2.9k', 1, 32, 0.125, -0.066), ('14.2k', 0, 35, 1.75, -0.991) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |

**vLLM defaults on the scaled kernel, job 23627540**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| cgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.3300 | ('short', 0, 17, 0.125, -0.061), ('2.9k', 1, 32, 0.0, -0.055) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |

**vLLM defaults on the scaled kernel, job 23627541**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| cgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.2606 | ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -1.168) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |

- **Stock is deterministic inside a job**: second repetition against first, 12 of 12 and logprob difference exactly 0, for both backends in every job.
- **Plugin arms against stock on the same backend: 8 to 12 of 12 identical.** The largest logprob difference of a chosen token before a first difference is 0.15 to 0.37 nats; bf16 logits near these magnitudes step by 0.125 to 0.25.
- **Stock against stock across backends is in the same range**: `s_tri` against `s_rocm` 8 to 10 of 12, 0.19 to 0.25 nats.
- Most first differences sit where the reference's top two candidates were 0 to 0.25 nats apart. The exceptions: the first 14.2k prompt at token 35 (the model decides between continuing the repeated filler sentence and ending it), where the reference margin is 1.0 to 1.75 nats and the reference token loses 0.6 to 1.1 nats in the plugin arm in one-step jobs; the same position flips between the two stock backends at vLLM's defaults (margin 0.125, 0.84 nats) and a 7.7k prompt flips between them at a margin of 2.1 nats in job 23627000. So the largest shifts also occur between stock configurations on these repetitive prompts.
- The scaled prefill kernel does not change this picture (10 of 12 in all three jobs): the equivalence set is dominated by near-ties and by the GEMM plugin's rounding ties, and does not resolve the 1e-2-level attention error that the float64 check on dumped activations measures directly (`evidence/P1/RESULTS.md`).
- Short prompts differ in some plugin arms (one of four, at margins of 0 to 0.125) although neither plugin takes a step below 512 rows. The plugin replaces vLLM's linear custom op by its own in the compiled graph; whether a compiler pass keyed on the op's name changes the surrounding code was not checked [hypothesis].
- The same stock arm does not give the same outputs in two different jobs (compare a stock-against-stock row across the two jobs of a setting), while it repeats exactly inside a job. Not investigated.

## 5. What the plugins took [observed, counters, identical in both workers and repetitions; time-to-first-token and equivalence requests together]

| Setting | GEMM: linear calls taken | rows per call | left to vLLM for the row count | padded rows | kernels used | errors |
|---|---|---|---|---|---|---|
| one-step | 16,768 | 5,739 | 12,928 (remainders of 512 to 663 rows, the 4-stream cell's 7,057 and 1,522) | 0 | 24 | 0 |
| vLLM defaults | 53,088 | 1,409 | 38,304 (remainders: 635, 663, 1,291 to 1,293, 512 rows) | 43,200 | 9 | 0 |

- At vLLM's defaults the share of eligible calls taken rose from 43% (G1: 44,672 of 104,704) to 58% (53,088 of 91,392; this count includes the equivalence requests, G1's did not): 784-row kernels at four shapes and the mixed steps of in_proj_qkvz (1,440 calls through the 1,600-row kernel). The mixed steps of the other four shapes and the remainder rows stay on vLLM's path.
- Prefill attention, one-step: 928 layer calls taken (58 steps), 160 mixed, on both backends; no guard trip, no error. Defaults with `HLP_CTX=1`: 928 fresh and 1,904 cached-context calls (context 1,568 to 7,840), 480 mixed, 0 errors.

## 6. Notes

- **Node failures.** Two nodes failed during the runs (jobs 23627000 and 23627538, both NODE_FAIL in sacct). Slurm requeued both under the same job ids, which would overwrite the result folders: the partial folders were copied to `raw/<job>-nodefail/` at once. 23627000 was not rerun; its 7 finished arms are used and labelled partial. 23627538 was left to rerun on another node and completed all four arms; the tables use the rerun. Its pre-failure arms (s_rocm once, fgp_rocm twice; 0.982, 0.928, 0.926, 0.908, 0.875, 0.981) are in `raw/23627538-nodefail/` and agree with the rerun to 0.5%.
- **Baselines.** Default attention backend and the stock `TRITON_ATTN` flag, both with vLLM's own dense-linear path. Not compared: any tuned hipBLASLt setting.
- **Energy** is the card's, for prefill-dominated requests of 8 output tokens; spreads of 5 to 13% at the short cells make those readings uninformative.

## Not tested

- Real prompts, batches of fresh prompts, prompts above 16k, long generations; a quality benchmark.
- Row counts outside the table (a prompt whose first step is not a multiple of 784 falls back to vLLM); M above 15,680.
- D 128 models; the scaled prefill kernel on the Triton backend.
- Decode in these jobs (G1 measured it: unchanged).
- TP other than 2, one-GCD serving, MI300.

Files: `candidates.md`, `landed.md`, `landed-check.md`, `tables-<job>.md`, `summary-*.md`, `raw/`, `src/` (scripts, both plugins, `gemm_bf16.py`, `gqa_fa_ps.py` as run; md5s in `src-md5.txt`).
