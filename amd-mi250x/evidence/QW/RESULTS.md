# QW: query-window prefill attention, so that every 1,568-token chunk of a default-chunked prefill runs our kernel (11 Nov 2026)

Labels: [observed], [computed], [inferred].

## Result in short [observed]

Two mirrored whole-node jobs per setting, every arm twice in each; verdicts from `summary-def.md` and `summary-one.md` (`qw_analyse.py`: LEAD only when the ratio is outside the larger spread of its two arms in both jobs).

| Time to first token, 1 stream | vLLM defaults: GEMM + query window / stock (job a, b) | GEMM alone / stock | GEMM + query window / GEMM alone | One-step: GEMM + prefill attention / stock | GEMM alone / stock |
|---|---|---|---|---|---|
| 1,419 tokens | 0.972, 0.979 | 0.975, 0.979 | 0.997, 1.000 tie | 0.972, 0.980 | 0.972, 0.981 |
| 2,859 | 0.967, 0.970 | 0.967, 0.971 | 1.000, 0.999 tie | 0.915, 0.925 | 0.918, 0.929 |
| 5,679 | 0.948, 0.951 | 0.964, 0.966 | 0.983, 0.984 LEAD | 0.909, 0.924 | 0.924, 0.939 |
| 7,719 | 0.941, 0.944 | 0.966, 0.967 | 0.974, 0.975 LEAD | 0.893, 0.905 | 0.915, 0.927 |
| 14,201 | **0.888, 0.893** | 0.967, 0.968 | **0.918, 0.923** LEAD | 0.870, 0.873 | 0.914, 0.920 |
| 4 x 2,859 | 0.988, 0.988 | 0.985, 0.988 | 1.003, 1.001 tie | 0.977, 0.980 | 0.978, 0.981 |

- **At vLLM's defaults the query window takes 7.7 to 8.2% off time to first token at 14,201 tokens on top of the GEMM plugin** (4.57 to 4.20 s; stock 4.72 s), 2.5% at 7,719 and 1.6% at 5,679, both orders, spreads at most 0.4%. Before this, prefill attention gave nothing at defaults (P1) and about 1 point with the square cached-context form. Prompts of one chunk (1,419) and the 4-stream cell are ties, as they must be: no single-sequence cached-context step, or mixed steps left to vLLM.
- **Less than computed** (11% expected at 16k, about 9.5% at 14.2k): 7.7 to 8.2% measured. The gather of the prefix K and V from the paged cache and the copies are the likely difference [inferred, not profiled].
- **One-step setting unchanged from G2/G3** (no cached-context chunks there, `qw_hit` 0): 0.870 / 0.873 at 14,201 (G3 read 0.873 / 0.872). Absolute time to first token at 14,201 tokens: stock defaults 4.72 s, defaults with both plugins 4.20 s, stock one-step 4.23 s, one-step with both plugins 3.68 s.
- **Card energy per cell, 14,201 tokens, GEMM + query window / stock:** defaults 0.917 (spreads 3.0%, 2.4%) and 0.895 (0.9%, 1.0%); one-step 0.910 and 0.879. Against GEMM alone at defaults [computed from the same cells]: 0.935 and 0.946, arm spreads 1.6 to 2.4% and 1.0 to 1.9%. Mean card power is unchanged (431 to 462 W at this cell), so the energy follows the time.
- **Counters:** 4,960 window calls per arm repetition at defaults (two workers), 0 errors, 0 guard, 0 misses, 0 too long; 960 mixed-step calls left to vLLM. Verify arm (job 23628970, untimed): 64 lines, totals 3,136 to 14,112, relative difference from vLLM's kernel on the same step 1.43e-3 to 1.54e-3, no NaN.
- **Equivalence set (12 greedy prompts x 48 tokens, against stock in the same job):** stock repeats 12 of 12 in all four jobs. Defaults: GEMM + query window 6 of 12 (job a) and 10 of 12 (job b); GEMM alone 7 and 10; verify arm 8. One-step: 10 of 12 for both plugin arms in both jobs. Each arm repeats itself exactly inside a job. Largest logprob difference before a first difference 0.17 to 0.26 nats for the window arm, 0.19 to 0.54 for GEMM alone. First differences of the window arm sit at top-2 margins of 0.125 to 0.375 nats at defaults; in the one-step jobs prompt ('14.2k', 0) flips at a margin of 1.75 for both plugin arms (GEMM alone flips the same token at defaults in job a, margin 1.875). Job a's 6 of 12 is the lowest count seen so far for any arm; it is one job, stock on that node differs from stock on job b's node too (different margins at the same token), and no quality benchmark was run.

## What was built (new files only; `gqa_fa.py` bb0808db, `gqa_fa_ps.py` 962f7c71 untouched)

- **`tilelang/gqa_fa_qw.py` (36fc593e).** A private copy of `gqa_fa.py`'s source with seven text changes (each must match exactly once or `load()` raises) that add one argument, `qw`: the kernel's grid gets ceil(qw / P) query blocks instead of ceil(L / P). With `rev=True` block index 0 is the last P positions, so the blocks that run are exactly the last ceil(qw / P) x P positions of the sequence, each over all keys up to its diagonal. Nothing inside a block changes: same tile, same K loop, same MFMAs, same hand edits; `gqa_fa_ps`'s weight scaling (2^10 before the fp16 cast) is applied to the device source as well. Own cache flags.
- **`horizon_lumi_prefill` version 4 (72b7cbb3), `HLP_QW=1`, off by default.** A step with one sequence, cached context and an even total length up to 16,384 runs the window kernel at the exact total length: prefix K and V gathered from the paged cache through the block table (as `HLP_CTX`), the chunk's K, V and queries from the call's arguments, the chunk's rows copied out. Kernels are precompiled for chunk 1,568 at totals 3,136 to 15,680 (9 kernels, `qw_precompile.py`); any other (total, chunk) stays on vLLM's kernel. Fresh steps use the same module's square form.
- **`g4.sbatch`, `g4_submit.sh`** (arm letter `w`; `fgw_rocm` = GEMM plugin + this; `fgwv_rocm` adds `HLP_VERIFY=4`), `p1_qw_check.py`.

## One-GCD gate, two arm orders [observed]: jobs 23628541 (forward), 23628593 (reversed)

`p1_qw_check.py`: arms window kernel, square kernel at the same total (the form it replaces), square kernel again (identical-code control); 7 interleaved rounds. Float64 reference of the chunk rows on the 6 dumped server steps with cached context and on 5 random cases. Table: `check-tables.md`.

| Case (total, chunk) | error vs float64: window / square / vLLM's kernel | window ms (fwd, rev) | square / window (fwd, rev) | vLLM's kernel in the server at that context, ms (profile, raw) |
|---|---|---|---|---|
| server dumps (4,704, 1,568) | 1.65e-3 to 1.69e-3 / same / 1.62e-3 to 1.67e-3 | 0.85, 0.88 to 0.90 | 1.57, 1.52 to 1.56 | 2.90 |
| server dumps (6,272, 1,568) | 1.65e-3 to 1.70e-3 / same / 1.62e-3 to 1.67e-3 | 1.14 to 1.15, 1.15 to 1.20 | 1.92 to 1.93, 1.85 to 1.92 | 3.90 |
| server dumps (7,840, 1,568) | 1.68e-3 to 1.69e-3 / same / 1.66e-3 to 1.67e-3 | 1.42 to 1.44, 1.45 to 1.48 | 2.34 to 2.35, 2.29 to 2.33 | 4.89 |
| random (3,136, 1,568) | 1.71e-3 / 1.71e-3 / - | 0.58, 0.59 | 1.040, 1.039 | 1.87 |
| random (9,408, 1,568) | 1.71e-3 / 1.71e-3 / - | 1.72, 1.82 | 2.77, 2.62 | 5.94 |
| random (15,680, 1,568) | 1.71e-3 / 1.71e-3 / - | 2.89, 3.06 | **4.42, 4.16** | **9.98** |
| random (4,096, 1,024) | 1.71e-3 / 1.71e-3 / - | 0.71, 0.71 | 1.40, 1.41 | - |
| random (6,272, 1,000; chunk not a block multiple) | 1.71e-3 / 1.71e-3 / - | 1.08, 1.07 | 1.97, 2.00 | - |

- **Correct:** the window kernel's rows are bitwise equal to the square kernel's rows in all 22 readings; error against float64 equals vLLM's kernel's on the server's own activations (1.65e-3 to 1.70e-3) and on random inputs (1.71e-3); all finite; `ok` true in both jobs.
- **Faster than the square form at every case, in both orders**, far outside the control noise (at most 1.1%): 1.04 at twice the chunk to 4.2 to 4.4 at ten times.
- **Against vLLM's kernel** the comparison crosses jobs (ours standalone without the cache gather of 0.1 to 0.2 ms and the copies; vLLM's inside the server, raw, about 0.08 ms of synchronise included): 0.58 against 1.87 ms at 1,568 cached tokens, 2.89 to 3.06 against 9.98 ms at 14,112. The server jobs measure it properly.
- The window form does not reach the ideal ratio (4.4 against 5.3 at the longest case): 49 blocks x 2 heads is 98 thread blocks on 110 compute units, so the GPU is not full and a block's latency sets the time. A smaller P for the window form would launch more blocks: not tried.

**Expected end to end [computed]:** a 16,115-token prompt at vLLM's defaults is 10 chunks of 1,568 and one of 435. vLLM's attention kernel: 59.4 ms per layer over the 11 steps (profile). Ours: 0.42 (fresh) + 0.58 + 0.85 + 1.14 + 1.43 + 1.72 + 2.0 + 2.3 + 2.6 + 2.89 (windows; the middle values interpolated) + 5.3 (the 435-row remainder stays on vLLM) = 21.2 ms. (59.4 - 21.2) x 16 layers = 610 ms of 5,347 ms: **about 11% of time to first token on top of the GEMM plugin's 3%**, minus gather and copies (about 0.25 ms x 9 x 16 = 36 ms).

## Server smoke [observed]: job 23628592 (dev-g, 2 GCDs, one arm `fgwv_rocm`, vLLM defaults)

- Server up after 409 s; `p1_client.py` 12 rows with `client_done`, `eq_client.py` 12 rows, both rc 0.
- Counters per worker (two workers, identical): `qw_hit` 1,008, fresh `hit` 528, `stock_mixed` 96, `stock_miss` 0, `stock_guard` 0, `error` 0. Window kernels used at totals 3,136 (336), 4,704 (336), 6,272 (176), 7,840 to 14,112 (32 each). Total 15,680 is not reached by this client (longest prompt 14,201).
- Verify (first 4 steps per total, vLLM's kernel run on the same step): 64 lines over totals 3,136 to 14,112, relative difference 1.43e-3 to 1.53e-3, no NaN. Not a timing arm.

## Server jobs [observed]: all COMPLETED, every arm 6 timed cells with `client_done` and 12 equivalence rows

Whole standard-g nodes, server on GCDs 0 and 1 (one card, no other tenant), every arm twice, mirrored orders, `--no-requeue`. Per arm: `p1_client.py` (time to first token at 1,419 to 14,201 tokens and 4 x 2,859), `eq_client.py` (12 greedy prompts with logprobs); `power.csv` for card energy. Per-job tables: `tables-<job>.md`; raw files in `raw/<job>/` (not included).

| Job | Name | Setting | Arm order |
|---|---|---|---|
| 23628970 | g4-def-a | vLLM defaults | fgwv (verify, untimed), s, g, fgw, fgw, g, s |
| 23628971 | g4-def-b | vLLM defaults | fgw, g, s, s, g, fgw |
| 23628972 | g4-one-a | `--max-num-batched-tokens 16384` | s, g, fgw, fgw, g, s |
| 23628973 | g4-one-b | `--max-num-batched-tokens 16384` | fgw, g, s, s, g, fgw |

Submit script: `g4_submit.sh` (9b1d713d).

Per job the server wrote `client-*.jsonl`, `eq-*.jsonl`, `ctr-*/hlp-*.json`, `ctr-*/hlg-*.json`, `power.csv`, `serve-*.log` and `header.txt`; the tables are built from these.

## Building the tables from the raw job folders

```
for j in 23628970 23628971 23628972 23628973; do python3 vllm-integration/g2_analyse.py evidence/QW/raw $j > evidence/QW/tables-$j.md; done
python3 vllm-integration/qw_analyse.py evidence/QW/raw 23628970 23628971 "vLLM defaults" > evidence/QW/summary-def.md
python3 vllm-integration/qw_analyse.py evidence/QW/raw 23628972 23628973 "one-step" > evidence/QW/summary-one.md
```


## Open

- Profile the window path inside the server (gather, copies, kernel) to find the 1.5 points between computed and measured.
- A smaller block (P 16) for the window form: 98 thread blocks do not fill 110 compute units.
- Half chunks (784) and the remainder chunk: precompile on demand or a ladder of window lengths; mixed steps.
- A quality benchmark. Without one the only accuracy statement is "same error against float64 as vLLM's kernel".

## Not tested

- Prompts above 14,201 tokens in the server (the 15,680 window kernel is compiled and gated on one GCD, not served). Real prompts. Decode. Mixed steps and batches (the plugin takes one sequence per step). Chunks other than 1,568 rows in the server (784-row half chunks and the remainder stay on vLLM's kernel).
- A smaller P for the window form; D 128; the Triton backend's cache layout (`RocmAttentionImpl` only).
- A quality benchmark.

## GPU use [sacct ElapsedRaw x GCDs]

Gate 23628541 (122 s) and 23628593 (50 s) x 1 GCD: 0.05. Smoke 23628592 (765 s) x 2: 0.43. Whole nodes 23628970 (4,006 s), 23628971 (3,368 s), 23628972 (3,314 s), 23628973 (3,326 s) x 8: 31.14. This round 31.6 GCD-hours.
