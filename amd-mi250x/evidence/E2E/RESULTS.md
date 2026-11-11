# E2E: tokens per second and energy per token of a served 27B, vendor kernels against ours (10 Nov 2026)

**Result [observed].** Against vLLM as served by default, our decode-attention kernel raises the decode token rate by 1.78x at one 8k-token stream, 1.48x at 8 and 1.19 to 1.20x at 32, and lowers GPU energy per output token in the decode window by 20 to 22% at one stream and 18 to 21% at 8. On short prompts the rate changes by +0.7 to +1.6% and energy per token does not change measurably. **Against stock vLLM with its own `--attention-backend TRITON_ATTN` flag there is no gain: our kernel is 0.3 to 2.4% slower in decode token rate on the same backend, and on that backend energy per token is inside the noise in every cell of the main job; in the second job one cell is 8.8% worse and one 2.1% better.** The stock flag alone gives 1.84 to 1.85x, 1.52 to 1.54x and 1.36 to 1.38x on the 8k cells. Time to first token on 8k prompts is unchanged by our kernels (ratio 0.999 to 1.001), because prefill does not run them.

So the end-to-end claim this supports is narrow: the default attention path of vLLM 0.30 on MI250X is slow on long contexts, and either our kernel or a stock flag removes that. It does not support "our kernels make a served model faster or cheaper than the vendor's best configuration".

## What ran

- **Model and server.** Public Qwen3.8-27B, BF16, vLLM 0.30.0 ROCm image, tensor parallel 2 on 2 GCDs that are the two dies of one MI250X card, `--max-model-len 65536 --gpu-memory-utilization 0.88`, no speculative decoding, prefix caching at its default (on). The same flags as track V1 (`vllm-integration/v1.sbatch`), which is where the earlier token-rate table came from. The earlier token-rate result used TP=2. On one GCD the model fits only with a 9,000-token limit and about 87k cache tokens, too small for 32 streams of 7.7k tokens, so TP=2 is the smallest configuration already validated for this workload.
- **Arms.** Each repetition is a fresh server start in the same allocation.

  | Arm | Flags | Decode attention kernel in the 16 full-attention layers |
  |---|---|---|
  | A `s_rocm` | stock, default backend (ROCM_ATTN) | vLLM's own Triton fallback inside ROCM_ATTN; the server logs that the ROCm custom paged kernel cannot be used for this model |
  | B `igs128_rocm` | A + overlay | ours |
  | T `s_tri` | stock + `--attention-backend TRITON_ATTN` | vLLM's Triton backend kernel |
  | Bt `igs128_tri` | T + overlay | ours |

  B against A and Bt against T differ only in the kernel. B against T also differs in the cache layout and backend. T is a control: it is stock vLLM with one stock flag.
- **What runs our kernels in B and Bt [observed from counters and source].** Only attention on decode-only steps, in the 16 full-attention layers of 64, and only when vLLM replays its FULL CUDA graph (batch sizes up to 64). The kernel is `gqa_decode.py` b4cfec43 through the overlay ec748b9c, in the form the overlay records into the graph: `paged_uniform` with 128 splits and dead-split skipping at the 65536-token bucket. Counters per worker: `capture_hit` 176 (16 layers x 11 graph sizes), `error` 0, in every B and Bt repetition. Replay calls cannot be counted from Python; that the kernels run on replay is inferred from the rate change in B.
- **What does not run our kernels.** Attention prefill (`gqa_fa.py` is not wired; prefill and mixed prefill+decode steps use vLLM's kernels). The 48 Gated-DeltaNet layers (vLLM's Triton/FLA kernels in every arm). Every dense linear layer, the MLPs and the output head (`gemm_prefetch.py` is not wired). The model has no MoE layers, so `moe_fused.py` has nothing to replace. Eager decode steps on batch buckets that were not preloaded ran vLLM's kernel (`stock_miss` 464 per worker per repetition, outside graph replay).
- **The 9 Nov DECODE-MAX forms are not exercised [inferred from source].** They are selected by `paged_kernels` for specific (batch, pages, heads) shapes on the Triton layout; the graph path calls `paged_uniform` at 4096 pages, which is the same code path V1 used. So this run measures the kernel form V1 measured, on the newer file.
- **Workload.** Synthetic prompts only, identical in every arm and repetition, greedy (temperature 0, seed 0), `ignore_eos`, fixed output length. Short prompt: 24 to 26 tokens. Long prompt: 7,708 to 7,711 tokens. Streams 1, 8, 32. Output 384 tokens per stream at 1 stream, 256 at 8 and 32. Every timed prompt starts with a tag that differs from the warm-up pass, so timed prompts miss the prefix cache (0 cache hits in every timed cell) and time to first token includes a full prefill. Cell order reverses on odd repetitions.
- **Windows.** Whole call: first request sent to last token. Decode window: after the last stream's first token to before the first stream's last token, so every stream is decoding.
- **Energy.** A host process samples at 5 Hz. Source 1: the Cray `pm_counters` accelerator energy counter of the card that holds both GCDs (10 Hz scan, 1 J steps, with its own timestamp); window energy is the counter difference interpolated at the window edges. Source 2: the amdgpu hwmon package power of the card, trapezoid. Source 3: the node energy counter, attributable to this run only in the whole-node job. Nothing is subtracted. The card was identified per job as the accel counter whose power correlates with hwmon (0.97 against at most 0.16 for the others).
- **Jobs.** 23586076: 2 GCDs of a dev-g node, 12 server starts, order A B T Bt, Bt T B A, A B T Bt, so 3 repetitions per arm. 23586127: a whole standard-g node with the server on 2 GCDs and the other 6 idle, order Bt T B A, A B T Bt, 2 repetitions per arm; it gives a second node, the reverse order and a clean node counter.

## Noise

Spread = (max - min) / median over the repetitions of one arm in one job, per cell: the A-against-A (and B-against-B) estimate [computed].

| Metric | Job 23586076 (3 repetitions) | Job 23586127 (2 repetitions) |
|---|---|---|
| tokens/s, decode window and whole call | 0.0 to 2.7% | 0.0 to 2.2% |
| J/token, card counter | 0.4 to 8.7% | 0.1 to 8.7% |
| J/token, hwmon | 0.2 to 3.2% | 0.1 to 3.4% |
| J/token, node counter | 0.1 to 2.7% | 0.1 to 2.3% |
| time to first token, 8k cells | 0.1 to 0.7% (4.8% once, Bt x1) | 0.0 to 0.3% (2.9% once, Bt x1) |
| time to first token, short cells | 0.4 to 22.9% | 0.1 to 19.6% |

A ratio is marked `~` (no measurable difference) when it is closer to 1 than the larger spread of its two arms in that cell. Energy on the card counter is the noisiest quantity: the card's power moves by several percent between restarts at the same token rate.

## Findings

1. **B against A, decode token rate [observed].** Short x1, x8, x32: 1.012 (inside noise), 1.015, 1.016 in the main job; 1.007 (noise), 1.009, 1.007 (noise) in the second. 8k x1: 1.776, 1.782. 8k x8: 1.475, 1.478. 8k x32: 1.202, 1.185.
2. **B against A, whole call (prefill included) [observed].** 8k x1 1.647, 1.653; 8k x8 1.171, 1.173; 8k x32 1.040, 1.042. At 8 and 32 long streams the call is mostly prefill (time to first token 13.5 s and 45.9 s median against 12 to 19 s of decoding), and prefill is unchanged.
3. **B against A, energy [observed].** Decode window, card counter: 8k x1 0.779, 0.804; 8k x8 0.820, 0.791; 8k x32 0.967 (noise), 0.942; short cells inside noise in the main job. Whole call: 8k x1 0.800, 0.832; 8k x8 0.954 (noise), 0.943; 8k x32 0.976, 0.993 (noise). On the node counter of the whole-node job, whole call: 0.695, 0.894, 0.973 at 8k x1, x8, x32 and 0.991 to 0.999 on short prompts. The energy gain is smaller than the rate gain because arm A draws less power on long contexts (333 W at 8k x1 against 456 to 477 W in B): the card is partly idle while the slow kernel runs.
4. **Bt against T, same backend, kernel only [observed].** Decode token rate 0.981 (noise), 0.988, 0.987, 0.984, 0.987, 0.977 in the main job; 0.980 (noise), 0.993 (noise), 0.996, 0.981, 0.997 (noise), 0.976 in the second. Ours is slower by 0.5 to 1.3 ms per step in the main job and 0.1 to 1.3 ms in the second [computed]. Energy per token: inside noise in all 12 cells of the main job; in the second job 8k x32 decode reads 1.088 and short x1 0.979, the rest inside noise.
5. **B against T [observed].** Decode token rate 0.956 to 0.988 except 8k x32, which reads 0.870 and 0.872: there B needs 8.0 to 8.1 ms more per step than T. Energy per token in that cell is 11 to 17% worse on the decode window. The cause was not investigated [the 128-split form on the default layout at 32 long rows is a hypothesis].
6. **T against A [observed].** Decode token rate 1.040 to 1.046 on short prompts and 1.847, 1.543, 1.382 at 8k x1, x8, x32 in the main job (1.019 to 1.032 and 1.835, 1.522, 1.359 in the second). Energy per token in the decode window 0.764, 0.806, 0.869 at 8k in the main job.
7. **Time to first token [observed].** 8k cells, B against A: 1.000 to 1.001 in both jobs. Short x1: B is 5 to 7% slower than A and Bt 5 to 7% slower than T, about 3 to 5 ms, outside the noise in both jobs. T against A at 8k x1 is 7% slower (2.59 s against 2.42 s) and 2.6 to 3.2% faster at 8k x32.
8. **Power [observed].** No process on the GCDs: 100.7 W and 96.4 W on the card counter (95.0 and 92.0 W hwmon), 554 and 551 W on the node counter. Server loaded, no requests: 100.4 to 100.9 W and 98.4 to 98.9 W on the card counter in all arms. Under load the card draws 387 to 504 W (cell medians; single repetitions 384 to 513 W) in every cell of every arm except A at 8k x1 and x8 (medians 333 to 356 W). The hwmon power limit of the card reads 500 W. hwmon energy is 1.5 to 14% below the card counter.
9. **Clock reading [observed].** The sclk reading is 1650 MHz in every decode window of every arm on both GCDs, and gpu_busy_percent reads 100. This is the driver's target level, not a measured frequency, so it does not show whether the card throttled near its 500 W limit.

## Share of step time [computed]

Step time = streams / decode-window token rate, medians, main job, ms:

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 31.63 | 31.26 | 30.25 | 30.85 |
| short x8 | 45.40 | 44.72 | 43.65 | 44.20 |
| short x32 | 49.70 | 48.93 | 47.63 | 48.26 |
| 8k x1 | 57.38 | 32.31 | 31.07 | 31.57 |
| 8k x8 | 70.06 | 47.49 | 45.41 | 46.00 |
| 8k x32 | 75.04 | 62.41 | 54.28 | 55.57 |

- The replaced kernel (default-path decode attention) accounts for at least (A - B) / A of arm A's step: 1.2 to 1.6% on short prompts, 43.7% at 8k x1, 32.2% at 8k x8, 16.8% at 8k x32. A lower bound, since our kernel's own time is not zero.
- In arm T the step grows from short to 8k by 0.8 ms at 1 stream, 1.8 ms at 8 and 6.7 ms at 32, that is 2.6%, 3.9% and 12.3% of the 8k step. Decode attention is the only part of a step that reads the long context [inferred], so this bounds what any decode-attention kernel can still change against T on these cells, apart from the short-context attention cost, which was not isolated.
- No profiler was attached. The split of the remaining 30 to 48 ms between the 48 linear-attention layers, the MLPs and the graph runtime is not measured here.

## Output equivalence [observed]

Fixed greedy set, one request at a time, with top-2 logprobs: 8 short prompts x 128 tokens and 4 long prompts x 64 tokens.

| Pair | Identical texts of 12, job 23586076 | job 23586127 |
|---|---|---|
| any arm against itself across restarts | 12, 12, 12 in all four arms | 12 in all four arms |
| B against A | 7, 7, 7 | 7, 7 |
| Bt against T | 8, 8, 8 | 9, 9 |
| B against T | 9, 9, 9 | 9, 9 |
| T against A (stock against stock) | 7, 7, 7 | 9, 9 |

Each arm is deterministic across restarts within a job. Our arms are not text-identical to stock, and stock is not identical to stock across backends. At the first differing token the top-2 logprob margin of the first-named run is 0, 0.125 or 0.25 nats, the BF16 logit step, in every case but one, in job 23586127: long prompt 0, token 61, where the gap between the same two candidates is -0.25 nats in A, +0.125 in B, +0.5 in T and +1.375 in Bt. Stock arms differ by 0.75 nats there and Bt differs from T by 0.875. No quality benchmark was run in this track, so this is a tiny documented divergence on near-tie tokens, not a quality claim.

## Limitations

- Only decode attention is replaced. The result says nothing about the prefill, MoE and GEMM kernels in a served model.
- One model, one card type, TP=2, BF16, synthetic prompts with a repeated filler sentence, two prompt lengths, fixed output length, no sampling, no speculative decoding. The long prompt is highly repetitive.
- Arm A is vLLM's default on this stack, where the ROCm custom paged-attention kernel is rejected for this model and a Triton fallback runs. It is the default a user gets, not AMD's hand-written kernel.
- Energy is GPU card energy (both dies of one card) unless the node counter is named. It excludes CPU, memory, the other three cards, cooling and conversion losses. The node counter includes six idle GCDs (about 3 x 95 W) that a fully used node would use.
- The card runs at 387 to 504 W against a 500 W limit, so energy per token mostly follows 1 / token rate; a power-capped or frequency-capped setting was not run.
- In each job the first start of an arm reports a KV cache of 631,330 tokens and later starts 684,487 to 685,215 [observed]. It is the same for every arm, and it adds to the repetition spread.
- The second job has 2 repetitions per arm. Jobs ran on two nodes; node-to-node differences are visible (T against A on short prompts 1.04 against 1.02 to 1.03).
- Counter energy over 11 to 22 s windows has edge error of about 0.1 s of power, under 1%.

## Reproduce

`src/e2e.sbatch` (arms, order and repetitions through `ARMS`), `src/e2e_client.py` (workload), `src/e2e_power.py` (host sampler), `src/e2e_analyse.py <raw>` then `src/e2e_paper_table.py <raw>/summary.json`. Raw per-run rows: `raw/<job>/client-<arm>-r<n>.jsonl`, texts and logprobs `texts-*.jsonl`, power samples `power.csv`, plugin counters `ctr-*/`. Per-repetition values of every metric: `TABLES-per-repetition.md`. Host names and the project id are removed from the raw copies; no device identifiers were recorded. Each job's `header.txt` holds the md5 of the sources it ran; the main job ran the sbatch before the `E2E_USE` lines were added (md5 878941588d42), the whole-node job the version in `src/` (c6ad4f8d6d21), with no other difference.

## Cost

sacct, elapsed x GCDs allocated [computed]: probe 7 s, smoke 990 s, five cancelled placements 78 s, main job 8,390 s, all on 2 GCDs = 5.26 GCD-hours; one whole-node start cancelled after 246 s (the `--export` value was split at a comma, so the server saw one GCD) and the whole-node job 5,826 s on 8 GCDs = 13.49 GCD-hours. **Total 18.75 GCD-hours.** Five small-g starts were cancelled within a minute: four had their two GCDs on different cards, which would have mixed another job's power into the card counter, and one was cancelled with the first of them. The second job ran on a whole node.

## Tables

Arms A, B, T, Bt as above. Median [min-max] over repetitions.

### Job 23586076 (2 GCDs of a node; repetitions A 3, B 3, T 3, Bt 3)

**Output tokens/s, decode window** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 31.6 [31.6-32.5] | 32.0 [31.9-32.6] | 33.1 [33.0-33.8] | 32.4 [32.3-33.1] |
| short x8 | 176.2 [176.0-176.3] | 178.9 [177.4-179.0] | 183.3 [183.0-183.7] | 181.0 [180.6-181.1] |
| short x32 | 643.9 [641.3-650.5] | 654.1 [653.9-659.6] | 671.8 [670.3-672.0] | 663.1 [661.5-667.2] |
| 8k x1 | 17.4 [17.4-17.6] | 31.0 [30.9-31.3] | 32.2 [32.2-32.6] | 31.7 [31.5-32.0] |
| 8k x8 | 114.2 [114.0-114.4] | 168.5 [167.5-169.4] | 176.2 [175.7-176.9] | 173.9 [173.9-174.5] |
| 8k x32 | 426.5 [424.1-430.0] | 512.8 [508.9-513.5] | 589.5 [588.1-592.3] | 575.8 [570.4-581.9] |

**Output tokens/s, whole call (prefill included)** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 31.5 [31.5-32.3] | 31.9 [31.8-32.5] | 32.9 [32.9-33.6] | 32.3 [32.2-33.0] |
| short x8 | 174.3 [174.2-174.6] | 176.9 [175.0-177.0] | 181.2 [181.1-181.5] | 179.0 [178.7-179.2] |
| short x32 | 628.4 [626.8-634.5] | 637.7 [637.5-642.7] | 654.6 [650.0-654.7] | 647.4 [644.9-650.6] |
| 8k x1 | 15.7 [15.7-15.9] | 25.9 [25.9-26.2] | 26.5 [26.5-26.7] | 26.0 [25.9-26.4] |
| 8k x8 | 51.9 [51.8-51.9] | 60.7 [60.6-60.8] | 63.5 [63.5-63.5] | 63.3 [63.1-63.3] |
| 8k x32 | 78.6 [78.6-78.6] | 81.8 [81.7-81.9] | 84.6 [84.5-84.7] | 84.3 [84.3-84.4] |

**Time to first token, median over streams, s** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 0.070 [0.069-0.072] | 0.075 [0.073-0.075] | 0.069 [0.068-0.070] | 0.072 [0.072-0.072] |
| short x8 | 0.193 [0.193-0.194] | 0.200 [0.196-0.241] | 0.188 [0.186-0.189] | 0.191 [0.190-0.193] |
| short x32 | 0.383 [0.381-0.383] | 0.396 [0.389-0.408] | 0.389 [0.379-0.428] | 0.379 [0.374-0.381] |
| 8k x1 | 2.424 [2.420-2.425] | 2.426 [2.424-2.442] | 2.593 [2.591-2.600] | 2.592 [2.592-2.717] |
| 8k x8 | 13.538 [13.519-13.554] | 13.555 [13.552-13.569] | 13.423 [13.417-13.453] | 13.421 [13.417-13.517] |
| 8k x32 | 45.883 [45.845-45.949] | 45.937 [45.856-45.946] | 44.703 [44.632-44.716] | 44.685 [44.616-44.720] |

**J per output token, decode window, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 14.28 [13.76-14.51] | 14.38 [14.32-14.40] | 14.48 [14.16-14.63] | 14.30 [13.86-14.80] |
| short x8 | 2.25 [2.24-2.26] | 2.33 [2.29-2.43] | 2.30 [2.20-2.37] | 2.27 [2.22-2.29] |
| short x32 | 0.70 [0.67-0.72] | 0.69 [0.68-0.70] | 0.66 [0.65-0.68] | 0.67 [0.66-0.70] |
| 8k x1 | 18.92 [18.19-19.38] | 14.73 [14.44-14.76] | 14.46 [14.22-14.53] | 14.83 [14.19-14.87] |
| 8k x8 | 3.06 [3.05-3.13] | 2.50 [2.43-2.65] | 2.46 [2.44-2.47] | 2.43 [2.39-2.53] |
| 8k x32 | 0.98 [0.98-1.02] | 0.95 [0.93-0.96] | 0.85 [0.85-0.86] | 0.84 [0.84-0.85] |

**J per output token, whole call, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 14.29 [13.77-14.51] | 14.41 [14.32-14.42] | 14.53 [14.20-14.65] | 14.35 [13.89-14.84] |
| short x8 | 2.27 [2.26-2.28] | 2.36 [2.31-2.44] | 2.33 [2.23-2.38] | 2.29 [2.24-2.30] |
| short x32 | 0.72 [0.69-0.73] | 0.70 [0.69-0.72] | 0.68 [0.67-0.70] | 0.68 [0.68-0.72] |
| 8k x1 | 22.09 [21.17-22.55] | 17.67 [17.65-17.92] | 17.79 [17.54-17.97] | 18.00 [17.61-18.32] |
| 8k x8 | 8.02 [7.98-8.38] | 7.65 [7.48-7.67] | 7.48 [7.48-7.62] | 7.55 [7.35-7.59] |
| 8k x32 | 6.09 [6.06-6.18] | 5.94 [5.93-5.96] | 5.96 [5.83-5.99] | 5.92 [5.90-5.95] |

**Average W, decode window, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 452 [446-458] | 461 [459-467] | 484 [468-489] | 461 [459-480] |
| short x8 | 396 [395-399] | 414 [409-434] | 420 [404-435] | 411 [402-413] |
| short x32 | 449 [434-460] | 451 [444-464] | 444 [439-456] | 442 [441-465] |
| 8k x1 | 333 [317-337] | 456 [453-456] | 465 [458-473] | 468 [454-470] |
| 8k x8 | 349 [348-358] | 419 [412-446] | 433 [430-437] | 422 [417-440] |
| 8k x32 | 421 [419-434] | 484 [476-495] | 504 [502-509] | 482 [481-492] |

**Ratios of medians** (`~`: inside the larger repetition spread of the two arms, no measurable difference)

| Metric | Cell | B / A | Bt / T | B / T | T / A |
|---|---|---|---|---|---|
| tok/s, decode | short x1 | 1.012 ~ | 0.981 ~ | 0.968 | 1.046 |
| tok/s, decode | short x8 | 1.015 | 0.988 | 0.976 | 1.040 |
| tok/s, decode | short x32 | 1.016 | 0.987 | 0.974 | 1.043 |
| tok/s, decode | 8k x1 | 1.776 | 0.984 | 0.962 | 1.847 |
| tok/s, decode | 8k x8 | 1.475 | 0.987 | 0.956 | 1.543 |
| tok/s, decode | 8k x32 | 1.202 | 0.977 | 0.870 | 1.382 |
| tok/s, call | short x1 | 1.012 ~ | 0.981 ~ | 0.968 | 1.045 |
| tok/s, call | short x8 | 1.014 | 0.987 | 0.976 | 1.040 |
| tok/s, call | short x32 | 1.015 | 0.989 | 0.974 | 1.042 |
| tok/s, call | 8k x1 | 1.647 | 0.982 | 0.978 | 1.684 |
| tok/s, call | 8k x8 | 1.171 | 0.996 ~ | 0.956 | 1.225 |
| tok/s, call | 8k x32 | 1.040 | 0.997 | 0.966 | 1.076 |
| J/token, decode | short x1 | 1.007 ~ | 0.988 ~ | 0.993 ~ | 1.014 ~ |
| J/token, decode | short x8 | 1.038 ~ | 0.990 ~ | 1.016 ~ | 1.022 ~ |
| J/token, decode | short x32 | 0.988 ~ | 1.008 ~ | 1.042 | 0.948 ~ |
| J/token, decode | 8k x1 | 0.779 | 1.026 ~ | 1.019 ~ | 0.764 |
| J/token, decode | 8k x8 | 0.820 | 0.985 ~ | 1.017 ~ | 0.806 |
| J/token, decode | 8k x32 | 0.967 ~ | 0.988 ~ | 1.113 | 0.869 |
| J/token, call | short x1 | 1.008 ~ | 0.988 ~ | 0.992 ~ | 1.017 ~ |
| J/token, call | short x8 | 1.040 ~ | 0.984 ~ | 1.016 ~ | 1.024 ~ |
| J/token, call | short x32 | 0.982 ~ | 1.007 ~ | 1.042 | 0.942 |
| J/token, call | 8k x1 | 0.800 | 1.012 ~ | 0.993 ~ | 0.805 |
| J/token, call | 8k x8 | 0.954 ~ | 1.010 ~ | 1.023 ~ | 0.933 |
| J/token, call | 8k x32 | 0.976 | 0.993 ~ | 0.997 ~ | 0.979 ~ |
| TTFT | short x1 | 1.067 | 1.051 | 1.086 | 0.983 ~ |
| TTFT | short x8 | 1.036 ~ | 1.015 ~ | 1.064 ~ | 0.974 |
| TTFT | short x32 | 1.034 ~ | 0.975 ~ | 1.018 ~ | 1.016 ~ |
| TTFT | 8k x1 | 1.001 ~ | 0.999 ~ | 0.935 | 1.070 |
| TTFT | 8k x8 | 1.001 ~ | 1.000 ~ | 1.010 | 0.992 |
| TTFT | 8k x32 | 1.001 ~ | 1.000 ~ | 1.028 | 0.974 |

### Job 23586127 (whole node; repetitions A 2, B 2, T 2, Bt 2)

**Output tokens/s, decode window** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 32.3 [31.9-32.6] | 32.5 [32.2-32.8] | 33.3 [32.9-33.7] | 32.6 [32.3-33.0] |
| short x8 | 178.7 [178.2-179.1] | 180.3 [180.2-180.5] | 182.7 [182.2-183.1] | 181.4 [180.7-182.1] |
| short x32 | 656.5 [655.9-657.0] | 661.0 [658.5-663.5] | 669.0 [667.9-670.1] | 666.4 [666.4-666.4] |
| 8k x1 | 17.6 [17.5-17.7] | 31.4 [31.3-31.5] | 32.3 [32.0-32.6] | 31.7 [31.5-31.9] |
| 8k x8 | 115.1 [115.0-115.3] | 170.1 [170.1-170.2] | 175.3 [175.1-175.4] | 174.8 [174.3-175.3] |
| 8k x32 | 432.5 [431.9-433.0] | 512.5 [510.4-514.5] | 587.6 [587.1-588.1] | 573.7 [570.9-576.5] |

**Output tokens/s, whole call (prefill included)** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 32.1 [31.8-32.5] | 32.4 [32.0-32.7] | 33.2 [32.8-33.6] | 32.5 [32.2-32.8] |
| short x8 | 176.4 [176.0-176.9] | 177.9 [177.6-178.3] | 180.6 [180.5-180.8] | 179.3 [178.7-179.8] |
| short x32 | 639.9 [639.2-640.6] | 643.0 [640.6-645.4] | 652.0 [650.8-653.1] | 649.5 [648.9-650.0] |
| 8k x1 | 15.9 [15.8-16.0] | 26.3 [26.2-26.4] | 26.6 [26.4-26.8] | 26.1 [26.0-26.3] |
| 8k x8 | 52.2 [52.1-52.2] | 61.2 [61.2-61.2] | 63.9 [63.8-63.9] | 63.7 [63.6-63.8] |
| 8k x32 | 79.1 [79.1-79.1] | 82.4 [82.3-82.4] | 85.4 [85.4-85.4] | 85.2 [85.1-85.2] |

**Time to first token, median over streams, s** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 0.071 [0.070-0.072] | 0.074 [0.074-0.074] | 0.069 [0.068-0.070] | 0.074 [0.074-0.074] |
| short x8 | 0.192 [0.192-0.192] | 0.217 [0.196-0.238] | 0.187 [0.187-0.187] | 0.192 [0.192-0.193] |
| short x32 | 0.382 [0.381-0.382] | 0.408 [0.385-0.432] | 0.378 [0.377-0.378] | 0.377 [0.371-0.382] |
| 8k x1 | 2.407 [2.406-2.409] | 2.406 [2.404-2.409] | 2.571 [2.569-2.574] | 2.607 [2.570-2.645] |
| 8k x8 | 13.471 [13.461-13.481] | 13.459 [13.458-13.460] | 13.294 [13.277-13.311] | 13.293 [13.277-13.308] |
| 8k x32 | 45.628 [45.602-45.654] | 45.611 [45.534-45.688] | 44.183 [44.154-44.212] | 44.191 [44.159-44.224] |

**J per output token, decode window, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 14.27 [14.14-14.39] | 14.00 [13.86-14.14] | 14.17 [14.12-14.23] | 13.88 [13.85-13.90] |
| short x8 | 2.17 [2.14-2.19] | 2.24 [2.22-2.25] | 2.19 [2.18-2.20] | 2.29 [2.23-2.35] |
| short x32 | 0.69 [0.67-0.71] | 0.66 [0.66-0.66] | 0.68 [0.67-0.70] | 0.68 [0.66-0.70] |
| 8k x1 | 18.91 [18.20-19.63] | 15.21 [15.20-15.21] | 14.46 [14.30-14.62] | 13.93 [13.38-14.47] |
| 8k x8 | 3.09 [3.03-3.15] | 2.45 [2.43-2.46] | 2.43 [2.39-2.48] | 2.45 [2.44-2.47] |
| 8k x32 | 1.00 [0.98-1.02] | 0.94 [0.92-0.96] | 0.81 [0.79-0.82] | 0.88 [0.86-0.89] |

**J per output token, whole call, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 14.30 [14.16-14.45] | 14.03 [13.89-14.16] | 14.20 [14.15-14.24] | 13.89 [13.87-13.91] |
| short x8 | 2.19 [2.17-2.21] | 2.26 [2.25-2.27] | 2.21 [2.20-2.23] | 2.32 [2.25-2.38] |
| short x32 | 0.71 [0.69-0.73] | 0.68 [0.68-0.68] | 0.69 [0.68-0.70] | 0.69 [0.67-0.72] |
| 8k x1 | 22.12 [21.27-22.96] | 18.39 [18.38-18.40] | 17.95 [17.64-18.26] | 17.21 [16.46-17.96] |
| 8k x8 | 8.19 [8.12-8.26] | 7.72 [7.60-7.84] | 7.60 [7.44-7.76] | 7.63 [7.58-7.68] |
| 8k x32 | 6.14 [6.08-6.19] | 6.09 [6.07-6.11] | 5.99 [5.99-5.99] | 6.02 [5.99-6.05] |

**Average W, decode window, card counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 460 [451-469] | 455 [446-464] | 472 [465-479] | 453 [448-458] |
| short x8 | 387 [384-390] | 403 [400-406] | 400 [397-403] | 416 [404-428] |
| short x32 | 455 [442-469] | 437 [436-439] | 457 [449-465] | 452 [438-467] |
| 8k x1 | 333 [319-348] | 477 [476-479] | 467 [467-468] | 442 [427-456] |
| 8k x8 | 356 [350-362] | 416 [414-419] | 426 [418-434] | 429 [428-431] |
| 8k x32 | 432 [424-440] | 483 [475-490] | 474 [465-483] | 503 [493-513] |

**J per output token, whole call, node counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 30.75 [30.41-31.10] | 30.66 [30.37-30.96] | 30.07 [29.73-30.42] | 30.50 [30.27-30.72] |
| short x8 | 5.24 [5.23-5.26] | 5.19 [5.18-5.21] | 5.16 [5.13-5.19] | 5.14 [5.12-5.15] |
| short x32 | 1.52 [1.51-1.52] | 1.51 [1.51-1.52] | 1.50 [1.49-1.52] | 1.50 [1.50-1.51] |
| 8k x1 | 54.64 [54.40-54.89] | 37.98 [37.70-38.26] | 37.70 [37.46-37.95] | 38.18 [37.92-38.45] |
| 8k x8 | 18.26 [18.22-18.30] | 16.33 [16.31-16.35] | 15.92 [15.87-15.97] | 15.95 [15.94-15.97] |
| 8k x32 | 12.91 [12.90-12.92] | 12.56 [12.55-12.58] | 12.30 [12.28-12.32] | 12.31 [12.30-12.31] |

**Average W, whole call, node counter** (median [min-max] over repetitions)

| Cell | A | B | T | Bt |
|---|---|---|---|---|
| short x1 | 989 [988-989] | 993 [992-993] | 998 [998-998] | 992 [990-994] |
| short x8 | 925 [924-925] | 924 [923-925] | 932 [928-937] | 921 [921-921] |
| short x32 | 970 [968-972] | 974 [974-974] | 979 [972-986] | 976 [976-977] |
| 8k x1 | 869 [869-869] | 999 [995-1002] | 1004 [1003-1005] | 998 [998-999] |
| 8k x8 | 953 [952-953] | 999 [998-1000] | 1017 [1015-1019] | 1017 [1016-1017] |
| 8k x32 | 1021 [1021-1022] | 1035 [1034-1035] | 1050 [1048-1052] | 1048 [1048-1049] |

**Ratios of medians** (`~`: inside the larger repetition spread of the two arms, no measurable difference)

| Metric | Cell | B / A | Bt / T | B / T | T / A |
|---|---|---|---|---|---|
| tok/s, decode | short x1 | 1.007 ~ | 0.980 ~ | 0.976 | 1.032 |
| tok/s, decode | short x8 | 1.009 | 0.993 ~ | 0.987 | 1.022 |
| tok/s, decode | short x32 | 1.007 ~ | 0.996 | 0.988 | 1.019 |
| tok/s, decode | 8k x1 | 1.782 | 0.981 | 0.971 | 1.835 |
| tok/s, decode | 8k x8 | 1.478 | 0.997 ~ | 0.971 | 1.522 |
| tok/s, decode | 8k x32 | 1.185 | 0.976 | 0.872 | 1.359 |
| tok/s, call | short x1 | 1.007 ~ | 0.980 ~ | 0.976 | 1.032 |
| tok/s, call | short x8 | 1.008 | 0.993 | 0.985 | 1.024 |
| tok/s, call | short x32 | 1.005 ~ | 0.996 | 0.986 | 1.019 |
| tok/s, call | 8k x1 | 1.653 | 0.982 | 0.987 ~ | 1.674 |
| tok/s, call | 8k x8 | 1.173 | 0.998 ~ | 0.958 | 1.224 |
| tok/s, call | 8k x32 | 1.042 | 0.998 | 0.965 | 1.080 |
| J/token, decode | short x1 | 0.981 ~ | 0.979 | 0.988 ~ | 0.993 ~ |
| J/token, decode | short x8 | 1.032 | 1.048 ~ | 1.021 | 1.010 ~ |
| J/token, decode | short x32 | 0.954 ~ | 0.993 ~ | 0.968 ~ | 0.986 ~ |
| J/token, decode | 8k x1 | 0.804 | 0.963 ~ | 1.052 | 0.764 |
| J/token, decode | 8k x8 | 0.791 | 1.009 ~ | 1.006 ~ | 0.786 |
| J/token, decode | 8k x32 | 0.942 | 1.088 | 1.168 | 0.807 |
| J/token, call | short x1 | 0.981 ~ | 0.978 | 0.988 ~ | 0.992 ~ |
| J/token, call | short x8 | 1.034 | 1.047 ~ | 1.021 | 1.012 ~ |
| J/token, call | short x32 | 0.957 ~ | 1.001 ~ | 0.979 ~ | 0.978 ~ |
| J/token, call | 8k x1 | 0.832 | 0.959 ~ | 1.025 ~ | 0.812 |
| J/token, call | 8k x8 | 0.943 | 1.004 ~ | 1.016 ~ | 0.928 |
| J/token, call | 8k x32 | 0.993 ~ | 1.006 ~ | 1.017 | 0.976 |
| TTFT | short x1 | 1.050 | 1.071 | 1.069 | 0.982 ~ |
| TTFT | short x8 | 1.134 ~ | 1.028 | 1.162 ~ | 0.975 |
| TTFT | short x32 | 1.071 ~ | 0.998 ~ | 1.082 ~ | 0.990 |
| TTFT | 8k x1 | 1.000 ~ | 1.014 ~ | 0.936 | 1.068 |
| TTFT | 8k x8 | 0.999 ~ | 1.000 ~ | 1.012 | 0.987 |
| TTFT | 8k x32 | 1.000 ~ | 1.000 ~ | 1.032 | 0.968 |
