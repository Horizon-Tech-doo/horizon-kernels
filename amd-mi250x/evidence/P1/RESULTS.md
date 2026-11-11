# P1: prefill attention (`gqa_fa` through `horizon_lumi_prefill`) in a served 27B, time to first token (10-11 Nov 2026)

Labels: [observed], [computed], [inferred]. Ratio convention: **plugin / stock time to first token, below 1 = plugin faster.**

**Result.**

- **vLLM at its defaults: no gain.** The plugin changes time to first token by -0.1 to +0.7% in the two clean jobs (11 of 12 readings on the slow side); three of twelve cell readings are outside the job's noise, all on the slow side, by 0.2 to 0.3%. The plugin takes 50 of the 210 prefill steps of 1,024 tokens or more (24%): vLLM cuts every prompt into chunks of at most 1,568 tokens, and only a prompt's first chunk is a fresh step.
- **Prompts prefilled in one step (`--max-num-batched-tokens 16384` in every arm): the plugin is faster on long prompts, in two opposite-order jobs with identical-code controls [observed].** Time to first token ratio, job 23620887 / job 23624336: 0.986 / 0.985 at 5,679 tokens, 0.979 / 0.981 at 7,719, **0.958 / 0.955 at 14,201**; all six outside the noise of their job (at most 1.0% and 0.4%). At 1,419 and 2,859 tokens and at 4 concurrent prompts the difference is at most 0.3% and is inside the noise in 4 of 6 readings.
- So the claim this supports: with one-step prefill, our prefill attention shortens time to first token by 1.4 to 1.5% at 5.7k tokens, 1.9 to 2.1% at 7.7k and 4.2 to 4.5% at 14k, against vLLM's default attention backend. It does not support a gain at vLLM's default scheduling, and nothing here compares against the stock `--attention-backend TRITON_ATTN` flag.
- One job (23620886) reads 4 to 7% in the plugin's favour at default settings. It is not evidence: a neighbour's load on a shared card changed exactly between its arms (below).

## What ran

- Server: vLLM 0.30.0 ROCm image, public Qwen3.8-27B, BF16, TP=2 on 2 GCDs of a small-g node, `--max-model-len 65536 --gpu-memory-utilization 0.88`, default attention backend. Arms: `s_rocm` stock (no plugin on the path), `f_rocm` + `horizon_lumi_prefill` (b08c95e5; `gqa_fa.py` bb0808db; ladder 1024:16384:0.06, guard on), `fv_rocm` the same with `HLP_VERIFY=16` (not a timing arm). Each repetition is a fresh server start in the same allocation. Script `p1.sbatch` 7bc1b46e, client `p1_client.py` 4b13b3d2.
- Load: synthetic prompts, greedy, 8 output tokens, a unique tag at the start of every prompt (no prefix-cache hit). Cells: one prompt at a time at 1,419, 2,859, 5,679, 7,719 and 14,201 tokens (8 timed requests each after 2 warm), and 4 prompts at once at 2,859 (8 rounds). Cell order reverses on odd repetitions.
- Jobs [sacct]: all COMPLETED, every arm 6 timed cells of 6 and `client_done`.

  | Job | flags | arm order | elapsed s | node |
  |---|---|---|---|---|
  | 23619582 | default | fv s f f s | 2,972 | node1 |
  | 23620886 | default | f s s f | 2,490 | node2 (contaminated, see below) |
  | 23624337 | default | f s s f | 2,573 | node3 |
  | 23620887 | MNBT 16384 | fv s f f s | 2,931 | node4 |
  | 23624336 | MNBT 16384 | f s s f | 2,264 | node5 |

  2 GCDs x 13,230 s = **7.35 GCD-hours**.
- Keep rule: two jobs in opposite arm order, each with its identical-code control (s twice and f twice in a job; spread = (max - min) / median of an arm's two repetition medians). A reading counts only when |ratio - 1| exceeds the larger spread of its two arms in that job.

## Time to first token: plugin / stock [observed; medians per repetition in `tables.md`]

vLLM defaults (chunked prefill):

| Cell | 23619582 (fv s f f s): ratio, noise | 23624337 (f s s f): ratio, noise | 23620886 (f s s f), contaminated |
|---|---|---|---|
| 1,419 x1 | 1.0020, 0.41% ~ | 1.0011, 1.91% ~ | 0.9444 |
| 2,859 x1 | 1.0028, 0.07% slower | 1.0012, 1.51% ~ | 0.9409 |
| 5,679 x1 | 1.0023, 0.12% slower | 1.0011, 1.56% ~ | 0.9336 |
| 7,719 x1 | 1.0016, 0.12% slower | 1.0012, 1.62% ~ | 0.9335 |
| 14,201 x1 | 1.0022, 0.32% ~ | 1.0034, 1.73% ~ | 0.9635 |
| 2,859 x4 | 1.0069, 0.89% ~ | 0.9989, 0.48% ~ | 0.9623 |

One-step prefill (`--max-num-batched-tokens 16384`):

| Cell | 23620887 (fv s f f s): s reps, f reps, ratio, noise | 23624336 (f s s f): s reps, f reps, ratio, noise |
|---|---|---|
| 1,419 x1 | 0.5013 0.5034, 0.5019 0.5040, 1.0012, 0.42% ~ | 0.4849 0.4883, 0.4853 0.4852, 0.9972, 0.70% ~ |
| 2,859 x1 | 0.9902 0.9802, 0.9850 0.9848, 0.9997, 1.02% ~ | 0.9397 0.9404, 0.9377 0.9377, 0.9975, 0.07% faster |
| 5,679 x1 | 1.8922 1.8737, 1.8572 1.8570, **0.9863**, 0.98% faster | 1.7894 1.7895, 1.7620 1.7631, **0.9850**, 0.06% faster |
| 7,719 x1 | 2.5412 2.5236, 2.4787 2.4775, **0.9786**, 0.69% faster | 2.4114 2.4103, 2.3693 2.3595, **0.9807**, 0.41% faster |
| 14,201 x1 | 4.7759 4.7686, 4.5758 4.5693, **0.9582**, 0.15% faster | 4.5595 4.5582, 4.3555 4.3535, **0.9552**, 0.05% faster |
| 2,859 x4 | 3.7441 3.7106, 3.7246 3.7165, 0.9982, 0.90% ~ | 3.5488 3.5467, 3.5443 3.5422, 0.9987, 0.06% faster |

- The saving at 14,201 tokens is 200 ms and 204 ms; at 7,719 it is 54 and 47 ms; at 5,679 it is 26 and 27 ms [computed]. The PROFILE track measures vLLM's attention kernel at 572 ms of a 15,680-token one-step prefill, so on the order of a third of it is removed (the two prompts differ in length, so this is rough).
- Absolute times differ by 5% between the two one-step jobs (4.77 s against 4.56 s at 14,201 tokens, stock). Ratios are taken inside a job.
- One-step prefill itself against vLLM's default chunking, stock arms: 4.77 and 4.56 s against 5.25 and 5.16 s at 14,201 tokens, in different jobs on different nodes, so only an indication.

## How many steps the plugin took [observed, counters, identical in every plugin repetition and both workers]

Counters count attention-layer calls (16 per step) on steps whose longest request has at least 1,024 new tokens.

| Flags | hit | taken steps | tokens per taken step | padding | `stock_context` | `stock_mixed` | short, long, miss, guard, shape, capture, error | buckets |
|---|---|---|---|---|---|---|---|---|
| default | 800 | 50 of 210 (24%) | 1,568 | 2.04% | 2,080 (130 steps) | 480 (30 steps) | 0 | 1600 only |
| MNBT 16384 | 800 | 50 of 60 (83%) | 6,272 on average | 2.24% | 0 | 160 (10 steps) | 0 | 2432 (2,352 tokens), 5760 (5,488), 7200 (7,056), 14240 (14,112) |

- `stock_context`: the step has a cached prefix, that is, every chunk of a prompt after its first. At default settings that is 130 of 210 steps and most of the attention cost (PROFILE: 937 ms of a 16k prefill, growing with the context of each chunk).
- `stock_mixed`: a decode row shares the step (the 4-stream cell, where streams that already started decode while others prefill).
- Steps whose longest request is below 1,024 tokens are not counted and not taken. The 1,419-token prompt is scheduled as 784 + 635 tokens, so the plugin never touches it.
- vLLM schedules at most 1,568 tokens of one prompt per step at its defaults and cuts at multiples of 784; with MNBT 16384 the first step is the largest multiple of 784 [observed in the PROFILE track's step records].
- No guard trip, no missing bucket, no error in any arm.

## Verify arm: ours against vLLM's kernel on the same step inputs [observed]

First 16 calls per bucket (one step, both workers, all 16 layers = 32 compared calls per bucket). Relative Frobenius difference of the layer output, largest absolute difference, largest reference magnitude:

| Job | bucket (tokens) | rel min to max (median) | max abs | max abs(ref) | NaN |
|---|---|---|---|---|---|
| 23619582 | 1600 (1,568) | 1.52e-3 to 2.26e-3 | 0.5 | 101 | 0 |
| 23620887 | 2432 (2,352) | 1.52e-3 to 2.97e-3 (1.94e-3) | 0.5 | 102 | 0 |
| 23620887 | 5760 (5,488) | 1.52e-3 to 6.35e-3 (2.69e-3) | 0.75 | 106 | 0 |
| 23620887 | 7200 (7,056) | 1.52e-3 to 7.91e-3 (3.21e-3) | 0.75 | 106.5 | 0 |
| 23620887 | 14240 (14,112) | 1.57e-3 to 1.29e-2 (4.94e-3) | 1.5 | 103 | 0 |

- 1.5e-3 is the size of bf16 rounding of the output. The difference grows with length: at 14,112 tokens 15 of 32 calls exceed 5e-3 and the largest is 1.3%. Both kernels are compared with each other, not with a reference, so this does not say which is closer; the one-GCD gate (`p1_check.py`, job 23619490) measured ours against float64 up to 6,144 tokens only. **Open: error against a float64 reference above 6,144 tokens, and whether the plugin's fp16 round trip or vLLM's kernel carries the growth.**
- Generated text was not compared between arms in this track (the client asks for 8 tokens). No quality benchmark.

## Why job 23620886 is not evidence [observed]

The two GCDs of a small-g job are dies of two different cards (`two_dies_of_one_card` false in every P1 job), and each card's other die belongs to another job; the 500 W cap is per card. A co-tenant's load on a card that this job shares changed during this job, exactly between its arms, so the job is not used as evidence.

The stock arms read 5 to 7% slower than the plugin arms and than the stock arms of the other default jobs; one attention layer at 1.5k tokens costs under a millisecond, so a 28 ms difference at 1,419 tokens cannot be the kernel. Job 23624337 repeats the same order: the load on its shared cards also changed, but across both arms, and it reads 0.999 to 1.003 with a noise of 0.5 to 1.9%. The two one-step jobs show no such change between their arms.

Consequence for the method: on small-g a job's noise is set by its neighbours. Server races that matter should run on a whole node with the server on the two dies of one card (`E2E_USE=0:1`), as the G1 jobs now do.

## Not tested

- Against stock `--attention-backend TRITON_ATTN` (E2E found that flag faster than the default backend in decode; prefill not compared).
- Chunks with cached context (the plugin leaves them to vLLM): 76% of the eligible steps at default settings.
- Batches of several fresh prompts in one step beyond the 4-stream cell; prompts above 16,384 tokens; other head counts; one-GCD serving.
- Energy (the sampler's `power.csv` is in every job folder, not integrated). Output equivalence and quality.
- Error against a reference above 6,144 tokens.

Files: `tables.md` (all repetitions, counters, verify), `raw/<job>/`, `src/` (scripts as run).

## Addendum 11 Nov: error against a float64 reference up to 16,384 tokens (job 23626788, one GCD) [observed]

`p1_ref64.py`: the plugin's own `dense()` path (bf16 -> fp16 kernel -> bf16, padded to its bucket) and torch SDPA on bf16, each against a float64 reference computed in chunks of 384 query rows. Hkv 2, G 6, D 256. Relative Frobenius error of the whole output; worst single row in brackets.

| Input | n | ours | SDPA bf16 | ours against SDPA |
|---|---|---|---|---|
| randn | 2,048 | 1.655e-3 (1.81e-3) | 2.012e-3 (2.32e-3) | 2.19e-3 |
| randn | 6,144 | 1.670e-3 (1.83e-3) | 2.067e-3 (2.41e-3) | 2.27e-3 |
| randn | 8,192 | 1.660e-3 (1.81e-3) | 2.060e-3 (2.40e-3) | 2.26e-3 |
| randn | 12,000 | 1.670e-3 (1.83e-3) | 2.087e-3 (2.43e-3) | 2.30e-3 |
| randn | 14,112 | 1.670e-3 (1.84e-3) | 2.087e-3 (2.43e-3) | 2.29e-3 |
| randn | 16,384 | 1.668e-3 (1.82e-3) | 2.087e-3 (2.43e-3) | 2.29e-3 |
| peaky (q, k x 2.5) | 2,048 | 1.690e-3 (1.92e-3) | 1.688e-3 (1.92e-3) | 1.54e-3 |
| peaky | 8,192 | 1.698e-3 (1.94e-3) | 1.698e-3 (1.92e-3) | 1.56e-3 |
| peaky | 14,112 | 1.701e-3 (1.93e-3) | 1.701e-3 (1.95e-3) | 1.57e-3 |
| peaky | 16,384 | 1.702e-3 (1.97e-3) | 1.701e-3 (2.01e-3) | 1.57e-3 |

- **Our path's error does not grow with length**: 1.66e-3 to 1.70e-3 from 2k to 16k tokens, the size of bf16 rounding of the output, and at or below SDPA's in every case. All outputs finite.
- So the growth the server's verify arm showed against vLLM's kernel (median 4.9e-3, maximum 1.3e-2 at 14,112 tokens) is not reproduced by our path on these inputs. It comes either from vLLM's kernel or from properties of the model's real activations that random inputs lack (reference magnitudes up to 100 there, 3 to 5 here). **Still open: the same reference check on activations captured from the server.** vLLM's own kernel was not run standalone.

## Addendum 11 Nov: prefill chunks with cached context: what it takes [design; costs observed in job 23626788 and in the profile]

**The problem.** At vLLM's defaults a prompt is prefilled in 1,568-token chunks. Every chunk after the first attends over cached prefix + chunk; the plugin left those to vLLM (`stock_context`, 130 of 210 steps). They are 17.5% of a 16k prefill (PROFILE).

**Two ways.**

1. *Plugin only, square kernel.* `gqa_fa`'s kernel takes Q, K, V of one length L and a causal mask. Put K and V = prefix (gathered from the paged cache through the step's block table) followed by the chunk, and the chunk's queries in rows context..context+chunk; causal masking then makes those rows exactly right, and the rows before them are computed and dropped. No kernel change. Cost grows with (context + chunk) squared, vLLM's with chunk x (context + chunk).
2. *Kernel change, query window.* The kernel's grid runs over all L / P query blocks (`make`: `NB = L // P`, block start `p0`). A form that launches only the blocks from the chunk's first row on costs chunk x (context + chunk / 2). That is a change to `gqa_fa.py` (grid size, `p0` offset, the hand edits that follow the generated source, the dispatch tables), so it needs the track's own races. Not done.

**Measured, per attention-layer call, ms** (ours: one GCD, job 23626788, wired = cache gather through a random block table + bf16 -> fp16 copies + kernel + copy out, block size 16; vLLM: the profile's per-chunk attention time inside the server, job 23621041, raw, which includes about 0.08 ms of synchronise; window form: computed as square x (1 - (context / Lk)^2)):

| context | Lk | vLLM's kernel | ours wired, square | of which gather and copies | window form (computed) |
|---|---|---|---|---|---|
| 1,568 | 3,136 | 1.87 | **0.74** | 0.11 | 0.47 |
| 3,136 | 4,704 | 2.90 | **1.53** | 0.11 | 0.78 |
| 4,704 | 6,272 | 3.90 | **2.40** | 0.11 | 0.99 |
| 6,272 | 7,840 | 4.89 | **3.65** | 0.13 | 1.25 |
| 7,840 | 9,408 | 5.94 | **5.03** | 0.15 | 1.48 |
| 9,408 | 10,976 | 6.94 | 6.72 | 0.17 | 1.73 |
| 10,976 | 12,544 | 7.95 | 8.63 (slower) | 0.18 | 1.97 |
| 12,544 | 14,112 | 8.93 | 10.81 (slower) | 0.20 | 2.22 |
| 14,112 | 15,680 | 9.98 | 13.16 (slower) | 0.22 | 2.45 |

- The gather is cheap: 0.09 to 0.24 ms for up to 14k cached tokens (two cards' worth of K and V, 14 MB each), block size 16 or 784 alike.
- **The plugin-only form leads up to about 9,400 tokens in total** and loses beyond. Summed over a 16k prompt's chunks it saves (1.14 + 1.37 + 1.50 + 1.24 + 0.90) x 16 layers = 98 ms, 1.8% of the prefill; over a 7.7k prompt's chunks 64 ms, 2.7% [computed]. The comparison crosses jobs (ours standalone, vLLM's inside the server).
- **The window form would remove about three quarters of the attention time**: the 11 chunks of a 16k prompt cost 59.4 ms per layer on vLLM's kernel and 14.7 ms on the window form (0.42 for the fresh chunk, the column above, 0.85 for the last 435 rows), so (59.4 - 14.7) x 16 = 715 ms of a 16k default prefill, 13% [computed from the square kernel's measured times; the form does not exist].
- Chunk rows of the square form against float64 (context 4,704): 1.71e-3, finite.

**Done.** The plugin-only form is implemented behind `HLP_CTX=1` (plugin version 2, 214fb2dc; off by default): one sequence per step, `RocmAttentionImpl` only, up to `HLP_CTX_MAX` (9,408) total tokens, verify and guard as for fresh steps, buckets at every multiple of 784 (`HLP_BUCKETS_784=1`). Server results: `evidence/G2/RESULTS.md`.

## Addendum 11 Nov: on the server's own activations our prefill path is 2 to 6 times less accurate than vLLM's kernel (jobs 23626921, 23626922 dumps; 23627040 reference) [observed]

The verify arm's growing difference had to be attributed. `HLP_DUMP` (plugin a89a779a) saves, in a verify step whose difference from vLLM's kernel exceeds a threshold, the step's queries, all keys and values, our output and vLLM's output. `p1_dump_ref.py` computes the float64 attention of the same rows in chunks. 12 dumped layer calls (3 per worker per job; fresh one-step calls of 5,488 tokens, and 1,568-token chunks with 3,136 to 6,272 cached tokens through the new cached-context path):

| Call | rows | cached context | largest abs(q), abs(k) | **ours vs float64** (worst row) | **vLLM's kernel vs float64** (worst row) | ours vs vLLM's |
|---|---|---|---|---|---|---|
| 63399-1-ctx3136 | 1568 | 3136 | 8.6, 17.8 | 8.42e-03 (2.13e-02) | 1.67e-03 (1.99e-03) | 8.45e-03 |
| 63399-2-ctx4704 | 1568 | 4704 | 8.7, 17.8 | 1.09e-02 (2.72e-02) | 1.67e-03 (1.95e-03) | 1.09e-02 |
| 63399-3-ctx6272 | 1568 | 6272 | 11.6, 13.8 | 6.33e-03 (1.34e-02) | 1.67e-03 (1.90e-03) | 6.43e-03 |
| 63400-1-ctx3136 | 1568 | 3136 | 10.5, 21.6 | 6.95e-03 (1.63e-02) | 1.62e-03 (2.34e-03) | 6.96e-03 |
| 63400-2-ctx4704 | 1568 | 4704 | 10.9, 21.6 | 8.92e-03 (1.86e-02) | 1.62e-03 (2.38e-03) | 8.94e-03 |
| 63400-3-ctx6272 | 1568 | 6272 | 9.9, 21.0 | 6.47e-03 (1.65e-02) | 1.66e-03 (1.88e-03) | 6.50e-03 |
| 36960-1-fresh5488 | 5488 | 0 | 12.7, 13.8 | 3.31e-03 (9.34e-03) | 1.69e-03 (1.94e-03) | 3.43e-03 |
| 36960-2-fresh5488 | 5488 | 0 | 9.1, 17.8 | 6.32e-03 (2.31e-02) | 1.68e-03 (1.98e-03) | 6.35e-03 |
| 36960-3-fresh5488 | 5488 | 0 | 10.2, 20.2 | 3.31e-03 (1.49e-02) | 1.68e-03 (2.04e-03) | 3.33e-03 |
| 36961-1-fresh5488 | 5488 | 0 | 10.9, 21.0 | 3.50e-03 (1.09e-02) | 1.67e-03 (2.09e-03) | 3.50e-03 |
| 36961-2-fresh5488 | 5488 | 0 | 11.3, 21.6 | 5.65e-03 (1.68e-02) | 1.63e-03 (2.31e-03) | 5.65e-03 |
| 36961-3-fresh5488 | 5488 | 0 | 13.8, 16.5 | 2.95e-03 (6.63e-03) | 1.72e-03 (2.36e-03) | 3.14e-03 |

- **vLLM's kernel sits at bf16 rounding on real activations: 1.62e-03 to 1.72e-03. Ours reads 2.95e-03 to 1.09e-02, with single rows up to 2.7% off.** The difference the verify arms logged is our error, not vLLM's.
- This corrects the reading of the first addendum: on random inputs (job 23626788) our path matches float64 to 1.7e-3 at every length; on the model's activations it does not. Only calls that differed from vLLM's by more than 3e-3 (one-step) or 6e-3 (chunks) were dumped, so these are the worse calls, not a sample; the verify arms show most calls near 1.5e-3 to 2e-3 at 1.5k to 2.4k tokens and medians of 2.7e-3 to 4.9e-3 from 5.5k to 14k.
- **Cause [observed by emulation, jobs 23627057 and 23627069]: the fp16 path drops attention weights that are subnormal in fp16.** `p1_dump_hyp2.py` recomputes each dumped call in float64 with one change at a time. Rounding the unnormalised weights exp(s - rowmax) to fp16 costs 2e-5 to 5e-5: not the cause. Rounding them to fp16 **and setting those below fp16's smallest normal number (6.1e-5 of the row maximum) to zero** reproduces our kernel's error in all 12 calls:

  | Call | our kernel, measured | float64 with fp16 weights, subnormals dropped | float64 with fp16 weights, subnormals kept | float64 with bf16 weights | vLLM's kernel, measured | mean share of a row's weight below 6.1e-5 |
  |---|---|---|---|---|---|---|
  | 63399-1-ctx3136 | 8.42e-03 | 8.35e-03 | 2.77e-05 | 2.23e-04 | 1.67e-03 | 4.96e-03 |
  | 63399-2-ctx4704 | 1.09e-02 | 1.09e-02 | 2.55e-05 | 2.07e-04 | 1.67e-03 | 6.47e-03 |
  | 63399-3-ctx6272 | 6.33e-03 | 6.52e-03 | 4.18e-05 | 3.30e-04 | 1.67e-03 | 5.06e-03 |
  | 63400-1-ctx3136 | 6.95e-03 | 6.76e-03 | 2.06e-05 | 1.66e-04 | 1.62e-03 | 4.64e-03 |
  | 63400-2-ctx4704 | 8.92e-03 | 8.77e-03 | 1.93e-05 | 1.53e-04 | 1.62e-03 | 6.17e-03 |
  | 63400-3-ctx6272 | 6.47e-03 | 6.47e-03 | 2.52e-05 | 2.03e-04 | 1.66e-03 | 4.94e-03 |
  | 36960-1-fresh5488 | 3.31e-03 | 3.07e-03 | 5.00e-05 | 3.99e-04 | 1.69e-03 | 2.33e-03 |
  | 36960-2-fresh5488 | 6.32e-03 | 6.17e-03 | 3.73e-05 | 2.96e-04 | 1.68e-03 | 3.35e-03 |
  | 36960-3-fresh5488 | 3.31e-03 | 2.95e-03 | 3.80e-05 | 3.04e-04 | 1.68e-03 | 2.21e-03 |
  | 36961-1-fresh5488 | 3.50e-03 | 3.20e-03 | 3.46e-05 | 2.77e-04 | 1.67e-03 | 2.28e-03 |
  | 36961-2-fresh5488 | 5.65e-03 | 5.41e-03 | 2.70e-05 | 2.16e-04 | 1.63e-03 | 3.22e-03 |
  | 36961-3-fresh5488 | 2.95e-03 | 2.54e-03 | 4.76e-05 | 3.81e-04 | 1.72e-03 | 2.20e-03 |

  The measured column also holds the bf16 rounding of the output (1.7e-3, added in quadrature). The model's long contexts put 0.2 to 0.6% of a row's attention mass (up to 8% in single rows) on keys whose weight is below 6.1e-5 of the row's maximum; random inputs do not, which is why the float64 check on random inputs saw nothing. Where the drop happens is settled below: in the fp16 MFMA.
- **Fix, a kernel change [inferred]:** keep the weights in bf16 (no underflow near 1e-5; emulated error 1.5e-4 to 4e-4) and run the P x V product on the bf16 MFMA, which the GEMM track showed works on gfx90a (`v_mfma_f32_16x16x16bf16_1k`); V is bf16 in the model anyway, so the plugin's fp16 copies go away too. Not done here: it changes `gqa_fa.py`.
- **Consequence.** The time-to-first-token gains of the prefill-attention plugin stand as measured, but its output is less accurate than vLLM's on long real prompts and it must not be described as numerically equivalent. The G2 equivalence set reports what that does to greedy outputs. The GEMM plugin is not affected: its error equals vLLM's on the server's activations (differences 4e-6 to 1.2e-4).

## Addendum 11 Nov: the fp16 MFMA of gfx90a treats subnormal fp16 operands as zero (job 23627461, one GCD) [observed]

`p1_subnormal.py`: a matrix of one value x (fp16) times a matrix of ones, K 4096, so the exact result is 4096 x.

| x | fp16 class | x after a float32 -> fp16 cast on the GPU | expected | hipBLASLt fp16 (`F.linear`) | our hand kernel fp16 (`gemm_prefetch`) | hipBLASLt bf16 |
|---|---|---|---|---|---|---|
| 1e-3 | normal | 1.0004e-3 | 4.0977 | 4.0977 | 4.0977 | 4.0938 |
| 1e-4 | normal | 1.0002e-4 | 0.40967 | 0.40967 | 0.40967 | 0.41016 |
| 6.2e-5 | normal (smallest is 6.1e-5) | 6.199e-5 | 0.25391 | 0.25391 | 0.25391 | 0.25391 |
| 3e-5 | subnormal | 2.998e-5 | 0.12280 | **0.0** | **0.0** | 0.12305 |
| 1e-5 | subnormal | 1.0014e-5 | 0.04102 | **0.0** | **0.0** | 0.04102 |
| 1e-6 | subnormal | 1.013e-6 | 0.00415 | **0.0** | **0.0** | 0.00409 |

- The cast keeps subnormals; the product returns exactly zero for them, in AMD's own library and in our kernel alike (both use `v_mfma_f32_*f16`). bf16 has no subnormals in this range and its product is right.
- So the prefill kernel's error on real prompts is the fp16 MFMA dropping every attention weight below 6.1e-5 of the row maximum in the P x V product. It is a property of fp16 MFMA arithmetic on this GPU, and it applies to **any** fp16 matrix product there: an fp16 GEMM silently ignores operand values below 6.1e-5 in magnitude.
- Two fixes for `gqa_fa.py` (the second is done as a separate file, next addendum): (1) the P x V product on the bf16 MFMA with bf16 weights; (2) keep fp16 and scale the weights by a power of two before the cast (2^10 moves the threshold to 6e-8 of the row maximum and stays far below fp16's largest value), dividing it out with the row sum. (2) is a one-constant change to the kernel's device source; both need the track's races and a float64 check on dumped activations (`p1_dump_ref.py` and the 12 dumped calls, which are not included, are the test set).
- The decode kernel (`gqa_decode.py`) keeps its weights in bf16 (`dt="bf16"`), so this does not apply to it [read from source, not measured].

## Addendum 11 Nov: the cached-context form in a server (job 23626852, small-g, 2 GCDs, order fv s f f s, vLLM defaults, `HLP_CTX=1`) [observed, one job]

| Cell | stock reps | plugin reps | plugin / stock | verdict against the job's noise |
|---|---|---|---|---|
| 1,419 | 0.5160, 0.5158 | 0.5147, 0.5177 | 1.0006 | inside (0.58%) |
| 2,859 | 1.0014, 1.0014 | 0.9992, 1.0023 | 0.9994 | inside (0.31%) |
| 5,679 | 1.9759, 1.9754 | 1.9493, 1.9540 | **0.9879** | faster |
| 7,719 | 2.6975, 2.7044 | 2.6444, 2.6493 | **0.9800** | faster |
| 14,201 | 5.2914, 5.3002 | 5.2200, 5.2249 | **0.9861** | faster |
| 4 x 2,859 | 2.6957, 2.6979 | 2.7074, 2.7013 | 1.0028 | slower (0.23%) |

Counters per worker and repetition: 800 fresh layer calls and 1,600 cached-context calls taken (context 1,568 to 7,840), 480 left to vLLM because the total exceeded 9,408 tokens, 480 mixed, 0 errors, 0 guard trips. Verify arm: differences from vLLM's kernel 1.5e-3 to 1.5e-2, the subnormal error above. Against the same plugin without the cached-context form (jobs 23619582, 23624337: 1.002 to 1.003 at these lengths) this is the computed 1 to 2%. One job on shared cards: the whole-node pair is in `evidence/G2/`.

## Addendum 11 Nov: the fix as a new file, `tilelang/gqa_fa_ps.py` (962f7c71): weights scaled by 2^10 before the fp16 cast (gate job 23627530, one GCD) [observed]

`gqa_fa.py` is not edited. `gqa_fa_ps.py` imports a private copy of it and adds `+ 10 ln 2` to the exponent of the weights in the device source that copy compiles (four lines, after `gqa_fa`'s own source edits; it raises if they are not found exactly). Every weight is multiplied by 1,024, the row sum is accumulated from the same scaled weights, so the factor cancels exactly in acc_o / l_i; the largest scaled weight is 1,024 (fp16 holds 65,504) and the drop threshold moves to 6e-8 of the row maximum. Tiles, schedule and MFMAs are unchanged. The plugin uses it with `HLP_PSHIFT=1` (version 3, dfe7407e; off by default).

`p1_ps_check.py` runs both kernels on the 12 dumped server calls exactly as the server does, each against float64, and times them interleaved with an identical-code control:

| Call | vLLM's kernel | landed `gqa_fa` | **`gqa_fa_ps`** (worst row) | landed ms | scaled ms | landed / scaled | control noise |
|---|---|---|---|---|---|---|---|
| 63399-1-ctx3136 | 1.67e-03 | 8.42e-03 | **1.69e-03** (2.02e-03) | 1.352 | 1.354 | 0.9983 | 0.06% |
| 63399-2-ctx4704 | 1.67e-03 | 1.09e-02 | **1.70e-03** (2.03e-03) | 2.154 | 2.198 | 0.9800 | 0.02% |
| 63399-3-ctx6272 | 1.67e-03 | 6.33e-03 | **1.68e-03** (1.93e-03) | 3.346 | 3.347 | 0.9996 | 0.03% |
| 63400-1-ctx3136 | 1.62e-03 | 6.95e-03 | **1.65e-03** (2.34e-03) | 1.342 | 1.342 | 1.0000 | 0.16% |
| 63400-2-ctx4704 | 1.62e-03 | 8.92e-03 | **1.65e-03** (2.40e-03) | 2.153 | 2.168 | 0.9929 | 0.03% |
| 63400-3-ctx6272 | 1.66e-03 | 6.47e-03 | **1.69e-03** (1.92e-03) | 3.351 | 3.334 | 1.0052 | 0.10% |
| 36960-1-fresh5488 | 1.69e-03 | 3.31e-03 | **1.68e-03** (1.94e-03) | 1.897 | 1.948 | 0.9739 | 0.01% |
| 36960-2-fresh5488 | 1.68e-03 | 6.32e-03 | **1.70e-03** (2.01e-03) | 1.913 | 1.952 | 0.9800 | 0.02% |
| 36960-3-fresh5488 | 1.68e-03 | 3.31e-03 | **1.69e-03** (2.17e-03) | 1.899 | 1.952 | 0.9729 | 0.05% |
| 36961-1-fresh5488 | 1.67e-03 | 3.50e-03 | **1.69e-03** (2.14e-03) | 1.880 | 1.947 | 0.9656 | 0.08% |
| 36961-2-fresh5488 | 1.63e-03 | 5.65e-03 | **1.65e-03** (2.35e-03) | 1.883 | 1.950 | 0.9659 | 0.01% |
| 36961-3-fresh5488 | 1.72e-03 | 2.95e-03 | **1.72e-03** (2.37e-03) | 1.890 | 1.957 | 0.9656 | 0.01% |

- **Accuracy is back at vLLM's level on all 12 calls: 1.65e-03 to 1.72e-03** (vLLM's kernel 1.62e-03 to 1.72e-03; the landed kernel 2.95e-03 to 1.09e-02). Random inputs unchanged: n 4096 x1: 1.657e-03 landed, 1.657e-03 scaled; n 16384 x1: 1.667e-03 landed, 1.667e-03 scaled; n 4096 x4: 1.488e-03 landed, 1.488e-03 scaled.
- **Cost: 0 to 3.4% of kernel time.** Landed / scaled 0.966 to 1.005 in the forward job (23627530) and 0.968 to 1.001 in the reversed one (23627606), control noise at most 0.19%: slower by 2.0 to 3.4% at the 5,488-token fresh calls and by 0.7 to 2.0% at a total length of 6,272 in both orders, within 0.5% at 4,704 and 7,840. The cost is the one added vector add per K tile.
- A first version that changed the Python expression instead read 1.4 to 2.2 times slower (job 23627483); that was the test, not the change: with two copies of `gqa_fa` in one process only the first installs the compile hook that applies the hand edits, so the copy's kernels were compiled without them. `gqa_fa_ps.load()` now stacks its hook; the docstring says in which order the modules may be loaded.
- Not done: other tiles than `d256_tile`'s; D 128 forms; lengths other than the dumped ones. Server results with `HLP_PSHIFT=1`: `evidence/G2/RESULTS.md` (G3 jobs).
