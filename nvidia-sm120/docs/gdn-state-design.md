# GDN recurrent-state work: design, cost model, candidates (29-09-2026)

Started 29-09-2026. This phase was CPU-only: analysis, kernels, offline
compile, CPU tests and a staged GPU job. **No GPU correctness or speed result exists yet.** Every speed number below
is a paper model unless it is marked as measured.

## Summary

- **The problem.** At 32 users, the GDN (linear attention) decode kernel took 30% of every decode step (16.7 of
  55.7 ms, profile of 29-09-2026 00:41, 3 drafts). The kernel is not slow. It moves too many bytes. Each request
  keeps a 3.1 MB fp32 state per layer. Every step, the kernel reads that state once and writes it back once per
  window token (1 + drafts). Speculative decoding needs this, because the kernel does not yet know which drafts
  will be accepted. Only one of those written states is ever read again. At 32 requests with 3 drafts, that is
  503 MB per layer per step, 80% of it writes. The kernel ran at 1.44 TB/s, about 80% of the DRAM peak.
- **Why `HORIZON_GDN=1` (G1) "never activated".** It did activate, in both G1-only arms on 27-09 (23:19:10 and
  23:41:38). Three things hid it:
  - The marker was printed during CUDA-graph capture, right after tqdm's carriage-return progress bar. The service log
    stored the second entry as binary, and a plain log grep shows such entries as `[N blob data]`.
  - The 29-09 "G" arms also set `HORIZON_GDN_RECOVER=1`. G2 replaces the whole GDN decode path, so G1's op is never
    called there.
  - The 29-09 profile ran on the baseline env, which does not set `HORIZON_GDN`.

  Fixed: the markers now print on their own line, and there is an install-time marker. Always search logs with
  a mode that prints binary entries.
- **Chosen approaches.** Both are implemented as TileLang kernels, compiled for SM120, CPU-tested and staged for the
  GPU:
  1. **G1S, 16-bit state** (`HORIZON_GDN_STATE=fp16|bf16`, plus `HORIZON_GDN2=1` for our kernel). It halves the
     state bytes and the per-request GDN memory with almost no integration work. vLLM already supports bf16.
     - fp16 is the recommended variant: the CPU study puts its error at about the level of the bf16 rounding that
       vLLM already applies to the layer output.
     - bf16's error is about 8x larger than fp16's.
     - Both change numerics, so they need the quality gate below.
  2. **G3, replay ring.** Each request keeps one checkpoint plus a small ring of exact per-token update records.
     Each step replays the committed tokens onto the checkpoint in registers and writes the checkpoint only every
     ~16 tokens. With an fp32 checkpoint the result is **bit-identical** to vLLM by construction, and the state
     traffic falls about 2.5x (1 draft) to 3.5x (3 drafts). An fp16 checkpoint roughly halves it again and rounds
     only once per ~16 tokens. The vLLM integration is real work (slot allocation, ring reset, materialize before
     non-decode paths). It is designed here but not built.
- **Expected effect (model).** At 32 users with the live draft schedule (1 draft), the GDN core would go from about
  10 ms per step to about 5 ms (G1S fp16), 4 ms (G3 fp32) or 2.2 ms (G3 fp16). At 8 users (3 drafts) the saving is
  about 1.5-2.1 ms of a ~19 ms step. At 1 user it is small (~2%), because that case is latency-bound.
- **Memory.** Each request currently holds 12 GDN cache blocks (3 GDN layer groups x (1 + 3 drafts) slots). That
  is 19,200 token-equivalents of the 1.3M-token pool, whatever its length. The 16-bit state halves this. G3 needs
  1 slot per group plus a 50 MB ring. At 4k-token conversations the pool holds about 54 requests today, 95 with a
  16-bit state, 116 with G3 fp32 and 162 with G3 fp16 (block accounting, not measured).

## 1. What the decode kernel does (vLLM 0.29 source)

Path: `QwenGatedDeltaNetAttention.forward_cuda` -> custom op `vllm.qwen_gdn_attention_core_fused_norm_packed` (opaque
to torch.compile) -> `_forward_core_fused_norm` -> pure speculative batches (`num_prefills == 0`, `num_decodes == 0`)
-> `_forward_core_decode_spec_fused_norm`:
1. `causal_conv1d_update` (Triton, conv state in slot `state_indices[r, 0]`, width 3 + drafts, bf16: ~123 KB per
   request and layer, separate kernel, 6 us per call at 32 users);
2. `ops.fused_gdn_decode_post_conv_mtp` (looked up on the `vllm._custom_ops` module at call time, which is why a
   plugin monkeypatch works) -> `torch.ops._C.fused_gdn_decode_post_conv_mtp`, CUDA kernel
   `gdn_decode_post_conv_mtp_kernel<StateT, ValueHeadsPerKeyHead, SigmoidGate>` from
   `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`. The profile's `<float, 3, false>` means **fp32 state, 3
   value heads per key head (48 / 16), SiLU gate**. The 3 is not the draft count, which is a runtime value (tokens
   per request from `cu_seqlens`).

Grid (requests, 48 value heads), 256 threads. Per (request r, head):
- source slot `state_indices[r, n_acc[r] - 1]` (the state after the last accepted token of the previous window);
- read the 128 x 128 fp32 head state (64 KB) through a 2-stage cp.async pipeline (32 rows per stage);
- per window token t: `h *= decay_t; delta = (v - h k) beta_t; h += k delta; o_t = h q`, then **store h into slot
  `state_indices[r, t]`** (every token, whether or not it will be accepted);
- gated RMSNorm of o over the head's 128 values, write bf16.

Mamba cache layout (from `mamba_get_block_table_tensor`, align mode = prefix caching on): each request's GDN blocks
are the last `1 + num_speculative_blocks` = 4 blocks of its block table (config maximum 3 drafts; the per-batch
schedule does not shrink the allocation). One block = one 17-layer page position; the page holds the conv state
(10240 x 6 bf16) and the SSM state (48 x 128 x 128 fp32) = 817,152 floats, padded to 819,200 (the state slot stride
seen in the logs). 64 layers + the MTP attention layer = 17 attention + 48 GDN, grouped in 4 groups of 17 (3 padding
layers): 3 GDN groups.

## 2. Cost model

State bytes per request and layer: S = 48 x 128 x 128 x 4 = 3.15 MB. Per step the kernel moves S x (1 + T), T =
tokens per request (1 + drafts). Inputs and outputs per token are ~45 KB (negligible).

| Requests | T | Bytes per layer | Floor at 1.63 TB/s | Measured per call | Note |
|---|---|---|---|---|---|
| 1 | 4 | 15.7 MB | 9.6 us | 14.1 (profile), 13.4 (standalone) | 48 CTAs on 188 SMs: latency / occupancy bound |
| 1 | 2 | 9.4 MB | 5.8 us | - | |
| 8 | 4 | 126 MB | 77 us | 56.1 (profile), 91.9 (standalone, states rotated past L2) | profile below the floor: the ~100 MB of writes land in the 128 MB L2 and are written back during later kernels |
| 8 | 2 | 75 MB | 46 us | - | |
| 32 | 4 | 503 MB | 309 us | **348.5 (profile)**, 371 (standalone) | 1.44 TB/s; 402 MB of the 503 are writes |
| 32 | 2 | 302 MB | 185 us | - (est. ~210) | draft schedule (1 draft above 10 requests), not yet profiled |

**Why 348 us at 32 requests.** The call moves 503 MB at 1.44 TB/s, which is 89% of the 1.63 TB/s this GPU sustains
in our GEMM races. There is no compute or latency problem to fix inside this algorithm. G1, our row-split port, only
wins where the grid is too small (1 user: 11.2 vs 13.4 us). At 32 requests it was 369.7 vs 370.9 us.

**Removable traffic.** Of the 1 + T state passes, only one read and one eventual write are needed per accepted state:
- the T - 1 speculative writes that are never read (acceptance is known only after sampling);
- the write of the state that IS used, which could be deferred and batched over many steps.

With the live schedule at 32 users (T = 2), that is 2 of the 3 passes. At <= 10 users (T = 4) it is 4 of the 5.

Paper model per layer and step (bytes and time at the 1.44 TB/s vLLM reaches). Acceptance means: mean committed tokens
per step 3.4 (T = 4) / 1.6 (T = 2). G3: ring half 16, record = 33 KB per token (fp32 delta 24.6 KB + k 8.2 KB +
decay):

| Requests, T | vLLM / G1 fp32 | G1S 16-bit | G3 fp32 checkpoint | G3 16-bit checkpoint |
|---|---|---|---|---|
| 1, T=4 | 15.7 MB, 10.9 us | 7.9 MB, 5.5 us | 4.4 MB, 3.1 us | 2.4 MB, 1.7 us |
| 8, T=4 | 126 MB, 87 us | 63 MB, 44 us | 35 MB, 25 us | 19 MB, 13 us |
| 8, T=2 | 75 MB, 52 us | 38 MB, 26 us | 31 MB, 21 us | 17 MB, 12 us |
| 32, T=4 | 503 MB, 350 us | 252 MB, 175 us | 142 MB, 98 us | 77 MB, 53 us |
| 32, T=2 | 302 MB, 210 us | 151 MB, 105 us | 123 MB, 85 us | 66 MB, 46 us |

Per decode step (48 layers) at 32 users with the schedule: ~10.1 ms today. The models give 5.0 ms (G1S), 4.1 ms
(G3 fp32) and 2.2 ms (G3 16-bit). Step time with the schedule has not been profiled; with the pre-schedule 55.7 ms
step as reference, that is 9-14% of a step. At 1 user every variant is latency-bound (~5-8 us per call is realistic),
worth ~0.3 ms of a ~15 ms step.

## 4. Candidates evaluated

| Candidate | Traffic | Memory per request | Numerics | Integration | Verdict |
|---|---|---|---|---|---|
| G1 (row split, fp32) | 1 + T passes | unchanged | bit-identical | done (plugin) | only 1-user latency; no change at load |
| **G1S bf16** (vLLM-native, or ours) | (1 + T) / 2 | half | rounded every step; CPU study median output error 0.3-1.2% | config only (vLLM's kernel has a bf16 instantiation) | fallback, if fp16 overflows |
| **G1S fp16** (ours) | (1 + T) / 2 | half | rounded every step; error ~8x below bf16, near the existing bf16 output rounding | plugin: state dtype + fused-dtype list + our kernel (vLLM has no fp16 kernel) | **first A/B candidate** |
| bf16 + stochastic rounding | as bf16 | half | unbiased, but ~40% larger error than RNE in the study | ours | rejected (study) |
| G2 (RecoverSSM: verify + commit) | 3 passes | 1 slot | bit-identical | done; measured 29-09 **3-6% slower** with the schedule | parked: at T = 2 it moves exactly as many bytes as vLLM |
| **G3 replay ring, fp32 checkpoint** | ~1.2-1.4 passes | 1 slot + 50 MB ring | **bit-identical** | kernel done; vLLM integration designed (section 6) | **main exact path** |
| **G3, 16-bit checkpoint** | ~0.6-0.7 passes | half slot + ring | rounded once per flush (~16 tokens): 2-3x below per-step bf16 | as G3 | **max-throughput path** |
| Discard dead writes in L2 | - | - | - | the next step comes ~20 ms later; the lines are evicted by then | not feasible |

## 5. CPU precision study (synthetic, `scripts/kernels/gdnstate/state_precision_sim.py`)

Recurrence as vLLM computes it (fp32 inside a step). The state is rounded where it leaves the registers: after every
decode step (1-4 accepted tokens), every 16 tokens (G3), or every 8192 tokens (prefill chunks). The inputs use the
**real gating parameters of every GDN layer** (A_log, dt_bias, and the scale of a / b from the in_proj_a/b row
norms), extracted from the checkpoint with numpy (`extract_gdn_params.py`). q, k and v are synthetic (512 recurring
key prototypes plus noise). 48 heads per layer, 32,768 tokens.

Gating facts from the real weights: at a = 0, 30% of all heads decay by less than 0.4% per token (1 - g < 4e-3) and
15% by less than 0.1%. Those are exactly the heads where rounding to 8 mantissa bits (bf16) can swallow a whole
step's decay.

Median / p99 relative error of the pre-norm layer output (per token and head, over 128 values), tokens 16k-32k.
Yardstick: the bf16 rounding vLLM already applies to that output is 0.16% / 0.21%.

| Layer | bf16, every step (T=4) | bf16 + stochastic rounding | **fp16, every step** | bf16 every 16 tokens (G3) | bf16 every 8192 (prefill) |
|---|---|---|---|---|---|
| 0 | 0.88% / 1.89% | 1.25% / 2.65% | **0.11% / 0.24%** | 0.39% / 0.84% | 0.00% / 0.11% |
| 1 | 0.11% / 1.75% | 0.15% / 2.46% | **0.01% / 0.22%** | 0.00% / 0.78% | 0.00% / 0.04% |
| 24 | 0.27% / 1.59% | 0.38% / 2.27% | **0.03% / 0.20%** | 0.12% / 0.71% | 0.00% / 0.08% |
| 45 | 0.29% / 1.54% | 0.41% / 2.19% | **0.04% / 0.19%** | 0.13% / 0.69% | 0.00% / 0.06% |
| 62 | 0.32% / 1.64% | 0.45% / 2.33% | **0.04% / 0.21%** | 0.15% / 0.73% | 0.00% / 0.07% |

With 1 draft (T = 2, rounding every ~1.6 tokens; all 5 layers): bf16 0.16-1.17% / 2.03-2.50%, **fp16
0.02-0.15% / 0.26-0.31%**, bf16 every 16 tokens 0.01-0.41% / 0.71-0.87%. That is ~30% more error than at T = 4,
because the state is rounded more often. 131,072 tokens (layer 0, 12 heads, T = 2): bf16 0.81% / 2.61%, fp16
0.10% / 0.33%, bf16 every 16 tokens 0.28% / 0.90%, flat from 4k to 128k. The state's absolute maximum was 5-11
in every run.

Findings:
- **No runaway drift.** The error reaches its level within ~1k tokens and stays flat to 32k, and to 128k in the
  long run. The delta rule corrects the state along every key it sees again.
- **fp16 is ~8x more accurate than bf16 at the same bytes**, and its p99 is at the yardstick. The largest state value
  was 11 (the fp16 limit is 65,504), but these are synthetic activations: real maxima must come from the GPU gate.
- Stochastic rounding makes bf16 worse (more variance), so it is not implemented in the kernels.
- Rounding once per prefill chunk is harmless. **A prefill-only log-prob gate cannot see this risk.** The gate has to
  exercise the decode path.

Results: `results/gdnstate_29-09-2026/precision_sim_*.jsonl`.

## 6. Kernels (fork `horizon/gdn-state`, `tilelang_contrib/horizon_ops/gdn_state.py`)

- **`gdn_window_kernel` (G1S).** vLLM's contract with the state stored as float32, bfloat16 or float16. G1's
  arithmetic is reused (`GDN_SRC`: vLLM's fast-math instructions spelled out in PTX), and so is G1's row split (VR
  rows per CTA) and the last-CTA gated norm. 16-bit states are addressed as int16, with explicit `cvt.rn.bf16.f32` /
  `cvt.rn.f16.f32` (fp16 saturates to +-65504 first). fp32 and bf16 are designed to be bit-identical to vLLM's
  kernel.
- **`gdn_replay_kernel` (G3).** Per (request, head, row chunk):
  - Compute `ncom = nbuf + min(n_acc, npend)` from the ring meta.
  - Replay: `h = h * g_j; h = fma(k_j, delta_j, h)` for the committed records. These are the verify's own two
    instructions with the saved fp32 delta, so the result is bit-exact.
  - Flush (write the checkpoint, switch ring halves) when `ncom + T > HALF` or on a forced flush.
  - Compute the window outputs as G1, saving delta (per row), k (per key head) and decay (per head) into the ring.
  - Two ring halves guarantee that a flush never overwrites a record another CTA of the same launch is still
    replaying.
  - Meta is kept per 32-row chunk, so each CTA only touches its own entries (no cross-CTA race and no extra ticket).
- **`gdn_materialize_kernel`.** Folds the committed records into the checkpoint and empties the ring. It is for
  mixed batches, prefix-cache copies and preemption.

Offline SM120a compile (`gdnstate_lower.py`, no CUDA context, production strides; results in
`results/gdnstate_29-09-2026/lower-summary.json`):

| Kernel | Rows per CTA | Registers | Spills | Shared memory |
|---|---|---|---|---|
| window f32 / bf16 / f16 | 32 | 79 / 78 / 79 | 0 | 8.8 KB |
| window f32 / bf16 / f16 | 64 | 93 / 95 / 95 | 0 | 9.3 KB |
| window f32 / bf16 / f16 | 128 | 94 / 95 / 95 | 0 | 10.3 KB |
| replay f32 / bf16 / f16, HALF 16 | 32 | 96 / 96 / 96 | 0 | 19.1 KB |
| replay f32 / bf16 / f16, HALF 32 | 32 | 96 / 96 / 96 | 0 | 29.4 KB |
| replay f32 / bf16 / f16, HALF 16 | 64 | 124 / 126 / 126 | 0 | 21.6 KB |
| replay f32 / bf16 / f16, HALF 32 | 64 | 124 / 126 / 126 | 0 | 34.0 KB |
| materialize f32 / bf16 | 32 | 92 / 95 | 0 | 0 |

For comparison, G1 used 80 registers. 16-bit state loads and stores compile to `short4` (8-byte) vector accesses,
and fp32 to `float4`. Compile success says nothing about GPU correctness or speed.

## 7. CPU tests (all pass, 29-09-2026)

`scripts/kernels/gdnstate/test_gdn_state_cpu.py` (torch reference `gdn_state_ref.py`), 29 checks:
- bf16 RNE bit trick equals torch's conversion (200k values incl. ties / denormals), and fp16 stores saturate;
- **G3 protocol equals vLLM's window protocol bit for bit** (fp32): outputs every step, and the checkpoint after
  materialize equals vLLM's slot `n_acc - 1`. Coverage:
  - 160-step chains, window lengths 1-4 and 1-2;
  - ring halves 8 / 16 / 32, forced flushes, padding requests, an invalid slot;
  - production geometry (48 / 16 heads);
- the ring invariants hold: reads and writes of one launch never overlap, and the meta entries agree;
- the flush rate matches the model;
- the 16-bit G3 checkpoint has a smaller error than the per-step 16-bit window (bf16 2.9e-3 vs 5.0e-3, fp16 7.1e-4 vs
  1.4e-3);
- the plugin (vLLM mocked):
  - `HORIZON_GDN_STATE=fp16` sets the state dtype and adds float16 to vLLM's fused-dtype list;
  - every dtype routes to our kernel, and an uncovered fp32 call falls back to vLLM;
  - an uncovered fp16 call raises;
  - with the switches unset, nothing is touched, and a bad value is refused;
- the GPU-result verifier accepts a complete synthetic set and rejects 6 kinds of broken sets.

`test_gdn_state_harness_cpu.py` runs the GPU test's own harness code on CPU, with the reference standing in for both
kernels. The replay chains at production geometry (2/8/32 requests, QL switching 2 <-> 4, padding, forced flushes,
materialize) give 0 diffs, and the verifier passes on its records.

## 9. In-vLLM A/B plan (draft, after the standalone gate passes)

**Phase 1: G1S (cheap to integrate).** Arms on a copy of the baseline env (ops_arm-style, order P1 K F B P2):
- **P**: production (fp32 state, vLLM kernel);
- **K**: `HORIZON_GDN2=1` (our kernel, fp32 state). It must be bit-identical to P: sequential decode fingerprint
  10/10 with bit-equal log-probs, plus prompt log-probs;
- **F**: `HORIZON_GDN_STATE=fp16 HORIZON_GDN2=1`;
- **B**: `HORIZON_GDN_STATE=bf16` (vLLM's own bf16 kernel).

Gates for F and B (all of them):
1. **Speed.** `bench_speed` 1 / 8 users and `bench_load` 32, A/B/A/B, with no loss above noise at 1 user. The
   32-user gain must be at least 5%.
2. **Memory / capacity.** The log's block size and KV tokens, plus a saturation run: 48 requests of 8k tokens
   against max-num-seqs 64 in a dev arm, counting running requests. The model predicts ~1.8x the requests per GB at
   short contexts.
3. **Long-context prefill quality.** The existing prose / code NLL at 16k / 32k, within PPL x1.10 (existing gate),
   run twice:
   - at the production chunk of 8192;
   - with `--max-num-batched-tokens 256`, so that the state is rounded every 256 tokens. Tighter threshold x1.02.
   This is still the prefill path.
4. **Decode-path quality** (the real risk; section 5):
   - AIME 2025 thinking (30 problems, up to 32k generated tokens) and a 100-problem MATH-500 thinking subset,
     paired vs P: no drop beyond paired noise (sign test p > 0.1 and at most 3 points);
   - greedy 4k-token continuations of 10 long prompts (8k / 32k / 64k). Report agreement with P and the first
     divergence position. This is report only: different numerics diverge by design.
5. **Needles** at 64k / 128k / 250k: 3/3 (existing).
6. **fp16 overflow check.** A probe (to add: `HORIZON_GDN_STATE_PROBE=1` logs the absmax of 4 layers' states every
   N steps) must stay below 1e4 on the long-generation runs. Any value at 65,504 fails F.

Use only if F (or B) passes every gate.

**Phase 2: G3 (after its standalone gate).** Integration, as a new plugin module `gdn_replay.py`:
- **Slots.** One checkpoint slot per request and GDN group: reuse G2's `use_kda_recoverssm` switch
  (`num_speculative_blocks = 0`).
- **Ring and meta storage.** A per-request side tensor indexed by the V2 runner's stable request-state index, or
  extra page tensors like KDA's RecoverSSM record.
- **Meta reset.** Set meta to 0 for every request that is prefilled or newly scheduled (in the metadata builder,
  graph-safe index fill).
- **Materialize** (the kernel exists) before:
  - any batch where a spec request goes through FLA's mixed-batch path;
  - the prefix-cache align copies (`postprocess_mamba` at block boundaries);
  - preemption.
- **Conv state:** vLLM's conv-compaction as in G2.

Gates:
- decode fingerprint bit-identical to P (sequential 10/10, long 3/3, log-probs bit-equal);
- prompt log-probs bit-identical;
- speed A/B/A/B at 1 / 8 / 32;
- capacity.

G3 with an fp16 checkpoint then takes the phase-1 quality gates.

## 11. First GPU gate (job 60, 29-09-2026 12:12 UTC) and the fix

**Result.** Marker `status=GATE_FAILED`. The records are in
`results/gdnstate_29-09-2026/gpu-results-job60.jsonl`.

| Check | Result |
|---|---|
| G1S window fp32 / bf16 vs vLLM | **not bit-identical** in all 56 cases (for example 1 request, QL 2: 6 of 12,288 outputs and 417k of 1.57M state elements differ) |
| G1S fp16 vs our fp32 kernel + torch rounding | **bit-identical** in 6/6 cases, incl. saturation (24,474 values clamped at +-65,504, none inf) |
| G3 chained vs vLLM | not bit-identical (it inherits the window arithmetic) |
| Determinism + CUDA-graph replay | pass, all 5 kernels |
| Speed | as modelled (table below) |

**Cause: a kernel arithmetic defect, not a harness bug and not an expected tolerance issue.** It was inherited from G1.
- `gdn_state` used `gdn_decode.GDN_SRC`: the explicit-PTX transcription of vLLM's fast-math helpers, introduced in
  56935fbe (27-09 17:20) so that the bodies do not depend on `--use_fast_math`.
- That transcription writes `mul.rn.ftz` / `add.rn.ftz`. vLLM's fast-math build emits `mul.ftz` / `add.ftz`
  without `.rn` (checked in our PTX), and ptxas is free to fuse those into FFMA. With the `.rn` forms it may not, so
  the roundings differ.
- G1's own gate shows the same break:
  - 27-09 14:24 (eb285529, verbatim helpers + `--use_fast_math`): 42/42 bit-identical, 1000 chained steps with a
    difference of 0;
  - 27-09 23:06 run (56935fbe): `bit_diffs_total 636,946,007`, `drift_ok false`, first bad step 0. This was
    recorded but never flagged.

  So **G1 (`HORIZON_GDN=1`) on `horizon/ops-bodies` is not bit-identical to vLLM either**. That is harmless for
  production, where it is off, but the "bit-identical" claims for it and for G2's verify body are stale.
- Fix (fork 6fcb6981): `gdn_state` carries vLLM's helpers verbatim, compiles with `--use_fast_math` and uses plain
  body arithmetic, exactly the eb285529 scheme. The G3 replay keeps `h * g` and `fma(k, delta, h)` as explicit
  instructions, because a lone multiply-add pair could otherwise be contracted differently from the window loop, where
  `h * g` has two uses.
- Offline compile: 23/23 variants, 0 spills.
- GPU confirmation is pending: job 64.

**Speed measured in job 60** (us per layer; CUDA-graph replay over more than 512 MB of rotated state). Speed does not
depend on the rounding detail that failed. At 32 users with the draft schedule (QL 2):

| Arm | us per layer | vs vLLM fp32 |
|---|---|---|
| vLLM fp32 | 219.0 | 1.00 |
| vLLM bf16 | 108.4 | 0.49 |
| G1S bf16 / fp16 | 106.4 / 106.6 | 0.49 |
| G3 fp32 | 87.9 | 0.40 |
| G3 bf16 / fp16 | 76.1 / 78.4 | 0.35 |

Per 48-layer step: vLLM fp32 10.5 ms, 16-bit 5.1 ms, G3 fp32 4.2 ms. Other points:
- **QL 4 at 32 requests:** 372.7 / 194.0 / 185.9 (G1S fp16) / 119.5 (G3 fp32) / 111.4 (G3 fp16).
- **8 requests, QL 4:** 91.3 / 48.2 / 41.1 / 35.4.
- **1 request, QL 4:** 13.5 / 11.8 / 8.4 / 11.8. G3 does not help at 1 user.
- **HALF 32** is slower than HALF 16 in every case, so keep 16.
- **G3 with a 16-bit checkpoint** gains little over G3 fp32 (88 vs 76 us): replaying the ring then dominates, not the
  checkpoint read.

Report-only drift at 2000 steps (real layer-24 gating), relative error of the outputs vs fp32:

| Arm | Median | p99 |
|---|---|---|
| bf16 window | 0.28% | 1.4% |
| fp16 window | 0.056% | 0.45% |
| G3 bf16 | 0.11% | 0.65% |
| G3 fp16 | 0.012% | 0.36% |

This matches the CPU study's ordering.

**Recommendation.** Run the in-vLLM A/B of the 16-bit state next, before the G3 integration:
- **B arm (vLLM's own bf16 kernel, `--mamba-ssm-cache-dtype bfloat16`).** Config only. It gets the whole 2x kernel
  win (108 vs 219 us), halves per-request GDN memory, and does not depend on our kernel's bit-identity.
- **F arm (fp16 on our G1S kernel).** About 8x more accurate than bf16 in both the CPU study and the GPU drift, at the
  same speed. It needs job 64 to pass first (the runner skips F otherwise).
- **G3 fp32.** The exact path, and the fastest at 8-32 users: 0.40x of vLLM at QL 2 and 0.32x at QL 4. It stays the
  follow-up, because its vLLM integration is real work. Its standalone gate is job 64's `replay_chain`.

  The arms are P, B, F and K (V with `FV=1`), in the order P1 B1 F1 K1 P2 B2 F2. Round 1 adds `logprob_eval`,
  needles, the long-decode fingerprint and AIME 2025 + MATH-500[:100] thinking.

## 12. G3v: G3 integrated into vLLM 0.30 (30-09-2026)

Started 30-09-2026: integrate, measure and report. The switch is
`HORIZON_GDN_G3=1`, default off. Nothing below was run on the GPU yet.

**How vLLM 0.30 lays the state out.** This is from the 0.30 source: the V2 model runner, `mamba_hybrid.py`,
`v1/worker/mamba_utils.py` and `gdn_attn.py`.
- **Window slots.** With prefix caching in "align" mode, every request has a window of SW = 1 + 3 state blocks per
  GDN KV-cache group, `slot[k] = block_table[(seq_len - 1) // B + k]`.
- **The fused decode kernel.** It reads `slot[n_acc - 1]` and writes the state after window token t into `slot[t]`.
- **Pre-copy.** When the window moves to a new block, the align pre-copy (`preprocess_state`) copies
  `slot[n_acc - 1]` into the new running block and resets n_acc to 1.
- **Post-copy.** The align post-copy (`postprocess_state`) saves the boundary state `slot[bias]` into the completed
  block.
- **Mixed batches** go through FLA, which reads `slot[n_acc - 1]`.
- **Conv state.** vLLM's conv kernels read and write only `slot[0]`'s conv region, so the conv regions of slots 1..3
  (122,880 B each) are never used.

**G3v (fork `horizon/g3v-0115`, `tilelang_contrib/horizon_ops/gdn_g3v.py`):**
- **Where the ring lives.** The replay ring sits in those unused conv regions, so it needs **no extra GPU memory**.
  Each record is 32,960 B (fp32 delta, normalised k and decay), so a slot holds 3 records. The ring has 9
  positions, half = 4 ≥ window 4.
- **Meta.** Meta is a 2.4 MB side table indexed by vLLM's request-state index. H4 hands the kernel that index through
  a static buffer that is safe inside CUDA graphs.
- **Mode A** is vLLM's layout. Entering mode B flushes the checkpoint into `slot[0]` when n_acc > 1.
- **Mode B** keeps the checkpoint in `slot[0]` plus the committed records.
- **Expand.** A persistent-grid kernel rewrites exactly vLLM's per-token slots and resets the ring. It runs before
  anything else reads the slots.
- **Arithmetic** is gdn_state's, the scheme that job 64 measured bit-identical.
- **Offline SM120a compile on TileLang 0.1.15:** step 117 registers, expand 133, 0 spills.

**Plugin (`horizon_kernels/gdn_g3.py`).**

| Hook | Where | What it does |
|---|---|---|
| L | layer decode | runs the G3v step kernel |
| H1 | `preprocess_state` | expands rows whose window moves, and every row when the batch is not pure speculative decode |
| H2 | `postprocess_state` | expands the rows the align post-copy saves |
| H3 | `add_request` | resets the meta of that request-state index |
| H4 | `prepare_attn` | writes the request-index buffer; builds the context and groups before the captured forward |

- All of this is asynchronous GPU work with no host sync.
- G3v disables itself (logged), and vLLM's kernel runs, when any of these hold:
  - not align mode;
  - fine-grained mamba prefix cache or `prefix_match_unit` is set;
  - `HORIZON_GDN_CKPT` or `HORIZON_PI` is set;
  - a non-fp32 state;
  - a conv region that is not contiguous.
- It refuses to run together with G1, G2 or G1S.
- The kernels ship inside the dev plugin (`horizon_kernels/g3v_ops` (not in this release)). The arm imports production's TileLang 0.1.15
  prefix unchanged, with no overlay.

**CPU tests.**
- `test_g3v_cpu.py`: 8/8. A world simulating vLLM 0.30's own align machinery vs the same world with G3v and hooks,
  which must match bit for bit on:
  - every step's outputs;
  - every copy source at copy time;
  - every completed block and every committed state.

  Coverage: QL 2 ↔ 4, block sizes 3-11, up to 30% mixed batches, a request running past its block table (window mode),
  request-index reuse, and production geometry. Negative controls fail as expected: without H2, or without the mixed
  expand, hundreds of thousands of elements differ.
- `test_g3v_hooks_cpu.py`: 18/18 against the installed vLLM 0.30 classes on the server, CUDA hidden. It checks:
  - the four method signatures;
  - that the layer hook falls through to vLLM when G3v is not integrated;
  - H1 and H2 row selection against a Python replica of vLLM's own pre-copy and post-copy rules (200 random batches
    each, including PP -1 rows);
  - the window-slot gather;
  - H3 and H4.

**Expected** (paper model; the ring is smaller than in the standalone G3 run):
- QL 2 flushes about every 2 steps: ~1.5 state passes instead of 3, about 2x less traffic.
- QL 4 flushes every step: ~2 passes instead of 5, about 2.4x less.
- The job 60/64 standalone G3 with a 16-token half measured 88 vs 219 µs (QL 2) and 120 vs 373 µs (QL 4) at 32
  requests. G3v will land between those numbers and vLLM's.

## 13. G3v with the caching package (PI, GDN checkpoint, fine-grained prefix caching), 01-10-2026

**Goal:** the caching package can deploy later without switching G3v off. The package is `HORIZON_PI=1`, `HORIZON_GDN_CKPT=1` (stop mode `fill`), `HORIZON_TAIL_MIN_BLOCK=1`,
`--prefix-match-unit 64` and `--enable-mamba-fine-grained-prefix-cache`. This section is CPU work; GPU work comes after it.

### 13.1 Why it is compatible

G3v defers exactly one kind of write: a **decoding** request's per-token window states. It keeps the committed state as
a checkpoint in slot 0 plus a ring stored in the conv regions of window slots 1..3. Every vLLM reader of those slots is
covered by a hook: H1 runs before a pre-copy or any step with prefill, H2 runs before a post-copy, and H3 resets the
meta on `add_request`.

The caching package adds state traffic only around **prefill**. Each piece was checked against vLLM 0.30's source:

| Feature | What it does with GDN state | Why G3v is not affected |
|---|---|---|
| **PI** (`prefill_invariant.py`) | Swaps kernels (fp4x2, K5 skinny, FA2 without the KV split) and graph sets in steps with prefill only. Decode-only steps keep the startup graphs. | No GDN state reads or writes. |
| **GDN checkpoint, export** (`gdn_ckpt.py`, all stop modes) | Wraps `_forward_core` for **prefill** rows only. Writes the conv and ssm state at the checkpoint into `ck.slots`, the prefill request's own reserved block (column `cdiv(seq_len, B) - 2`). | Calls vLLM's `_forward_core` underneath, which reaches the fused decode method through `self.`, so G3v still runs on pure decode steps. |
| **GDN checkpoint, `fill`** | Writes boundary states into the same request's non-null columns below the checkpoint column. | Prefill request's own blocks only. |
| **GDN checkpoint, manager side** | `num_prefill_checkpoint_blocks=1` reserves a block and skips the scratch relocation **during prefill**. | In decode no checkpoint is pending, so allocation is stock (relocation of exclusively owned scratch blocks). Window slots 1..3 are always the request's own scratch blocks, never cached, which vLLM itself requires: it writes their ssm per token at QL 4. |
| **Fine-grained / `prefix_match_unit`** (`cache_partial_block`) | Registers a partial tail only for the prompt chunk being computed (vLLM: "Bounded to the prompt chunk being computed"). The source is the running block (slot 0). The copy-on-write is queued and applied in `update_states`, before the step's hooks and forward (`model_runner.py` 1208). | When that copy runs, the request has `npend = 0` and `n_acc = 1`, and G3v writes nothing to slot 0 then. The junction stop is also prompt-only. |
| **Prefix-cache hits** | Copy a cached state into a **new** request's block. | H3 resets that request-state index's meta. |
| **Block reuse** | The zeroing of freshly allocated blocks also happens in `update_states`. | A freed block's ring is dead. Rings live only in the scratch blocks of requests still in the batch. |

**The only state-touching interaction is a step with prefill.** There, H1 (mode 1) already expands every decode row to
vLLM's layout before the FLA path runs.

### 13.2 Code changes (plugin `gdn_g3.py`)

- **Disables removed:** G3v no longer disables itself for fine-grained prefix caching, `prefix_match_unit`,
  `HORIZON_GDN_CKPT` (stop modes `lcp`, `cont`, `fill`) or `HORIZON_PI`.
- **New marker:** "horizon_kernels G3v: caching features on (...): decode-only protocol, compatible".
- **`HORIZON_GDN_G3_COMPAT_STRICT=1`** restores the refusal.
- **Unknown stop mode:** an unknown `HORIZON_GDN_CKPT_STOP` still disables G3v.
- **New `HORIZON_GDN_G3_CHECK=1` guards** (debug, sync), on top of the old "no pending ring in a non-fused forward" check:
  1. a checkpoint export must never hit a spec-decode row's window block in the same step;
  2. a copy-on-write block copy (source or destination) must never touch a live ring block;
  3. zeroing a freshly allocated block must never touch a live ring block.

  The live ring blocks are window slots 1..3 of in-batch requests with pending records, recomputed after every H2.
- **Refactor:** the H2 per-group path moved into `_expand_rows_post`, with unchanged semantics.

### 13.3 CPU evidence

- **`test_g3v_compat_cpu.py`** (real vLLM 0.30 classes plus the caching package `horizon-pkgs-pi10-30-09-2026`):
  **22/22**.
  - Wrapper chain in plugin order: G3v CHECK, then the checkpoint, then vLLM.
  - The fused decode is reached dynamically.
  - The checkpoint's builder and spec hooks and PI's runner hooks coexist with every G3v hook.
  - The compatibility decisions, the strict mode and the unknown stop mode behave as specified.
  - All CHECK guards: hit and miss cases, and live-ring computation across groups.
- **`test_g3v_caching_sim_cpu.py`** (protocol simulator):
  - **Setup:** two worlds (vLLM vs G3v) over the same request stream. It covers:
    - prompts sharing prefixes;
    - mixed prefill and decode steps;
    - in-step checkpoint and fill exports;
    - partial-tail copy-on-write;
    - prefix hits;
    - block freeing, reuse and zeroing;
    - request-state index reuse;
    - the 3/1 draft schedule;
    - random acceptance;
    - pre- and post-copies, with conv state modelled (slot 0's conv region, shifted copies).
  - **Result:** 5 configurations all bit-identical: decode outputs, every state and conv history read through a copy or
    kernel, every cache entry's ssm and conv, and committed states. In total: 1,281 G3v steps, 341 flushes,
    707 expands, 52 checkpoints, 76 fills, 47 copy-on-writes, 67 hits, 638 reused blocks.
  - **Negative controls:** both are detected. Skipping H1 in mixed steps changes outputs, copies and cache entries; a
    hypothetical whole-page copy of a decoding request's scratch block changes 17.6M read floats. 8/8 checks pass.
- **Old hook checks** still pass with the changed plugin: 18/18 and 24/24.
