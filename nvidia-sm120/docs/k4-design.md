# K4: the persistent decode-step kernel (design)

Written 27-09-2026, K4-foundations track. Status: **design only; nothing deployed.**
Code base: fork `horizon-tilelang`, branch `horizon/k4-foundations` (PDL flags and HKT in-kernel timing).

## 0. Changes found while building

Changes from the design below, found while building it:
- **CTA shape:** 2 CTAs of 128 threads per SM (376 CTAs), not 1 CTA of 256 threads with producer warps.
  - Every one of our GEMM configurations then maps onto 4 warps without re-tiling.
  - Two CTAs per SM give the memory-level parallelism that the per-op kernels get from 2–4 CTAs per SM.
  - Instead of producer warps, the weight stream runs ahead in the cp.async pipeline itself (the `hw=2` mode).
- **Bit-identity target:** each output depends only on the split-K factor (tasks.py). A K4 GEMM task therefore uses
  the phase-1 split for that M with its own 128-column tiling, and the result is bit-identical by construction.
- **Hand-over cost:** 0.9 µs per counter hop (acquire spin + release-add), about a kernel boundary. The gain has to
  come from overlap (weights streaming during waits, K-sliced edges), not from removing hops.

## 1. What K4 is, in one paragraph

Today one decode step (1 user: the main model verifies 4 tokens, then the MTP head drafts 3 more) is about **800 separate
GPU kernels** replayed from a CUDA graph. Each kernel starts on an empty GPU, ramps up, streams its weights, and ends
with a tail where most SMs sit idle waiting for the slowest block. The next kernel cannot start its weight stream until
the previous one has completely finished. K4 replaces that sequence with **one kernel that stays resident on all 188
SMs** ("persistent") and walks a list of small jobs ("tasks": one tile of one GEMM, one row of one norm, ...). Blocks
signal each other through counters in global memory instead of through kernel boundaries, so an SM that finishes early
immediately starts streaming the next operation's weights. This document designs that runtime on top of the pieces we
already have.

## 2. What we measured first (27-09-2026, standalone, our kernels only)

Test bench (`scripts/kernels/k4/` (not in this release)): one Qwen3.8 GDN decoder layer's chain of OUR kernels,
`add+RMSNorm -> qkvz (W4A16 16384x5120) -> ba (BF16 96x5120) -> out_proj (W4A16 5120x6144) -> add+RMSNorm ->
gate_up (W4A16 34816x5120) -> SiLU*mul -> down (W4A16 5120x17408)` (the GDN core in the middle is vLLM's and left
out), 4 layers with their own weights (4 x 216 MB, far above the 128 MB L2), one CUDA graph per variant, bit-identical
outputs in every variant.

**The chain's floor.** A layer reads 216.3 MB of weights and scales; at the measured 1.63 TB/s that is **132.7 us**.
Measured without PDL: **157 us at M = 4** (84% of the floor) and **177 us at M = 32** (75%). K4 lives in that gap.

**Where the gap is** (HKT stamps, M = 4, per layer; M = 32 in brackets):

| Item | us per layer | What it is |
|---|---|---|
| Kernel-to-kernel gaps | 8 x 0.5-0.8 = **~5.5** | last block of kernel N exits -> first block of N+1 enters, inside the graph |
| Small kernels | norms 2 x 1.3-1.5, ba 2.1-2.6 (4.9), SiLU 0.8 (1.3) = **~5.6** (~9) | work that could ride inside the GEMMs (norm, SiLU) or is tiny (ba reads 1 MB = 0.6 us at the floor) |
| GEMM tails | qkvz 2.3, out 1.8-3.2, gu 3.5-4.4, down 2.0-2.6 = **~10** (~13) | last block exit minus median block exit: the GPU drains while the next op waits |
| GEMMs below the floor | out 88%, qkvz 87%, down 88%, gu 92% (M = 32: 76-84%) | steady-state efficiency; not a K4 topic (kernel tuning), except that idle SMs could prefetch |
| SMs never used | gate_up 136 of 188 SMs (both M), qkvz at M = 32 128, out/down at M = 32 160 | the tile grids leave SMs empty for the whole kernel |

**PDL on its own** (paired A/B, 300 iterations, every iteration replays every variant back to back so clock drift
cancels; median us per layer and the median paired difference with its 95% interval):

| Batch rows M | off (`pdl=0`), us per layer | `pdl=1` trigger at entry | `pdl=3` + weight prefetch | `pdl=5` late trigger | `pdl=7` late + prefetch |
|---|---|---|---|---|---|
| 4 (1 user, 3 drafts) | 166.5 | -0.3% [-0.5, -0.2] | **-3.4%** [-3.4, -3.2] (faster in 100% of iterations) | -0.3% | -1.2% |
| 8 | 169.8 | – | **-4.4%** [-4.5, -4.4] (100%) | 0.0% | -1.6% |
| 16 | 170.4 | – | +7.1% (slower in 100%) | **-1.1%** [-1.1, -1.0] | -0.6% |
| 32 (8 users) | 193.9 | +11.6% | +8.9% | **-1.7%** [-1.7, -1.6] (faster in 99%) | +0.5% |

(Paired medians; brackets = 95% bootstrap interval in % of the `pdl=0` time. Absolute times here run 6-9% above the
round-based race (157 / 177 us at M = 4 / 32) because every replay starts cold after a different graph; the paired
differences are what count.)

**Why early triggering hurts at larger batches.** HKT recorded which SM each block ran on. With the trigger at kernel
entry the next kernels are launched while the current GEMM still occupies the GPU; the hardware then places the
waiting kernel's blocks wherever there is room *at that moment* and never moves them. At M = 32 that packed out_proj
and down onto **108 SMs instead of 160** (two blocks per SM), and their main loops ran 15-20% slower (down 34.0 ->
41.0 us). Triggering late (after the main loop, all kernels) keeps the normal placement (160 SMs) but then the next
GEMM's weight prefetch has almost no time to overlap. **PDL cannot choose where blocks run; K4 can** (one resident CTA
per SM, explicit task placement). This is the single most important input to the design below.

## 3. The task graph of one decode step

A decode step = main-model forward over M rows (M = 4 for 1 user with 3 drafts, 32 for 8 users) + verification +
3 sequential MTP draft steps. Operations per layer (Qwen3.8-27B: 64 layers = 48 GDN + 16 full attention, pattern
3 GDN : 1 attention; hidden 5120, MLP 17408):

```
GDN layer (48x)                                   attention layer (16x)
  norm1 = add_rmsnorm(h, res)          [ours K5b]   norm1                               [ours K5b]
  qkvz  = W4A16 16384x5120             [regfrag]    qkv+gate = W4A16 14336x5120         [regfrag]
  ba    = BF16 96x5120                 [ours K5a]   q/k RMSNorm + rotary                [vLLM]
  conv1d update (4-tap causal)         [vLLM]       KV-cache write                      [vLLM]
  GDN recurrent + gating + gated norm  [vLLM]       attention decode (XQA)              [FlashInfer]
  out   = W4A16 5120x6144              [regfrag]    sigmoid(gate) * attn                [vLLM/Inductor]
  norm2 = add_rmsnorm(out, res)        [ours K5b]   o_proj = W4A16 5120x6144            [regfrag]
  gu    = W4A16 34816x5120             [regfrag]    norm2, gu, silu*mul, down           (as GDN)
  act   = silu(gate) * up              [Inductor]
  down  = W4A16 5120x17408             [regfrag]
final norm -> lm_head (BF16 248k, cuBLAS) -> verify/accept (vLLM sampler)
MTP x3: fc(cat(norm(emb), norm(h))) -> 1 attention layer + MLP -> K1a FP8 draft head (48k) -> argmax -> embed
```

Dependencies are op-level (a GEMV needs its whole input vector) with two useful exceptions:
- **K-sliced edges.** A W4A16 tile that covers k range [k0, k1) only needs that slice of its input. `down` split-K
  slice s needs only silu columns of slice s, which need only `gu` columns (s, I + s). With per-slice counters `down`
  can start while `gu` is still running. This is the overlap no kernel boundary (and no PDL) can express.
- **Row edges.** `add_rmsnorm` is per row; with M = 4 rows the norm and the following GEMM's A-tile load can be fused
  (each GEMM CTA recomputes rstd for its M rows: 4 x 10 KB from L2, cheap) and the norm task disappears.

## 4. Runtime

### 4.1 Execution model

- One persistent kernel per step (phase 3; segments before that, section 7): grid = 188 CTAs, one per SM, 256 threads
  (8 warps), up to 99 KB shared memory, launched with the PDL attribute so it overlaps the previous step's tail.
- Co-residency is required (CTAs spin on each other): the host checks occupancy (1 CTA/SM) and launches exactly 188
  CTAs; a cooperative launch is not needed if the grid never exceeds one wave.
- **Warp roles inside a CTA:** warps 0-3 = consumer (run the current task's body: dequant + MMA / reductions),
  warps 4-7 = producer (cp.async the NEXT task's weight tiles into a second shared-memory stage set while the current
  task computes, and poll the next task's dependency counters). Weights never depend on anything, so the producer can
  run ahead across op and even layer boundaries; only activation loads wait for counters. This is the PDL "prefetch
  before the sync" idea, but landing in shared memory instead of L2 and never paying a kernel boundary. Consequence
  for the op bodies: the regfrag configs that use 8 compute warps today (n_warps = 8 / warps_n = 8) are re-tiled to 4
  consumer warps with twice the tiles per CTA; the configs with 4 warps map directly.

### 4.2 Scheduler

- **Static schedule first.** The host builds, per captured batch size M, the task list of the whole step and assigns
  tasks to CTAs: per op, tile t goes to CTA (base + t) mod 188, bases rotated per op so load evens out; each CTA's list
  is in topological order. Stored as a device array of 16-byte descriptors `{op id, tile index, arg-block pointer,
  wait counter id, wait target, signal counter id}`. The CTA walks its own list: no queue atomics, no contention, and
  a fixed mapping keeps every result deterministic (our split-K sums are already fixed-order).
- **Dynamic later, only if HKT shows imbalance:** a global atomic ticket per op ("next tile") instead of the static
  map. Deterministic numerics are unaffected (a tile's math does not depend on which CTA runs it), only the timing.
- Deadlock freedom: every CTA's list is a subsequence of ONE global topological order. The globally earliest
  unfinished task then has all its inputs done, and every task before it in its own CTA's list is earlier globally,
  hence done; so its CTA is running it. Some task always makes progress. (This needs all 188 CTAs resident, 4.1.)

### 4.3 Cross-CTA flags (acquire / release)

- Counters: one int32 per (op instance, slice) in a zeroed step workspace; reset by the last task of the step (the
  same trick as our split-K tickets, which reset themselves, so the graph replays without a memset).
- **Producer (task finished):** all threads store their outputs -> `bar.sync` -> thread 0:
  `red.release.gpu.global.add.s32 [cnt], 1`. The release is cumulative: it orders the other threads' stores that
  the CTA barrier ordered before it.
- **Consumer (task start):** one thread spins `ld.acquire.gpu.global.s32` until `cnt >= target` (with `nanosleep`
  backoff 32 -> 256 ns so spinning CTAs do not steal issue slots from co-resident work) -> `bar.sync` -> everyone
  reads.
- **Stale data traps (the same hazard PDL showed today):** dependent activations must be read with L2-coherent
  loads (`cp.async.cg`, `ld.global.cg`, or `ld.acquire`), never through the non-coherent read-only path
  (`ld.global.nc`, which `const __restrict__` pointers invite). Weights may use any path. Our PDL work already
  separates the two classes of kernel parameters (the `HZ_PDL_RESTRICT` list): K4 reuses exactly that split.

### 4.4 How the op bodies become tasks

| Op | Body today | Task form |
|---|---|---|
| W4A16 GEMM (all projections) | `decode_regfrag` / `decode_regfrag_xl` kernels | Refactor the `with T.Kernel` body (already a `T.macro` on this branch: `_kernel(...)`) into `regfrag_body(cfg)(x, Y, Wg, Sg, alpha, P, counters, bx, bk, tid, smem)`: grid indices become task arguments, shared buffers become views of one arena. The split-K last-arriver reduction is already a cross-CTA protocol (ticket + fence); in K4 it signals the op counter instead of ending the kernel. |
| in_proj_ba | `horizon_fused.skinny_body` | Already a task body (tile (m0, n0), `red` buffer passed in, barrier at the end). |
| add + RMSNorm | `horizon_fused.add_rmsnorm_body` | Already a task body (one row); phase 2 fuses it into the next GEMM's A prologue. |
| SiLU * mul | `horizon_fused.silu_mul_body` (new, this branch) | Task body per (row, 1024 columns); phase 2 fuses it into `down`'s A-tile load (read gate/up, write the product into the swizzled A tile). |
| GDN conv1d, recurrent update | vLLM CUDA kernels | Must be rewritten as TileLang bodies (phase 3). The recurrent update is memory-bound on its 3.1 MB fp32 state per call; the body is a per-head task. |
| Attention decode | FlashInfer XQA | Rewrite as a split-KV decode body (per head x KV chunk + a fixed-order combine) (phase 3). |
| lm_head (BF16 248k) / K1a draft head (FP8 48k) + argmax | cuBLAS / our K1a | K1a is ours; argmax becomes per-CTA partial max + counter + last-arriver final (deterministic tie-break by index). |

Register budget: one kernel has ONE register count, the maximum over its bodies. Measured (ptxas, sm_120a, 27-09):
regfrag M=4 bodies 72-88, regfrag_xl M=32 bodies 102-112, skinny 95 (M=4) / 181 (M=32), add_rmsnorm 32, the stream-K
range kernel 255. So K4 is specialised per batch bucket (M <= 8, <= 32; above that the normal kernels run), and the
range kernel stays out of K4.

### 4.5 The MTP draft loop

The 3 draft steps are strictly sequential (draft i+1 embeds the argmax of draft i). Today vLLM runs them as separate
CUDA-graph pieces driven from Python. In K4 they are just more tasks: fc -> MTP attention layer -> MTP MLP -> K1a head
tiles -> argmax (cross-CTA) -> embedding gather -> next draft. No host round trip between drafts. Verification
(rejection sampling, KV/state bookkeeping for accepted tokens) stays in vLLM until phase 4.

## 5. Expected gain

Numbers per GDN layer from section 2 (M = 4, 1 user), then scaled to a decode step with the K5 profile (1 user:
16,122 us per step, 64 layers; our W4A16 GEMMs + small ops of the 64 layers ~10 ms of it).

| Source of the gap (per layer, M = 4) | us | PDL (measured) | K4 phase 2 (segment kernel) | K4 phase 3-4 |
|---|---|---|---|---|
| Kernel-to-kernel gaps (8) | ~5.5 | ~half (hand-over 0.26-0.5 instead of 0.5-0.8 us) | all inside the segment | all |
| Norms + SiLU as separate kernels | ~3.5 | - | fused into the GEMM prologue / A load | same |
| GEMM tails (next op idles) | ~10 | only with early trigger, which breaks placement at M >= 32 | next op's weights stream during the tail (producer warps) | same |
| GDN core / attention boundaries and tails | not in the chain | - | - | yes (phase 3) |
| MTP draft loop (3 x small graphs, host round trips) | not in the chain | - | - | yes (phase 4) |
| **Chain total** | 157 us, floor 133 | **-3.4% (M=4), -4.4% (M=8); -1.1 / -1.7% (M=16 / 32, late trigger)** | **~-10 to -12% of the chain** (est.) | |

Step-level estimate (1 user): phase 2 removes ~15-19 us per layer x 64 layers = **1.0-1.2 ms of 16.1 ms (6-7.5%)**;
phase 3 adds the GDN / attention boundaries (4 per layer x 64 x ~1.5-2.5 us of gap + tail = 0.4-0.6 ms, 2.5-4%);
phase 4 removes the draft loop's per-draft graph launches and host synchronisation. **Total ~9-12% at 1 user**, in
line with the earlier "+8-15%" roadmap estimate. At 8 users (M = 32) the chain runs at only 75% of the floor
(44 us/layer of headroom vs 24.5 at M = 4), so the ceiling there is larger, but it is mostly GEMM efficiency (below the
floor in steady state) plus placement, which K4's explicit per-SM scheduling addresses.

What PDL alone is worth in vLLM (phase 1): in serving, only boundaries between two of OUR kernels benefit (vLLM's GDN
kernels, FlashInfer attention and Inductor's SiLU sit in between). Per GDN layer that is 5 of the chain's 8 boundaries
(norm1 -> qkvz -> ba, out -> norm2 -> gu, down -> next norm1). Expected: **~1-2% at 1 user** with pdl=3, ~1% at 8+
users with pdl=5; a batch-size-dependent choice (early trigger + prefetch for M <= 8, late trigger without prefetch above) is possible
because the plugin already picks kernels per M.

## 6. Risks

1. **gpushare and spinning.** gpushare time-slices the GPU between processes and hooks allocations; a persistent kernel that spins while another process holds the GPU could stall both.
2. **Registers and shared memory** are the maximum over all bodies (section 4.4); a single heavy body lowers occupancy
   for everything. Mitigation: per-bucket specialisation, bodies with explicit register caps.
3. **Memory-model bugs** (stale L1 / non-coherent loads, missing release) give wrong results only sometimes. Mitigation:
   the load-class split above, a debug mode that checks every counter wait with a sequence number, and bit-identity
   against the kernel-per-op path on every test (the same gate the PDL work used).
4. **GDN and attention rewrites** are the long pole (phase 3) and must match vLLM's numerics exactly (the GDN state is
   carried across steps: an error accumulates).
5. **vLLM integration.** K4 needs FULL CUDA-graph mode for decode and all per-step metadata (sequence lengths, block
   tables, GDN state slots, sampling tensors) as device pointers; vLLM upgrades move these structures.
6. **Compile time and debuggability:** one huge kernel compiles slowly and Nsight Compute crashes on this driver. HKT
   stamps per task (section 8) are the profiler; per-op bit-identity tests stay mandatory.
7. **Gain is capped** by what the kernel-per-op path already achieves (section 5): the weights stay the same bytes.

## 7. Phased plan

| Phase | What | Gate | Effort |
|---|---|---|---|
| **0 (done 27-09)** | PDL flags in all our kernels (default off, bit-identical), `__restrict__` fix, HKT stamps, chain benchmark, this design | bit-identity, graph capture | done |
| **1** | PDL in vLLM behind `HORIZON_PDL` (plugin, section 9); A/B speed, log-prob bit-identity; HKT/k5_trace profile of the gaps | +x% at 1 user, no loss at 8/32 | 1-2 days |
| **2** | **Segment kernel** for the all-ours stretch `norm2 -> gu -> silu -> down -> next norm1 -> qkvz + ba` (per GDN layer), persistent over 188 CTAs with the static scheduler and counters; K-sliced edges gu -> silu -> down; norm fused into the GEMM prologue; the GDN core and attention stay vLLM kernels linked by PDL | bit-identical to phase 1, HKT shows no gaps inside the segment | ~1 week |
| **3** | TileLang bodies for GDN conv1d + recurrent update and attention decode; whole layer, then the whole forward in one kernel | bit-identical (or log-prob/eval gate where the numerics change) | 2-3 weeks |
| **4** | MTP draft loop and argmax inside the kernel; then consider dynamic scheduling if HKT shows imbalance | speed A/B | 1 week |

## 8. Profiling K4 with HKT

Every task writes `{ENTRY, SYNC (dependencies satisfied), LOOP (main loop done), EXIT, SMID, task id}` into a
per-step record buffer when built with `stamps=1` (the same `tilelang.contrib.horizon_k4.hkt` layout used for the
kernels today). `hkt.chain_summary` turns that into per-op spans, waits and gaps; `hkt.activity_curve` into the
"how many SMs are busy" curve. A task that waits long on its counter is the critical path; an SM with a long idle
stretch means the static schedule is unbalanced.
