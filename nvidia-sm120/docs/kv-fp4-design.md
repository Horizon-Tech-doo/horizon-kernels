# FP4 KV cache: design (29-09-2026)

Goal: store the attention KV cache in 4 bits instead of FP8, so about twice as many full-context
users fit on one GPU, without changing the model. This document covers the format, the kernels, the vLLM
integration plan, the gates and the risks.

**Status on 29-09-2026:**

- The format has an exact PyTorch reference.
- The write kernel and the decode kernel are written and compile for SM120 with no register spills.
- The CPU tests pass.
- **Nothing has run on the GPU.** GPU correctness, speed and model quality are all still unmeasured.

## 1. Where the KV bytes go today

Qwen3.8-27B has 64 layers. Only 16 of them are full-attention layers (ids 3, 7, …, 63); the other 48 are GDN (linear
attention), which keeps a fixed-size state and is not affected by this work. The MTP draft layer adds one more
attention layer, so 17 layers use the KV cache.

- Attention: 24 query heads, 4 KV heads (GQA 6), head_dim 256.
- The cache is FP8 e4m3 with one scale per tensor. `k_scale = v_scale = 1.0` because the checkpoint has
  `kv_cache_scheme: null`.

| | Per token, per KV head, K+V | Per token, per layer (4 heads) | Per token, 16 layers | Per token, 17 layers (with MTP) |
|---|---|---|---|---|
| FP8 today | 2 × 256 = **512 B** | 2,048 B | 32,768 B | 34,816 B |
| NVFP4 (this design) | 2 × (128 + 16) = **288 B** | 1,152 B | 18,432 B | 19,584 B |

**The capacity gain is exactly 512 / 288 = 16 / 9 = 1.778x for the attention KV bytes.** Each side (K or V) needs
128 B of 4-bit data (256 values × 4 bits) and 16 B of E4M3 block scales (one per 16 values). For K+V that is 256 B of
data plus 32 B of scales. The global scale is static per (layer, KV head), so it adds no bytes per token.

**Paper estimate for an example configuration, not measured:**

- **FP8:** a 46 GiB KV pool gives 1,297,539 tokens. At 34,816 B/token those 46 GiB could hold 1,418,660 tokens, so
  about 4.2 GB of the budget goes to other things (mostly GDN state pages).
- **With NVFP4:** if those 4.2 GB stay the same, the remaining 45.2 GB at 19,584 B/token gives about **2.31 M tokens
  (1.78x)**. That is about 8.8 full 262,144-token conversations instead of 4.95.
- **Block size:** vLLM's hybrid allocator makes an attention page as large as a GDN state page (that is why the block
  is 1,600 tokens with FP8). With 1.78x smaller tokens the block becomes about 2,850 tokens. Per-request rounding
  therefore grows a little, and the real token count has to be measured.

## 2. Format: hz-kvfp4 v1

The exact rules are in `tilelang_contrib/horizon_ops/kv_fp4_ref.py` on branch `horizon/kv-fp4`. The kernels are
bit-identical to it by construction; the GPU gate G1 checks that.

**Per (token, KV head, side K or V), D = 256:**

- `data`: 128 bytes of E2M1 codes. Element 2j is the low nibble of byte j, and bit 3 of each code is the sign. This is
  the same packing as our NVFP4 weights in `sm120_nvfp4`.
- `sf`: 16 bytes, one E4M3 block scale for each 16 consecutive values.
- `g`: an fp32 global scale per (layer, KV head, side). It is static and not stored per token. For now it is vLLM's
  `k_scale` / `v_scale` (1.0).
- The stored value is `e2m1(code) × e4m3(sf) × g`.

**Quantising one block of 16 fp32 values:**

Every step is a single IEEE fp32 operation with round-to-nearest-even and no FMA. The kernels use `__fdiv_rn`,
`__fmul_rn`, `__fadd_rn` and `__fsub_rn`, so compiler flags cannot change the bytes.

1. `amax = max |x|`
2. `c0 = e4m3_satfinite(amax / (g·6))`
3. **Block-scale search (on by default):** the writer tries the candidates `c0`, `c0+1`, `c0−1` and
   `e4m3(amax / (g·4))` in that order. It keeps the one with the smallest squared error, summed in order over the 16
   values; on a tie, the earlier candidate stays. This "4-over-6 plus neighbours" search only changes the writer, so
   readers are unaffected.
   - On synthetic Gaussian, Laplace and Student-t data it lowers the relative L2 error from 0.095 to 0.083 (24% less
     MSE).
   - In the CPU test, 64% of blocks pick a scale other than `c0`, and no block gets worse.
4. `s = e4m3(sf) · g`, then `code = e2m1_rne(|x| / s) | sign`. The E2M1 grid is {0, .5, 1, 1.5, 2, 3, 4, 6}; ties go
   to the even code, and values above 6 saturate. When `s = 0`, the code is 0.

**Why g = 1.0 is safe:** the block scales carry the dynamic range. With g = 1, a block keeps full scale precision for
amax between 0.094 and 2,688. Today's FP8 cache at scale 1 cannot hold anything above 448, so the top has at least 6x
headroom. Smaller blocks get subnormal scales, which lose precision but are tiny in absolute terms. We rejected a
per-token fp32 global scale (296 B, 1.73x) because it adds bytes for no measured benefit. It is the fallback if real
K/V shows a range problem.

**Exactness trick in the reader:** `e2m1 × e4m3` has at most 5 significant bits, so `code × sf` is exact in bf16. The
decode kernel therefore builds bf16 tiles for the tensor cores with no rounding. It folds `g` into the fp32 score
scale for K and into the output for V, the same way the FP8 path handles `k_scale` and `v_scale`. CPU test T8 checks
this over every code and scale.

### K versus V: the Hadamard option

Keys have outlier channels (the low-frequency RoPE pairs); values mostly do not. The design supports an optional
rotation of the keys:

- The writer stores `k' = FWHT(k ⊙ signs) · D^-0.5`, with a fixed random ±1 sign per channel.
- The decode kernel receives `q' = FWHT(q ⊙ signs) · D^-0.5`, rounded once to bf16 by `rotate_q_kernel`.
- `q'·k' = q·k` exactly in real arithmetic, so the model's math is unchanged and only the quantisation error is
  spread over all 256 channels.
- The FWHT butterfly order is fixed, so the kernel and the reference are bit-identical.
- Values are never rotated, because rotating V would need an inverse rotation of the output.

**What the synthetic data shows (CPU test T5, 5 seeds, 4,096 keys, 64 queries):**

| Format | Gaussian: K err | Gaussian: attn-out err | Outlier keys: K err | Outlier keys: attn-out err |
|---|---|---|---|---|
| FP8 e4m3, scale 1 (today) | 0.027 | 0.056 | 0.027 | 0.203 |
| NVFP4, plain RN scale | 0.095 | 0.211 | 0.096 | 0.412 |
| **NVFP4 + scale search** | 0.083 | 0.180 | 0.089 | 0.320 |
| **NVFP4 + scale search + K Hadamard** | 0.083 | 0.182 | 0.083 | **0.252** |
| FP8 K + NVFP4 V (K8V4-like) | 0.027 | 0.097 | 0.027 | 0.218 |
| int4 per token-head, asymmetric | 0.109 | 0.221 | 0.185 | 0.552 |
| int4 per token-head + RHT (like vLLM `int4_per_token_head`) | 0.109 | 0.241 | 0.089 | 0.244 |

All errors are relative L2.

**How to read the table:**

- Every 4-bit format has about 3x the per-element error of FP8. That is inherent to 4 bits.
- The attention output is far more sensitive to K errors than to V errors. In a separate seed sweep, V-only NVFP4 gave
  about 0.083 attention error, while K-only gave 0.14–0.36.
- The 16-value blocks of NVFP4 already contain most outliers: on Gaussian keys the Hadamard makes no difference.
- With structured outlier channels plus queries that use them, the Hadamard cuts the attention error from 0.32 to
  0.25.
- These are synthetic numbers, and the attention-output metric exaggerates errors for near-uniform attention. Only
  the real-model gates in §6 decide.

**Default: K Hadamard on, V plain, scale search on.** Both kernels also support Hadamard off, so the model A/B can
measure it.

### Optional hot window (designed, not built)

If the quality gate fails, the next step is to keep the most recent W tokens in FP8 as well:

- **Storage:** a per-sequence ring buffer, W = 128 tokens × 17 layers × 2 KB. At 32 sequences that is 143 MB
  (0.3% of 46 GiB).
- **Writer:** it always writes FP4 into the paged cache, and also writes FP8 into ring slot `pos % W`. Eviction
  therefore needs no requantisation.
- **Reader:** positions `≥ L − W` are read from the ring and older positions from FP4.
- **Catch:** on a prefix-cache hit the ring does not hold the reused prefix. The window has to be defined as "tokens
  written by this request", or cached and uncached runs will differ, and that breaks the determinism rule in §6.

The other fallback tier is K8V4: FP8 keys plus NVFP4 values, 400 B per token-head, 1.28x.

### Page layout

We reuse **vLLM 0.29's own NVFP4 page geometry**: `FlashInferBackend.customize_spec` gives 2 × KVH head slots of
D/2 + D/16 = 144 bytes, allocated as uint8. vLLM already allocates, splits and prefix-caches these pages. A 64-token
kernel page is one dense 73,728-byte block, HND:

```
[ K data  (KVH, PS, 128) | K scales (KVH, PS, 16) | V data (KVH, PS, 128) | V scales (KVH, PS, 16) ]
    32,768 B                 4,096 B                  32,768 B                4,096 B           (KVH 4, PS 64)
```

The standalone tests index each part as a `[page, position, head, byte]` view with its own strides
(`kv_fp4_ref.split_views`, which places bytes the same way as vLLM's `nvfp4_split_data_scale`). **In serving (§4) each
144-byte slot is `[128 data | 16 scales]` for one (token, head, side) instead;** the kernels are stride-generic, so
both are the same format with a different byte placement.

We keep vLLM's page geometry, but the byte meaning is ours. The scales are in linear order, not trtllm-gen's
swizzled order. Nothing else reads this cache while our backend is active, so that is fine, but the cache is not
readable by FlashInfer's SM100 kernels.

## 3. Kernels

These live on TileLang fork branch `horizon/kv-fp4` (worktree `<src>/wt-kvfp4`), in
`tilelang_contrib/horizon_ops/`.

| Kernel | File | What it does | Status |
|---|---|---|---|
| Write / quantise | `kv_write_fp4.py` | One CTA per token, one thread per 16-value block (4 KV heads × 16 blocks × K and V = 128 threads). Optional FWHT of the keys in 4 KB of shared memory; scale search; writes data and scale bytes; skips slot < 0. `kv_write_fp4_body` is a `T.macro` for K4 reuse. | compiled |
| Decode attention | `attn_decode_fp4.py` | Same split-KV structure as our FP8 `attn_decode.py`. One task is a (request, KV head, split), and all G·QL ≤ 32 query rows are handled together, which covers QL 1–4 for MTP verify with the causal draft mask. The fp4 bytes and scales are expanded exactly into bf16 tiles. Scores, the online softmax and the accumulator are fp32; P is bf16. The fixed-order combine is reused from `attn_decode`, so it is deterministic and batch-invariant. `pipe=1` (default) stages raw bytes with a 2-deep cp.async pipeline and needs PS % 32 == 0; `pipe=0` does direct loads. | compiled |
| q rotation | `attn_decode_fp4.rotate_q_kernel` | FWHT of the 24 query heads of one token in fp32, one bf16 rounding. Later this can be fused into our q/k-norm + RoPE kernel, which already touches q and k. | compiled |
| Prefill over cached context | not written | See §4, step 5. | planned |
| Decode attention v2 (fast) | `attn_decode_fp4_v2.py` | Register-level FP4 unpack straight into mma fragments; see §3b. | compiled, CPU-emulated |
| Reference | `kv_fp4_ref.py` | Pure torch: quantise, dequantise, FWHT, paged write, fp64 attention. | CPU-tested |

**Offline SM120 compile (29-09-2026):** `nvcc -arch=sm_120a -Xptxas -v`, no CUDA context, run on the server. The logs
are in `results/kvfp4_29-09-2026/compile/`.

| Variant | Registers | Spills | Shared memory |
|---|---|---|---|
| write, Hadamard + search | 44 | 0 | 4 KB |
| write, Hadamard, no search | 40 | 0 | 4 KB |
| write, no Hadamard, search / no search | 44 / 40 | 0 | 0 |
| rotate_q (24 heads) | 63 | 0 | 24 KB |
| decode QL 1, pipe 1, block 32 | 134 | 0 | 68 KB |
| decode QL 1, pipe 0, block 64 | 131 | 0 | 84 KB |
| decode QL 4, pipe 1, block 32 | 178 | 0 | 68 KB |
| decode QL 4, pipe 0, block 64 | 186 | 0 | 84 KB |

Compiling proves only that the code lowers and builds. It says nothing about GPU correctness or speed.

**Known speed weakness:** the fp4 → bf16 expansion is scalar in the generated code. Each thread does a shared-memory
byte load, about 10 integer operations and one bf16 store per element. The first optimisations to try:

- packed conversion of two nibbles at a time (SM120's `cvt.rn.f16x2.e2m1x2`, or a `prmt` lookup table) with bf16x2
  stores;
- expanding straight into registers instead of shared memory.

Our FP8 decode kernel is already 1.36–3.15x slower than XQA (28-09-2026). Reading 44% fewer bytes helps only once the
kernel is limited by memory bandwidth.

The GPU confirmed this on 29-09-2026: v1 is 5–10x slower than XQA, and pipe0 is 10x slower still. pipe0 does
per-element global byte loads with a page-table lookup per element and no async copies, so it is latency-bound. **It
is dropped:** it stays in the code only for A/B history.

### 3b. Decode v2: register-level FP4 unpack (29-09-2026)

`attn_decode_fp4_v2.py` has the same arguments and output semantics as v1:

- the same dequantised values;
- fp32 scores, online softmax and accumulators;
- P rounded to bf16 for P @ V;
- the same opart / lse split outputs and the same fixed-order combine.

It never builds bf16 tiles in shared memory. The core is a hand-written CUDA device routine (inline PTX `mma`,
`movmatrix`, `cp.async`) that a TileLang kernel launches.

**Design:**

- **Staging:** raw K/V bytes and scales go through a per-warp cp.async ring of 3–4 stages of 16 tokens each. There is
  no CTA barrier for staging.
- **FP4 → bf16 in registers, 8 values per 32-bit word:**
  - Marlin's placement trick `(y & 0x80008000) | ((y & 0x70007000) >> 6)` gives two e2m1 values × 2^-126 per bf16x2.
  - One `HMUL2.BF16` by the block scale × 2^117 turns that into `code × sf × 2^-9` **exactly**: at most 5 significant
    bits, no rounding.
  - The 2^9 is folded into the fp32 score scale and the output scale. Both are exact powers of two.
- **K:** `S^T = K·Q^T`, with tokens as the MMA M dimension and 8 query rows per n-tile. QL 4 dense fills 24/24 rows;
  QL 1 fills 6/8. Each lane unpacks its own bytes straight into A fragments. Channels map to k-slots in the order the
  bytes arrive, and Q's fragments (built once per task) use the same permutation, since a dot product ignores channel
  order.
- **P:** P^T moves from the S accumulator layout to the B-fragment layout with `movmatrix.trans`, without shared
  memory.
- **V:** `O^T = V^T·P^T`. The token pairs a V fragment needs are gathered with one `byte_perm` per byte.
- **Head dim:** it is split over NCW warps (2 for up to 24 query rows, 4 for 32), so the fp32 O accumulator fits in
  registers. The per-slice partial scores are summed through shared memory in fixed warp order, with one
  `__syncthreads` per 16-token block. Softmax rescaling is skipped when no row maximum changed; multiplying by 1 is
  exact.
- **Robustness:** scale bytes are decoded with an integer formula, so garbage bytes in a partial block stay finite and
  masked tokens contribute exactly 0.

**Offline SM120 compile (29-09-2026), 0 spills for all eight variants.** "Loop" is the SASS instructions of the
16-token main loop per warp, including the conditional rescale; the warp count is NCW.

| Variant | Registers | Shared memory | Loop instructions per warp | Per 16 tokens × 1 KV head (warps) |
|---|---|---|---|---|
| dense (4 KV heads, GQA 6), QL 1, NCW 2, NST 4 / 3 | 123 / 119 | 22.0 / 17.0 KB | 670 / 672 | ~1,340 |
| dense, QL 4, NCW 2, NST 4 / 3 | 233 / 235 | 26.0 / 21.0 KB | 998 / 1,001 | ~2,000 (~1,740 without rescale) |
| MoE (2 KV heads, GQA 8), QL 1, NCW 2, NST 4 / 3 | 123 / 119 | 22.0 / 17.0 KB | 669 / 671 | ~1,340 |
| MoE, QL 4, NCW 4, NST 4 / 3 | 187 / 186 | 38.0 / 32.5 KB | 873 / 876 | ~3,490 |

- **Instruction counts:** the dense QL 4 loop per warp has 48 HMMA, 64 HMUL2, about 175 LOP3, 60 SHF and 45 LDS.
  That is about 7.8 lane-instructions per stored K/V value, all in (QL 1: 5.2).
- **Occupancy:** 64-thread CTAs at ≤ 235 registers and ≤ 26 KB allow 4 CTAs (8 warps) per SM.
- **CPU emulation:** a lane-level emulation of the whole kernel (`test_kvfp4_v2_cpu.py`) matches the fp64 reference
  within 0.0019–0.0021 across 6 cases. Deliberately wrong K or V mappings give 0.78 and 1.09, so the emulation catches
  mapping errors. The bit tricks are exact over every packed byte × every scale code, and every staged data load is
  bank-conflict-free.
- **Not covered by the CPU checks:** the PTX fragment layouts are taken from the ISA manual, and whether `HMUL2.BF16`
  keeps bf16 subnormal inputs is unverified. The 0.5 code is a bf16 subnormal before the scale multiply; if the
  hardware flushed it, G3v2 would fail clearly.

**Paper speed estimate (not measured):**

- **8 × 32k:** 302 MB of FP4 plus about 36 MB of split workspace (23 splits at 4 CTAs/SM) comes to about 217 µs at
  XQA's measured 1.56 TB/s. About 66 µs of instruction issue has to overlap with that. Expected 220–320 µs, against
  XQA 345 and v1 2,543.
- **32 × 32k:** expected 800–1,000 µs, against XQA 1,340.
- **1 × 1k:** latency-bound: two launches (split + combine), about 2–4 dependent 16-token blocks per CTA, and the Q
  fragment setup. We capped the splits at ceil(max_seq_len / min_chunk) so the combine does not read empty splits.
  Expected 10–15 µs against XQA 14.9. `rotate_q` is one more launch when keys are rotated; it could be fused into
  q-norm/RoPE.

**Knobs swept in G4v2:** NST 3/4, CTAs per SM 2/4 (fewer splits means less workspace traffic but less latency
hiding), and min_chunk 64/32.

## 4. vLLM integration (built 29-09-2026, dev-tested; A/B prepared, not run)

**Route taken: our own attention backend, registered as vLLM's `CUSTOM` backend by a separate plugin package.** No
vLLM file is patched and the production plugin package is unchanged.

- **Package:** `horizon_kvfp4` (`horizon_kvfp4/`; its `kernels/` subpackage holds copies of the six kernel modules of
  `tilelang_contrib/horizon_ops/`).
- **Entry point:** the package has its own `vllm.general_plugins` entry point. With `HORIZON_KVFP4=1` its
  `register()` calls `register_backend(AttentionBackendEnum.CUSTOM, "horizon_kvfp4.backend.HorizonFp4Backend")`.
- **Env:** the env adds the package to `PYTHONPATH`, `VLLM_KV_CACHE_DTYPE=nvfp4` and
  `VLLM_EXTRA_ARGS=--attention-backend CUSTOM`.

**What vLLM does for us:** the `nvfp4` cache dtype gives `KVQuantMode.NVFP4` and uint8 storage.

- **Head slots:** our backend's `customize_spec` gives 2 × KVH slots of 144 B per token.
- **Hybrid alignment works unchanged:** "Setting attention block size to 2848 tokens to ensure that attention page
  size is >= mamba page size" (FP8: 1,600).
- **Layout:** vLLM resolves the layout to `LBHNC`. The kernels take any strides, and NHD was tested standalone too.
- **Pages:** allocation, kernel-block splitting (64-token pages) and prefix caching are vLLM's.
- **Only our backend accepts nvfp4 on SM120;** FlashInfer's SM100-only check is untouched.

**Serving byte layout** (differs from the region layout of the standalone tests; the kernels are stride-generic): each
144-byte slot is `[128 B E2M1 data | 16 B E4M3 scales]` for one (token, head, side). The views are plain strided
slices of vLLM's `(blocks, 2*KVH, PS, 144)` tensor.

| Piece | Where | What it does |
|---|---|---|
| KV write | `do_kv_cache_update` → `kv_write_fp4.write` | every write: prefill, decode, MTP layer; scale search on; bit-exact to the reference |
| Decode | `kv_fp4_serving.decode_ragged` + `combine_ragged` | see below |
| Prefill | `kv_fp4_serving.dequant_tokens` + FlashInfer FA2 `single_prefill_with_kv_cache` + `merge_state` | see below |
| Metadata | `HorizonFp4Builder` | see below |
| Warm-up | `process_weights_after_loading` | compiles the TileLang kernels and JIT-warms FlashInfer's bf16 FA2 prefill before any CUDA graph |

**Decode:** `decode_ragged` handles every request with ≤ 1 + drafts query tokens in **one launch over the whole
batch**:

- It reads per-request query offsets and seq_lens on the device.
- Rows with more tokens are skipped, so batch order does not matter. This matters because vLLM's V2 runner does not
  reorder the batch, pads CUDA-graph batches with 0-token requests, and async scheduling makes the CPU seq_lens only
  an upper bound for decode rows.
- It is capture-safe.

**Prefill:**

- Rows with more than 1 + drafts tokens are dequantised exactly to bf16 in 64k-token chunks.
- The chunk that holds the new tokens runs causal (bottom-right aligned); earlier chunks are fully visible.
- It runs eagerly, and only for true prefill rows, whose CPU lengths are exact.

**Metadata builder:**

- A thin builder with UNIFORM_BATCH CUDA-graph support.
- It supports in-place MTP draft-step metadata: the decode path reads only device tensors that the speculator
  advances.

**Options:**

| Switch | Default | Effect |
|---|---|---|
| `HORIZON_KVFP4_HAD` | 0 | Hadamard keys + `rotate_q`; not exercised in vLLM yet |
| `HORIZON_KVFP4_SEARCH` | 1 | block-scale search in the writer |
| `HORIZON_KVFP4_NST` | 3 | decode pipeline stages |
| `HORIZON_KVFP4_CPS` | 4 | CTAs per SM for the split plan |
| `HORIZON_KVFP4_CHUNK` | 65536 | prefill chunk size |
| `HORIZON_KVFP4_PF32` | 0 | fp32 partials; FlashInfer 0.6.18 has no such variant |
| `HORIZON_KVFP4_DUMP` | off | real K/V/Q capture, gated by a `GO` file |

**Development checks on the GPU.**

- **a0:** two real bugs found and fixed by vLLM's own start-up:
  - `get_name()` must return `CUSTOM`;
  - the TileLang wrappers shared one size symbol for K and V buffers of different lengths.
  The arm ended "unclean start", correctly, because the failed starts counted.
- **a1:**
  - Standalone GPU pretest (`test_kvfp4_serving_gpu.py`), S1 ragged decode: within 0.0025 (worst 0.00200) with batches
    mixing 4/2/1-token requests, a 0-token padding request and a prefill row. Covers HND and NHD, dense and MoE
    geometry and NST 3/4; the prefill rows stay untouched; deterministic; two graph replays exact.
  - S2 dequantisation: bit-exact.
  - vLLM dev start clean; 391 on both ports; needle 128k 3/3.
- **a2:**
  - Clean start again; coherent answers (Rayleigh scattering, a correct iterative Fibonacci, "Canberra").
  - **Needles 64k / 128k / 250k all 3/3** (TTFT per request about 13 s / 34 s / 96 s).
- **a3:**
  - S3 prefill scheme against fp64: one FA2 call 0.0021–0.0026, so FA2's own bf16 precision already sits at the
    0.0025 edge (XQA also exceeded this limit on 28-09).
  - Merges: 2 × 32k/64k chunks about 0.0026, 3 chunks 0.0029, 6 tiny chunks 0.0037 (partials are bf16).
  - Result: the chunk size was raised to 64k. The strict 0.0025 limit is **not met** by FA2-based prefill; the
    model-level long-context gates decide.

**Measured capacity:**

| | FP8 baseline | FP4 |
|---|---|---|
| KV cache tokens in the same pool | 1,297,539 | **2,148,124 (1.656x)** |
| Full 262,144-token requests | 4.95 | 8.19 |

The per-token attention bytes shrink exactly 16/9 = 1.778x. vLLM's figure is lower because it reports *max
concurrency at max_model_len × max_model_len*, and each request also holds a fixed number of GDN-state blocks (about
15, the same bytes in both formats). Shorter requests gain less than 1.656x for the same reason.

**Real K/V** (a2 dump: first chunk of the 64k needle prompt, which is random words, 5,696 tokens; attention-output
relative L2 vs bf16 for the last 256 queries):

| Layer | FP8 today | NVFP4 + search | + K Hadamard |
|---|---|---|---|
| 3 | 0.0099 | 0.0249 | 0.0239 |
| 31 | 0.0419 | 0.0489 | 0.0398 |
| 63 | 0.0106 | 0.0361 | 0.0353 |

Scale search helps on every layer (plain: 0.031 / 0.054 / 0.042). The Hadamard is kept off for the first A/B because
it costs one launch per layer and is untested in vLLM; it is a one-line F variant (`F_EXTRA=HORIZON_KVFP4_HAD=1`).

**Still open:**
- fused `rotate_q` (for the Hadamard);
- decode speed at short context with one request (improved by the tuning in §4b, still behind XQA at 4-token rows);
- an FP4-native prefill with fp32 merges (phase 2);
- ~~the write kernel's latency (parallel scale search)~~ done in §4b.

### 4b. Speed tuning (29-09-2026, after the first A/B)

The first comparison passed capacity and quality but was slower than FP8, so the kernels were tuned.

**Profile (vLLM torch profiler, `prof_step.py`, `kvfp4_trace.py`; old candidate vs same-session FP8 production),
per steady decode step:**

| Users | FP8: XQA + write (µs) | FP4: decode + combine + write (µs) | Step wall FP8 → FP4 (µs) |
|---|---|---|---|
| 1 | 567 + 69 | 419 + 283 + 380 | 15,128 → 15,390 |
| 8 | 369 + 75 | 886 + 188 + 403 | 18,621 → 19,626 |
| 32 | 268 + 72 | ~605 + 107 + 378 | 30,488 → 30,850 |

First prefill step (mixed): 8 users 121 → 136 ms, 32 users 210 → 349 ms. At 32 users GPU busy time was about equal
(139 vs 143 ms); the extra 134 ms was GPU idle, i.e. the per-request Python prefill loop (32 requests × 17 layers of
dequant + FA2 + merge launches). That explains the F2 TTFT loss.

**Causes and fixes:**

| Cause | Fix | Evidence |
|---|---|---|
| per-request prefill loop | batched prefill: one `dequant_multi` for all prefill rows + one FlashInfer ragged FA2 call per layer, planned once per step per model | S5 GPU gate (≤ 0.0021 vs fp64); t5: 32-user TTFT p50 1.91 s vs FP8 1.88 s |
| empty splits wrote zero partials, combine read all S splits | empty splits skip Q loads and partial writes; combine v2 reads the S log-sum-exps once per block and sums only non-empty splits | t0 → t3 standalone: B=1 1k QL1 18.0 → 13.1 µs |
| decode variant always QLMAX 4 | per step: the smallest of 1 / 2 / 4 ≥ the longest decode row | – |
| TileLang write ~20 µs per call (64-byte stack frame) | CUDA write, 4 threads per block (one scale candidate each), register selects; bit-exact | S4 exact; 7.4 µs vs FP8 7.3 µs at 1–128 tokens |
| short contexts: 2 warps per CTA, serial 16-token loop | token groups (TW): TW groups of warps per CTA take interleaved blocks, merged in fixed order; split plan with 1 CTA/SM | S1 (54 + 20 cases); t4 sweep below |
| at ≥ 24 requests one split: the combine is pure overhead | with one split the decode writes the bf16 output itself (same arithmetic), no combine launch. First try failed to start in vLLM (t6): the output shared the query's size symbol, but vLLM passes a padded output buffer; fixed with its own symbol, and S1 now uses a padded output | S1 single-split cases (t6, t7) |
| prefill precision: S3 (the chunked path, for contexts that do not fit the 64k batched scratch) reached 0.0026–0.0029 vs fp64 with 2–3 chunks, and even one FA2 call 0.00256 | prefill computes in fp16 (`HORIZON_KVFP4_PF16=1`, default): the dequantised K / V are exact in fp16 as in bf16, and fp16 keeps 3 more mantissa bits in FA2's P matrix, its partial outputs and the merges; only the final store rounds to bf16. S3 is now gated (0.0025) and S5's gate tightened from 0.003 to 0.0025 | t7 |

**Decode configuration** is a table by batch size and decode-row length, graph-safe (it depends on the request
count only): `DECODE_TABLE = "16:128:4:2:1,1024x2:128:4:2:1,1024:256:1:3:4"` (maxB[xmaxQL]:min-chunk:groups:stages:
CTAs per SM), overridable with `HORIZON_KVFP4_DECODE_TABLE`. The draft schedule used here sends 4-token rows only up to
10 requests.

**Standalone decode, µs (CUDA graphs, >512 MiB rotated caches, median of 10, best of 3 rounds; t0 = before, t4 =
chosen table config):**

| Shape (requests, context, row) | FP4 t0 best | FP4 now | XQA FP8 |
|---|---|---|---|
| 1, 1k, QL1 | 16.3 | 10.7 | 10.5 |
| 1, 1k, QL4 | 24.8 | 21.6 | 15.0 |
| 1, 16k, QL1 | – | 24.3 | 33.9 |
| 8, 1k, QL4 | 40.1 | 29.4 | 18.5 |
| 8, 1k, QL1 | 20.1 | 15.4 | 15.6 |
| 8, 4k, QL4 | 71.8 | 53.9 | 52.7 |
| 8, 16k, QL4 | – | 161.2 | 176.9 |
| 32, 1k, QL1 | 38.6 | 35.7 | 44.1 |
| 32, 4k, QL1 | 124.9 | 119.7 | 168.5 |
| 32, 16k, QL1 | – | 439.5 | 659.3 |

Still behind XQA: 4-token rows at short context (1–16 requests, ≤ 4k). Ahead from ~4k (1 draft) or ~16k (3 drafts).

**Serving, dev arm vs same-session FP8 control (`pair_arm.sh`, bench_speed + bench_load 32, one sample per side):**

| Run | 1 user | 8 users | 32 users | 32-user TTFT p50 |
|---|---|---|---|---|
| First A/B (old build, mean of 2 per side) | ×0.970 | ×0.935 | ×0.941 | 3.1 vs ~1.9 s |
| t5 (fixes 1–5) | ×0.991 | ×0.985 | ×0.969 | 1.91 vs 1.88 s |
| t8 (final build = `horizon-pkgs-kvfp4-cand2-29-09-2026`) | 155.9 / 155.0 = ×1.006 | 884.6 / 877.1 = ×1.009 | 1,195.8 / 1,212.0 = ×0.987 | 1.89 vs 1.90 s |

**Why S3 was over the gate (29-09-2026):** the error is FlashInfer FA2's own bf16 arithmetic, not the FP4 format. Our
dequantisation is exact (S2, S5), and the reference is fp64 over the same dequantised values. One bf16 FA2 call already
sits at 0.0021–0.0026 (bf16 P matrix and bf16 output), and each extra chunk adds one bf16 rounding of a partial output
before `merge_state`. Serving used this path only for prefill rows whose context does not fit the batched 64k scratch
(long prompts), and those are exactly the long contexts where precision matters most.

### 4c. Fused Hadamard rotation

**Why:** the two gated A/Bs on vLLM 0.30 (max-num-seqs 64, warm compile cache):

| | Plain FP4 KV (run 1757) | FP4 KV + Hadamard, separate rotation (run 1947) |
|---|---|---|
| GSM8K under load, FP4 vs FP8 | 95.53 vs 96.36, p 0.019 ❌ | 95.98 vs 96.29, p 0.45 ✅ |
| Prompt dNLL | +0.0035 | +0.0019 |
| bench_speed 1 / 8 users | ×1.010 / ×0.963 | ×0.985 / ×0.954 |
| bench_load 32 users (gate ≥ 0.97) | ×1.004 ✅ | ×0.906 ❌ |

The Hadamard fixes quality, but the old path pays for it three times per layer and step: a separate `rotate_q` launch
(TileLang, one CTA per token), the slower TileLang write (the CUDA write had no rotation, 14.7 vs 7.4 µs), and in
prefill steps a rotation over every prompt token.

**What changed (kernels `kv_fp4_serving.py` / `attn_decode_fp4.py`, backend `HORIZON_KVFP4_FUSED_ROT=1`, default):**

- **Write:** the CUDA write kernel rotates the keys itself (`HAD` template). The CTA stages the token's K rows ×
  signs in shared memory (fp32), runs the FWHT with one butterfly per thread per stage in the reference order
  (IEEE adds, no contraction), and quantises from fp32 `x · 2^-4`. Values are never rotated. Bit-exact to
  `kv_fp4_ref.write_reference(..., signs)`.
- **Decode:** the ragged decode kernel rotates the query rows while loading them. It stages the CTA's (G · QL, 256)
  rows in shared memory (reusing the pipeline buffers before the pipeline starts), rotates, rounds once to bf16
  and builds the MMA fragments from that. So its output is bit-identical to `rotate_q` + the plain kernel, without
  the launch. Empty splits skip the rotation.
- **Prefill:** `rotate_q` writes fp16 directly (`F16` variant), replacing the cast to fp16 that the prefill already
  launched. There is no extra launch compared with plain FP4 KV.
- `HORIZON_KVFP4_FUSED_ROT=0` keeps the old path for comparison.

**Gates (GPU test `test_kvfp4_serving_gpu.py`):**
- S6: fused write, bit-exact (both layouts, random / outlier / edge tokens, search on / off, padded slots, graph replay).
- S7: fused decode, bitwise equal to separate rotation + plain kernel; rotate_q equal to the reference; ≤ 0.0025 vs
  fp64; deterministic; graph replay.
- S8: fp16 rotation equal to the reference rounded to fp16; batched prefill over rotated keys ≤ 0.0025.

**Offline compile (SM120, zero spills):**
- Every HAD decode variant (dense and MoE geometry, QL 1 / 2 / 4, 1 / 2 / 4 token groups) compiles. The one
  exception is the MoE QL4 × 4 groups variant, which is already excluded: 123 KB of shared memory.
- HAD write: 46 registers, the same as the plain write.
- The QL4 single-group decode needs 24 KB of shared memory for the query rows (21 KB before).

**Results (t9 / t10):**
- **Standalone (µs):** the fused decode is 8–9 µs faster than the separate rotation in every swept shape and costs
  about as much as the plain kernel. Example: 1 request, 1k context, QL4: 22.4 vs 30.1 separate (plain 21.6). The
  fused write is 7.5 µs (FP8 7.3).

### 4d. K8V4: FP8 keys + FP4 values (30-09-2026)

**Why:** run e showed that 4-bit K/V loses accuracy far into a 131k-token document, while GSM8K and ≤ 32k were fine.
Keys carry the far-context precision (attention scores), values only set how much each token contributes. K8V4
keeps production's FP8 keys byte for byte and quantises only the values.

**Format (hz-kv K8V4):**
- One 400-byte slot per (token, KV head): `[256 B K e4m3 | 128 B V E2M1 | 16 B V E4M3 block scales]`.
- vLLM spec: `num_head_slots = KVH`, `state_content_bytes = 400`, cache dtype string still `nvfp4`
  (`HORIZON_KVFP4_K8=1` selects the layout).
- K bytes are exactly vLLM's fp8 KV bytes: `e4m3(k / k_scale)`, round to nearest even, saturating (k_scale = 1 here,
  logged at start).
- V is the unchanged hz-kvfp4 v1 format. Hadamard off (it is for FP4 keys; the backend refuses both).

**Capacity:** 400 vs 512 bytes per token-head = ×1.28 on attention. vLLM reports 1,624,192 vs 1,297,539 KV tokens
= **×1.252** (the GDN state blocks per request are fixed; the hybrid attention block becomes 2,048 tokens, FP8 1,600).

**Kernels (`kv_fp4_serving.py`, `K8` template parameter):**
- Decode: K tiles are staged as bytes (twice the FP4 K bytes, no K scales). Each 8-channel group is expanded in
  registers: `__byte_perm` pairs channels (j, j+4) like the FP4 path, and one masked shift plus one bf16 multiply
  by 2^120 gives the exact e4m3 value, subnormals included. The softmax scale takes k_scale without the FP4 path's
  2^9. V is unchanged. Offline SM120: every variant 147–254 registers, zero spills (except the MoE QL4 × 4-group
  variant, which is excluded anyway).
- Write: K-side threads convert 4 values each with `__nv_cvt_float_to_fp8(__fdiv_rn(x, k_scale), SATFINITE, E4M3)`.
- Dequant (prefill): K bytes go to exact bf16 / fp16 like V. The batched fp16 FA2 prefill and the 512 MB
  workspace are unchanged.

**Gates (GPU test):**
- S10: write bit-exact to `write_reference_k8`, and K bytes equal to vLLM's own `reshape_and_cache_flash` fp8.
- S11: decode ≤ 0.0025 vs fp64, deterministic, graph replay, padded output untouched.
- S12: dequant bit-exact (bf16 and fp16).
- S13: fp16 batched prefill ≤ 0.0025.
- S1: plain FP4 decode regression.

**Standalone decode vs XQA FP8 (k1 sweep, µs, best config of the default table):**

| Shape (requests, context, row) | K8V4 | XQA FP8 |
|---|---|---|
| 1, 1k, QL1 | 10.4 | 10.5 |
| 8, 1k, QL4 | 28.1 | 18.4 |
| 32, 4k, QL1 | 145 | 168 |
| 64, 4k, QL2 | 307 | 367 |
| 64, 16k, QL1 | 1,111 | 1,330 |

The K8V4 write takes 8.0 µs, like FP8's 7.9.

## 5. Compared with vLLM's portable 4-bit options

The byte counts below come from the vLLM 0.29 source on the server.

| Option | Bytes per token-head (K+V), D 256 | vs FP8 | Key idea | Decode path on SM120 |
|---|---|---|---|---|
| FP8 (today) | 512 | 1.00x | per-tensor scale | FlashInfer XQA |
| **hz-kvfp4 (ours)** | **288** | **1.78x** | NVFP4 with 16-value E4M3 blocks, optional K Hadamard, scale search | our TileLang split-KV, tensor cores on exact bf16 tiles |
| `turboquant_4bit_nc` | 130 + 132 = 262 | 1.95x | K: Hadamard + Lloyd-Max 4-bit + fp16 norm; V: 4-bit uniform + fp16 scale/zero | Triton; continuation prefill = full dequant + flash-attn |
| `turboquant_k8v4` | 256 + 132 = 388 | 1.32x | FP8 K, 4-bit V | Triton |
| `int4_per_token_head` | about 256 data + per-token-head scale/zero in a separate tensor (exact size not checked) | about 1.9x | RHT + asymmetric int4, one scale per 256 values | Triton attention backend |

**Our trade-off:**

- We spend 0.5 bit per value on scales, which gives slightly less compression than TurboQuant 4-bit (1.78x vs 1.95x).
- In return the blocks are 16x finer than one scale per head vector. On synthetic data NVFP4 + search has a K error of
  0.083, against 0.109 for int4 per token-head.
- Theory says 4-bit Lloyd-Max on a Gaussian gives 0.0975. That is a theoretical figure, not a TurboQuant measurement.
- The reader is a plain exact expansion into tensor-core tiles, with no centroid lookups.

A separate job measures the Triton options. **Our kernels must beat the
fastest of them on SM120 at 8 and 32 users and at least match their quality gates. Otherwise we should use theirs.**

## 6. Gates

**Standalone (GPU, `scripts/kernels/kvfp4/test_kvfp4_gpu.py`, not run yet):**

- **G1:** write bytes are bit-identical to the reference, for Hadamard 0/1 × search 0/1 and for random,
  outlier-heavy, zero, tiny and huge tokens. Padded slots are skipped and other bytes stay untouched. The output is
  deterministic and correct after two graph replays.
- **G2:** `rotate_q` is bit-identical to the reference, including after graph replay.
- **G3:** decode has relative L2 ≤ 0.0025 against fp64 attention over the **same dequantised cache**, with the
  kernel's own bf16 q'.
  - Covers pipe 0/1, QL 1/4, the dense (4×6) and MoE (2×8) geometries, and non-unit g.
  - The output must be finite, deterministic and correct after two NaN-poisoned graph replays.
  - The CPU emulation of the algorithm (T7) reaches 0.00214, so the slack is only about 15%. The FP8 kernel passes the
    same limit at about 0.0022.
  - Also reported: the error against unquantised bf16 K/V and against an FP8 cache.
- **G4 (speed):**
  - Measured with graph replay, more than 512 MiB of rotated KV copies per format, and 3 interleaved rounds.
  - Compared against FlashInfer XQA on FP8 and against our FP8 kernel, at B 1/8/32, context 1k–128k and QL 4/1.
  - The write kernel is compared against vLLM's FP8 `reshape_and_cache_flash`.

**Real K/V, offline (before integration):** capture K/V of a few attention layers from the live model on long prose
and code prompts. Measure round-trip and attention-output error with Hadamard on/off and search on/off. It is cheap
and decides the defaults on real data.

**Model quality, as an A/B against production FP8 KV:**

- **Prompt log-prob drift** (`logprob_eval.py`, prose and code docs from 1k to 32k):
  - top-1 agreement ≥ 98.5%;
  - mean |Δ log-prob| ≤ 0.03 per token;
  - aggregate NLL change ≤ +0.01 nats/token.
  - For comparison, the FP8 output head gave 99.13% / 0.0141 and was still not accepted.
- **Long context:** `long_context_vs_production` requires 16k and 32k prose + code NLL within PPL ×1.10. This is the
  gate the NVIDIA FP8 checkpoint failed; short evals do not catch it.
- **Needles:** 64k, 128k and 250k, 3/3.
- **Paired evals:** GSM8K 1,319, HumanEval, MBPP, IFEval, MMLU-Pro 200. None may drop significantly (paired test);
  the baselines are 96.3 / 97.0 / 76.6 / 81.1 / 65.5.
- **MTP acceptance:** the drafts also read the FP4 cache. Acceptance per position must stay within 0.02 of baseline
  for prose, maths and code.
- **Determinism:** within the FP4 arm, cached (prefix-hit) and uncached prompt log-probs must be bit-identical, and the
  same request twice must be identical.
- **Greedy identity with production is NOT expected**, because the numbers change. Report the first-divergence
  statistics only.

**Speed and capacity (A/B, interleaved):**

- vLLM's reported KV tokens at the same 46 GiB must be ≥ 1.7x of 1,297,539.
- tok/s at 1 and 8 users must be no worse than −2%.
- 32 users and long-context load (8 users × 32k) must be faster.
- TTFT at 64k must be no worse than +5%.
- It must beat the best Triton 4-bit option at 8 and 32 users.

## 7. Risks and open questions

1. **K at 4 bits is the quality risk.** Sharp attention amplifies key errors, and every 4-bit scheme has about 3x
   FP8's element error. Fallbacks, in order: hot window, then K8V4 (1.28x).
2. **Long-context accumulation:** the same failure class as the FP8 attention/GDN checkpoint. The 16k/32k NLL gate is
   decisive.
3. **Decode speed:** the scalar expansion and our split-KV kernel, which is already behind XQA. The capacity win can
   be real while the speed loses at short context; the gates cover both.
4. **The 0.0025 standalone gate is tight** (T7 needs 85% of it). A failure there would first need to be separated
   into kernel bug versus bf16-P rounding.
5. **vLLM plumbing:**
   - nvfp4 is SM100-only in several places;
   - hybrid page-size unification with the new page size;
   - FP8 q quantisation for nvfp4;
   - backend selection per layer;
   - keeping it working across vLLM upgrades.
6. **Prefill:** memory for the scratch pieces (bounded to 128 MiB per layer by the LSE merge), and the numerics of the
   merge.
7. **The Hadamard signs are part of the cache format.** The writer and reader must use the same vector; changing it
   invalidates cached data, so a restart is required.
8. **SM120 native FP4 MMA:** `mma.sync kind::f8f6f4` needs both operands in FP8/FP6/FP4, and the block-scaled kinds
   use UE8M0 scales, not E4M3. Using the tensor cores directly on FP4 K would need an FP8 q and a different scale
   type. That is research, not v1.
