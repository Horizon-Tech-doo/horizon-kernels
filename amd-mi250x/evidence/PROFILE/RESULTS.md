# PROFILE: where a serving step of the 27B goes on MI250X, and what to wire next (10-11 Nov 2026)

Labels: [observed] measured in a job; [computed] arithmetic on observed numbers; [inferred] reasoning from source or numbers; [hypothesis].

**Result.** On the stock server (vLLM 0.30.0 ROCm image, public Qwen3.8-27B, BF16, TP=2 on 2 GCDs), a prefill is a GEMM workload and a decode step is an attention-plus-weight-streaming workload:

- **Prefill** (1.4k to 16k tokens): the dense linear layers are 63 to 74% of the step time [computed], the MLP alone 43 to 48%. All of them run torch `F.linear` on bf16, which calls hipBLASLt [observed]. The attention kernel is 3 to 18%, the Gated-DeltaNet core 6 to 17%, the tensor-parallel all-reduce 5% at vLLM's default chunking and 16 to 18% when a prompt is prefilled in one step. Norms, activations, residual adds, RoPE and reshapes together are 1 to 3%; sampling, the output head, KV-cache writes and the runner's Python are each under 1%.
- **Decode** at 8k contexts, default attention backend: the attention kernel is 27 to 28 ms of a 56 to 73 ms step at 1, 8 and 32 streams (the default-path problem the E2E track already found; the stock `--attention-backend TRITON_ATTN` flag removes it). The MLP GEMMs are 13 to 15 ms at 1 stream (vLLM's own `wvSplitK` kernel) and 19 to 24 ms at 8 and 32 streams (hipBLASLt): the step streams the weights once and is bandwidth bound.
- vLLM at its defaults does **not** prefill a long prompt in one step with this model: `max_num_batched_tokens` is 2048 and the scheduler cuts at multiples of 784 tokens, so the server's GEMM row count is 1568 (and 784, and a remainder) at every prompt length [observed].

**What to wire next [computed from this profile and the two-job GEMM race below].**

| Candidate | Share it touches | Kernel ratio measured | Expected end-to-end gain | Status |
|---|---|---|---|---|
| bf16 GEMM (`tilelang/gemm_bf16.py`, new) for the five large linear shapes, prompts prefilled in one step (`--max-num-batched-tokens 16384`) | 63% of a 16k prefill | 1.03 to 1.13 per shape at 15680 rows, 1.06 to 1.13 at 7056 | computed: 4.8% of time to first token at 16k, 5.8% at 7.7k, 1.9% at 4k | **wired and measured (`evidence/G1/`): 5.8 to 8.0% at 2.9k to 14k tokens in two opposite-order whole-node jobs** |
| the same at vLLM's default chunk (1568 rows) | 65% | 224x256 tile: 0.73 to 1.02, no lead; per-shape tiles from the sweep: 1.02 to 1.14 at four shapes, down_proj none | computed: 3.5% at 16k with the sweep's tiles; 9 to 10% slower with the 224x256 tile | **wired with the sweep's tiles and measured (`evidence/G1/`): 2.7 to 3.4% at 2.9k to 14k tokens** |
| prefill attention on chunks that have cached context (today's plugin takes only the first chunk) | 17.5% of a 16k prefill at default chunking | not measured (rectangular causal attention is not in `gqa_fa.py`) | upper bound 17.5% | kernel form missing |
| decode GEMM at 8 to 32 rows | 41% (MLP) to 64% (all five shapes) of the 45 ms 8-stream step of E2E's stock Triton-flag arm [computed from standalone call times] | ours 0.99 to 1.01 of hipBLASLt (tie) | none today; 8.7% of the step if 8-row calls reached the rate `wvSplitK` reaches at 1 row | kernel work, not wiring |
| fp16 `gemm_prefetch.py` through casts | as row 1 | 0.84 to 1.02 as wired (bare kernel 0.93 to 1.12) | negative | rejected: the casts cost more than the kernel gains |
| Gated-DeltaNet core | 6 to 17% of prefill | nothing of ours exists on AMD | - | no kernel; vLLM runs Triton/FLA (`fla_chunk_gated_delta_rule` is 58% of the core's time, the causal conv 14%) |
| tensor-parallel all-reduce | 17% of a one-step prefill | not a kernel of ours | - | vLLM's `PYNCCL` backend over two GCDs; noted, not investigated |

This file holds the profile, the kernel races and the estimates. The end-to-end measurements of the GEMM plugin are in `evidence/G1/RESULTS.md`; the estimates here were written before those jobs ran and are left as they were. One more row was found while probing decode: **stock vLLM is 14% slower per decode step at exactly 5 concurrent streams than with its own small-row kernels switched off** (its `wvSplitK` kernel at 5 rows, reached through the output head); see the decode GEMM section at the end.

## Method

- **Server.** Exactly the P1 / E2E flags: `vllm serve <27B> --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88 --reasoning-parser qwen3 ...`, two GCDs of small-g / dev-g nodes. Arms `_d` leave `max_num_batched_tokens` at its default (2048); arms `_m16` add `--max-num-batched-tokens 16384`.
- **Plugin `horizon_lumi_profile`** (new, `vllm-integration/overlay/horizon_lumi_profile/`, a `vllm.general_plugins` entry point, changes no result). vLLM compiles the model with torch.compile, so module forward hooks do not fire and torch.profiler in this image reports CPU activity only (`supported_activities()` = {CPU}) [observed]. But every heavy operator is an opaque Python call at run time even inside the compiled model: every dense linear layer goes through the custom op `vllm.rocm_unquantized_gemm`, attention through `RocmAttentionImpl.forward`, the Gated-DeltaNet core through `QwenGatedDeltaNetAttention._forward_core*`, the all-reduce through `GroupCoordinator._all_reduce_out_place` [read from source in the image]. The plugin wraps those and the runner's `execute_model` / `sample_tokens` (the V2 runner, `vllm/v1/worker/gpu/model_runner.py`, is the one this server uses) [observed].
- **Modes.** `step`: clock reads only. `sync`: one `torch.cuda.synchronize()` per step. `ops`: a synchronise before and after every wrapped operator; the time between two operators is booked as glue (the Inductor-compiled code and Python), so operators plus glue partition the step exactly (timed parts sum to the step within 0.1%) [observed].
- **Arms**, one fresh server start each: `s` no plugin; `ps` step mode; `py` sync mode; `pyn` sync mode with `--compilation-config '{"cudagraph_mode":"NONE"}'`; `po` ops mode with the same flag. Decode steps replay a HIP graph in the default mode, which runs no Python, so the per-operator numbers for decode are from a server without graphs. Prefill steps are not affected by the flag (py and pyn agree within 0.1% at 1.4k, 7.7k and 16k; 4.6% apart at 4k) [observed].
- **Load** (`pf_client.py`): synthetic prompts, greedy, unique tag at the start of every prompt (no prefix-cache hit). Prefill cells: one prompt at a time at 1,413, 3,993, 7,713 and 16,115 tokens, 2 warm and 4 timed. Decode cells: 1, 8, 32 prompts of 7,709 to 7,713 tokens at once, 96 output tokens; steps are taken from the window in which every stream decodes. Server step records are matched to cells by wall-clock time.
- **Correction of the synchronise cost [computed, a model].** Synchronising after every operator costs time: the ops arm reads 4 to 50% above the sync arm in prefill and 1.8 to 2.3x in decode. c = (ops-arm step - `pyn` step) / number of synchronises is 76 to 101 us in prefill and 48 to 70 us in decode. The corrected column subtracts c per synchronise charged to a class (one per call, one per glue segment, two per nested timer). It assumes every synchronise costs the same. It fails where an operator is a few small kernels around nested timers: the decode Gated-DeltaNet core comes out at -5 to -1 ms (raw 16 to 21 ms) and the compiled glue at -13 ms in one cell; read those as "between zero and raw". The large entries move little: MLP GEMM raw against corrected differs by 1 to 12% in prefill.
- **Jobs.** 23621041 (dev-g, 5,696 s): `po_d s_d ps_d py_d pyn_d po_m16 py_m16 s_m16`; `po_m16` failed to start (a shared-memory error at engine start, not the plugin) and was rerun in 23624390 (dev-g): `po_m16 pyn_m16`. 14 client rows in every arm that started [observed]. Smoke jobs 23620213 (the plugin wrapped the V1 runner, which this server does not use: no records) and 23620876 (client argument error). Sources and md5s: `src/`, `src-md5.txt`. Raw: `raw/<job>/` (`hpf-<arm>/hpf-<pid>.jsonl` one line per step and worker, `client-*.jsonl`, `pf-summary.json`). Tables: `report.md` (every cell, raw and corrected, GEMM by shape, glue by neighbours), `tables-<job>.md`.

## Cross-check against the unprofiled server [observed]

Client time to first token (s) or decode step (ms); server = step time summed over the prefill's steps, worker 0 (worker 1 agrees within 0.1%).

| Cell | steps (tokens per step) | `s_d` client | `ps_d` client | `py_d` client / server | `pyn_d` client / server | `po_d` client / server |
|---|---|---|---|---|---|---|
| prefill 1,413 | 784, 629 | 0.4547 | 0.4572 | 0.4646 / 452.9 | 0.4645 / 453.0 | 0.6869 / 679.0 |
| prefill 3,993 | 1568, 1568, 784, 73 | 1.2405 | 1.2479 | 1.2635 / 1243.0 | 1.3196 / 1299.8 | 1.6873 / 1674.5 |
| prefill 7,713 | 4 x 1568, 784, 657 | 2.3688 | 2.3798 | 2.4038 / 2371.2 | 2.4015 / 2370.7 | 3.0142 / 2994.7 |
| prefill 16,115 | 10 x 1568, 435 | 5.3387 | 5.3585 | 5.4069 / 5351.4 | 5.4013 / 5346.6 | 6.4872 / 6447.6 |
| decode x1, ms | 1 | 55.78 | 56.34 | 58.88 / 58.7 | 70.81 / 70.6 | 135.24 / 135.0 |
| decode x8, ms | 8 | 68.51 | 69.99 | 72.43 / 71.9 | 93.67 / 93.2 | 167.83 / 166.9 |

| Cell (`m16`) | steps | `s_m16` client | `py_m16` client / server | `pyn_m16` client / server | `po_m16` client / server |
|---|---|---|---|---|---|
| prefill 1,413 | 784, 629 | 0.4566 | 0.4628 / 451.5 | 0.5135 / 505.0 | 0.7634 / 755.5 |
| prefill 3,993 | 3920, 73 | 1.2200 | 1.2286 / 1208.5 | 1.4109 / 1397.3 | 1.6110 / 1597.5 |
| prefill 7,713 | 7056, 657 | 2.2373 | 2.2499 / 2218.9 | 2.5397 / 2522.8 | 2.7591 / 2737.3 |
| prefill 16,115 | 15680, 435 | 4.7254 | 4.7409 / 4687.2 | 5.3303 / 5292.4 | 5.5472 / 5507.5 |
| decode x1, ms | 1 | 55.83 | 58.16 / 58.0 | 63.98 / 63.8 | 144.56 / 144.1 |
| decode x8, ms | 8 | 69.50 | 71.50 / 71.4 | 84.99 / 84.9 | 173.47 / 173.9 |
| decode x32, ms | 32 | 72.95 | 75.37 / 75.3 | 86.53 / 86.2 | 180.85 / 180.0 |

- The step-mode plugin changes the client numbers by 0.4 to 0.6% in prefill and 1.0 to 2.2% in decode; one synchronise per step by 1.3 to 2.2% and 2.9 to 5.7%.
- Outside the runner's steps (tokenising, scheduling, the engine's IPC, streaming) a prefill spends client minus server = 11 to 56 ms (1.0 to 2.6%) and a decode step 0.1 to 0.5 ms [computed, `py` arms].
- Replaying graphs is worth 12 ms of a 1-stream decode step and 21 ms at 8 streams (`pyn_d` against `py_d`) [observed].
- `py_m16` and `s_m16` ran in job 23621041, `pyn_m16` and `po_m16` in job 23624390 on another node, which reads 12% slower in every prefill cell including the one whose steps are identical (1,413 tokens: 505.0 against 451.5 ms). The two GCDs of a small-g or dev-g job are dies of two different cards, each shared with another tenant and capped at 500 W per card, so step time moves with the neighbours' load (the P1 track shows this directly). Shares are computed inside one arm; absolute times are not comparable across jobs.
- Decode x32 at default chunking is not in the table: 32 prompts of 7.7k tokens are prefilled in 1568-token chunks interleaved with the decoding of the streams that started first, so no window exists in which all 32 decode.

## Prefill: where the time goes [corrected ms per prefill and share, computed; raw in `report.md`]

vLLM defaults (`_d`, job 23621041):

| Class | 1,413 | 3,993 | 7,713 | 16,115 |
|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 218.2 (48.3%) | 593.6 (45.7%) | 1112.1 (47.0%) | 2290.3 (42.9%) |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 61.3 (13.5%) | 171.9 (13.2%) | 318.5 (13.5%) | 653.7 (12.2%) |
| attention projection GEMM (qkv_proj, o_proj) | 53.7 (11.9%) | 138.6 (10.7%) | 250.7 (10.6%) | 506.8 (9.5%) |
| attention kernel | 16.1 (3.6%) | 88.1 (6.8%) | 228.5 (9.6%) | 936.9 (17.5%) |
| Gated-DeltaNet core (conv + delta rule) | 76.0 (16.8%) | 181.7 (14.0%) | 274.6 (11.6%) | 528.1 (9.9%) |
| tensor-parallel all-reduce | 24.4 (5.4%) | 70.3 (5.4%) | 121.0 (5.1%) | 274.1 (5.1%) |
| compiled glue: norms, activations, residual adds, RoPE, embedding, reshapes | -9.7 (model fails; raw 84.0) | 26.7 (2.1%) | 23.0 (1.0%) | 78.1 (1.5%) |
| sampling incl. lm_head GEMM and all-gather | 3.0 (0.7%) | 6.3 (0.5%) | 9.2 (0.4%) | 16.9 (0.3%) |
| KV-cache write | 1.7 (0.4%) | 4.9 (0.4%) | 6.7 (0.3%) | 13.4 (0.3%) |
| runner: input prep, bookkeeping, Python | 7.6 (1.7%) | 15.8 (1.2%) | 23.5 (1.0%) | 44.3 (0.8%) |
| sum = `pyn_d` step | 452.3 | 1297.7 | 2367.8 | 5342.5 |

One step per prompt (`_m16`, job 23624390):

| Class | 1,413 | 3,993 | 7,713 | 16,115 |
|---|---|---|---|---|
| MLP GEMM | 216.8 (43.0%) | 626.5 (44.9%) | 1148.5 (45.5%) | 2305.9 (43.6%) |
| GDN projection GEMM | 59.8 (11.9%) | 146.9 (10.5%) | 288.4 (11.4%) | 585.8 (11.1%) |
| attention projection GEMM | 52.6 (10.4%) | 129.2 (9.3%) | 232.4 (9.2%) | 442.2 (8.4%) |
| attention kernel | 16.4 (3.2%) | 65.0 (4.7%) | 155.3 (6.2%) | 571.8 (10.8%) |
| Gated-DeltaNet core | 77.9 (15.4%) | 122.5 (8.8%) | 175.2 (6.9%) | 319.3 (6.0%) |
| tensor-parallel all-reduce | 80.9 (16.0%) | 252.7 (18.1%) | 440.6 (17.5%) | 891.2 (16.8%) |
| compiled glue | -13.1 (model fails; raw 90.8) | 37.8 (2.7%) | 65.8 (2.6%) | 152.9 (2.9%) |
| sampling incl. lm_head | 3.0 (0.6%) | 3.2 (0.2%) | 3.1 (0.1%) | 3.1 (0.1%) |
| KV-cache write | 1.7 (0.3%) | 3.1 (0.2%) | 3.7 (0.1%) | 5.7 (0.1%) |
| runner | 8.4 (1.7%) | 9.6 (0.7%) | 10.7 (0.4%) | 14.4 (0.3%) |
| sum = `pyn_m16` step | 504.5 | 1396.4 | 2523.6 | 5292.3 |

Findings:

1. **GEMM is the prefill.** All dense linears: 73.7%, 69.6%, 71.1%, 64.6% at default chunking and 65.3%, 64.7%, 66.1%, 63.1% in one step [computed]. Measured rates of the two MLP shapes inside the server: 110 to 127 TFLOP/s per GCD from 1568 rows up, 83 to 108 at 435 to 784 rows, 28 to 40 at 73 rows [observed, raw per-call times].
2. **Attention is a small share in one step and a large one in chunks.** One step at 15,680 tokens: 571.8 ms over 16 layers and 2 steps, 10.8%. The same prompt in ten 1568-token chunks: 936.9 ms, 17.5%, because every chunk after the first attends over the cached context with vLLM's default-path kernel (5.4 ms per layer call on average, raw). The P1 plugin takes only the first, fresh chunk.
3. **The Gated-DeltaNet core** costs 1.4 ms per layer call at 1568 tokens (raw): `fla_chunk_gated_delta_rule` 0.81 ms, `causal_conv1d_fn` 0.20 ms, the rest gating and state handling [observed, nested timers]. We have no kernel for it on AMD (the E2E track lists it as vLLM Triton/FLA in every arm; `tilelang/` holds GQA attention, GEMM and MoE only) [observed].
4. **All-reduce** is 258 calls of 3.5 ms at 15,680 rows (one after every `down_proj`, `o_proj` and `out_proj`): 891 ms, 16.8%. At 1568 rows it is 0.19 ms a call (corrected) and 5.1%. Per byte the large call is about half as fast (46 against 84 GB/s) [computed]. vLLM logs `Using ['PYNCCL'] all-reduce backends`. In this job the two GCDs are on different cards [inferred from the power metadata of the same partition; not checked for this job].
5. **Everything else is small**: norms, SiLU, residual adds, RoPE, QK-norm, embedding and reshapes together 1.0 to 2.9%; sampling with the 124,160-row output head 0.1 to 0.7%; KV-cache writes 0.1 to 0.4%; the runner's Python 0.3 to 1.7%.

## Decode: where the time goes

Per decode step, 8k contexts, **no graph replay, default attention backend** (job 23624390 `po_m16` / 23621041 `po_d`). Raw [observed] and corrected [computed]; steps without graphs are 63.8, 84.9, 86.2 ms (`pyn_m16`), with graphs 58.0, 71.4, 75.3 ms (`py_m16`), unprofiled 55.8, 69.5, 72.9 ms (`s_m16`).

| Class | x1 raw / corrected | x8 raw / corrected | x32 raw / corrected |
|---|---|---|---|
| attention kernel (16 calls) | 29.2 / 28.3 | 28.4 / 27.4 | 28.8 / 27.7 |
| MLP GEMM (128 calls) | 21.0 / 13.3 | 31.8 / 23.3 | 32.0 / 23.0 |
| GDN projection GEMM (96) | 8.4 / 2.6 | 15.5 / 9.1 | 15.5 / 8.8 |
| attention projection GEMM (80) | 7.0 / 2.1 | 12.9 / 7.5 | 12.7 / 7.1 |
| tensor-parallel all-reduce (129) | 15.9 / 8.2 | 17.3 / 8.7 | 18.9 / 9.8 |
| Gated-DeltaNet core (48) | 17.0 / below zero | 18.0 / below zero | 21.4 / below zero |
| compiled glue (513 segments) | 38.6 / 7.7 | 42.3 / 8.0 | 42.4 / 6.4 |
| sampling incl. lm_head | 1.9 / 1.5 | 2.3 / 2.0 | 2.7 / 2.3 |
| KV-cache write | 2.3 / 1.4 | 2.4 / 1.3 | 2.5 / 1.3 |
| runner | 2.8 / 1.9 | 2.9 / 2.0 | 3.1 / 2.0 |

- **Attention, 1.8 ms a layer call, is the only entry that needs no correction** (the per-call time is 30 times c): 27 to 28 ms of every step at 8k contexts whatever the stream count. This is the default `ROCM_ATTN` path, whose log says the ROCm paged kernel cannot be used and a Triton fallback runs; E2E measured that the stock `--attention-backend TRITON_ATTN` flag or our decode kernel removes 16 to 25 ms of it.
- **GEMM in decode is weight streaming.** Standalone on one GCD, the same calls (job 23620226, bf16, no profiler) [observed]: 1 row: `wvSplitK` 0.153 ms (gate_up, 178 MB of weights: 1.17 TB/s) and 0.078 ms (down); 8 rows: hipBLASLt 0.190 and 0.103 ms; 32 rows: 0.194 and 0.109 ms. So the MLP costs 64 x (0.153 + 0.078) = 14.8 ms at 1 stream and 64 x (0.190 + 0.103) = 18.8 ms at 8 [computed], which brackets the corrected 13.3 and 23.3 ms. vLLM sends 1 to 5 rows to its own `wvSplitK` kernel and everything larger to hipBLASLt (`rocm_unquantized_gemm_impl`; the 10 to 128 row kernel `wvSplitKrc` is enabled for gfx950 only) [read from source].
- The per-operator split of the remaining 10 to 20 ms (Gated-DeltaNet core, all-reduce, glue) is not resolved: these are 700 small launches a step, each shorter than a synchronise.

## Which GEMMs the server issues [observed: plugin records and the weight list of worker 0]

C[M, N] = A[M, K] @ W[N, K]^T, activations bf16, weights bf16, no bias, per GCD under TP 2:

| Layer | N x K | per step | weight bytes per GCD |
|---|---|---|---|
| MLP `gate_up_proj` | 17408 x 5120 | 64 | 10.62 GiB |
| MLP `down_proj` | 5120 x 8704 | 64 | 5.31 GiB |
| GDN `in_proj_qkvz` | 8192 x 5120 | 48 | 3.75 GiB |
| GDN `out_proj` / attention `o_proj` | 5120 x 3072 | 48 + 16 | 1.41 + 0.47 GiB |
| attention `qkv_proj` | 7168 x 5120 | 16 | 1.09 GiB |
| GDN `in_proj_ba` | 48 x 5120 | 48 | 0.02 GiB |
| `lm_head` (in sampling, one row per sequence) | 124160 x 5120 | 1 | 1.18 GiB |

All parameters on one GCD: 25.05 GiB, all bf16.

M (rows = tokens in the step):

- vLLM defaults: a prompt is cut at multiples of 784 tokens and at most 1568 are scheduled per step for one prompt: 1,413 -> 784 + 629; 3,993 -> 1568, 1568, 784, 73; 7,713 -> 4 x 1568, 784, 657; 16,115 -> 10 x 1568, 435. So **M = 1568 dominates**, then 784 and one remainder below 784.
- `--max-num-batched-tokens 16384`: the first step takes the largest multiple of 784: 3920, 7056, 15680 (P1's prompts: 2352, 5488, 7056, 14112), then the remainder.
- Decode: M = number of streams.
- Library: `F.linear` on bf16 and on fp16 calls hipBLASLt at every M from 1 to 2048 (`rocblaslt_matmul` in the API log, 16 of 16 calls; no rocBLAS call) [observed, job 23620226 `liblog.txt`]. MIOpen is not on the path.

## GEMM: our kernels at the server's shapes

**fp16 kernel through casts: rejected [observed, one job, 23620226, 5 rounds].** `gemm_prefetch.py` is fp16 only and the server is bf16. Bare kernel against hipBLASLt fp16 at 1024 to 16384 rows: 1.04 to 1.12, and 0.93 at 1536 x 5120 x 8704. As it would be wired (activations cast to fp16, an fp16 weight copy of 2 bytes per weight, result cast back): 0.84 to 0.98 of the server's path in 11 of 12 large cells and 1.02 at 4096 x 17408 x 5120. The two casts cost about 10% of the GEMM. A bf16 weight is not always exactly representable in fp16 (`fp16_copy_exact` false: values below fp16's range) [observed].

**gfx90a has the bf16 MFMA.** `v_mfma_f32_16x16x16bf16_1k` assembles and runs in the image's toolchain: same 16x16 tile and K 16 per instruction as the fp16 form, so loads, LDS layout and schedule do not change [observed]. `tilelang/gemm_bf16.py` (new; `gemm_prefetch.py` is not edited) loads a private copy of `gemm_prefetch` and rewrites the generated source in two places: the MFMA mnemonic, and the epilogue (fp32 to bf16 by bits, round to nearest even). Tensors are passed reinterpreted, without a copy.

**Race [observed]: jobs 23624388 (forward order) and 23624389 (reversed), one GCD each, 7 rounds, 60 cells each, 0 errors, 0 failed error checks.** Ratio = vLLM's path over ours in time (above 1: ours faster). noise(job) = largest |stock_b / stock - 1| over the job's cells = 3.25% and 2.80% (one cell, 1024 x 5120 x 3072; every other cell is at most 1.2%). LEAD needs the ratio above 1 + noise(job) in both jobs, LOSS below 1 - noise(job) in both. Error against an fp32 product: 1.66e-3 for both ours and stock in every cell (bf16 rounding of the result). Full table: `gemm-bf16-tables.md`.

Rows form on a 224 x 256 tile with M as given (no padded copy), forward / reversed:

| M | gate_up 17408x5120 | down 5120x8704 | qkvz 8192x5120 | qkv 7168x5120 | out/o 5120x3072 |
|---|---|---|---|---|---|
| 784 | 0.862 / 0.864 LOSS | 0.886 / 0.886 LOSS | 0.707 / 0.698 LOSS | 0.654 / 0.648 LOSS | 0.997 / 0.986 TIE |
| **1568** | 0.970 / 0.966 TIE | 0.743 / 0.729 LOSS | 0.853 / 0.848 LOSS | 1.018 / 1.003 TIE | 0.778 / 0.766 LOSS |
| 3920 | 1.093 / 1.088 LEAD | 0.989 / 0.984 TIE | 0.957 / 0.959 LOSS | 1.078 / 1.067 LEAD | 1.024 / 1.017 TIE |
| 7056 | 1.133 / 1.132 LEAD | 1.108 / 1.099 LEAD | 1.072 / 1.067 LEAD | 1.065 / 1.063 LEAD | 1.131 / 1.134 LEAD |
| 15680 | 1.115 / 1.113 LEAD | 1.030 / 1.031 TIE | 1.112 / 1.108 LEAD | 1.131 / 1.123 LEAD | 1.041 / 1.045 LEAD |
| 8192 | 1.120 / 1.114 LEAD | 1.098 / 1.092 LEAD | 1.117 / 1.109 LEAD | 1.069 / 1.065 LEAD | 1.112 / 1.106 LEAD |
| 16384 | 1.117 / 1.117 LEAD | 1.098 / 1.094 LEAD | 1.102 / 1.097 LEAD | 1.116 / 1.110 LEAD | 1.111 / 1.108 LEAD |

- `gemm_prefetch`'s own dispatch (`gemm_auto`, M padded up to a multiple of 128 through a copy) at 1568 rows, i.e. at 1664: 0.87 to 0.98, LOSS at four shapes and TIE at one. At 1024, 2048, 4096, 8192, 16384 it leads at 21 of 25 cells (1.03 to 1.14), ties 2 (8192 x 5120 at 1024 and 4096) and loses 2 (7168 x 5120 at 1024 and 2048).
- 8 and 32 rows (the small-M forms, bf16): 0.987 to 1.006 at 17408 x 5120, TIE. No small-M form exists for the other four shapes.
- **Padding M to a bucket costs the padding** (time is linear in the padded M) plus a copy of the activations of under 1%: 1568 -> 1664 is 6.1%, more than the kernel's lead at any shape. The rows form avoids the copy but its tile count still rounds up (1568 = 7 x 224 exactly).
- All 19 multiples of 784 from 1568 to 15680 through the plugin's own path (job 23624793, one job, `evidence/G1/raw/23624793/check.jsonl`): 2352 rows lead at four shapes (1.10 to 1.21) and lose at 8192 x 5120 (0.89); 3136 rows 0.95 to 1.05; from 4704 rows up 73 of 75 cells are at or above 0.98 and 64 are above 1.04 (the two below: 0.977 at 4704 x 8192 x 5120 and 0.939 at 6272 x 5120 x 8704); cell noise at most 0.75%; error equal to stock's in all 95 cells.

**Tile sweep at 1568 and 784 rows** (job 23624491): see the section at the end of this file.

## Expected gain from wiring the bf16 GEMM [computed]

Gain in time to first token = sum over shapes of (that shape's corrected ms) x (1 - 1 / ratio) / (step time), with the per-row-count ratios of job 23624793 and the shares of the `_m16` profile:

| Prompt | first step | gate_up | down | qkvz | out/o | qkv | sum | of step |
|---|---|---|---|---|---|---|---|---|
| 16,115 | 15680 | 1533.7 ms x (1 - 1/1.109) = 150.8 | 700.5 x (1 - 1/1.026) = 17.8 | 554.4 x (1 - 1/1.111) = 55.4 | 264.2 x (1 - 1/1.042) = 10.6 | 159.7 x (1 - 1/1.126) = 17.9 | 252.5 ms | **4.8%** of 5292 |
| 7,713 | 7056 | 703.3 x (1 - 1/1.128) = 79.8 | 342.8 x (1 - 1/1.103) = 32.0 | 250.7 x (1 - 1/1.067) = 15.7 | 133.8 x (1 - 1/1.130) = 15.4 | 73.4 x (1 - 1/1.061) = 4.2 | 147.1 ms | **5.8%** of 2523 |
| 3,993 | 3920 | 402.4 x (1 - 1/1.085) = 31.5 | 198.8 x (1 - 1/0.980) = -4.1 | 133.5 x (1 - 1/0.960) = -5.6 | 80.0 x (1 - 1/1.020) = 1.6 | 40.7 x (1 - 1/1.068) = 2.6 | 26.0 ms | **1.9%** of 1397 |

At vLLM's default chunking the same sum with the 1568-row ratios is negative (gate_up 0.968, down 0.736, qkvz 0.848, out/o 0.777, qkv 1.011): wiring today's forms there would slow the server by 9 to 10% of a 16k prefill [computed]. Decode: no gain (ties).

Ceiling for reference: if every prefill GEMM gained what our best cell gains (1.13 at 4096 x 17408 x 5120 and at 7056 x 17408 x 5120), the 16k one-step prefill would shorten by about 63% x (1 - 1 / 1.13) = 7.2% [computed].

## Power-limit sweep: not possible as a user

On a small-g compute node, as the job's user (job 23620226, `raw/23620226/power-controls.txt`) [observed]:

- amdgpu hwmon `power1_cap` (500 W; `power1_cap_max` 560 W), `power_dpm_force_performance_level` (reads `manual`), `pp_dpm_sclk` (500 / 800 / 1650 MHz) and `pp_od_clk_voltage` are root-owned with mode 0644: not writable. These are the files `rocm-smi --setpoweroverdrive`, `--setperflevel`, `--setsclk` and `amd-smi set -o / -l / -L` write.
- Cray `pm_counters` `accel*_power_cap` and `power_cap` are read-only (they read 0 W).
- No set command was issued: writability was tested with `test -w` only, because a card's cap is shared with whatever other job holds its second die.

So neither the power cap nor the clock or performance level can be changed by a user on LUMI. No energy sweep was run. What exists instead: every P1, G1 and PROFILE server job has the host sampler's `power.csv`; the G1 whole-node jobs are analysed for card energy per prefill cell (`evidence/G1/RESULTS.md`), the others are not.

## Not tested

- Energy in the profile jobs (the G1 track has card energy for the prefill cells). Any quality benchmark.
- The per-operator split of a decode step with graph replay (no Python runs there) and of decode's Gated-DeltaNet core, all-reduce and glue beyond raw upper bounds.
- `TRITON_ATTN` backend profile; batch prefill (several prompts in one step); prompts above 16k; other models; one-GCD serving.
- Whether the two GCDs of the profile jobs shared a card, and how all-reduce behaves when they do.
- GEMM: real activations (the race uses random bf16 inputs and random weights scaled by 0.02), placed weights (`place_b`), concurrent load on the second die, 5120 x 8704 small-M forms, M between the measured values.
- torch.profiler / roctracer kernel names (CPU activity only in this image).

## Tile sweep at the default chunk: 1568 and 784 rows (job 23624491, one GCD, one job: a search, not a race) [observed]

`gm_tiles.py`: `gemm_bf16`'s copy of `gemm_hand_rows` over 14 tiles x 5 schedules per (rows, shape), each timed against vLLM's path in the same process, 5 interleaved rounds; stock_b is the identical-code control. 36 of 70 forms per cell built and passed the error check (error at most 1.25x stock's); 340 builds were refused by the generator (the tile does not fit the schedule or the LDS), 0 wrong results. All 10 cells ran.

| M x N x K | stock TFLOP/s | cell noise | best forms: stock / ours | 224x256 xs4 (the form raced above) |
|---|---|---|---|---|
| 1568 x 17408 x 5120 | 117.4 | 0.59% | 224x224:xs4 1.089, 224x224:plw16 1.075, 224x160:xs4 1.059, 224x160:plw 1.056 | 0.9742 |
| 784 x 17408 x 5120 | 116.1 | 1.31% | 224x224:xs4 0.980, 224x160:xs4 0.971, 224x160:plw 0.968, 224x224:plw16 0.959 | 0.8739 |
| 1568 x 5120 x 8704 | 119.0 | 1.95% | 160x256:xs4 1.000, 160x256:plw16 0.978, 128x128:xs4 0.969, 128x128:x8_24 0.951 | 0.752 |
| 784 x 5120 x 8704 | 104.8 | 0.41% | 160x256:xs4 1.156, 160x256:plw16 1.123, 192x256:x8_24 1.063, 224x192:xs4 1.063 | 0.8922 |
| 1568 x 8192 x 5120 | 106.4 | 0.70% | 160x256:xs4 1.146, 160x256:plw16 1.125, 224x192:xs4 1.093, 224x192:plw16 1.076 | 0.8562 |
| 784 x 8192 x 5120 | 103.0 | 1.05% | 224x160:plw 1.040, 224x160:xs4 1.037, 224x160:plw16 1.012, 96x256:xs4 0.962 | 0.7155 |
| 1568 x 7168 x 5120 | 116.0 | 1.12% | 224x160:xs4 1.042, 224x160:plw 1.041, 224x160:plw16 1.025, 224x256:xs4 1.024 | 1.0243 |
| 784 x 7168 x 5120 | 96.4 | 0.96% | 128x128:xs4 1.063, 128x128:x8_24 1.054, 128x128:x8_42 1.050, 128x128:plw 1.030 | 0.6612 |
| 1568 x 5120 x 3072 | 108.1 | 2.70% | 160x256:xs4 1.050, 160x256:plw16 1.028, 128x128:xs4 1.025, 128x128:x8_24 1.007 | 0.7944 |
| 784 x 5120 x 3072 | 88.9 | 0.32% | 160x256:xs4 1.288, 160x256:plw16 1.260, 224x192:xs4 1.198, 192x256:x8_24 1.188 | 0.9994 |

- At 1568 rows a tile other than 224x256 leads at four of five shapes in this search: 224x224 at `gate_up_proj` (1.089), 160x256 at `in_proj_qkvz` (1.146) and at 5120 x 3072 (1.050, cell noise 2.7%), 224x160 at `qkv_proj` (1.042). `down_proj` (5120 x 8704) has no leading form: best 1.000.
- At 784 rows: 160x256 leads at `down_proj` (1.156), 224x160 at `in_proj_qkvz` (1.041), 128x128 at `qkv_proj` (1.063); 160x256 at 5120 x 3072 (1.288; this cell finished after the server jobs were submitted, so they leave it to vLLM); `gate_up_proj` has none (best 0.980).
- These are the best of 36 forms in one job, so they are biased upward by selection; the standalone check of plugin version 2 (`evidence/G1/`) re-times the chosen forms, and only a two-order race would land them.
- Expected gain at vLLM's defaults with these forms, 16k prompt [computed, same formula, `_d` profile, corrected ms]: gate_up 1489 x (1 - 1/1.089) = 121.7; qkvz 602 x (1 - 1/1.146) = 76.7; out/o 327 x (1 - 1/1.050) = 15.6; qkv 162 x (1 - 1/1.042) = 6.5; down left to vLLM: 220 ms of 5347 = **4.1% of time to first token**, before the selection bias.

## Decode GEMM: vLLM's own small-row kernel at 1 to 32 rows (job 23625791, one GCD, one job, 7 interleaved rounds) [observed]

`gm_skinny.py`. vLLM sends 1 to 5 rows to `wvSplitK` and everything larger to torch `F.linear` (hipBLASLt). Times in ms per call, bf16; `server` = what `rocm_unquantized_gemm_impl` runs at that row count, `hipBLASLt` = `F.linear` on the same inputs. Identical-code noise at most 2.4% (4 rows), under 0.7% elsewhere. `wvSplitK` called directly refuses 6 rows and more (`Unsupported N value`). Its error against fp32 equals hipBLASLt's (1.6e-3 to 1.7e-3).

| rows | 17408x5120 server / hipBLASLt | 5120x8704 server / hipBLASLt | 8192x5120 server / hipBLASLt | 7168x5120 server / hipBLASLt | 5120x3072 server / hipBLASLt |
|---|---|---|---|---|---|
| 1 | 0.154 / 0.189 | 0.070 / 0.102 | 0.066 / 0.094 | 0.057 / 0.086 | 0.028 / 0.044 |
| 2 | 0.223 / 0.189 | 0.075 / 0.102 | 0.100 / 0.094 | 0.086 / 0.085 | 0.029 / 0.044 |
| 4 | 0.162 / 0.190 | 0.094 / 0.104 | 0.075 / 0.094 | 0.072 / 0.086 | 0.053 / 0.044 |
| 5 | 1.568 / 0.191 (8.2x slower) | 0.325 / 0.103 (3.1x slower) | 0.726 / 0.094 (7.7x slower) | 0.638 / 0.086 (7.4x slower) | 0.092 / 0.045 (2.1x slower) |
| 6 | 0.191 / 0.191 | 0.104 / 0.103 | 0.094 / 0.094 | 0.086 / 0.086 | 0.068 / 0.044 (1.5x slower) |
| 8 | 0.191 / 0.190 | 0.105 / 0.105 | 0.095 / 0.095 | 0.086 / 0.086 | 0.067 / 0.045 (1.5x slower) |
| 16 | 0.193 / 0.193 | 0.107 / 0.107 | 0.096 / 0.096 | 0.087 / 0.087 | 0.068 / 0.045 (1.5x slower) |
| 32 | 0.195 / 0.195 | 0.112 / 0.111 | 0.096 / 0.096 | 0.087 / 0.087 | 0.067 / 0.046 (1.5x slower) |

Sum over the step's 256 large linear calls (64 gate_up, 64 down, 48 qkvz, 16 qkv, 64 out/o), ms per decode step [computed]: 1 rows: server 20.1, all-hipBLASLt 27.4; 2 rows: server 27.0, all-hipBLASLt 27.4; 4 rows: server 24.5, all-hipBLASLt 27.5; 5 rows: server 172.1, all-hipBLASLt 27.6; 6 rows: server 29.1, all-hipBLASLt 27.6; 8 rows: server 29.2, all-hipBLASLt 27.7; 32 rows: server 30.0, all-hipBLASLt 28.6.

- **At exactly 5 rows the server's kernel is 2.1 to 8.2 times slower than hipBLASLt** at every one of the five shapes (gate_up 1.568 against 0.191 ms). vLLM routes 5 rows to it (`m > 8 and 0 < n <= 5`). Per decode step that is 172 ms of GEMM against 28 ms [computed]. At 2 rows it is 18% slower at gate_up and 6% at qkvz, equal at qkv and faster at the other two shapes; at 1 row it is faster at all five (1.23 to 1.59x) and at 4 rows at four of five.
- At 6 to 32 rows the server's path is hipBLASLt and equals it within 0.3% at four shapes; at 5120 x 3072 it reads 0.067 against 0.045 ms (1.5x slower) at every row count from 6 to 32. Cause not traced [the dispatch has an `aiter.tuned_gemm` branch; hypothesis].
- hipBLASLt reads the weights at 0.80 to 0.94 TB/s at 8 to 32 rows on the four large shapes; `wvSplitK` at 1.16 to 1.30 TB/s at 1 row. A kernel that kept the 1-row rate up to 32 rows would save 64 x (0.190 - 0.153) + 64 x (0.105 - 0.070) + 48 x (0.095 - 0.066) + 16 x (0.086 - 0.057) = 6.5 ms of an 8-stream step [computed]. Our small-M forms do not do that today (0.99 to 1.01 of hipBLASLt).
- **In the served model the 5-row case is reached through the output head only [observed job 23625797; cause inferred].** Decode steps replay graphs captured at batch sizes 1, 2, 4, 8, 16, 24, 32, ... (`cudagraph_capture_sizes` in the server log), so a batch of 5 or 6 streams runs its 64 layers at 8 rows, on hipBLASLt. Sampling runs outside the graph, and its `lm_head` GEMM (124160 x 5120) sees the true stream count [inferred from the runner's source and the profile, where `lm_head` appears under sampling; not timed at 5 rows].

### Served decode step with vLLM's small-row kernels on and off (job 23625797, small-g, 2 GCDs, arm order s sk sk s, one job) [observed]

`s` = stock; `sk` = stock with `VLLM_ROCM_USE_SKINNY_GEMM=0` (vLLM's own switch: every linear through hipBLASLt). No plugin of ours in either arm. 8k contexts, 96 output tokens, default attention backend, decode step in ms, repetition 0 / repetition 1:

| streams | stock | small-row kernels off | stock / off |
|---|---|---|---|
| 1 | 57.92 / 58.58 | 69.20 / 69.65 | 0.839 |
| 2 | 67.62 / 68.74 | 69.83 / 69.46 | 0.979 |
| 4 | 63.98 / 63.31 | 69.02 / 69.49 | 0.919 |
| **5** | **79.56 / 79.89** | 69.33 / 70.22 | **1.143** |
| 6 | 70.09 / 69.38 | 69.28 / 69.85 | 1.002 |
| 8 | 69.71 / 70.06 | 69.95 / 69.67 | 1.001 |

- **Stock vLLM is 14% slower per decode step at exactly 5 concurrent streams than with its own small-row kernels switched off, and 10 ms slower than at 6 or 8 streams.** Spread between repetitions is at most 1.7% in every cell. One job; the two arms' repetitions are in mirrored order inside it.
- The same kernels are worth 16% at 1 stream and 8% at 4, so the switch is not a fix: the fix is routing by row count (5 rows, and whatever else is slower, to hipBLASLt). That is a change in vLLM's `rocm_unquantized_gemm_impl`, or a three-line general plugin; it needs none of our kernels. **Candidate for an upstream issue with this table and the standalone times above.** Not tested: 3 and 7 streams, short contexts, the `TRITON_ATTN` backend, gfx942 (MI300), where the kernel was presumably tuned.

## Tile sweeps at the one-step row counts where the 224x256 tile does not lead (jobs 23626594: 2352, 3136, 3920 rows; 23626595: 6272, 12544 rows; one GCD, one job each: a search) [observed]

Same script and rule as the 1568-row sweep. Ratio = vLLM's path / ours.

| M x N x K | cell noise | 224x256 xs4 | best three forms |
|---|---|---|---|
| 2352 x 17408 x 5120 | 0.21% | 1.1452 | 224x256:xs4 1.145, 224x224:xs4 1.136, 224x224:plw16 1.118 |
| 3136 x 17408 x 5120 | 0.46% | 1.0623 | 224x224:xs4 1.088, 224x224:plw16 1.075, 224x256:xs4 1.062 |
| 3920 x 17408 x 5120 | 0.24% | 1.0979 | 192x256:xs4 1.160, 192x256:x8_24 1.155, 224x224:xs4 1.146 |
| 2352 x 5120 x 8704 | 1.49% | 1.1931 | 224x256:xs4 1.193, 128x128:xs4 1.120, 128x128:x8_24 1.101 |
| 3136 x 5120 x 8704 | 1.03% | 0.9601 | 224x224:xs4 1.084, 224x224:plw16 1.064, 96x256:xs4 0.993 |
| 3920 x 5120 x 8704 | 0.45% | 0.9907 | 192x256:x8_24 1.136, 192x256:xs4 1.134, 192x256:x8_42 1.128 |
| 2352 x 8192 x 5120 | 0.86% | 0.9105 | 192x256:xs4 1.048, 192x256:x8_24 1.048, 192x256:x8_42 1.038 |
| 3136 x 8192 x 5120 | 0.54% | 0.978 | 192x256:xs4 1.139, 192x256:x8_24 1.133, 192x256:x8_42 1.124 |
| 3920 x 8192 x 5120 | 0.64% | 0.9739 | 96x256:xs4 0.996, 96x256:plw 0.995, 128x128:xs4 0.994 |
| 2352 x 7168 x 5120 | 1.08% | 1.1233 | 160x256:xs4 1.130, 224x256:xs4 1.123, 160x256:plw16 1.106 |
| 3136 x 7168 x 5120 | 0.57% | 1.0092 | 224x192:xs4 1.043, 224x160:plw 1.031, 224x160:xs4 1.030 |
| 3920 x 7168 x 5120 | 0.50% | 1.0804 | 128x128:xs4 1.089, 224x256:xs4 1.080, 128x128:x8_24 1.072 |
| 2352 x 5120 x 3072 | 1.59% | 1.2397 | 224x256:xs4 1.240, 128x128:xs4 1.165, 128x128:x8_24 1.148 |
| 3136 x 5120 x 3072 | 1.07% | 1.0034 | 224x224:xs4 1.137, 224x224:plw16 1.120, 96x256:plw 1.034 |
| 3920 x 5120 x 3072 | 1.40% | 1.038 | 192x256:x8_24 1.194, 192x256:xs4 1.181, 192x256:x8_42 1.180 |
| 6272 x 17408 x 5120 | 0.03% | 1.0947 | 224x224:xs4 1.120, 192x256:xs4 1.106, 192x256:x8_24 1.095 |
| 12544 x 17408 x 5120 | 0.09% | 1.1078 | 224x224:xs4 1.108, 224x256:xs4 1.108, 192x256:xs4 1.106 |
| 6272 x 5120 x 8704 | 0.43% | 0.9305 | 192x256:xs4 1.068, 192x256:x8_24 1.058, 192x256:x8_42 1.055 |
| 12544 x 5120 x 8704 | 0.27% | 0.9848 | 192x256:xs4 1.042, 192x256:x8_24 1.034, 192x256:x8_42 1.031 |
| 6272 x 8192 x 5120 | 0.99% | 1.0446 | 224x192:xs4 1.126, 192x256:x8_24 1.111, 224x192:plw16 1.106 |
| 12544 x 8192 x 5120 | 0.26% | 1.0827 | 224x224:xs4 1.113, 224x224:plw16 1.093, 160x256:xs4 1.090 |
| 6272 x 7168 x 5120 | 0.07% | 1.0435 | 128x128:xs4 1.104, 192x256:xs4 1.089, 192x256:x8_24 1.088 |
| 12544 x 7168 x 5120 | 0.22% | 1.0673 | 192x256:x8_24 1.101, 192x256:xs4 1.097, 192x256:x8_42 1.089 |
| 6272 x 5120 x 3072 | 1.43% | 0.978 | 192x256:x8_24 1.128, 192x256:xs4 1.118, 224x224:xs4 1.115 |
| 12544 x 5120 x 3072 | 0.48% | 0.996 | 192x256:x8_24 1.065, 192x256:x8_42 1.064, 192x256:xs4 1.056 |

25 cells. In 11 of them the 224x256 form reads below 1.02; in 10 of those another tile reads above 1.03 (range 1.042 to 1.139); the exceptions: 3920 x 8192 best 0.996.

## Follow-up (11 Nov)

- The per-(rows, shape) tile table was swept at all 21 row counts, raced in two orders and landed: `evidence/G2/RESULTS.md`.
- Prefill chunks with cached context: plugin-only form built and measured, query-window form designed: `evidence/P1/RESULTS.md`, addenda.
- The stock `wvSplitK` finding has a stand-alone reproduction and an issue draft: `evidence/UPSTREAM/vllm-rocm-wvsplitk-5-rows/` (3 rows is slow as well: 3.3x).
- New: the fp16 MFMA of gfx90a returns zero for subnormal fp16 operands (`evidence/P1/RESULTS.md`). The fp16 GEMM path rejected above for speed would also have dropped operand values below 6.1e-5.
