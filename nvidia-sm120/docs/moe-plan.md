# MoE plan, phase 0: nvidia/Qwen3.6-35B-A3B-NVFP4 on the RTX PRO 6000 (SM120)

27-09-2026. Research and design only: no GPU runs. Sources are the vLLM 0.29
install on the server (`<venv>/lib/python3.12/site-packages/vllm`; all `file:line` below are
relative to that `vllm/` directory unless a path says otherwise), the downloaded checkpoint
(`<data>/models/nvidia-Qwen3.6-35B-A3B-NVFP4`, complete, 23.4 GB in 3 shards), our plugin
(`horizon_kernels/`, identical to `<data>/horizon-pkgs/horizon_kernels/` by md5) and the TileLang fork at
upstream `7a5f446f` + our commits (`<src>/horizon-tilelang`, read only).

## 0. Summary

| Question | Answer |
|---|---|
| What does vLLM run for the experts today? | **Marlin-MoE (W4A16), and nothing else can be picked.** The checkpoint declares its experts `W4A16_NVFP4`, so vLLM asks for "NVFP4 weights, no activation quantization"; every FP4-tensor-core MoE backend (FlashInfer TRT-LLM / CuTe-DSL / CUTLASS, vLLM CUTLASS) requires NVFP4 activations and rejects itself. The FlashInfer "TMA descriptor 999" crash is therefore not in play for the experts. |
| Is Marlin-MoE NVFP4 supported on sm_120? | Yes: device check is "CUDA and cc >= 7.5", NVFP4 weights are in its supported list, and `_moe_C_stable_libtorch.abi3.so` ships sm_120 SASS containing the `marlin_moe_wna16` kernels (it uses `mma.sync`, no TMA). |
| New crash risks in this model | (1) The **FP8** attention/GDN layers pick `FlashInferFP8ScaledMMLinearKernel` first, whose `bmm_fp8(backend="auto")` resolves to FlashInfer's `cutlass_sm12x`, the same family that crashes on driver 570. (2) The **BF16 MTP experts** auto-select FlashInfer CUTLASS BF16 MoE (JIT build + same risk). Both have one-line workarounds. |
| Plug-in point for our own experts kernel | Wrap `ModelOptNvFp4FusedMoE.process_weights_after_loading` from the plugin (as we already wrap Marlin's linear kernel), prepare our own weight copy, and set `self.moe_kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), HorizonNvFp4Experts(...))`. No vLLM file changes are needed. |
| Our plugin on this model | Safe only with **K1a off** (it would read the NVFP4 `lm_head` as if it were BF16 and assert on Qwen3.8's 5120-wide trained head) and **K5 off** until validated (both K5 kernels *do* fire on this model, at shapes never tested: `in_proj_ba` 64x2048 and RMSNorm 2048). The NVFP4 linear hook is inert here (ModelOpt layers fall back to stock Marlin). Capture mode `thread_local` stays on. |
| Bytes per decode step | **2.25 GB at 1 token**, 3.85 GB at 4 tokens (1 user + 3 drafts), 13.3 GB at 32 tokens (8 users), ~19.8 GB once every expert is touched (M >= ~128). Qwen3.8-27B today: ~16.2 GB at any M. Routed experts are 25% of the bytes at M=1, 56% at M=4, 87% at M=32. |
| Kernel design headline | One grouped W4A16 NVFP4 kernel family on regfrag's fragment layout, with a device-side work list (CUDA-graph safe), the shared expert folded in as a 9th always-on expert, SiLU fused into the up-projection epilogue and the top-k weighting + ordered combine fused into the down-projection: **~3 kernels per MoE layer instead of ~13**, one weight copy (18.1 GB of experts). FP4 grouped prefill (two-pass, K3/rsN3 style) comes later: it only pays when experts see >= ~128 tokens each. |
| Plan | P1 stock bring-up + profile (1 day) → P2 plugin plumbing + ModelOpt dense layers on regfrag (2-3 days) → P3 grouped W4A16 decode kernel (1.5-2 weeks) → P4 fused routing + PDL (3-5 days) → P5 grouped FP4 prefill (1 week) → P6 draft head / FP8 dense decode. Go/no-go for P3 comes from P1's measured Marlin-MoE bandwidth. |

## 1. What is actually in the checkpoint

`hf_quant_config.json` / `config.json` (`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`, producer ModelOpt 0.44.0),
tensor headers read on the CPU:

| Part | Format on disk | Shape (per layer) | Count |
|---|---|---|---|
| Routed experts `mlp.experts.{0..255}.{gate,up,down}_proj` | **W4A16_NVFP4**: uint8 packed e2m1, fp8 e4m3 block scale per 16, fp32 `weight_scale_2` (= amax/2688, a *multiplier*), fp32 `input_scale` (present, unused by W4A16) | gate/up 512x2048, down 2048x512 | 40 layers x 256 |
| Shared expert `mlp.shared_expert.{gate,up,down}_proj` | W4A16_NVFP4 | same as one expert | 40 |
| Router `mlp.gate`, `mlp.shared_expert_gate` | BF16 (not quantized) | 256x2048, 1x2048 | 40 |
| GDN `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` | **FP8** e4m3, per-tensor static weight + input scale | qkv 8192x2048, z 4096x2048, out 2048x4096 | 30 |
| GDN `in_proj_a`, `in_proj_b` | BF16 | 32x2048 each (vLLM fuses them to `in_proj_ba`, 64x2048) | 30 |
| Full attention `self_attn.{q,k,v,o}_proj` | FP8 per-tensor | q 8192 (with output gate), k/v 512, o 2048x4096 | 10 |
| `lm_head` | W4A16_NVFP4 | 248320x2048 | 1 |
| MTP (`mtp.*`) | **BF16, excluded** (`exclude_modules: ["mtp.layers.0*", "mtp*"]`), including **256 BF16 experts** as fused 3-D tensors (`gate_up_proj` 256x1024x2048, `down_proj` 256x2048x512) | one full-attention MoE layer | 1 |
| Vision tower, embeddings | BF16 | | |
| KV cache | `kv_cache_quant_algo: FP8` | 10 attention layers x 2 KV heads x 256 | |

- **gate/up `weight_scale_2` are tied in all 10,240 experts** (checked: 0 mismatches), so vLLM's use of the gate's
  global scale for both halves (`quantization/modelopt.py:1571`) is exact for this checkpoint.
- Every NVFP4 shape fits our tiling rules as-is: w13 N=1024 / K=2048, w2 N=2048 / K=512, lm_head N=248320 / K=2048
  (N % 128 == 0 and K % 256 == 0, the plugin's condition for K3/regfrag).
- Architecture facts that matter below: hidden 2048, 40 layers (30 GDN + 10 full attention, every 4th), 256 experts,
  top-8 softmax with renormalisation (`norm_topk_prob` default True, `models/qwen3_next.py:224`), expert intermediate 512,
  one shared expert (512) scaled by `sigmoid(shared_expert_gate(x))` (`models/qwen2_moe.py:109-118`), SiLU-and-mul.

## 2. Which kernels vLLM 0.29 picks on SM120, and how to force them

### 2.1 Quantization config resolution

1. `ModelOptMixedPrecisionConfig.override_quantization_method` turns `MIXED_PRECISION` into `modelopt_mixed`
   automatically (`quantization/modelopt.py:2225-2231`); no `--quantization` flag is needed.
2. Per layer, `get_quant_method` (`quantization/modelopt.py:2408-2462`) resolves the prefix in `quantized_layers`
   (`_resolve_quant_algo`, `:2312-2398`; the `.experts` prefix is resolved at `:2349-2362`):
   - `FusedMoE` experts with `W4A16_NVFP4` → `ModelOptNvFp4FusedMoE(w4a16_nvfp4_config)` (`:2440`, `:2451-2455`).
   - Linear layers with `W4A16_NVFP4` (shared expert, lm_head) → `ModelOptNvFp4W4A16LinearMethod` (`:2433-2434`).
   - Linear layers with `FP8` (GDN, attention) → `ModelOptFp8LinearMethod` (`:2427-2428`).
   - Anything not listed (router gate, `in_proj_ba`) → `UnquantizedLinearMethod` (`:2438`).
   - `mtp*` is excluded → `UnquantizedLinearMethod` for linears (`:2419-2421`), `None` → unquantized MoE for the MTP
     experts.

### 2.2 Routed experts: Marlin-MoE, by elimination

- `ModelOptNvFp4FusedMoE.__init__` sets `use_a16 = quant_method == "W4A16_NVFP4"` and calls
  `select_nvfp4_moe_backend(weight_key=kNvfp4Static, activation_key=None)` (`quantization/modelopt.py:1425-1430`).
- The auto order is TRT-LLM, CuTe-DSL, CuTe-DSL batched, FlashInfer CUTLASS, vLLM CUTLASS, **Marlin**, Humming,
  emulation (`layers/fused_moe/oracle/nvfp4.py:196-205`); the first class whose `is_supported_config` passes wins
  (`:293-305`). Every backend before Marlin requires `(kNvfp4Static, kNvfp4Dynamic)`:
  `experts/trtllm_nvfp4_moe.py:229-231`, `experts/flashinfer_cutedsl_moe.py:93-95`,
  `experts/flashinfer_cutedsl_batched_moe.py:83-85`, `experts/flashinfer_cutlass_moe.py:173-179` (NVFP4 needs
  cc >= 100 *and* NVFP4 activations), `experts/cutlass_moe.py:724`. With `activation_key=None` they all fail.
- Marlin passes: `_supports_current_device` = CUDA and cc >= 7.5 (`experts/marlin_moe.py:605-607`),
  `_supports_quant_scheme` checks only the weight key and lists `kNvfp4Static` (`:614-633`).
- **sm_120 binaries exist:** `_moe_C_stable_libtorch.abi3.so` has 26 sm_120 cubins; the ones with `marlin_moe_wna16`
  symbols hold 240-1,200 kernel instances each (checked with `cuobjdump -xelf`). Marlin runs on `mma.sync` BF16 with
  in-register dequant, no TMA, so the driver-570 limitation does not apply (dense Marlin NVFP4 ran our production
  until 26-09-2026).
- Weight conversion: `convert_to_nvfp4_moe_kernel_format` → `prepare_nvfp4_moe_layer_for_marlin`
  (`oracle/nvfp4.py:448-467`, `quantization/utils/marlin_utils_fp4.py:342-400`), which repacks every expert into
  Marlin's layout and logs "Your GPU does not have native support for FP4 computation" (cosmetic, `:354`).
  Activation scales are dropped (`a13_scale = a2_scale = None`).
- Quant config: `nvfp4_w4a16_moe_quant_config(g1_alphas=w13_scale_2, g2_alphas=w2_scale_2, w1_scale, w2_scale)`
  (`oracle/nvfp4.py:539-550`).
- Kernel object: `make_nvfp4_moe_kernel` builds `FusedMoEKernel(prepare_finalize, MarlinExperts)`
  (`oracle/nvfp4.py:595-642`); on one GPU the prepare/finalize is `MoEPrepareAndFinalizeNoDPEPModular`
  (`prepare_finalize/no_dp_ep.py:40-94`).
- What one Marlin-MoE call does (`experts/marlin_moe.py:234-397`): pick `block_size_m` from {8, 16, 32, 48, 64} by
  `M*topk/E` (`:333-338`; decode always lands on 8), `moe_align_block_size` (sort token-expert pairs into per-expert
  blocks, `:340-346`), `moe_wna16_marlin_gemm` for w13 (`:135`), SiLU-and-mul (`:179`), `moe_wna16_marlin_gemm` for
  w2 with the top-k weights multiplied in (`:202-220`), then `moe_sum` over the 8 slots (`:393-397`).
- `VLLM_DISABLED_KERNELS` **does not affect MoE selection at all**: it is only read in
  `model_executor/kernels/linear/__init__.py:585,815,858,900,953,1080`.

### 2.3 Shared expert and lm_head: dense Marlin (forced)

`ModelOptNvFp4W4A16LinearMethod.__init__` calls `init_nvfp4_linear_kernel(use_a16=True)`
(`quantization/modelopt.py:1289`), which forces `MarlinNvFp4LinearKernel` when `--linear-backend` is `auto`
(`kernels/linear/__init__.py:1053-1055`). `process_weights_after_loading` collapses `weight_scale_2` with `max()` and
passes it on un-reciprocated as `weight_global_scale` (`quantization/modelopt.py:1379-1395`).

### 2.4 FP8 attention/GDN: a crash risk the Qwen3.8 stack never had

- The FP8 linear candidates on CUDA are, in order, `FlashInferFP8ScaledMMLinearKernel`, `CutlassFP8ScaledMMLinearKernel`,
  `B12xTensorFP8…`, torch, Marlin, Humming (`kernels/linear/__init__.py:402-410`).
- FlashInfer's is accepted for cc >= 100, so also for SM120 (`kernels/linear/scaled_mm/flashinfer.py:41-53`), and
  calls `bmm_fp8(..., "auto")` (`utils/flashinfer.py:1008-1015`). In FlashInfer 0.6.18 the auto heuristic puts
  **`cutlass_sm12x` first** on SM120 (`flashinfer/gemm/gemm_base.py:7337-7400`): FlashInfer's CUTLASS path, the family
  that fails on driver 570 with "TMA descriptor 999". (vLLM's own CUTLASS kernels and TileLang's TMA loads work on
  this driver, so the failure is FlashInfer-specific.)
- **Fix:** add `FlashInferFP8ScaledMMLinearKernel` to `VLLM_DISABLED_KERNELS` → vLLM's own
  `CutlassFP8ScaledMMLinearKernel`. Can be A/B'd back later if FlashInfer's FP8 path turns out to work.

### 2.5 MTP drafter: BF16 MoE

- The MTP layer is excluded from quantization, so its 256 experts use the unquantized oracle. On CUDA the order is
  FlashInfer TRT-LLM, **FlashInfer CUTLASS**, Triton, batched Triton (`fused_moe/oracle/unquantized.py:68-72`).
  TRT-LLM BF16 needs cc family 100 (`experts/trtllm_bf16_moe.py:88-95`); FlashInfer CUTLASS accepts cc family 120 and
  `(None, None)` BF16 (`experts/flashinfer_cutlass_moe.py:129-138`, `:155-163`) and would be chosen: a FlashInfer JIT
  build on first start (RAM) and the same crash family.
- **Fix:** `"moe_backend": "triton"` inside `--speculative-config` (`config/speculative.py:406-410`: the drafter's own
  MoE backend; without it the drafter inherits `--moe-backend`, and `marlin` is not valid for unquantized MoE,
  `oracle/unquantized.py:166-180`). NVIDIA's own model card uses exactly this for the MTP.
- The MTP `lm_head` is built with the model's quant config (`models/qwen3_5_mtp.py:245-250`), so it resolves to the
  NVFP4 `lm_head` → dense Marlin. The draft step reads it in full: 286 MB of a 415 MB draft step (section 6).

### 2.6 Knobs

| Knob | Where | Effect here |
|---|---|---|
| `--moe-backend marlin` | `config/kernel.py:122-143,235`; `engine/arg_utils.py:1682,2545` | Pins the experts to Marlin (same as auto today, but fails loudly instead of silently changing). `humming` (`humming_kernels 0.1.12` is installed; `experts/fused_humming_moe.py:310` lists `(kNvfp4Static, None)`, device cc >= 7.5 `:385-391`) is the only other stock W4A16 NVFP4 MoE path: use it as the second baseline. |
| `--speculative-config '{…,"moe_backend":"triton"}'` | `config/speculative.py:406` | Drafter MoE on Triton instead of FlashInfer CUTLASS. |
| `VLLM_DISABLED_KERNELS=…,FlashInferFP8ScaledMMLinearKernel` | `kernels/linear/__init__.py:585` | FP8 dense layers on vLLM CUTLASS. Keep the four NVFP4 entries from production for the (shared expert / lm_head) NVFP4 linears. |
| `--linear-backend` | `config/kernel.py:185-207,269` | Leave `auto` (it forces Marlin for W4A16). |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` | `envs.py:1998`; `runner/shared_experts.py:69-75` | Only if CUDA-graph capture misbehaves: by default the shared expert runs on an aux stream for <= 256 tokens (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`, `envs.py:2011`; `shared_experts.py:121-127`). |
| `--language-model-only` | `engine/arg_utils.py:1368`; `config/multimodal.py:101-103,495-496` | Text only; should skip the vision tower (the model wraps it in `_mark_tower_model`). |

**If a W4A4 NVFP4 MoE checkpoint comes later** (llm-compressor, other vendors), auto selection *would* pick FlashInfer
CUTLASS on SM120 (`flashinfer_cutlass_moe.py:173-179`: `has_device_capability(100)` is true for 120) and hit the
driver-570 crash. Then pass `--moe-backend marlin` (W4A16 execution, activation scales dropped) or try
`--moe-backend cutlass` (vLLM's own `CutlassExpertsFp4`, cc family 120 allowed, `experts/cutlass_moe.py:707-712`;
untested here).

## 3. vLLM's modular MoE interface, and where our plugin plugs in

### 3.1 Call chain for one MoE layer

1. `Qwen3NextSparseMoeBlock.forward` → `self.experts(hidden_states, router_logits=hidden_states)`
   (`models/qwen3_next.py:128-270`; the gate, shared expert and experts are created at `:170-231`).
2. The runner dispatches through the opaque custom op `torch.ops.vllm.moe_forward[_shared]`
   (`fused_moe/runner/moe_runner.py:116-213`). It is **not** a CUDA-graph splitting op
   (`config/compilation.py:765-782`), so the whole MoE layer is captured inside the piecewise/full graphs.
3. `MoERunner._forward_impl` (`moe_runner.py:859-935`): starts the shared expert on the aux stream if allowed, applies
   the router gate (`:897-902`, BF16 256x2048 via `UnquantizedLinearMethod`), then `_apply_quant_method`
   (`:592-645`): top-k selection with `fused_topk` (softmax, renormalise; `topk_weights` fp32 and `topk_ids` int32,
   `router/fused_topk_router.py:80-110`), then `RoutedExperts.forward_modular` → `quant_method.apply`
   (`fused_moe/routed_experts.py:1224-1259`).
4. `ModelOptNvFp4FusedMoE.apply` → `self.moe_kernel.apply(x, layer.w13_weight, layer.w2_weight, topk_weights,
   topk_ids, activation, global_num_experts, expert_map, apply_router_weight_on_input, shared_experts, …)`
   (`quantization/modelopt.py:1662-1686`).
5. `FusedMoEKernelModularImpl.apply` (`fused_moe/modular_kernel.py:1437-1531`): `prepare` (activation quantization;
   none for W4A16), workspace allocation from the experts' `workspace_shapes` through vLLM's workspace manager
   (`:1120-1181`), `experts.apply(...)`, `finalize` (top-k weighting/reduction per the experts'
   `finalize_weight_and_reduce_impl`, `prepare_finalize/no_dp_ep.py:80-94`).

### 3.2 The contract an experts class must honour (`FusedMoEExpertsModular`, `modular_kernel.py:777-976`)

- **Selection metadata** (`modular_kernel.py:551-690`): `_supports_current_device`, `_supports_no_act_and_mul`,
  `_supports_quant_scheme(weight_key, activation_key)`, `_supports_activation`, `_supports_parallel_config`,
  `activation_format()` (`Standard` for us), optional `_supports_shape`, `_supports_batch_invariance`,
  `supports_lora`.
- **`moe_problem_size(a1, w1, w2, topk_ids)`** → `(E, M, N, K, topk)` (`:787-828`). Override it (Marlin does,
  `experts/marlin_moe.py:666-690`) when `w1/w2` are not in `(E, N, K)` layout, which will be the case for our prepared
  copy.
- **`workspace_shapes(M, N, K, topk, …)`** → `(workspace13, workspace2, output)` shapes (`:837-876`); the output must
  be exactly `(M, K)` for `Standard` format.
- **`apply(output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
  a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input)`**
  (`:922-976`): write the MoE output into `output` (M, K) in the activation dtype.
  - `hidden_states`: (M, 2048) BF16, unquantized (W4A16: `quant_config.quant_dtype` is None, so
    `_quantize_input` is a no-op, `no_dp_ep.py:14-37`).
  - `topk_ids` (M, 8) int32, `topk_weights` (M, 8) fp32, already renormalised.
  - `expert_map`: None on one GPU (only set with expert parallelism); assert it or honour it (-1 = not local).
  - `activation`: `MoEActivation.SILU` with `is_act_and_mul` (w13 is `[gate; up]`, N = 2 x 512).
  - `apply_router_weight_on_input`: False for this model (only valid for top-1 anyway, `no_dp_ep.py:67-74`).
- **`finalize_weight_and_reduce_impl()`**: return `TopKWeightAndReduceNoOP()` if our kernel already applies the top-k
  weights and sums the 8 slots (as Marlin does, `marlin_moe.py:697-698`); `TopKWeightAndReduceContiguous` makes vLLM do
  it with an extra kernel.
- **Shared expert:** by default vLLM runs it itself (aux stream or before the experts) and adds it after
  (`moe_runner.py:583-645`, `runner/shared_experts.py`). With `NoDPEP` prepare/finalize `supports_async()` is False,
  so the experts class never sees it. Folding it into our kernel therefore means also taking it away from the runner
  (see 3.4).
- **CUDA graphs:** the op is captured. Everything inside `apply` must be graph-safe: no host reads of device data
  (no `.item()`, no Python-side group sizes), fixed grid per captured M, no first-call JIT or module load during
  capture (pre-warm every capture size, as the NVFP4 linear plugin already does), deterministic and replayable
  (self-resetting counters, as regfrag's split-K tickets).
- **Monolithic alternative** (`FusedMoEExpertsMonolithic`, `modular_kernel.py:978-1101`): `apply(hidden_states, w1,
  w2, router_logits, …)`. The runner then skips `fused_topk` and hands us the router logits, which is the hook for
  fusing routing into our kernels (`ModelOptNvFp4FusedMoE.apply_monolithic`, `quantization/modelopt.py:1638-1660`,
  already delegates; `is_monolithic` follows the kernel, `fused_moe_method_base.py:137-143`). Pair it with
  `MoEPrepareAndFinalizeNoDPEPMonolithic` (`no_dp_ep.py:97-135`).

### 3.3 Where the plugin plugs in (no vLLM file edits)

There is no public registry for MoE experts classes (`backend_to_kernel_cls` is a hard-coded `if` chain,
`oracle/nvfp4.py:72-152`). The clean seam is the same one our NVFP4 linear plugin uses: wrap the quant method's
`process_weights_after_loading`.

```python
# horizon_kernels/moe.py, installed from vllm_plugin.register() when HORIZON_MOE=1 (sketch)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular

orig_pw = ModelOptNvFp4FusedMoE.process_weights_after_loading

def pw(self, layer):
    if not (self.use_a16 and horizon_moe_supported(layer)):      # shapes, E, topk, SILU, no EP, no LoRA
        return orig_pw(self, layer)                                # stock Marlin-MoE
    # layer.w13_weight (E, 2*I, H/2) uint8, w13_weight_scale (E, 2*I, H/16) fp8, w13_weight_scale_2 (E, 2) fp32,
    # w2_* likewise: still the checkpoint layout here (Marlin's repack happens inside orig_pw)
    layer.horizon_moe = prepare_grouped(layer)                     # our regfrag-layout copy, per-column alpha
    free_checkpoint_tensors(layer)                                  # keep ONE copy (no Marlin repack at all)
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    self.moe_kernel = mk.FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(),
                                        HorizonNvFp4Experts(self.moe, self.moe_quant_config, layer.horizon_moe))
    warm_all_capture_sizes(layer)                                   # outside capture, every M vLLM will record

ModelOptNvFp4FusedMoE.process_weights_after_loading = pw
```

- `apply()` and `apply_monolithic()` keep working unchanged because they only call `self.moe_kernel`.
- `free_checkpoint_tensors` must leave `w13_weight`/`w2_weight` as (small) parameters because `apply` passes them;
  our `moe_problem_size` override takes the real sizes from `layer.horizon_moe`.
- The same wrapper can later cover `CompressedTensorsW4A4Nvfp4MoEMethod` (llm-compressor checkpoints; global scale is
  a divisor there, as for our dense layers).
- Folding the shared expert into the grouped kernel additionally needs the runner not to run it: set the runner's
  `_shared_experts` to None for that layer and have our kernel add `sigmoid(shared_expert_gate(x)) * shared(x)`. That
  is a runner-level patch, so it comes after the plain experts class works (phase 4).
- Scale convention for ModelOpt: dequantised weight = e2m1 x e4m3 block scale x `weight_scale_2` (multiplier). For our
  kernels `alpha = weight_scale_2` per output column (per expert; per shard for w13, exact even if gate/up differed).
  Compressed-tensors stores the reciprocal; the prepare function must branch on the format.

## 4. Our plugin on a Qwen3_5Moe model: what fires, risks, switches

| Feature (switch) | Hook | Fires on this model? | Risk | First bring-up |
|---|---|---|---|---|
| Adaptive NVFP4 linear, regfrag_xl, drop-Marlin, native prefill (`HORIZON_NVFP4_ADAPTIVE`, `HORIZON_REGFRAG`, `HORIZON_DROP_MARLIN`, `HORIZON_NATIVE_PREFILL`) | `CompressedTensorsW4A4Fp4.process_weights_after_loading` + `MarlinNvFp4LinearKernel.process_weights_after_loading/apply_weights` (`vllm_plugin.py:182-301`) | The Marlin hook is reached by the ModelOpt shared expert and `lm_head` (and the MTP `lm_head`), but `horizon_shard_gs` is only set by the compressed-tensors scheme hook, so `marlin_pw` takes the "unexpected NVFP4 layer format" branch → **stock Marlin** (`vllm_plugin.py:207-212`). No layer is registered, no calibration runs. FP8 layers and the experts never touch it. | None functionally; no speed-up either. With a *compressed-tensors* MoE checkpoint, dense NVFP4 layers **would** register with unknown shapes and `HORIZON_AUTO_CALIBRATE=1` would calibrate on the GPU at load and write into the production table dir. | `HORIZON_NVFP4_ADAPTIVE=0` (or keep 1: inert). Always point `TILELANG_ADAPTIVE_DIR` at a scratch copy for MoE tests. |
| K1a fused FP8 draft head (`HORIZON_FP8_DRAFT_HEAD`, `HORIZON_DRAFT_VOCAB`, `HORIZON_DRAFT_HEAD_WEIGHTS`) | vLLM patch in `Qwen3_5MTP.get_top_tokens` (`patches/vllm029-qwen3_5_mtp-horizon-draft-head.patch`), **inherited by `Qwen3_5MoeMTP`** (`models/qwen3_5_mtp.py:368`) | Yes, if the env is set | **Breaks.** It takes `self.lm_head.weight` as a BF16 matrix, but here it is the Marlin-repacked NVFP4 `uint8` tensor; and the trained 48k head is (48k, 5120) vs hidden 2048 → assert on the first draft call (or the "must be built before CUDA graph capture" error). | `HORIZON_FP8_DRAFT_HEAD=0`, **unset** `HORIZON_DRAFT_VOCAB` and `HORIZON_DRAFT_HEAD_WEIGHTS`. Leave out `use_local_argmax_reduction` at first (the stock path would work with the NVFP4 head, `models/interfaces.py:1562-1571`, but it is one variable less). |
| K5a skinny BF16 GEMM (`HORIZON_K5`, `HORIZON_K5_SKINNY`) | `UnquantizedLinearMethod`, prefix ends with `in_proj_ba` (`k5.py:63-83`) | **Yes**: Qwen3.5-MoE's GDN has `in_proj_ba` (`layers/mamba/gdn/qwen_gdn_linear_attn.py:433-438`), BF16 unquantized, N=64, K=2048; `skinny_bf16.supported(64, 2048)` is True | Never tested at this shape (tuned and verified for 96x5120). Generic code, so probably right, but unverified. | `HORIZON_K5=0`; validate `test_skinny.py` at (64, 2048) in P2, then extend the match to `mlp.gate` (N=256) and `mlp.shared_expert_gate` (N=1). |
| K5b add+RMSNorm (`HORIZON_K5_NORM`) | replaces `GemmaRMSNorm.forward_native/forward_cuda` globally (`k5.py:155-178`) | **Yes**: every decoder and MTP norm (hidden 2048; `add_rmsnorm.supported(2048)` is True, 128-thread config instead of production's 640). q/k norms (256) fall back. | Untested config on every layer of the model. | Off with `HORIZON_K5=0`; validate `test_norm.py` at H=2048 in P2. |
| Capture mode (`HORIZON_CAPTURE_MODE=thread_local`) | `torch.cuda.graphs.CUDAGraph.capture_begin` (`vllm_plugin.py:160-181`) | Yes | None; model-agnostic and still needed (gpushare breaks "global" captures). | **Keep on.** It needs the plugin to be importable: `PYTHONPATH=…:<data>/horizon-pkgs`. |
| Compile-cache key (`register()`, `vllm_plugin.py:143-158`, `:304-312`) | `vllm.envs.compile_factors` | Yes | None. Still clear the torch compile cache on model switches, or use a separate `VLLM_CACHE_ROOT`. | Separate `VLLM_CACHE_ROOT`. |

## 6. Bytes per decode step (memory-bandwidth arithmetic)

Per-parameter bytes: NVFP4 0.5625 (4-bit code + one e4m3 scale per 16), FP8 1, BF16 2. One expert (gate + up + down,
3 x 2048 x 512 = 3.15 M params) = **1.77 MB**; all 10,240 experts = **18.1 GB**. Distinct experts per layer for M
tokens assume independent uniform routing, 256 x (1 - (1 - 8/256)^M); real routing is skewed and the draft tokens of one
sequence route alike, so real numbers will be somewhat lower.

Non-routed bytes per forward (any M): GDN FP8 30 x 33.8 MB = 1.01 GB, attention FP8 10 x 27.3 MB = 0.27 GB, router
BF16 40 x 1.05 MB = 0.04 GB, shared experts 40 x 1.77 MB = 0.07 GB, lm_head 0.29 GB → **1.69 GB**.

| M (tokens in the step) | Distinct experts / layer | Tokens per active expert | Routed experts | Total | Routed share | Roofline at 1.8 TB/s |
|---|---|---|---|---|---|---|
| 1 | 8.0 | 1.00 | 0.57 GB | 2.25 GB | 25% | 1.25 ms |
| 4 (1 user, 3 drafts) | 30.5 | 1.05 | 2.16 GB | 3.85 GB | 56% | 2.14 ms |
| 8 | 57.4 | 1.11 | 4.06 GB | 5.75 GB | 71% | 3.19 ms |
| 16 | 102.0 | 1.26 | 7.22 GB | 8.90 GB | 81% | 4.95 ms |
| 32 (8 users x 4) | 163.3 | 1.57 | 11.56 GB | 13.25 GB | 87% | 7.36 ms |
| 64 | 222.4 | 2.30 | 15.74 GB | 17.43 GB | 90% | 9.68 ms |
| 128 (32 users x 4) | 251.6 | 4.07 | 17.81 GB | 19.49 GB | 91% | 10.83 ms |
| 256 | 255.9 | 8.00 | 18.11 GB | 19.80 GB | 91% | 11.00 ms |

- "3B active" = about **2.25 GB per token** at M=1, of which only a quarter is experts; the FP8 GDN/attention weights
  are the biggest single block (1.29 GB). At M=1 the FP8 dense path matters as much as the experts.
- **The regime that matters is "many experts, 1-4 tokens each"**: with speculative decoding every user contributes 4
  tokens, so 8 users already touch ~64% of all experts per layer. It is a grouped-GEMV problem, not a grouped-GEMM one.
- Draft step (MTP, BF16): attention 54.5 MB + fc 16.8 MB + 8 BF16 experts 50 MB + shared/router 7 MB + NVFP4 lm_head
  286 MB = **415 MB, 69% of it the lm_head**; 3 drafts = 1.25 GB. A 48k FP8 draft head (K1a style, rows taken from the
  dequantised NVFP4 lm_head, same 248,320-token vocabulary family) brings a draft step to ~227 MB (-45%).
- Versus Qwen3.8-27B (dense, ~13.7 GB main + 2.5 GB BF16 lm_head per verification step at any M): the MoE reads ~7x
  fewer bytes at 1 user, ~4x at 4 tokens, only ~1.2x fewer at 8 users x 4 tokens, and *more* from ~64 tokens on,
  when nearly every expert is read (and every expert then has few tokens). High concurrency on the MoE lives or dies
  by how close the expert kernel streams 18 GB to the 1.8 TB/s roofline.
- Capacity side: KV is 10 attention layers x 2 KV heads x 256 x 2 x 1 byte =
  **10 KB per token** (262k context = 2.7 GB per full-length sequence); GDN state 30 layers x 32 heads x 128 x 128 x
  fp32 = 63 MB per sequence.

## 7. Kernel design: horizon grouped NVFP4 MoE for SM120

### 7.1 What upstream TileLang has (base `7a5f446f`) and what is missing

| Upstream | Where | Usable for us? |
|---|---|---|
| FP16 grouped GEMM fwd/bwd/pointer variant | `examples/grouped_gemm/example_grouped_gemm_fwd.py:39-114` | Pattern only: the grid and group loop come from a **Python list of group sizes** (`total_m_blocks` computed on the host, `batch_sizes_list`), so it cannot be CUDA-graph captured with data-dependent routing. BF16 weights. |
| Fused MoE example (shared + routed, BF16) | `examples/fusedmoe/example_fusedmoe_tilelang.py:11-190` | Same host-side `group_sum` / `group_count` (`:117-120`); unquantized; SiLU fused into the up GEMM epilogue (the idea we reuse). |
| SM100 MXFP8 block-scaled grouped GEMM | `examples/blockscaled_gemm_sm100/grouped_gemm_mxfp8_blockscaled_1d1d.py` | Device-side `offsets` + static `max_M_per_E` bound (the graph-safe pattern), but tcgen05 / 2-CTA: SM100 only. |
| SM120 block-scaled MMA (`T.gemm_blockscaled`, `T.mma_gemm_blockscaled`, register-A #3257) | `tilelang/language/gemm_op.py:270`, `tilelang/cuda/language/gemm_op.py:84,398` | Yes: the FP4 prefill building block (K3 / rsN3 already use it). |
| PDL, persistent scheduler, grid sync | `tilelang/cuda/language/pdl.py:10-17`, `tilelang/language/tile_schedule.py:93` (`PersistentTileScheduler`), `tilelang/language/builtin.py:1181` (`T.sync_grid`) | Yes: overlap between the MoE kernels (P4) and a possible single-launch MoE. |
| **Missing** | | Grouped **NVFP4** anything on SM120; any W4A16 grouped kernel; a device-side (graph-safe) group scheduler; routing/top-k/permute kernels. All of it is ours to build (upstream policy of 27-09-2026: build in `contrib/`). |

Ours to build on: `tilelang.contrib.sm120_nvfp4.decode_regfrag` / `decode_regfrag_xl` (W4A16 NVFP4, weights
dequantised straight into `mma.sync` B fragments at ~2.1 instructions per weight, deterministic split-K tickets,
graph-safe, tile heights in 16-row steps, a run-time-sized "range" kernel), `prefill_rs` (two-pass FP4 on regfrag's own
layout, `pu=2`: +0.4-1.4% vs K3 layer-weighted, bit-identical), and `horizon_fused.skinny_bf16` (K5a).

### 7.2 (a) Grouped W4A16 decode kernel (the main piece)

**Weights (one copy, prepared at load):**
- Experts laid out as regfrag 32(n) x 32(k) blocks of 512 bytes + 64 bytes of re-encoded scales, experts
  concatenated: `Wg[E', N/32, K/32, 512 B]`, `Sg[E', N/32, K/32, 64 B]`, `alpha[E', N]` fp32 (= ModelOpt
  `weight_scale_2`, per output column). E' = 257: the **shared expert is expert 256**.
- w13 rows interleaved in 8-column groups `[g0..g7, u0..u7, g8..g15, u8..u15, …]`: in the m16n8k16 C fragment a lane
  then holds gate columns (2t, 2t+1) in n8 tile 0 and the matching up columns in n8 tile 1, so `silu(g) * u` is computed
  in registers in the epilogue and the up kernel writes the 512-wide intermediate directly.
- Memory: 18.1 GB experts + 0.07 GB shared, i.e. the checkpoint's size. Marlin's repack is never built (no transient
  2x at load either, because the hook runs before `orig_pw`).

**Work list (device-side, graph-safe):** from `topk_ids` (plus slot 8 = expert 256 for every token), count tokens per
expert, exclusive scan, scatter `(token, slot)` pairs into per-expert segments, and emit work items `(expert, row
offset, rows)` in blocks of BM rows (BM = 16 by default: at decode an expert has 1-4 tokens). The item count stays on
the device; the grid is the static worst case for the captured M: `min(9M, 257) + ceil(9M / BM)` items (M=4: 39 items,
~32 real; M=32: 275 items). Surplus CTAs exit at once. The alternative is a persistent grid of 188 x c CTAs looping over
the device-side count (regfrag_xl's range kernel with the item count read on the device), which removes wave
quantisation. Race both.

**Up kernel** (grid `items x N13/BN`, N13 = 1024): per CTA, cp.async-gather the BM token rows `x[token_idx]` into the
swizzled A tile (padding rows zero-filled, as the XL range kernel already does), stream the expert's weight blocks +
scales through the regfrag pipeline, dequantise into B fragments, `mma.sync`, epilogue `alpha`, SiLU-and-mul → BF16
intermediate `[pairs, 512]` (9 KB per token, L2-resident).

**Down kernel** (grid `items x 2048/BN`, K = 512: 8 k-tiles of 64, no split-K): same main loop on the intermediate;
epilogue multiplies by the slot weight (`topk_weights`, or `sigmoid(shared_gate)` for slot 8) and writes fp32
partials `[M, 9, 2048]`; a per-token ticket makes the last CTA sum the 9 slots **in fixed slot order** (deterministic,
no fp32 atomics) and write BF16 `output`. `finalize_weight_and_reduce_impl` = NoOP. (First version: write BF16
`[M, 9, 2048]` and reuse vLLM's `moe_sum`, then fuse.)

**Why this should beat Marlin-MoE at decode:**
- Marlin-MoE runs `block_size_m = 8` blocks padded to its 16-row tile, one expert per block, and a separate
  `moe_align_block_size`, SiLU and `moe_sum` kernel around its two GEMMs; the shared expert is 3 more dense Marlin /
  elementwise kernels plus the gate GEMM and the sigmoid-multiply-add. That is ~13 kernels per MoE layer, ~520 per
  forward. At M=4 one MoE layer's roofline is only ~57 MB / 1.8 TB/s = 32 µs, so a few µs of launch gap and tail per
  kernel is the same order as the useful work.
- The down projection has K = 512: pipelines are short, so per-expert latency (not bandwidth) dominates; grouping all
  (expert, n-block) tiles of all active experts in one grid gives 188 SMs enough independent loads in flight.
- regfrag's dequant is already ~2.1 instructions per weight and beat Marlin by 3-4% on the dense 27B shapes at the
  same M values (decode 4 and 32 tokens).

### 7.3 (c) Fused routing and combine

- **Route kernel** (monolithic experts class, router logits in): the router GEMM itself (BF16 257 x 2048 including the
  shared-expert gate row; 1 MB) stays a K5a-style skinny GEMM (it needs many CTAs to stream 1 MB); its last CTA (ticket)
  runs softmax, top-8, renormalise, sigmoid of the shared gate, the per-expert histogram, the scan and the work list.
  For M <= 256 (every decode and verification step) the post-processing fits in one CTA. This replaces `fused_topk` +
  `moe_align_block_size` (+ its sort) + the shared-gate GEMM.
- **PDL** (`T.pdl_trigger` / `T.pdl_sync`) between route → up → down → next layer's norm lets each kernel's prologue
  (weight prefetch does not depend on the previous kernel) overlap the previous kernel's tail.
- **Target: 3 kernels per MoE layer** (route, up, down+combine) instead of ~13; optionally the residual add + next
  RMSNorm fused into the combine (K5b's kernel), and later a single persistent launch with a grid barrier between up
  and down (`T.sync_grid`, the K4 megakernel direction).
- Worth it: yes at decode (launch count is comparable to the useful time at M <= 32); marginal at prefill.

### 7.4 (b) Grouped FP4 prefill (two-pass residual)

- Tokens per expert in prefill ≈ chunk x 8 / 256 = chunk / 32 (popular experts more). The dense calibration table puts
  the FP4 two-pass crossover at 128-512 rows per GEMM, so grouped FP4 only helps for chunks of ~4k-16k tokens. Below
  that, the grouped W4A16 kernel with block_M up to 128 (regfrag_xl's tile heights) is the prefill path.
- Design: per work item, `prefill_rs`'s two-pass kernel on **regfrag's layout** (`pu=2`, producer-side untwiddle,
  3 activation stages), with the activation quantiser fused over the gathered rows (one per-row scale per token, as
  `fp4_quant` does), same epilogues (SiLU for up, slot weight for down). Adaptive dispatch by *tokens per expert*
  (from M, known on the host per shape) between W4A16-grouped and FP4-grouped, calibrated like `tilelang.adaptive`.
- **One weight copy:** a native-order copy for prefill would add another 18.1 GB, while `pu=2` costs only +0.4-1.4%
  prefill GEMM time on the dense shapes. For the MoE, one copy (regfrag layout) is the right trade.
- Quality: FP4 two-pass has ~1% GEMM error (K3); it needs the same gate as K3 (long-prompt log-probs + paired eval).
  The W4A16 path is exact-class (same maths as Marlin up to summation order).

### 7.5 Expected gain vs Marlin-MoE (assumptions stated; P1 replaces them with measurements)

Assumptions: stock Marlin-MoE streams expert weights at 45% of 1.8 TB/s at M <= 32 (plausible for 1-2 tokens per
512-wide expert; dense Marlin reached ~65% on the 27B's large shapes), our grouped kernel at 70% (regfrag_xl level);
stock ~13 kernels per MoE layer vs 3-4 for ours, 1.5-3 µs of gap/tail each.

| M | Expert + shared bytes | Stock at 45% | Horizon at 70% | Saved (GEMM) | Saved (launch, 40 layers x ~9 kernels) |
|---|---|---|---|---|---|
| 4 (1 user) | 2.23 GB | 2.75 ms | 1.77 ms | ~1.0 ms | ~0.5-1.1 ms |
| 32 (8 users) | 11.63 GB | 14.4 ms | 9.2 ms | ~5.1 ms | ~0.5-1.1 ms |
| 128 (32 users) | 17.88 GB | 22.1 ms | 14.2 ms | ~7.9 ms | ~0.5-1.1 ms |

- End-to-end that would be roughly **+25-35% decode tok/s at 1 user and +30-40% at 8 users** on top of stock, *if*
  Marlin-MoE really sits near 45%. If P1 measures Marlin-MoE at >= 65% on these shapes, the GEMM gain shrinks to
  ~5% and the fusion (launch count, PDL) is the only lever left: then P3 is a smaller job (fusion around stock-level
  GEMMs) and priority moves to the FP8 dense path (1.29 GB per step) and the draft head.

## 8. Phased plan

| Phase | Work | Effort | Exit criterion |
|---|---|---|---|
| **P0** | This document | done | |
| **P1** stock bring-up + profile | Stage A smoke + quality vs NVIDIA's card; stage B speed arms (3 drafts, 0 drafts, Humming); step profile; Marlin-MoE microbench (achieved GB/s per M) | 1 day, ~3 h GPU | Model serves with CUDA graphs + MTP; quality within noise of the card; measured split of step time; **go/no-go for P3** (go if routed-expert time at M=4/32 is >= 1.4x its 70%-roofline time, or MoE launch gaps >= 15% of the step) |
| **P2** plumbing | `horizon_kernels/moe.py`: the `process_weights_after_loading` wrapper, `HorizonNvFp4Experts` with a *reference* implementation (per-expert `decode_regfrag_xl` calls, graph-safe via fixed per-M loops) to prove the interface, capture, determinism, one weight copy; ModelOpt support in the dense NVFP4 hook (alpha = `weight_scale_2`, no max collapse) so the shared expert and lm_head run regfrag_xl; K5 validated at (64, 2048) / H=2048 and extended to the router gate | 2-3 days | Bit-identical to a PyTorch reference; vLLM starts clean 4/4 with capture; log-probs vs stock within noise |
| **P3** grouped W4A16 decode (a) | Route-lite (histogram/scan/work list), up kernel with SiLU epilogue, down kernel with ordered combine, shared expert as expert 256; worst-case-grid vs persistent variants; race vs Marlin-MoE at M 1-256 with rotated weights; then in-vLLM A/B | 1.5-2 weeks | >= 1.3x Marlin-MoE on expert time at M = 4 and 32, bit-exact vs the P2 reference, deterministic, quality gate passed |
| **P4** fused routing + PDL (c) | Monolithic experts class; router skinny GEMM + top-k + work list in one launch chain; PDL across route/up/down; shared expert taken off the runner; combine fused with residual add + norm | 3-5 days | 3-4 kernels per MoE layer; measured step-time gain |
| **P5** grouped FP4 prefill (b) | `prefill_rs pu=2` over work items + gathered two-pass quantiser; adaptive switch by tokens per expert | 1 week | TTFT gain at 4k-64k prompts; K3-level quality gate |
| **P6** the rest of the step | 48k FP8 draft head for this model (K1a with rows from the NVFP4 lm_head; draft step 415 → ~227 MB); FP8 W8A16/W8A8 skinny decode kernel for the GDN/attention projections (1.29 GB per step; K1a's `draft_head_w8a16` is the starting point); Triton config (or our kernel) for the BF16 MTP experts; later MTP retrain for this model | 1-2 weeks | per-item A/B |

## 10. Upstream angle

- **TileLang:** upstream has no graph-safe grouped GEMM and nothing grouped for SM120 / NVFP4. Generic pieces worth
  offering (per the 27-09-2026 policy: building blocks upstream, tuned kernels stay Horizon IP in `contrib/`):
  (1) an example "grouped GEMM with device-side group offsets" (static worst-case grid, early exit or a persistent
  loop over a device-side count) that works under CUDA graphs, on SM80-SM120; (2) an SM120 block-scaled (NVFP4)
  grouped GEMM example on `T.gemm_blockscaled`; (3) any lowering fixes found on the way (as #3286). Check open upstream
  PRs first.
- **vLLM:** two small, generic bug reports / PRs are justified by this study: FP8 `bmm_fp8(backend="auto")` choosing
  FlashInfer CUTLASS on SM12x without a fallback, and the MTP BF16 MoE auto-selecting FlashInfer CUTLASS on SM12x.
  A TileLang-based SM120 NVFP4 MoE backend for vLLM would be a larger conversation and would expose the kernel; not before P3 has numbers.

## 11. P1 stage B results (27-09-2026 12:20–12:50 CEST): speed, quality, profile, Marlin-MoE bandwidth

The MoE model was the only GPU process (CUDA graphs on, KV 28 GiB,
max-num-seqs 32, stock kernels: Marlin-MoE for the experts, dense Marlin for the shared expert and lm_head, vLLM CUTLASS
for the FP8 layers, horizon plugin only for `thread_local` capture). Scripts: `scripts/kernels/moe/`, results:
`results/moe_p1b_27-09-2026/` (not in this release).

### 11.2 Marlin-MoE bandwidth (microbenchmark, vLLM stopped)

`moe_marlin_bench.py`: vLLM's own `fused_marlin_moe` on this model's shapes (E=256, top-8, hidden 2048, expert width
512), weights repacked by vLLM's own `prepare_nvfp4_moe_layer_for_marlin`, 8 expert sets used round-robin (453 MB each,
so every call streams from HBM), uniform random routing, CUDA-graph replay. Bytes = distinct experts actually touched x
1.769 MB (codes + scales). "GEMMs" = the two Marlin-MoE kernels only (from the profiler); "whole op" adds align, SiLU
and the top-k sum.

| M (tokens) | Distinct experts | MB read | GEMMs µs | GEMMs GB/s | **% of 1.8 TB/s** | Whole op µs | Whole op % |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 14.2 | 15.8 | 896 | **50%** | 26.7 | 29% |
| 2 | 16 | 28.3 | 29.3 | 968 | 54% | 40.9 | 38% |
| 4 | 32 | 56.6 | 44.9 | 1,261 | **70%** | 56.2 | 56% |
| 8 | 59 | 104.4 | 77.2 | 1,352 | 75% | 89.6 | 65% |
| 16 | 95 | 168.1 | 116.6 | 1,442 | 80% | 129.0 | 72% |
| 32 | 160 | 283.1 | 189.4 | 1,495 | **83%** | 202.4 | 78% |
| 64 | 220 | 389.3 | 260.7 | 1,493 | 83% | 270.5 | 80% |
| 128 | 252 | 445.9 | 305.4 | 1,460 | 81% | 310.1 | 80% |
| 256 | 256 | 453.0 | 333.8 | 1,357 | 75% | 328.6 | 77% |

**The 45% assumption of section 7.5 is wrong for M >= 4.** Stock Marlin-MoE streams the experts at 70% of the
1.8 TB/s peak at M=4 (1 user + 3 drafts) and 80–83% from M=16 on, which is about as fast as any kernel gets on this
GPU (dense Marlin reached ~65%, regfrag_xl ~80%). Only M=1–2 is poor (50–54%, and 36% inside the real model, see
below). The non-GEMM part of the op (align, SiLU, sum: ~11 µs per layer) is a bigger lever at small M than the GEMMs.

### 11.3 Where a decode step goes (torch profiler, no MTP, `moe_step_breakdown.py`)

Per decode step, summed over all kernels (40 layers; steps counted from the Marlin-MoE launches; profiler overhead
~10%, so the steps are 5.0 / 8.5 / 11.4 ms instead of the unprofiled ~4.5 / 7.4 / ~11 ms):

| Category | 1 user | 8 users | 32 users |
|---|---|---|---|
| Routed experts: Marlin-MoE GEMMs (80 launches) | 883 µs (17.6%) | 2,192 µs (25.9%) | 3,012 µs (26.4%) |
| **Shared expert: dense Marlin (80 launches, 8.1 µs each for 1.8 MB)** | **643 µs (12.9%)** | 678 µs (8.0%) | 872 µs (7.6%) |
| lm_head: dense Marlin NVFP4 (286 MB, 1.53 TB/s) | 187 µs (3.7%) | 189 µs (2.2%) | 198 µs (1.7%) |
| **MoE routing glue** (top-k, align, sort, SiLU, top-k sum, shared-gate sigmoid and multiply; ~355–426 launches) | **667 µs (13.3%)** | 918 µs (10.9%) | 1,035 µs (9.1%) |
| **BF16 GEMVs** (router gate 256x2048, shared-expert gate, GDN in_proj_ba; cuBLAS gemv/dot + wmma) | **539 µs (10.8%)** | 1,225 µs (14.5%) | 1,036 µs (9.1%) |
| FP8 dense GEMMs (GDN + attention projections, CUTLASS sm120; 1.29 GB at ~1.1 TB/s = 62%) | 1,158 µs (23.1%) | 1,186 µs (14.0%) | 1,306 µs (11.4%) |
| Norms + FP8 activation quant (Inductor-fused) | 282 µs (5.6%) | 340 µs (4.0%) | 405 µs (3.5%) |
| GDN core (recurrent, conv1d, gated norm) | 287 µs (5.7%) | 613 µs (7.2%) | 2,040 µs (17.9%) |
| Full attention (10 layers) + KV write + RoPE | 341 µs (6.8%) | 186 µs (2.2%) | 183 µs (1.6%) |
| Other (copies, casts, sampling) | 171 µs (3.4%) | 418 µs (4.9%) | 718 µs (6.3%) |
| Idle gaps (GPU waiting, incl. profiler overhead) | 824 µs (16.5%) | 2,120 µs (25.1%) | 2,630 µs (23.0%) |
| Kernels per step | 1,079 | 1,177 | 1,212 |

- In the real model at 1 user the Marlin-MoE GEMMs take 22 µs per layer for 8 experts (14.2 MB): **36% of peak**,
  worse than the microbenchmark (50%). At 8 and 32 users real routing touches fewer experts than uniform routing
  (popular experts), so the in-model time per layer (55 µs and 75 µs) sits between the microbenchmark's M=4 and M=8
  rows: the kernel runs at the microbenchmark's ~70–80% there.
- **The routed-expert GEMMs are only 18–26% of the step.** The MoE *overhead around them* is larger: the shared
  expert on dense Marlin (8 µs per launch for 1.8 MB = 110 GB/s, 13% of the 1-user step), the routing glue (13%) and
  the tiny BF16 router/gate GEMVs (11–15%). Together ~30–37% of the step, all launch- and latency-bound.

### 11.4 Go/no-go for P3, with the expected gain recomputed

Section 8's criterion: go if routed-expert time at M=4/32 is >= 1.4x its 70%-roofline time, **or** MoE launch gaps
>= 15% of the step.
- **GEMM part: no-go.** Measured Marlin-MoE GEMM time vs the time at 70% of peak: M=4 44.9 vs 44.9 µs (1.00x), M=32
  189.4 vs 224.7 µs (0.84x, i.e. already faster than the target). A new grouped W4A16 GEMM can gain at most ~5–15% of
  the expert GEMM time at M >= 4 (to a practical ceiling of ~85%), i.e. **~1–4% of a decode step**. Only the M=1–2
  regime (no spec decoding, 1–2 users; 36–54%) has real headroom (~+9% of the step at 1 user without MTP).
- **Overhead part: go.** The non-GEMM MoE work (shared expert + routing glue + router/gate GEMVs) is ~1.8 ms of a
  4.5 ms step at 1 user and ~2.8 ms of a 7.4 ms step at 8 users.

Recomputed expected gains (estimates from the profile, not measured):

| Item | Saves per step (1 user / 8 users) | Tok/s effect |
|---|---|---|
| Shared expert folded into the routed-expert call as a 257th expert (first on stock Marlin-MoE: append its weights, top-k 9 with weight sigmoid(gate); later in our kernel) | ~0.55 / ~0.6 ms | +12% / +8% |
| Router gate + shared gate as one skinny BF16 GEMM (K5a extended to N=257, K=2048; plus in_proj_ba at 64x2048) | ~0.35 / ~0.9 ms | +8% / +12% |
| Fused routing (top-k + align + sort in one kernel, SiLU and the weighted top-k sum in the GEMM epilogues, PDL) | ~0.45 / ~0.6 ms | +10% / +8% |
| New grouped W4A16 GEMM (the original P3 core) | ~0.1–0.4 / ~0.1–0.3 ms | +2–9% / +1–4% |
| FP8 W8A16 decode kernel for the GDN/attention projections (62% → ~80% of peak) | ~0.25 / ~0.25 ms | +5% / +3% |

**Recommendation: GO for P3, re-scoped to fusion first.** Order: (1) shared expert as a 257th expert inside the
existing Marlin-MoE call (small, no new kernel, biggest single item), (2) skinny router/gate GEMV (extends K5a),
(3) fused routing + epilogues + PDL (the old P4), and only then (4) a grouped GEMM, justified by M=1–2 only.
Expected together: roughly **+25–35% at 1 user and +20–30% at 8 users** on top of stock (the old plan's +25–40%
figure survives, but the gain comes from fusion, not from GEMM bandwidth). The MTP arm will add the M=4 in-model
numbers; it does not change the ordering.

## 12. Build phase, steps 0–3 (27-09-2026, from 15:40 CEST)

### 12.2 Step 0b: MTP on the MoE model (V2 runner)

The MoE MTP test (`run_moe_spec.sh`, 15:44–16:45) settled it:
- **V2 runner, no `--moe-backend`:** the NVFP4 experts still pick Marlin, but the drafter's BF16 experts auto-select
  `FlashInferExperts` (FlashInfer CUTLASS BF16 MoE); vLLM did not come up within 12 minutes (JIT build / hang, the
  driver-570 family predicted in §2.5).
- **V1 runner + `--moe-backend marlin` + drafter `moe_backend: triton`:** works. `bench_load` 1 / 8 / 32 users =
  **255.6 / 1,153.8 / 2,622.6 tok/s** (no MTP: 208 / 832 / 2,104), acceptance length prose 2.42, maths 3.67, code
  3.53, `bench_speed` 284.6 / 1,525.6 on the second run. The first `bench_speed` after the start was 113.5 / 568.9:
  the drafter's Triton MoE kernels JIT-compile while serving (a warm-up item for step 6).
- **Our fix, `HORIZON_MOE_SPEC_BACKEND=1`:** wraps V2's `load_eagle_model` (and the names the mtp / eagle /
  multi-module speculators imported) so the drafter is built with `kernel_config.moe_backend =
  speculative_config.moe_backend`, exactly what V1's `_create_draft_vllm_config` does. The runner stays V2 (as in
  production). Upstream bug report pending.

### 12.3 Step 1: the shared expert as routed expert #256 (`HORIZON_MOE_FSE=1`)

vLLM 0.29 already has the machinery for this, written for ROCm AITER and switched off on CUDA. The plugin adds only
what is missing on CUDA:
1. **Enable** `resolve_layer_fused_shared_expert` for a block whose experts and shared expert are both ModelOpt
   `W4A16_NVFP4` (the MTP block, excluded from quantisation, stays as it is). The loader then maps
   `mlp.shared_expert.*` to `mlp.experts.256.*`, the layer allocates 257 experts, and the runner receives the
   shared-expert gate.
2. **Loader rename without vLLM's split:** `maybe_fuse_shared_experts` slices every tensor per shared expert, which
   fails on ModelOpt's 0-d `weight_scale_2` / `input_scale`; with one shared expert the split is the identity, so the
   plugin only renames.
3. **Router:** vLLM's CUDA router would take the top-8 of all 257 logits. Ours: `topk_softmax` over the 256 routed
   logits (the stock op, same numerics), then slot 9 = (256, sigmoid(shared logit)) with the sigmoid rounded to bf16
   as `torch.sigmoid` on the bf16 gate output does.
4. **One gate GEMM:** the router gate and the shared-expert gate are concatenated into the router gate's own weight
   (257 rows + 7 zero rows = 264, so rows stay 16-byte aligned and skinny's N % 4 holds). The runner's own fused-gate
   path (a separate `F.linear`) is switched off for these layers.
5. **Marlin-MoE counts expert #256:** vLLM passes `global_num_experts = 256`, and `moe_align_block_size` would drop
   slot 9 as invalid; the plugin calls the kernel with `-1` (E taken from the weights, 257).

Checked on the CPU against the checkpoint: gate/up `weight_scale_2` are identical in 40 of 40 shared experts, so
vLLM's single w13 scale is exact for the folded expert. Numerics: the shared product is weighted in fp32 inside
Marlin's down-projection epilogue and summed with the routed slots by `moe_sum`, instead of a bf16 multiply and add
after it: exact-class, not bit-identical to stock.

### 12.4 Step 2: the fused gate on K5a skinny (`HORIZON_MOE_GATE=1`)

The 264-row gate runs on `horizon_fused.skinny_bf16` for M <= `HORIZON_MOE_GATE_MAX_M` (default skinny's 48), cuBLAS
above. Fixed-order fp32 accumulation, one bf16 rounding: same rounding class as cuBLAS, different summation order.

### 12.5 Step 3: the fused route kernel (`HORIZON_MOE_ROUTE=1`, `tilelang.contrib.horizon_moe.route`)

- `route_shared` (any M): one warp per row does softmax, top-8, renormalisation and the sigmoid slot, and writes
  weights (M, 9) fp32 and ids (M, 9) int32. It replaces 6 launches (contiguous copy, `topkGating`, sigmoid, fill,
  two concatenations). It mirrors vLLM's `topkGating<VPT=8, 256 experts>` step by step (lane-owned 8-expert chunks,
  butterfly max / sum / argmax with lower-id tie-break, `expf` precise via an extern call because TVM lowers `T.exp`
  to `__expf`, `p = e * (1/sum)`, renormalisation `w * (1/sum_selected)`), so it is meant to be bit-identical.
- `route_align` (M <= 128, one CTA of 1,024 threads): the same routing plus Marlin-MoE's block alignment
  (per-expert histogram in shared memory, warp-shuffle scan of the padded counts, `expert_ids`, sentinel fill, pair
  scatter), in vLLM's exact `moe_align_block_size` format. The router stashes it; the plugin's wrapper of
  `marlin_moe.moe_align_block_size` returns it when Marlin asks for the alignment of the same ids tensor (otherwise
  vLLM's own op runs). Saves the `moe_align_block_size` + `count_and_sort` launches as well (8 launches -> 1).
- ptxas (sm_120a): `route_shared` 56 registers, `route_align` 55 registers, no spills.

### 12.6 vLLM bug report draft: the V2 model runner ignores `speculative_config.moe_backend`

Not filed yet.
> **Title:** [Bug][V2 model runner] `speculative_config.moe_backend` is ignored for MTP/EAGLE drafters.
> **Summary:** In 0.29.0, `v1/spec_decode/llm_base_proposer.py:_create_draft_vllm_config` applies
> `speculative_config.moe_backend` to the drafter's `kernel_config`, but the V2 runner's loader
> (`v1/worker/gpu/spec_decode/eagle/utils.py:load_eagle_model`, used by the mtp / eagle / multi-module speculators) only
> overrides the KV-cache dtype. The drafter is therefore built with the target's `--moe-backend`.
> **Repro:** Qwen3.5/3.6-MoE NVFP4 (ModelOpt, W4A16 experts), `--moe-backend marlin`,
> `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"moe_backend":"triton"}'` →
> `ValueError: moe_backend='marlin' is not supported for unquantized MoE` (the MTP experts are BF16). Without
> `--moe-backend`, the drafter auto-selects FlashInfer CUTLASS BF16 MoE, which does not come up on an SM120 box with
> driver 570.
> **Fix:** in `load_eagle_model`, `replace(vllm_config, kernel_config=replace(vllm_config.kernel_config,
> moe_backend=speculative_config.moe_backend))` when it is set (as V1 does). A one-line PR.

### 12.7 Op-level gate results (27-09-2026 16:21–16:48; details in our run notes)

- **Routing:** the route kernel is bitwise identical to stock in 60 of 60 cases.
- **Accuracy vs the fp32 reference:** all fused arms are in the same class as stock (rel. L2 0.0043–0.0049 vs
  0.0045–0.0047).
- **Speed:** MoE block time falls 55.3 → 32.1 µs at M=1, 90.7 → 61.6 µs at M=4 and 216 → 188 µs at M=32, i.e.
  0.9–1.4 ms saved per 40-layer decode step.
- **Open items:**
  - The route+align scan bug is fixed.
  - The align test is fixed (it compared per block; it must compare per expert segment).
  - Arms whose gate runs on cuBLAS are not run-to-run bit-identical from M=32. The candidate configuration therefore
    runs the gate on skinny up to M=256 (`HORIZON_MOE_GATE_MAX_M=256`), and a probe confirms or refutes cuBLAS as the
    cause.

### 12.8 Step 4: our grouped W4A16 NVFP4 MoE GEMM (written, compiled, race pending)

- **Module:** `tilelang_contrib/horizon_moe/grouped_w4a16.py`, fork `d4b5dc75`.
- **Work list:** built on the device from the route kernel's alignment with 16-row blocks. The grid is the worst case
  for the captured M; blocks at or past `ntp` exit at once, so it is graph-safe.
- **Per block:** decode_regfrag's main loop (fragment-order weights, ~2.1 instructions per weight dequant, `mma.sync`).
  - Up projection: w13 rows interleaved `[g0..7, u0..7, g8..15, u8..15, …]`, so each lane holds gate and up of the
    same column. The epilogue writes `silu(g) * u` with vLLM's rounding (bf16 g/u, bf16 silu, bf16 product).
  - Down projection: the epilogue writes `bf16(acc · alpha_e · topk_weight)`; vLLM's `moe_sum` does the fixed-order
    slot sum.
  - Split-K: fp32 partials plus a ticket, and the last CTA sums the splits in fixed order (the SiLU is applied after
    the sum).
- **Build:** 54–70 registers, no spills, all 25 race variants precompiled CPU-only. The body is a K4 `T.macro`.
- **Plugin:** `HORIZON_MOE_GROUPED=1` (default off) runs it for M ≤ 128. Our expert copy is built before Marlin's
  repack; Marlin stays for larger M (prefill), which costs ~18 GB more while both copies exist.
- **Race:** `moe_grouped_race.py` checks accuracy vs fp32 and vs Marlin, determinism and capture, and sweeps configs
  for M 1–128. Kill criterion (master plan §5.3): ≥ 1.1× Marlin-MoE at M = 1–4, otherwise keep Marlin and stop at
  step 3.

### 12.9 Step 6: draft head for the MoE model

The Qwen3.6 and Qwen3.8 vocabularies are identical (248,077 tokens, same map), so production's 48k draft vocabulary
applies unchanged. `make_moe_draft_head.py` builds the 49,152 × 2048 BF16 head from the MoE's own NVFP4 lm_head (exact
dequantisation; every row is finite and non-zero) at `<work>/moe-k/draft_head_moe_lmhead.safetensors`.
The draft step then reads ~100 MB (FP8) instead of the 286 MB NVFP4 lm_head. It needs the V2 runner (K1a's
`use_local_argmax_reduction` path), hence step 0b first.
