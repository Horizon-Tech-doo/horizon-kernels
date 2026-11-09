# Model portability requirements

28-09-2026: this is an explicit requirement for horizon_kernels. Qwen 27B is the current development
and performance target, not the boundary of the library. Preserve this scope
when reporting completion of the overall kernel goal.

Reusable operation kernels must dispatch from tensor shapes, layouts, dtypes,
quantization metadata and hardware capabilities, not model names. Keep model
adapters responsible for architecture semantics, state management and integration.
A model-specific optimization may remain available as a specialization, but it
does not satisfy the general implementation requirement.

## Requirements

This applies to
the current WP1.3 state/cache work as well as subsequent kernel development.
Keep the 27B capacity goal and quality gates; portability is an additional
requirement, not a replacement task.

- Shared kernels select by operation semantics, tensor geometry, strides,
  precision, quantization layout and hardware. Derive dimensions from tensors
  and model configuration; keep optimized shape specializations explicit.
- Architecture adapters own model-specific hooks, cache/state lifetimes,
  positional conventions and speculative-decoding semantics. A model without
  GDN has no GDN state to optimize. GDN and other state-space recurrences must
  not be treated as interchangeable merely because tensor shapes agree.
- Validate capability before changing persistent-state dtype or allocation.
  A missing fp16 backend cannot safely fall back to a kernel that expects
  fp32/bf16. Unsupported state precision must be rejected or remain at the
  native precision before allocation, with a clear support decision.
- Apply the cache-equivalence fix to a documented recurrence/cache-boundary
  contract. Separate reusable rounding/state operations from Qwen/vLLM hooks;
  expose head dimensions, head counts, checkpoint alignment and storage dtype.
  Do not spread the current 48-head/128-dimension/64-alignment assumptions to
  unrelated model families.
- Report support per operation and architecture: optimized and verified,
  compatible through native fallback, experimental, or unsupported. Loading
  successfully through fallback is not evidence of optimized coverage.
- Validation must include another architecture and different geometries for
  shared operations. Cover standard dense attention, alternate normalization
  conventions, MoE, MLA, recurrent operations and multimodal components as
  their implementations/adapters become available. Each support claim requires
  actual dispatch, correctness, quality, cache and performance evidence.

## Required work

| Area | Required portable behavior | Current evidence / remaining gap |
|---|---|---|
| Linear and output projections | Parameterized M/N/K, group/block scales, quantized layouts and dtype variants | Adaptive linear dispatch exists; calibration identity and benchmark coverage still contain Qwen-specific assumptions |
| Normalization and activations | Hidden dimensions, epsilon, weight conventions, gated activations and supported precision explicit | Separate RMSNorm and Gemma-style conventions; do not assume adding one to weights is universal |
| Attention and positions | Correct implementations/adapters for MHA, GQA, MQA, MLA, supported sparse/windowed attention and positional conventions | Qwen attention hooks do not establish DeepSeek MLA support |
| Recurrent/state-space operations | Architecture-specific state-update semantics with parameterized geometry and state precision | Experimental commit6822462a parameterizes key/value heads and derived packed/state layouts; CPU checks and four compile controls pass, GPU gate23 pending. Head dimension128 and FP32 recurrent state remain required |
| MoE | Variable expert count, top-k, shared experts, routing weights, grouped GEMM, deterministic reduction and expert-parallel layouts | Existing MoE work has unresolved determinism gates; not portable-performance evidence |
| Persistent execution | Shared task/runtime infrastructure; model-derived operation graphs, dimensions and schedules | Phase-2 geometry is parameterized in experimental commit91f3ef55; seven offline compilations passed and GPU gate21 is pending. GDN heads are parameterized experimentally; model adapters remain specialized and full generic persistent execution is open |
| Quantization and calibration | Format/layout-specific implementations; shape-based calibration keyed by hardware/runtime and numeric semantics | NVFP4, FP8 and BF16 paths need explicit coverage; new formats require implementation and gates |
| Speculative decoding | Model-native MTP where present; separate draft-model adapters otherwise | Qwen MTP integration must not be assumed valid for other architectures |
| Vision/OCR and embeddings | Reusable compatible primitives plus dedicated encoder/image/patch/convolution adapters as required | Accepted platform scope; specific optimized model coverage unverified |
| Distributed execution | Tensor, pipeline, expert and context parallel integration according to architecture | Single-GPU measurements do not prove multi-GPU support or scaling |

## Evidence required per model

Record exact checkpoint revision, architecture, tensor shapes, quantization/layout,
state/cache precision, adapter/backend versions and GPU/runtime. Distinguish:

- Optimized and verified: the intended kernels execute, pass correctness/state/
  determinism/graph gates and improve measured end-to-end performance.
- Compatible through fallback: the model runs, but some operations use the backend.
- Experimental: implementation exists but required evidence is incomplete.
- Unsupported: required implementation or adapter is missing.

Fallback is necessary for continuity but does not count as optimized coverage or
completion of “works with any models”. Unknown architectures need capability
inspection and, when necessary, new operation implementations/adapters. Do not
promise that an arbitrary future architecture is already supported.

Test each new implementation against the relevant backend/reference on real model
weights and representative decode/prefill shapes, mixed lengths, padding, cache
reuse, long context and concurrency. Include state updates where applicable.
Measure kernel selection and fallback frequency, not only whether a model loads.
Repeat end-to-end quality and speed checks; do not extrapolate Qwen gains to other
models. Shared operation tests should include multiple geometries and real layers
from different architecture families as checkpoints become available.

