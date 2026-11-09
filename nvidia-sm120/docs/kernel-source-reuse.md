# Kernel source reuse and reference inventory

Updated 28-09-2026 01:17 CEST.

Direction: use relevant open-source code and locally available comparable kernels/addons as implementation references and reusable components. Likely names are Marlin and DFlash; DFlash is a speculative-decoding method requiring a compatible draft model, not a drop-in matrix kernel.

Verified server inventory (read-only): FlashInfer 0.6.18 with bundled CUDA source `flashinfer/data/include/flashinfer/attention/prefill.cuh`; vLLM 0.29.0 includes Marlin adapters for NVFP4/MXFP4/FP8 and Qwen DFlash/DFlash2 model implementations; TileLang 0.1.12; Triton 3.7.1; CUTLASS DSL 4.6.2; CUTLASS source under TileLang; FLA 0.5.2 metadata in the calibration environment. A Qwen3.8-27B-DFlash2 cache directory exists, which does not establish checkpoint completeness or runtime compatibility.

The current prefill benchmark calls FlashInfer BatchPrefillWithPagedKVCacheWrapper with automatic backend selection, HND layout, BF16 queries and FP8 KV. The bundled prefill header is available to inspect, but the precise dispatched backend must be recorded before attributing timings to a particular CUDA kernel.

Workflow: inspect installed implementations first; record upstream revision, license and notices before copying; compare matching hardware, layout, precision and shape; reuse the fastest verified implementation through capability-based dispatch; specialize only for measured gaps. Preserve upstream attribution. Correctness, determinism, graph replay, long-context quality and end-to-end speed gates still apply. Neither fallback support nor published speedups prove optimized support for every model.

Priority: investigate the roughly 2x attention-prefill gap against FlashInfer; repair the decode benchmark's unsupported page size baseline; study Marlin quantization/layout and reduction strategies; compare DFlash against existing MTP only with a compatible verified draft and identical workloads. No live configuration changed by this source review.

| Reference | Intended use | Evidence/status |
|---|---|---|
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Paged attention, prefill/decode, quantized operators | Installed source confirmed; benchmark baseline |
| [Marlin](https://github.com/IST-DASLab/marlin) and vLLM Marlin | Quantized GEMM, packing/scales, reduction | Local benchmark scripts and installed adapters confirmed; original Marlin and vLLM extensions differ |
| [DFlash](https://github.com/z-lab/dflash) | Parallel speculative drafting | Installed vLLM model code and cached draft directory confirmed; new evaluation pending |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Attention tiling and pipelining | Upstream repository inspected; hardware-specific implementation eligibility must be checked |
| CUTLASS / CuTe, Triton, TileLang | Matrix primitives, layouts, scheduling, code generation | Installed packages/source confirmed |
| Flash Linear Attention | GDN and recurrent-state references | Existing baseline and installed metadata confirmed |
| DeepGEMM / DeepEP, TensorRT-LLM, SGLang, llama.cpp | Further GEMM/MoE, communication, serving/cache and quantization references | Candidate sources for subsequent inventory; not yet audited in this pass |
