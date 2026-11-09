> **Status:** Posted on tile-ai/tilelang#1592 (27-09-2026 04:36); prefill multiple, #3290 decode ranges and the 111.4 baseline corrected on GitHub 27-09-2026 14:22. This file is the local draft; the live text on GitHub is authoritative.

At Horizon Tech I've been working on SM120 NVFP4 for LLM inference on an RTX PRO 6000 Blackwell (96 GB, sm_120) and would like to contribute the parts that complement the new block-scaled GEMM support (#3237, #3257).

For serving, the new `T.gemm_blockscaled` path covers W4A4. Two gaps remain in practice: the small-M decode path, where most tokens are generated, and prefill on real checkpoints. Here is what I have, all written in TileLang against current `main`:

**1. W4A16 NVFP4 GEMM for decode (M = 1…1024)**
- FP4 weights are dequantized in registers from a fragment-ordered layout (~2 instructions per weight); activations stay BF16.
- vs Marlin (vLLM's NVFP4 W4A16), layer-weighted over the Qwen3.8-27B shapes: **1.2–4.6% faster at M = 1–32** and 6.8% at M = 96. It is within ±1% at M = 33–64 and 1–2% slower at 192–256, where Marlin is also near the tensor-core limit; M = 544 is 6–7% slower and still needs retuning.
- At M = 1 a GEMV variant reaches ~96% of memory bandwidth on the widest layer.
- Correctness: 288/288 cases within tolerance vs a reference, deterministic, CUDA-graph replay tested.

**2. W4A4 prefill with two-pass residual activations**
- Activations are quantized in two NVFP4 passes (value + residual) to recover near-BF16 accuracy on the native `mxf4nvf4` tensor cores: ~1% GEMM error, vs ~10% for a single pass.
- **1.5–1.9× faster than Marlin at 512–8,192 tokens** on the same shapes. A single pass reaches ~960–1,036 TFLOPS (3.1–4.0×), but its error is too high for serving.
- Validated end-to-end with log-prob and paired-eval checks.

**3. A register-A speed-up for the #3257 path** (small, ready first): unroll the K atoms so the A fragment stays in registers (no local-memory spills), and load the scale factors once per atom instead of once per MMA. Outputs are unchanged.

End to end in vLLM (Qwen3.8-27B NVFP4, one RTX PRO 6000, MTP speculative decoding), single-stream decode went from 111.4 tok/s (NVFP4 + Marlin) to 143.3 tok/s with these kernels plus a few fused decode ops, quality-neutral in paired evals (GSM8K, MMLU-Pro; no significant change) and log-prob checks over ~108k tokens.

Plan, unless you'd prefer a different shape:
1. A PR with the register-A speed-up (it touches `gemm_mma_sm120.py`, so I'll rebase after #3284 and coordinate with #3081).
2. A PR adding the W4A16 decode GEMM, with tests gated on sm_120 and a benchmark vs Marlin.
3. A PR with the two-pass W4A4 prefill.

Questions before I open 2 and 3:
- Would you like these in the library (e.g. an SM120 W4A16 lowering or intrinsics) or as `examples/` + `maint/` benchmarks first?
- Is there any direction for small-M W4A16 on SM120 already in progress that I should build on?

I can run tests and benchmarks on sm_120 hardware for these PRs.
