"""P1 follow-up, one GCD: where do fp16 subnormals get lost, in the float32 -> fp16 cast or in the fp16 matrix product? python p1_subnormal.py
A [1, K] fp16 row of one value x times a [N, K] fp16 matrix of ones, through torch F.linear (hipBLASLt, fp16 MFMA on gfx90a) and through gemm_prefetch's hand kernel (v_mfma_f32_16x16x16f16),
for x normal (1e-3, 1e-4), just above the smallest normal (6.2e-5) and subnormal (3e-5, 1e-5, 1e-6). Expected result K * x16 (x16 = x rounded to fp16, computed on the CPU in float64).
Also the cast alone: a float32 GPU tensor of x converted to fp16 on the GPU and read back. And the same products with bf16 operands (bf16 has no subnormals near these values)."""
import json, sys
import torch, torch.nn.functional as F
K, N = 4096, 4096
res = []
try:
    import gemm_prefetch as g
    kern = g.gemm_auto(N, N, K)
except Exception as e:
    kern = None; print("no hand kernel:", repr(e)[:200])
for x in (1e-3, 1e-4, 6.2e-5, 3e-5, 1e-5, 1e-6):
    x16cpu = float(torch.tensor(x, dtype=torch.float32).to(torch.float16).double())
    cast_gpu = float(torch.full((4,), x, device="cuda", dtype=torch.float32).to(torch.float16).double()[0])
    a = torch.full((N, K), x, dtype=torch.float32, device="cuda").to(torch.float16); w = torch.ones(N, K, device="cuda", dtype=torch.float16)
    lin = float(F.linear(a[:1], w).float()[0, 0]); lin_big = float(F.linear(a, w).float()[0, 0])
    hand = float(kern(a, w).float()[0, 0]) if kern is not None else None
    ab = torch.full((N, K), x, dtype=torch.float32, device="cuda").bfloat16(); wb = torch.ones(N, K, device="cuda", dtype=torch.bfloat16)
    linb = float(F.linear(ab, wb).float()[0, 0])
    r = dict(x=x, subnormal_in_fp16=x16cpu < 6.103515625e-05, x_fp16_cpu=x16cpu, x_fp16_gpu_cast=cast_gpu, expected=K * x16cpu, hipblaslt_fp16_1row=lin, hipblaslt_fp16_4096rows=lin_big, hand_fp16=hand,
             expected_bf16=K * float(torch.tensor(x).bfloat16().double()), hipblaslt_bf16=linb)
    res.append(r); print(json.dumps(r), flush=True)
