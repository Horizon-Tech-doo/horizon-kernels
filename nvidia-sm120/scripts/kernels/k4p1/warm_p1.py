# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Pre-compile (on the GPU, vLLM idle) every Horizon kernel variant the phase-1 arm (HORIZON_PDL policy + K5 SiLU) can
# pick, into the arm's TILELANG_CACHE_DIR, so vLLM's start does not compile ~40 new kernels (ab_arm.sh allows 12 min).
# Shapes: Qwen3.8-27B NVFP4 layers (synthetic weights: kernels depend only on shapes). usage: python warm_p1.py POLICY
import sys, time
import torch
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as nr, silu_mul as sm
from tilelang.contrib.horizon_k4 import pdl as P

P.set_policy(sys.argv[1] if len(sys.argv) > 1 else "3@8,5@32")
t0 = time.time()
for n, k in [(16384, 5120), (14336, 5120), (5120, 6144), (34816, 5120), (5120, 17408)]:
    packed, scale, alpha = synthetic_layer(n, k, seed=1)
    prep = xl.prepare(packed, scale, alpha)
    del packed, scale
    for m in range(1, 65):
        xl.linear(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16), prep)
    torch.cuda.synchronize()
    print(f"{n}x{k} warmed at {time.time() - t0:.0f} s", flush=True)
    del prep
W = torch.zeros(96, 5120, device="cuda", dtype=torch.bfloat16)
for m in range(1, sk.SUPPORTS_M + 1):
    sk.linear(torch.zeros(m, 5120, device="cuda", dtype=torch.bfloat16), W)
for m in P.warm_ms(1024):
    x = torch.zeros(m, 5120, device="cuda", dtype=torch.bfloat16)
    nr.add_rmsnorm(x, x, x[0], 1e-6); nr.rmsnorm(x, x[0], 1e-6)
    sm.silu_mul(torch.zeros(m, 34816, device="cuda", dtype=torch.bfloat16), mode=1)
torch.cuda.synchronize()
print(f"WARM_P1_DONE {time.time() - t0:.0f} s")
