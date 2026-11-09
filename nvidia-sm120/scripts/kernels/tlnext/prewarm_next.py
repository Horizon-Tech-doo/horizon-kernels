# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Pre-compile, into the rebased build's kernel cache, every kernel the vLLM plugin warms at load (so the A/B arm on the
# new build starts as fast as production): regfrag_xl at every M 1..1024 for the 5 NVFP4 shapes, K3 (quantizer + FP4
# GEMM passes 2) per shape, K5 (all skinny configs, add+RMSNorm / RMSNorm at H 5120) and the K1a draft head (48k x 5120).
import sys, time, torch
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as an
t0 = time.time()
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
for name, (N, K) in SHAPES.items():
    packed, scale, alpha = synthetic_layer(N, K, seed=0)
    prep = xl.prepare(packed, scale, alpha)
    for m in range(1, 1025):
        xl.linear(torch.zeros(m, K, device="cuda", dtype=torch.bfloat16), prep)
    w = nv.prepare_weight(packed.view(torch.uint8), scale, torch.tensor([2688.0], device="cuda"))
    nv.linear(torch.randn(128, K, device="cuda", dtype=torch.bfloat16), w, passes=2)
    torch.cuda.synchronize(); del prep, w
    print(f"{name}: warm after {time.time() - t0:.0f} s", flush=True)
W = torch.zeros(96, 5120, device="cuda", dtype=torch.bfloat16)
for cfg in sk.all_configs(96, 5120):
    for m in range(1, sk.SUPPORTS_M + 1):
        if sk.config(96, 5120, m) == cfg:
            sk.linear(torch.zeros(m, 5120, device="cuda", dtype=torch.bfloat16), W, cfg=cfg); break
for eps in (1e-6,):
    x = torch.zeros(4, 5120, device="cuda", dtype=torch.bfloat16); wn = torch.zeros(5120, device="cuda", dtype=torch.bfloat16)
    an.add_rmsnorm(x, x.clone(), wn, eps); an.rmsnorm(x, wn, eps)
sys.path.insert(0, "$HK_DATA/horizon-pkgs")
from horizon_kernels.draft_head import DraftHeadFP8
head = DraftHeadFP8(torch.zeros(49152, 5120, device="cuda", dtype=torch.bfloat16), act="fp8")
head(torch.zeros(4, 5120, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
print(f"PREWARM_DONE {time.time() - t0:.0f} s", flush=True)
