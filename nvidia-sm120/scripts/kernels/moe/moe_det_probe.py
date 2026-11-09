# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Determinism probe (27-09-2026, MoE step 1 gate): which op is not run-to-run bit-identical at M >= 32?
# cuBLAS F.linear for the router gate at 256 / 257 / 264 rows, skinny for 264 rows, Marlin-MoE with fixed routing,
# the stock block and the FSE block. 20 repeats each; prints the number of distinct outputs.
import json, os, sys, torch, torch.nn.functional as F
sys.argv = [sys.argv[0], "--layers", "3", "--no-timing"]
import moe_fse_test as t
from vllm.config import VllmConfig, set_current_vllm_config

def distinct(fn, n=20):
    outs = [fn().contiguous().view(torch.int16 if fn().dtype == torch.bfloat16 else torch.int32).clone() for _ in range(n)]
    return len({o.cpu().numpy().tobytes() for o in outs})

with set_current_vllm_config(VllmConfig()):
    P = t.Layer(3)
    torch.manual_seed(5)
    for M in (8, 16, 32, 48, 64, 128, 256):
        x = (torch.randn(M, t.H, device="cuda") * 0.5).to(torch.bfloat16)
        g257 = P.gate264[:257].contiguous()
        tw, ti = t.router_torch(x, F.linear(x, P.gate264))
        r = {"M": M,
             "cublas256": distinct(lambda: F.linear(x, P.gate)),
             "cublas257": distinct(lambda: F.linear(x, g257)),
             "cublas264": distinct(lambda: F.linear(x, P.gate264)),
             "skinny264": distinct(lambda: t.sk.linear(x, P.gate264)),
             "marlin257_fixed_routing": distinct(lambda: t.moe(P.m257, x, tw, ti)),
             "stock_block": distinct(lambda: t.arm_stock(P, x)),
             "fse_block": distinct(lambda: t.arm_fse(P, x))}
        print(json.dumps(r), flush=True)
print("DET_PROBE_DONE")
