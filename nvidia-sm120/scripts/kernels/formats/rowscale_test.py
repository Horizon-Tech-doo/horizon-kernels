# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Fused row scale for the native FP4 prefill quantizer (prefill_rs.row_scale): bit-identity vs the torch glue and
# speed of quantize_rs old vs new (graph replay). Standalone, <= 2 min.
import json, time, torch
from tilelang.contrib.sm120_nvfp4 import prefill_rs as rs_mod
from tilelang.contrib.sm120_nvfp4 import _ROW_TARGET

def torch_glue(x):
    amax = x.abs().amax(dim=1).float()
    ok = amax > 0
    return (torch.where(ok, _ROW_TARGET / amax.clamp_min(1e-30), torch.ones_like(amax)),
            torch.where(ok, amax / _ROW_TARGET, torch.ones_like(amax)))

def gus(fn, iters=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(iters): fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(3):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3 / iters)
    return min(ts)

ok_all = True
for K in (5120, 6144, 17408):
    for M in (1, 127, 128, 1024, 8192):
        x = (torch.randn(M, K, device="cuda") * torch.logspace(-3, 2, M, device="cuda")[:, None]).to(torch.bfloat16)
        x[M // 2] = 0                                   # an all-zero row
        x[0, 7] = 3.0e4                                 # an outlier
        a1, b1 = torch_glue(x)
        a2, b2 = rs_mod.row_scale(x.contiguous(), _ROW_TARGET)
        same = bool(torch.equal(a1, a2) and torch.equal(b1, b2))
        row = {"K": K, "M": M, "bit_identical": same}
        if M in (1024, 8192):
            Mp = M
            rs_mod.FUSED_ROW_SCALE = False
            X1, S1, r1 = rs_mod.quantize_rs(x, 2)
            t_old = gus(lambda: rs_mod.quantize_rs(x, 2))
            rs_mod.FUSED_ROW_SCALE = True
            X2, S2, r2 = rs_mod.quantize_rs(x, 2)
            t_new = gus(lambda: rs_mod.quantize_rs(x, 2))
            rs_mod.FUSED_ROW_SCALE = False
            row.update({"quantize_identical": bool(torch.equal(X1, X2) and torch.equal(S1, S2) and torch.equal(r1, r2)),
                        "quantize_us_old": round(t_old, 1), "quantize_us_new": round(t_new, 1)})
            same &= row["quantize_identical"]
        ok_all &= same
        print(json.dumps(row), flush=True)
print("ROWSCALE_TEST", "ALL PASS" if ok_all else "FAIL")
