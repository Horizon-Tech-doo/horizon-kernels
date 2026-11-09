# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Row-scale fusion gate: horizon_ops.row_scale.quantize_rs vs sm120_nvfp4.prefill_rs.quantize_rs (bit-identity of
# X, SX, rsinv; zero rows; tiny rows) and speed of the whole quantisation step at prefill sizes.
import importlib, json, sys
import torch
OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
rsf = importlib.import_module(f"tilelang.contrib.{PKG}.row_scale")
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs
dev = torch.device("cuda")


def log(r):
    print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")


gen = torch.Generator(device=dev); gen.manual_seed(0)
tot = 0
for M in (128, 1000, 8190, 8192):
    for K in (5120, 6144, 17408):
        x = (torch.randn(M, K, device=dev, generator=gen) * torch.logspace(-6, 3, M, device=dev)[:, None]).to(torch.bfloat16)
        x[3] = 0
        a = prs.quantize_rs(x, 2); b = rsf.quantize_rs(x, 2)
        torch.cuda.synchronize()
        d = [int((p.view(torch.int8) != q.view(torch.int8)).sum()) if p.dtype != torch.float32 else
             int((p.view(torch.int32) != q.view(torch.int32)).sum()) for p, q in zip(a, b)]
        tot += sum(d)
        rec = {"test": "rowscale_bits", "M": M, "K": K, "X_diff": d[0], "SX_diff": d[1], "rsinv_diff": d[2]}
        if M in (8190, 8192):
            for name, fn in (("torch", lambda: prs.quantize_rs(x, 2)), ("fused", lambda: rsf.quantize_rs(x, 2))):
                fn(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                best = 1e30
                for _ in range(5):
                    e0.record()
                    for _ in range(10):
                        fn()
                    e1.record(); e1.synchronize(); best = min(best, e0.elapsed_time(e1) * 100)
                rec[name + "_us"] = round(best, 1)
        log(rec)
log({"summary": "rowscale", "bit_diffs_total": tot})
print("TEST_ROWSCALE_DONE", flush=True)
