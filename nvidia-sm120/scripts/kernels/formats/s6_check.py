# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S6 pre-check: the retrained MTP layer's BF16 linears, quantized per row to FP8 (horizon_kernels.mtp_quant), on the
# W8A16 kernel: exact class vs an FP32 reference of the quantized weight, deterministic; plus the quantization error
# itself (vs the BF16 weight) and speed vs cuBLAS BF16 at M = 1, 4, 8, 32.
import json, os, sys, torch
from safetensors import safe_open
sys.path.insert(0, os.path.expanduser("$HK_WORK/formats/hpkgs"))
from horizon_kernels.mtp_quant import quantize_rows
from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
F = "$HK_DATA/models/Qwen3.8-27B-NVFP4-hmtp/model_mtp.safetensors"
SEP = torch.zeros(1, device="cuda")
def gus(fns, iters=4):
    fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for f in fns: f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(iters):
            for f in fns: f()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(3):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3 / (iters * len(fns)))
    return min(ts)
sep = gus([lambda: None] * 8)
with safe_open(F, "pt", device="cpu") as f:
    names = [k for k in f.keys() if k.endswith(".weight") and len(f.get_slice(k).get_shape()) == 2]
    for k in sorted(names):
        w = f.get_tensor(k).to("cuda")
        if w.dtype != torch.bfloat16 or w.shape[0] % 128 or w.shape[1] % 64:
            print(json.dumps({"name": k, "shape": list(w.shape), "skipped": True})); continue
        q, s = quantize_rows(w)
        p = w8.prepare(q, s)
        wq = q.float() * s[:, None]
        N, K = w.shape
        qerr = ((wq - w.float()).norm() / w.float().norm()).item()
        copies = max(2, int(512e6 // (N * K * 2)) + 1)
        ws = [w.clone() for _ in range(copies)]; ps = [w8.prepare(q.clone(), s) for _ in range(copies)]
        for m in (1, 4, 8, 32):
            x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ wq.T
            y, y2 = w8.linear(x, p), w8.linear(x, p)
            t_o = gus([lambda pp=pp: w8.linear(x, pp) for pp in ps]) - sep
            t_b = gus([lambda ww=ww: torch.nn.functional.linear(x, ww) for ww in ws]) - sep
            print(json.dumps({"name": k, "N": N, "K": K, "M": m, "w_quant_rel_err": round(qerr, 5),
                              "err_vs_fp32_of_quant": round(((y.float() - ref).norm() / ref.norm()).item(), 6),
                              "det": bool(torch.equal(y, y2)), "us_w8a16": round(t_o, 2), "us_cublas_bf16": round(t_b, 2),
                              "speedup": round(t_b / t_o, 2)}), flush=True)
        del ws, ps
print("S6_CHECK_DONE")
