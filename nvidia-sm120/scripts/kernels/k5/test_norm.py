# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K5: add+RMSNorm (Gemma) kernel vs vLLM's native code, eager and Inductor-compiled (the stock serving path):
# bit-exactness stats, determinism, CUDA-graph capture/replay, and timing (graph replay with separator).
import json, statistics, torch
from tilelang.contrib.horizon_fused import add_rmsnorm as an
torch.manual_seed(0)
H, EPS, dev = 5120, 1e-6, "cuda"


def native(x, r, wg):          # vLLM ir.ops.fused_add_rms_norm with weight = w.float() + 1 (GemmaRMSNorm)
    weight = wg.float() + 1.0
    s = x.float() + r.float()
    r_out = s.to(x.dtype)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    y = s * torch.rsqrt(var + EPS)
    return (y.to(weight.dtype) * weight).to(x.dtype), r_out


comp = torch.compile(native, dynamic=True)
w = (torch.randn(H, device=dev) * 0.1).to(torch.bfloat16)
res = {}
for M in (1, 4, 8, 16, 32, 64, 128):
    x = (torch.randn(M, H, device=dev) * 2).to(torch.bfloat16); r = (torch.randn(M, H, device=dev) * 8).to(torch.bfloat16)
    y, ro = an.add_rmsnorm(x, r, w, EPS)
    ye, roe = native(x, r, w); yc, roc = comp(x, r, w)
    d = lambda a, b: int((a != b).sum())
    res[M] = {"resid_eq_eager": torch.equal(ro, roe), "y_diff_vs_eager": d(y, ye), "y_diff_vs_inductor": d(y, yc),
              "eager_vs_inductor": d(ye, yc), "max_abs_vs_inductor": (y.float() - yc.float()).abs().max().item(),
              "n": M * H, "det": all(torch.equal(an.add_rmsnorm(x, r, w, EPS)[0], y) for _ in range(3))}
    print(M, res[M], flush=True)
x = torch.randn(4, H, device=dev).to(torch.bfloat16); r = torch.randn(4, H, device=dev).to(torch.bfloat16)
out = [None]
an.add_rmsnorm(x, r, w, EPS); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, capture_error_mode="thread_local"):
    out[0] = an.add_rmsnorm(x, r, w, EPS)
x.copy_(torch.randn(4, H, device=dev)); g.replay(); torch.cuda.synchronize()
res["graph_replay_ok"] = torch.equal(out[0][0], an.add_rmsnorm(x, r, w, EPS)[0]); print("graph", res["graph_replay_ok"])
SEP = torch.zeros(1, device=dev)


def gtime(fn, n=200, reps=5):
    f = lambda: (fn(), SEP.add_(1.0))
    f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(n): f()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
    for _ in range(reps):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3 / n)
    return statistics.median(ts)


sep = gtime(lambda: None)
for M in (4, 32, 128):
    x = torch.randn(M, H, device=dev).to(torch.bfloat16); r = torch.randn(M, H, device=dev).to(torch.bfloat16)
    ti = gtime(lambda: comp(x, r, w)) - sep; to = gtime(lambda: an.add_rmsnorm(x, r, w, EPS)) - sep
    print(f"M={M} inductor {ti:.2f} us  ours {to:.2f} us"); res[f"t{M}"] = (ti, to)
print("RESULT", json.dumps(res))
