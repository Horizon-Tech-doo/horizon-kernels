# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K5: skinny BF16 GEMM (GDN in_proj_ba) - exactness vs cuBLAS / fp64, determinism, batch invariance, CUDA-graph
# capture (thread_local) + replay, and a race vs cuBLAS (graph replay, rotated weight copies > L2, separator kernel).
# Run under the GPU lock: gpu_lock.sh k5-skinny python test_skinny.py [--cfg "(MB,BN,thr)"]
import json, statistics, sys, torch, torch.nn.functional as F
from tilelang.contrib.horizon_fused import skinny_bf16 as sk
torch.manual_seed(0)
N, K = 96, 5120
MS = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]
dev = "cuda"
W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)


def ulp_diff(a, b):
    ai = a.view(torch.int16).int(); bi = b.view(torch.int16).int()
    ai = torch.where(ai < 0, -32768 - ai, ai); bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


res = {"exact": {}, "time_us": {}}
for M in MS:
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    y = sk.linear(x, W)
    ref = (x.double() @ W.double().T)
    yc = F.linear(x, W)
    d_c = ulp_diff(y, yc); d_ref_ours = (y.double() - ref).abs().max().item(); d_ref_cub = (yc.double() - ref).abs().max().item()
    det = all(torch.equal(sk.linear(x, W), y) for _ in range(3))
    inv = torch.equal(sk.linear(x[:1], W)[0], y[0])
    res["exact"][M] = {"vs_cublas_max_ulp": int(d_c.max()), "vs_cublas_frac_diff": round((d_c > 0).float().mean().item(), 4),
                       "abs_err_vs_fp64_ours": d_ref_ours, "abs_err_vs_fp64_cublas": d_ref_cub,
                       "deterministic": det, "batch_invariant_row0": inv}
    print(M, res["exact"][M], flush=True)

# CUDA graph capture (thread_local) + replay
xs = torch.randn(4, K, device=dev).to(torch.bfloat16); out = torch.empty(4, N, device=dev, dtype=torch.bfloat16)
sk.linear(xs, W, out=out); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
for attempt in range(3):
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            sk.linear(xs, W, out=out)
        break
    except RuntimeError as e:
        print("capture retry", e); g = torch.cuda.CUDAGraph()
xs.copy_(torch.randn(4, K, device=dev).to(torch.bfloat16)); g.replay(); torch.cuda.synchronize()
res["graph_replay_ok"] = torch.equal(out, sk.linear(xs, W))
print("graph replay equal to eager:", res["graph_replay_ok"], flush=True)

# race: rotated weight copies (256 x 0.98 MB = 252 MB > 128 MB L2), separator kernel between calls
SEP = torch.zeros(1, device=dev)
Ws = [(torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(256)]


def gtime(fns, reps=5):
    fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for f in fns: f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for f in fns: f()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
    for _ in range(reps):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3 / len(fns))
    return statistics.median(ts)


sep = gtime([lambda: None] * 256)
cfg = eval(sys.argv[sys.argv.index("--cfg") + 1]) if "--cfg" in sys.argv else None
for M in MS:
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    y1 = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    t_c = min(gtime([lambda w=w: F.linear(x, w) for w in Ws]) for _ in range(3)) - sep
    t_o = min(gtime([lambda w=w: sk.linear(x, w, cfg=cfg, out=y1) for w in Ws]) for _ in range(3)) - sep
    res["time_us"][M] = {"cublas": round(t_c, 2), "ours": round(t_o, 2), "cfg": cfg or sk.config(N, K, M)}
    print(f"M={M:4d} cuBLAS {t_c:6.2f} us  ours {t_o:6.2f} us  cfg {cfg or sk.config(N, K, M)}", flush=True)
res["separator_us"] = round(sep, 2)
print("RESULT", json.dumps(res))
