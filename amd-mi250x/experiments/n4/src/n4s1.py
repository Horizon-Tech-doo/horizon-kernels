# N4 round 3, step 1: does the 16384^3 low mode flip with the allocation of A and B? Real script kernel gemm_hand(16384,16384,16384,128,256), hipBLASLt through torch in the same process.
# argv: reallocs iters. Every reallocation frees A, B and the torch cache, holds a dummy of a different size, clones the master tensors, drops the dummy.
import json, sys, time
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
import torch
import gemm_prefetch as g
M = N = K = 16384
reallocs, iters = int(sys.argv[1]), int(sys.argv[2])
torch.manual_seed(0)
a0 = torch.randn(M, K, device="cuda", dtype=torch.float16)
b0 = torch.randn(N, K, device="cuda", dtype=torch.float16)
k = g.gemm_hand(M, N, K, 128, 256)
DUMMY_MB = [0, 64, 300, 900, 2048, 5120, 128, 1500]
a = b = None
flops = 2 * M * N * K


def arm(fn):
    c = fn()
    torch.cuda.synchronize()
    cp = hex(c.data_ptr())
    del c
    t0 = time.time()
    p0 = time.perf_counter()
    for _ in range(iters):
        c = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - p0) / iters * 1e3
    return dict(ms=round(ms, 4), TF=round(flops / ms * 1e-9, 2), t0=round(t0, 3), t1=round(time.time(), 3), C=cp), c


for r in range(reallocs):
    a = b = None
    torch.cuda.empty_cache()
    dmb = DUMMY_MB[r % len(DUMMY_MB)]
    d = torch.empty(dmb * 1024 * 512, device="cuda", dtype=torch.int16) if dmb else None
    if r % 2:
        b = b0.clone(); a = a0.clone()
    else:
        a = a0.clone(); b = b0.clone()
    del d
    torch.cuda.empty_cache()
    print(json.dumps({"realloc": r, "dummy_mb": dmb, "A": hex(a.data_ptr()), "B": hex(b.data_ptr())}), flush=True)
    arms = [("hb", lambda: a @ b.T), ("hand", lambda: k(a, b))]
    for i in range(2):
        nm, fn = arms[(i + r) % 2]
        rec, c = arm(fn)
        print(json.dumps({"impl": nm, "realloc": r, "pos": i, **rec}), flush=True)
        if nm == "hb":
            ch = c
        else:
            ck = c
    print(json.dumps({"check": r, "rel": round(((ck.float() - ch.float()).norm() / ch.float().norm()).item(), 7)}), flush=True)
    del c, ch, ck
print(json.dumps({"done": 1}), flush=True)
