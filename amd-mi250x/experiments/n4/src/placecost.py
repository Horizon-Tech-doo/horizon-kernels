# Cost of place_b against one GEMM at 16384^3 and 8192^3, and the resulting address offset. Prints ms.
import json, sys, time
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
import torch
import gemm_prefetch as g
for n in (8192, 16384):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16); b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t = time.perf_counter(); x = g.place_b(a, b); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3); assert x is not b
        again = g.place_b(a, x); assert again is x
        del x, again
    print(json.dumps({"n": n, "place_b_ms": [round(v, 2) for v in ts]}), flush=True)
