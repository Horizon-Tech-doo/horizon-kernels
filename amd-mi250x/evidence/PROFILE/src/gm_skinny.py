"""PROFILE / decode GEMM probe, one GCD: vLLM's own small-row kernel (ops.wvSplitK, which rocm_unquantized_gemm_impl uses for 1 to 5 rows only) called directly at 1 to 32 rows, against what the server
runs at those row counts (torch F.linear = hipBLASLt above 5 rows), bf16, the served 27B's five large linear shapes. python gm_skinny.py <out.jsonl>
Rows per cell are written before the next row count runs, in ascending order, so a row count the kernel does not take leaves the earlier ones on disk. Error against an fp32 product; a wrong or
non-finite result is recorded as such and not timed. stock_b = the server's path again (identical-code control). Bandwidth = weight bytes / time."""
import json, statistics, sys, time
import torch, torch.nn.functional as F
import vllm._custom_ops as ops
from vllm.model_executor.layers import utils as U
out = open(sys.argv[1], "a"); ROUNDS = 7
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it=200):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
cu = torch.cuda.get_device_properties(0).multi_processor_count
P(kind="env", cu=cu, torch=torch.__version__)
for N, K in [(17408, 5120), (5120, 8704), (8192, 5120), (7168, 5120), (5120, 3072)]:
    g = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
    for n in (1, 2, 4, 5, 6, 8, 12, 16, 24, 32):
        x = torch.randn(n, K, device="cuda", generator=g).bfloat16()
        ref = x.float() @ w.float().T; nr = ref.norm()
        row = dict(kind="cell", n=n, N=N, K=K)
        arms = {"stock": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "stock_b": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "linear": lambda: F.linear(x, w)}
        try:
            c = ops.wvSplitK(w, x, cu, None)
            e = float((c.reshape(n, N).float() - ref).norm() / nr); es = float((arms["stock"]().float() - ref).norm() / nr)
            row.update(err_wvsplitk=e, err_stock=es, wv_shape=list(c.shape), wv_ok=bool(torch.isfinite(c).all()) and e < 1.5 * es + 1e-4)
            if row["wv_ok"]: arms["wvsplitk"] = lambda: ops.wvSplitK(w, x, cu, None)
        except Exception as ex:
            row["wv_error"] = repr(ex)[-200:]
        for f in arms.values(): tm(f, 5)
        ts = {a: [] for a in arms}; names = list(arms)
        for rd in range(ROUNDS):
            for i in range(len(names)):
                a = names[(i + rd) % len(names)]; ts[a].append(tm(arms[a]))
        for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 5); row[a + "_TBs"] = round(N * K * 2 / statistics.median(ts[a]) * 1e-9, 3)
        row["noise"] = round(abs(row["stock_b_ms"] / row["stock_ms"] - 1), 4)
        if "wvsplitk" in arms: row["stock_over_wvsplitk"] = round(row["stock_ms"] / row["wvsplitk_ms"], 4)
        P(**row); del x, ref
    del w; torch.cuda.empty_cache()
P(kind="done")
