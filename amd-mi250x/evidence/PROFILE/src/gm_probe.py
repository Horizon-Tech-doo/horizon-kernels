"""PROFILE / GEMM gate, one GCD, in the server's own environment (vLLM image + TileLang venv). python gm_probe.py <out.jsonl> [M list] [N:K list]
For every dense-linear shape the served 27B issues under TP 2 (C[M,N] = A[M,K] @ W[N,K]^T): the server's own path (vllm rocm_unquantized_gemm_impl on bf16), torch F.linear bf16 and fp16 (hipBLASLt),
and tilelang/gemm_prefetch.py (fp16 only) both bare and as it would be wired (bf16 activations cast to fp16, fp16 copy of the weight, result cast back to bf16).
Interleaved rounds, order rotated; ms per call from ITERS calls between two synchronisations. Error against an fp32 product of the same bf16 inputs.
GM_LIBLOG=1 (a second, separate run): no timing; prints what hipBLASLt / rocBLAS log for F.linear at these shapes (set HIPBLASLT_LOG_MASK / ROCBLAS_LAYER in the environment)."""
import json, os, statistics, sys, time
import torch, torch.nn.functional as F
out = open(sys.argv[1], "a")
MS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 8, 16, 32, 64, 128, 512, 1024, 1536, 2048, 4096, 8192, 16384]
NK = [tuple(map(int, x.split(":"))) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [(17408, 5120), (5120, 8704)]
ROUNDS = 5
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
if os.environ.get("GM_LIBLOG") == "1":
    for dt in (torch.bfloat16, torch.float16):
        for N, K in NK:
            w = torch.randn(N, K, device="cuda", dtype=dt)
            for M in (1, 8, 32, 2048):
                print(f"## liblog dtype {dt} M {M} N {N} K {K}", flush=True); sys.stderr.write(f"## liblog dtype {dt} M {M} N {N} K {K}\n"); sys.stderr.flush()
                x = torch.randn(M, K, device="cuda", dtype=dt); F.linear(x, w); torch.cuda.synchronize()
    sys.exit(0)
from vllm.model_executor.layers import utils as U
import gemm_prefetch as g
P(kind="env", torch=torch.__version__, hip=torch.version.hip, dev=torch.cuda.get_device_name(0), blas=str(torch.backends.cuda.preferred_blas_library()), skinny=os.environ.get("VLLM_ROCM_USE_SKINNY_GEMM"),
  prefer_lt=os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"))
for N, K in NK:
    gen = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=gen) * 0.02).bfloat16()
    w16 = w.to(torch.float16)
    P(kind="weight", N=N, K=K, fp16_copy_exact=bool((w16.float() == w.float()).all()), fp16_copy_bytes=w16.numel() * 2)
    for M in MS:
        x = torch.randn(M, K, device="cuda", generator=gen).bfloat16(); x16 = x.to(torch.float16)
        ref = x.float() @ w.float().T; nr = ref.norm()
        err = lambda c: float((c.float() - ref).norm() / nr)
        t0 = time.time(); kern, how, Mb, wk = None, None, M, w16
        try:
            if M < 1024:
                for Mc in [M] + [b for b in (16, 32, 64) if b > M]:
                    kern = g.gemm_small_m(Mc, N, K)
                    if kern is not None: how, Mb, wk = "small_m", Mc, g.small_m_b(Mc, w16); break
            else:
                for Mc in sorted({M, -(-M // 128) * 128, -(-M // 256) * 256}):
                    kern = g.gemm_auto(Mc, N, K)
                    if kern is not None: how, Mb = "auto:" + str(g.select_mode(Mc, N, K)), Mc; break
        except Exception as e:
            P(kind="kernel_error", M=M, N=N, K=K, err=repr(e)[:300]); kern = None
        tb = time.time() - t0
        arms = {"stock_bf16": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "hb_bf16": lambda: F.linear(x, w), "hb_fp16": lambda: F.linear(x16, w16)}
        if kern is not None:
            if Mb != M:
                xb = torch.zeros(Mb, K, device="cuda", dtype=torch.float16)
                def wired():
                    xb[:M].copy_(x); return kern(xb, wk)[:M].to(torch.bfloat16)
                xk = torch.zeros(Mb, K, device="cuda", dtype=torch.float16); xk[:M].copy_(x16)
            else:
                def wired(): return kern(x.to(torch.float16), wk).to(torch.bfloat16)
                xk = x16
            arms["ours_fp16"] = lambda: kern(xk, wk)
            arms["ours_wired"] = wired
        row = dict(kind="cell", M=M, N=N, K=K, ours=how, M_bucket=Mb if kern is not None else None, build_s=round(tb, 1))
        try:
            for a, f in arms.items():
                c = f(); c = c[:M] if c.shape[0] != M else c
                row["err_" + a] = err(c); row["finite_" + a] = bool(torch.isfinite(c).all()); del c
            it = max(3, min(200, int(2e12 / (2.0 * M * N * K)) or 3)) if M >= 512 else 200
            for f in arms.values(): tm(f, 3)
            ts = {a: [] for a in arms}; names = list(arms)
            for rd in range(ROUNDS):
                for i in range(len(names)):
                    a = names[(i + rd) % len(names)]; ts[a].append(tm(arms[a], it))
            for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 5); row[a + "_all"] = [round(v, 5) for v in ts[a]]; row[a + "_TF"] = round(2.0 * M * N * K / statistics.median(ts[a]) * 1e-9, 2)
            row["iters"] = it
            if kern is not None:
                row["stock_over_ours_wired"] = round(row["stock_bf16_ms"] / row["ours_wired_ms"], 4); row["hbfp16_over_ours"] = round(row["hb_fp16_ms"] / row["ours_fp16_ms"], 4)
            row["hbbf16_over_hbfp16"] = round(row["hb_bf16_ms"] / row["hb_fp16_ms"], 4); row["stock_over_hbbf16"] = round(row["stock_bf16_ms"] / row["hb_bf16_ms"], 4)
        except Exception as e:
            row["error"] = repr(e)[:300]
        P(**row)
        del x, x16, ref, kern, arms; torch.cuda.empty_cache()
    del w, w16; torch.cuda.empty_cache()
P(kind="done")
