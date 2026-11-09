"""TileLang GEMM on one MI250X GCD (gfx90a): correctness vs torch, timing vs torch (hipBLASLt / hipBLAS).

C[M,N] = A[M,K] @ B[N,K]^T  (nn.Linear weight layout), fp16/bf16 in, fp32 accumulate.
Prints one JSON line per (dtype, shape, config) plus a summary; never aborts on a single failure.
"""
import itertools, json, os, re, sys, time, traceback
import torch
import tilelang
import tilelang.rocm.language as T  # ROCm dialect: T.gemm(..., k_pack=) lives here


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, bM, bN, bK, dtype, threads=256, stages=0, k_pack=1, swizzle=0):
    accum = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), accum)
            if swizzle:
                T.use_swizzle(panel_size=swizzle)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[bx * bN, k * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=k_pack)
            T.copy(C_l, C[by * bM, bx * bN])

    return main


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (4096, 5120, 5120), (4096, 17408, 5120), (64, 5120, 5120)]
CONFIGS = [  # bM, bN, bK, threads, stages, k_pack, swizzle
    (128, 128, 32, 256, 0, 1, 0),
    (128, 128, 32, 256, 0, 2, 0),
    (128, 128, 64, 256, 0, 2, 0),
    (256, 128, 32, 256, 0, 2, 0),
    (128, 256, 32, 256, 0, 2, 0),
    (128, 128, 32, 256, 2, 2, 0),
    (256, 128, 32, 256, 0, 2, 8),
]


def main():
    dev = torch.cuda.get_device_properties(0)
    print(json.dumps({"device": dev.name, "arch": getattr(dev, "gcnArchName", "?"), "torch": torch.__version__,
                      "hip": torch.version.hip, "tilelang": tilelang.__version__}), flush=True)
    feature_dump_done = False
    best = {}
    for dtype, (M, N, K) in itertools.product(["float16", "bfloat16"], SHAPES):
        tdt = getattr(torch, dtype)
        a = torch.randn(M, K, device="cuda", dtype=tdt)
        b = torch.randn(N, K, device="cuda", dtype=tdt)
        ref = (a.float() @ b.float().T)
        flops = 2 * M * N * K
        torch_ms = {}
        for lib in ["cublaslt", "cublas"]:  # on ROCm: hipBLASLt / hipBLAS(rocBLAS)
            try:
                torch.backends.cuda.preferred_blas_library(lib)
                torch_ms[lib] = bench(lambda: a @ b.T)
            except Exception as ex:  # noqa
                torch_ms[lib] = None
        torch.backends.cuda.preferred_blas_library("cublaslt")
        rec = {"dtype": dtype, "M": M, "N": N, "K": K,
               "torch_hipblaslt_ms": torch_ms["cublaslt"], "torch_hipblas_ms": torch_ms["cublas"]}
        for cfg in CONFIGS:
            bM, bN, bK, th, st, kp, sw = cfg
            r = dict(rec, cfg=cfg)
            t0 = time.time()
            try:
                k = matmul(M, N, K, bM, bN, bK, dtype, th, st, kp, sw)
                r["compile_s"] = round(time.time() - t0, 1)
                c = k(a, b)
                err = (c.float() - ref).abs().max().item()
                rel = err / ref.abs().max().item()
                r["max_rel_err"] = rel
                r["ok"] = bool(rel < 1e-2)
                ms = bench(lambda: k(a, b))
                r["tl_ms"] = ms
                r["tl_tflops"] = flops / ms / 1e9
                r["vs_hipblaslt"] = torch_ms["cublaslt"] / ms if torch_ms["cublaslt"] else None
                if not feature_dump_done:
                    src = k.get_kernel_source()
                    feats = {p: len(re.findall(p, src)) for p in
                             ["mfma", "__builtin_amdgcn", "buffer_load", "global_load_lds", "async", "ds_read", "__shared__", "s_waitcnt"]}
                    print(json.dumps({"kernel_source_features": feats}), flush=True)
                    with open("gemm_kernel_example.hip", "w") as f:
                        f.write(src)
                    feature_dump_done = True
                key = (dtype, M, N, K)
                if r["ok"] and (key not in best or ms < best[key]["tl_ms"]):
                    best[key] = r
            except Exception as ex:  # noqa
                r["ok"] = False
                r["error"] = (type(ex).__name__ + ": " + str(ex))[-600:]
                traceback.print_exc(limit=2, file=sys.stdout)
            print(json.dumps(r), flush=True)
    print("SUMMARY best TileLang config per shape (TFLOPS, ratio vs torch hipBLASLt):")
    for key, r in best.items():
        print(f"  {key}: tl {r['tl_tflops']:.1f} TF ({r['tl_ms']:.3f} ms) cfg={r['cfg']} | "
              f"hipBLASLt {r['torch_hipblaslt_ms']:.3f} ms hipBLAS {r['torch_hipblas_ms'] or float('nan'):.3f} ms | "
              f"ratio {r['vs_hipblaslt']:.2f}")


# Best kernel per shape on gfx90a, ROCm 6.2.4 (clang 18), stock wheel. Measured in evidence/C1. Do not carry to another compiler untested.
# regstage is gemm_prefetch.py's register-staged kernel; rr turns its register arrays into scalars; waves is the second __launch_bounds__ argument.
# memory_clause, when set, compiles with -mllvm -amdgpu-max-memory-clause=N. None of the winners needs it.
BEST_GFX90A = {
    (4096, 4096, 4096): dict(variant="regstage", bM=256, bK=32, waves=2, rr=True, memory_clause=None),
    (8192, 8192, 8192): dict(variant="regstage", bM=256, bK=32, waves=2, rr=True, memory_clause=None),
    (4096, 17408, 5120): dict(variant="regstage", bM=256, bK=32, waves=2, rr=True, memory_clause=None),
    (32768, 1024, 2048): dict(variant="regstage", bM=128, bK=64, waves=3, rr=True, memory_clause=None),
    (16384, 16384, 16384): dict(variant="base", bM=256, bK=32, waves=2, rr=False, memory_clause=None),
}
_HOOK = {"waves": 0, "rr": False}


def _scalar_registers(s):
    """regstage's a_r[]/b_r[] arrays as scalars, so the K loop loads straight into registers (evidence/E1/gen_e1.py)."""
    m = re.search(r"half_t a_r\[(\d+)\];\s*\n\s*half_t b_r\[(\d+)\];", s)
    na, nb = int(m.group(1)) // 8, int(m.group(2)) // 8
    decl = " ".join(f"uint4 a_r_{i};" for i in range(na)) + " " + " ".join(f"uint4 b_r_{i};" for i in range(nb))
    s = s.replace(m.group(0), decl)

    def expand(mm):
        v, n, body = mm.group(1), int(mm.group(2)), mm.group(3)
        out = []
        for i in range(n):
            t = re.sub(r"\*\(uint4\*\)\(([ab])_r \+ \(%s \* 8\)\)" % v, lambda q, i=i: f"{q.group(1)}_r_{i}", body)
            out.append(re.sub(r"\b%s\b" % v, f"({i})", t))
        return "\n".join(out)

    s = re.compile(r"for \(int (v(?:_\d+)?) = 0; \1 < (\d+); \+\+\1\) \{\n(.*?)\n\s*\}", re.S).sub(expand, s)
    assert "a_r +" not in s and "b_r +" not in s
    return s


def _postproc(code, target):
    if _HOOK["rr"]:
        code = _scalar_registers(code)
    if _HOOK["waves"]:
        code = re.sub(r"__launch_bounds__\((\d+)\)", lambda m: f"__launch_bounds__({m.group(1)}, {_HOOK['waves']})", code)
    return code


def best_kernel(M, N, K):
    """Compile BEST_GFX90A's choice for this shape through the JIT. The kernel cache sees neither the hook nor the
    environment, so a per-choice compile_flags entry (unused on the HIP path) keeps the cache entries apart."""
    import tvm_ffi
    from gemm_prefetch import gemm

    tvm_ffi.register_global_func("tilelang_callback_hip_postproc", _postproc, override=True)
    c = BEST_GFX90A[(M, N, K)]
    _HOOK.update(waves=c["waves"], rr=c["rr"])
    old = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
    if c["memory_clause"] is not None:
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = f"-mllvm -amdgpu-max-memory-clause={c['memory_clause']}"
    try:
        tag = f"-DBEST_{c['variant']}_w{c['waves']}_rr{int(c['rr'])}_mc{c['memory_clause']}"
        return tilelang.compile(gemm.get_tir(M, N, K, c["bM"], 128, c["bK"], 256, 2, c["variant"]), out_idx=[-1], compile_flags=[tag])
    finally:
        _HOOK.update(waves=0, rr=False)
        if old is None:
            os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
        else:
            os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = old


def run_best(rounds=4):
    """Each BEST_GFX90A shape against hipBLASLt through torch: same data, order rotated per round, fp32 check on every round."""
    print(json.dumps({"device": torch.cuda.get_device_properties(0).name, "tilelang": tilelang.__version__}), flush=True)
    torch.manual_seed(1234)

    def mk(n, m):
        r = torch.randint(0, 65536, (n, m), dtype=torch.int32, device="cuda")
        return ((r & 0x83ff) | (14 << 10)).to(torch.int16).view(torch.float16)

    for (M, N, K), c in BEST_GFX90A.items():
        a, b = mk(M, K), mk(N, K)
        ref = a.float() @ b.float().T
        flops = 2 * M * N * K
        iters = 5 if M >= 16384 else (20 if flops > 4e11 else 40)
        k = best_kernel(M, N, K)
        rec = {"M": M, "N": N, "K": K, **c, "resource": {n: u.extra for n, u in k.resource_usage.items()}, "bounds": re.search(r"__launch_bounds__\([^)]*\)", k.get_kernel_source()).group(0)}
        print(json.dumps(rec), flush=True)
        arms = [("hipblaslt_torch", lambda: a @ b.T), ("best", lambda: k(a, b))]
        for r in range(rounds):
            for name, fn in arms[r % 2:] + arms[:r % 2]:
                ms = bench(fn, iters=iters, warmup=5)
                out = fn()
                torch.cuda.synchronize()
                rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                print(json.dumps({"round": r, "M": M, "N": N, "K": K, "arm": name, "ms": round(ms, 4), "TF": round(flops / ms * 1e-9, 2), "fp32_rel": rel}), flush=True)
        del a, b, ref, k, arms, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--best":
        run_best(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        main()
