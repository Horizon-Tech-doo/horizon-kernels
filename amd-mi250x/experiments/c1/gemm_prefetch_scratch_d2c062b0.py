"""Dense GEMM on gfx90a with hand-written overlap of global loads and MFMA work (Track 4; also the bottleneck of the
fused MoE and attention kernels).

On HIP, TileLang's T.Pipelined(num_stages > 1) falls back to a sequential loop and gfx90a has no async global->LDS copy,
so every K step is: load A/B tiles, barrier, MFMA, barrier. Variants:
  base      : the bench_gemm.py kernel (stages 0)
  regpref   : register prefetch: the next K tile is loaded global -> registers before the MFMAs of the current tile, then
              stored to LDS after them, so the global-load latency overlaps the math (the CK / "software pipelining
              without async copy" pattern)
  dbuf      : two LDS buffers alternating, loads of tile k+1 into the other buffer before computing tile k
C[M,N] = A[M,K] @ B[N,K]^T, fp16, fp32 accumulate. Correctness vs torch fp32, speed vs torch hipBLASLt.
"""
import json
import sys

import torch
import tilelang
import tilelang.rocm.language as T


@tilelang.jit
def gemm_splitk(M, N, K, bM, bN, bK, threads, kp, S):
    """Small-M (decode) GEMM: grid (N/bN, M/bM, S); each block does K/S of the reduction and atomically adds its fp32
    partial into C32 (zeroed by the caller); the caller casts to fp16. Few rows means few blocks otherwise."""
    dtype, acc = T.float16, T.float32
    nk = K // bK // S

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C32: T.Tensor((M, N), acc)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), S, threads=threads) as (bx, by, bz):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            for k in T.serial(nk):
                kk = bz * nk + k
                T.copy(A[by * bM, kk * bK], A_s)
                T.copy(B[bx * bN, kk * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            for i, j in T.Parallel(bM, bN):
                T.atomic_add(C32[by * bM + i, bx * bN + j], C_l[i, j])

    return main


@tilelang.jit(out_idx=[-1])
def gemm(M, N, K, bM, bN, bK, threads, kp, variant):
    dtype, acc = T.float16, T.float32
    nk = K // bK
    VA = max(1, bM * bK // 8 // threads)          # 16-byte vectors per thread per tile (regstage)
    VB = max(1, bN * bK // 8 // threads)

    @T.prim_func
    def base(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            for k in T.serial(nk):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[bx * bN, k * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def regpref(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            A_r = T.alloc_fragment((bM, bK), dtype)
            B_r = T.alloc_fragment((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            T.copy(A[by * bM, 0], A_r)
            T.copy(B[bx * bN, 0], B_r)
            for k in T.serial(nk):
                T.copy(A_r, A_s)
                T.copy(B_r, B_s)
                if k + 1 < nk:                       # issue the next tile's global loads before this tile's MFMAs
                    T.copy(A[by * bM, (k + 1) * bK], A_r)
                    T.copy(B[bx * bN, (k + 1) * bK], B_r)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def dbuf(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A0 = T.alloc_shared((bM, bK), dtype)
            B0 = T.alloc_shared((bN, bK), dtype)
            A1 = T.alloc_shared((bM, bK), dtype)
            B1 = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            T.copy(A[by * bM, 0], A0)
            T.copy(B[bx * bN, 0], B0)
            for kk in T.serial(nk // 2):
                k = kk * 2
                T.copy(A[by * bM, (k + 1) * bK], A1)
                T.copy(B[bx * bN, (k + 1) * bK], B1)
                T.gemm(A0, B0, C_l, transpose_B=True, k_pack=kp)
                if k + 2 < nk:
                    T.copy(A[by * bM, (k + 2) * bK], A0)
                    T.copy(B[bx * bN, (k + 2) * bK], B0)
                T.gemm(A1, B1, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def regstage(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        """Register-staged pipelining (CK style) with an explicit per-thread copy map: thread t owns 16-byte vectors
        v*threads + t of each tile, so no fragment-layout inference is involved. Tile k+1 is loaded global -> registers
        before the MFMAs of tile k and stored to LDS after them; the hardware overlaps the loads with the math."""
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            a_r = T.alloc_local((VA * 8,), dtype)
            b_r = T.alloc_local((VB * 8,), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            tx = T.get_thread_binding()
            T.clear(C_l)
            for v in T.serial(VA):
                for j in T.vectorized(8):
                    a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for v in T.serial(VB):
                for j in T.vectorized(8):
                    b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for k in T.serial(nk):
                for v in T.serial(VA):
                    for j in T.vectorized(8):
                        A_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = a_r[v * 8 + j]
                for v in T.serial(VB):
                    for j in T.vectorized(8):
                        B_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = b_r[v * 8 + j]
                if k + 1 < nk:
                    for v in T.serial(VA):
                        for j in T.vectorized(8):
                            a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                    for v in T.serial(VB):
                        for j in T.vectorized(8):
                            b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    return {"base": base, "regpref": regpref, "dbuf": dbuf, "regstage": regstage}[variant]


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


def run_splitk():
    """Track 4 small-M sweep: M in {1, 16, 32, 64} x Qwen-like N/K, split-K S in {1, 2, 4, 8, 16}."""
    torch.manual_seed(0)
    for M, N, K in [(64, 5120, 5120), (32, 5120, 5120), (16, 17408, 5120), (64, 17408, 5120), (64, 5120, 17408), (1, 5120, 5120)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)
        ref = a.float() @ b.float().T
        tms = bench(lambda: a @ b.T)
        nbytes = (N * K + M * K) * 2
        print(json.dumps({"impl": "hipblaslt", "M": M, "N": N, "K": K, "ms": round(tms, 4), "gbps": round(nbytes / tms / 1e6, 1)}), flush=True)
        bM = 16 if M <= 16 else (32 if M <= 32 else 64)
        for bN, bK, th, S in [(64, 64, 64 if bM == 16 else 128, 1), (64, 64, 64 if bM == 16 else 128, 4), (64, 64, 64 if bM == 16 else 128, 8),
                              (128, 64, 128 if bM >= 32 else 64, 4), (64, 128, 64 if bM == 16 else 128, 4), (32, 128, 64, 8), (64, 64, 64 if bM == 16 else 128, 16)]:
            r = {"impl": "splitk", "M": M, "N": N, "K": K, "bM": bM, "bN": bN, "bK": bK, "threads": th, "S": S}
            if K % (bK * S):
                continue
            try:
                Mp = max(M, bM)
                ap = a if M == Mp else torch.cat([a, torch.zeros(Mp - M, K, device="cuda", dtype=a.dtype)])
                if S == 1:                      # no split: plain fp16 kernel, no fp32 zero/atomics/cast
                    k = gemm(Mp, N, K, bM, bN, bK, th, 1, "base")

                    def run():
                        return k(ap, b)[:M]
                else:
                    k = gemm_splitk(Mp, N, K, bM, bN, bK, th, 1, S)
                    c32 = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)

                    def run():
                        c32.zero_()
                        k(ap, b, c32)
                        return c32[:M].half()
                c = run()
                r["rel_err"] = round(((c.float() - ref).norm() / ref.norm()).item(), 6)
                ms = bench(run)
                r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), vs_hipblaslt=round(tms / ms, 3))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-160:]
            print(json.dumps(r), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "splitk":
    run_splitk()
    sys.exit(0)

if __name__ == "__main__":
    shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (4096, 17408, 5120), (32768, 1024, 2048), (64, 5120, 5120)]
    configs = [(128, 128, 32, 256, 2), (256, 128, 32, 256, 2), (128, 128, 64, 256, 2)]
    variants = ["base", "regpref", "dbuf"]
    if len(sys.argv) > 1:
        variants = sys.argv[1].split(",")
    torch.manual_seed(0)
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)
        ref = a.float() @ b.float().T
        tms = bench(lambda: a @ b.T)
        flops = 2 * M * N * K
        print(json.dumps({"impl": "hipblaslt", "M": M, "N": N, "K": K, "ms": round(tms, 4), "tflops": round(flops / tms * 1e-9, 1)}), flush=True)
        for bM, bN, bK, th, kp in configs:
            if M < bM:
                bM = 64 if M >= 64 else M
            for v in variants:
                r = {"impl": v, "M": M, "N": N, "K": K, "bM": bM, "bN": bN, "bK": bK, "threads": th, "k_pack": kp}
                try:
                    k = gemm(M, N, K, bM, bN, bK, th, kp, v)
                    c = k(a, b)
                    r["rel_err"] = round(((c.float() - ref).norm() / ref.norm()).item(), 6)
                    ms = bench(lambda: k(a, b))
                    r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 1), vs_hipblaslt=round(tms / ms, 3))
                except Exception as e:  # noqa: BLE001
                    r["err"] = (type(e).__name__ + ": " + str(e))[-160:]
                print(json.dumps(r), flush=True)
