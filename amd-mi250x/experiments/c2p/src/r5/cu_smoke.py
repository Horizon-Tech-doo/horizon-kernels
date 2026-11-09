"""CPU only, CUDA-enabled build of the series tree. Generates CUDA kernel source; no GPU, nothing is executed."""
import os, traceback
TGT = {"kind": "cuda", "arch": "sm_90a"}
import tilelang as tl, tilelang.language as T
print("tilelang", tl.__file__)
ok = bad = 0
def check(name, fn):
    global ok, bad
    try:
        print("PASS", name, fn()); ok += 1
    except Exception as e:
        bad += 1; print("FAIL", name, repr(e)[:400]); traceback.print_exc(limit=3)
def min_blocks():
    # Moved attribute key (0001): CUDA keeps emitting the annotation. Same kernel as testing/python/language/test_tilelang_language_min_blocks_per_sm.py.
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        with T.Kernel(128, threads=128) as bx:
            T.annotate_min_blocks_per_sm(2)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]
    s = tl.compile(main, out_idx=[2], target=TGT).get_kernel_source()
    assert "__launch_bounds__(128, 2)" in s
    return "__launch_bounds__(128, 2)"
def reducer(scale, TGT=TGT):
    # 0008 changes shared reducer passes: plain sum, and reducer_scale, on CUDA.
    @T.prim_func
    def main(A: T.Tensor((64, 64), "float32"), F: T.Tensor((64,), "float32"), out: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            acc = T.alloc_reducer((64,), T.float32, op="sum")
            r = T.alloc_fragment((64,), T.float32)
            T.reducer_init(acc)
            if scale:
                for i in T.Parallel(64):
                    T.reducer_scale(acc[i], F[i])
            for i, j in T.Parallel(64, 64):
                T.reducer_update(acc[i], A[i, j])
            T.finalize_reducer(acc, r)
            for i in T.Parallel(64):
                out[i] = r[i]
    s = tl.compile(main, out_idx=[2], target=TGT).get_kernel_source()
    return len(s)
check("cuda min_blocks launch bounds", min_blocks)
check("cuda sum reducer", lambda: reducer(False))
check("simple reducer_scale kernel on hip, same source as above", lambda: reducer(True, {"kind": "hip", "mcpu": "gfx90a", "mtriple": "amdgcn-amd-amdhsa-hcc", "thread_warp_size": 64}))
check("simple reducer_scale kernel on cuda", lambda: reducer(True))
def gemm_form(tgt):
    # The kernel form of testing/python/amd/test_tilelang_hip_reducer_scale.py, on the given target.
    from tilelang.tileop.base import GemmWarpPolicy
    M, N, steps, threads = 64, 64, 3, 64
    @T.prim_func
    def main(A: T.Tensor((steps, M, 32), T.float16), B: T.Tensor((N, 32), T.float16), out: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, 32), T.float16)
            B_s = T.alloc_shared((N, 32), T.float16)
            s = T.alloc_fragment((M, N), T.float32)
            m = T.alloc_fragment((M,), T.float32)
            f = T.alloc_fragment((M,), T.float32)
            r = T.alloc_fragment((M,), T.float32)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            T.copy(B[0, 0], B_s)
            for k in T.serial(steps):
                T.copy(A[k, 0, 0], A_s)
                T.clear(s)
                T.gemm(A_s, B_s, s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.reduce_max(s, m, dim=1)
                for i in T.Parallel(M):
                    f[i] = T.exp(-T.abs(m[i]))
                for i in T.Parallel(M):
                    T.reducer_scale(acc[i], f[i])
                for i, j in T.Parallel(M, N):
                    T.reducer_update(acc[i], s[i, j])
            T.finalize_reducer(acc, r)
            for i, j in T.Parallel(M, N):
                out[i, j] = r[i]
    return len(tl.compile(main, out_idx=[-1], target=tgt).get_kernel_source())
check("gemm form reducer_scale on hip", lambda: gemm_form({"kind": "hip", "mcpu": "gfx90a", "mtriple": "amdgcn-amd-amdhsa-hcc", "thread_warp_size": 64}))
check("gemm form reducer_scale on cuda", lambda: gemm_form(TGT))
print("passed", ok, "failed", bad)
