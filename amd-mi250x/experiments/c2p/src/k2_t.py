"""Compile a plain TileLang GEMM with T.Kernel(gemm_schedule="pgr2"), compare its asm with the hand kernel's. Login node."""
import os, sys, json, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import tilelang, tilelang.rocm.language as T
def make(M, N, K, bM, bN, sched="pgr2"):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=256, gemm_schedule=sched) as (bx, by):
            A_s = T.alloc_shared((bM, 64), T.float16)
            B_s = T.alloc_shared((bN, 64), T.float16)
            C_l = T.alloc_fragment((bM, bN), T.float32)
            T.clear(C_l)
            for k in T.serial(T.ceildiv(K, 64)):
                T.copy(A[by * bM, k * 64], A_s)
                T.copy(B[bx * bN, k * 64], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * bM, bx * bN])
    return main
k = tilelang.compile(make(8192, 8192, 8192, 256, 128), out_idx=[-1], target="hip")
s = k.get_kernel_source()
print(len(s), "hand_body" in s, s.count("v_mfma"), s[-700:])
print(dict(k.resource_usage))
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
import importlib.util
from tilelang.rocm.transform.schedule_gemm import _schedule_source
a = _schedule_source(8192, 8192, 8192, 256, 128)[0]
try:
    import gemm_prefetch
    b = gemm_prefetch.hand_source(8192, 8192, 8192, 256, 128)[0]
    print("generator equal to gemm_prefetch.hand_source:", a == b, len(a), len(b))
except Exception as e:
    print("no compare", e)
for bad in [(8192, 8192, 128, 256, 128), (8192, 8200, 8192, 256, 128)]:
    k2 = tilelang.compile(make(*bad), out_idx=[-1], target="hip")
    print(bad, "hand_body" in k2.get_kernel_source())
