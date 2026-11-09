# The GEMM kernel used in C1 (same body as E1 gemm_prefetch.py `base`); blocks>0 adds the annotation. Needs a real file: the JIT reads source lines.
import tilelang
import tilelang.rocm.language as T
@tilelang.jit
def _plain(M, N, K, bM, bN, bK, threads, kp):
    dtype, acc = T.float16, T.float32
    nk = K // bK
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
    return base

@tilelang.jit
def _ann(M, N, K, bM, bN, bK, threads, kp, blocks):
    dtype, acc = T.float16, T.float32
    nk = K // bK
    @T.prim_func
    def base(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            T.annotate_min_blocks_per_sm(blocks)
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            for k in T.serial(nk):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[bx * bN, k * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])
    return base

def gemm(M, N, K, bM, bN, bK, threads, kp, blocks):
    # An annotation placed under a python `if` only scopes that branch, so the two forms are separate kernels.
    return _ann(M, N, K, bM, bN, bK, threads, kp, blocks) if blocks else _plain(M, N, K, bM, bN, bK, threads, kp)
