# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K3: residual two-pass NVFP4 GEMM for SM120 (prefill / large batches).
#   y = alpha * W * (q1 + q2),   q1 = nvfp4(x),   q2 = nvfp4(x - dq(q1))
# Both activation passes use the native FP4 tensor-core MMA (mma.sync kind::mxf4nvf4.block_scale, m16n8k64)
# against the checkpoint's untouched NVFP4 weights. Two 4-bit passes act like a ~7-8 bit activation: ~1% relative
# error per GEMM on real Qwen3.8 layers vs ~10% for plain W4A4. alpha = 1 / weight_global_scale.
#
# Layout: the two passes are interleaved along K (pass 1 of k-block 0, pass 2 of k-block 0, pass 1 of k-block 1,
# ...). Only one activation tile lives in shared memory per stage, which keeps 2-stage pipelining inside SM120's
# ~99 KB/block limit, and the second read of each weight tile comes straight from L2.
# A is the two passes concatenated along K: A_cat[:, :K] = q1, A_cat[:, K:] = q2 (packed FP4, shape (M, 2K)).
import tilelang
import tilelang.language as T

SF_GRANULARITY_K = 16  # one ue4m3 scale per 16 FP4 values (NVFP4)


@tilelang.jit
def fp4_residual_gemm(M, N, K, block_M=128, block_N=128, block_K=256, num_stages=2, out_dtype=T.bfloat16):
    assert block_M == 128 and block_N == 128, "SM120 blockscaled chunk layout uses 128-row tiles"
    assert N % 8 == 0 and K % block_K == 0 and K % 256 == 0 and block_K % 64 == 0
    in_dtype = T.float4_e2m1fn
    words = block_K // 64
    M_pad = -(-M // block_M) * block_M
    N_pad = -(-N // block_N) * block_N
    k_blocks = K // block_K

    @T.prim_func
    def main(
        A_cat: T.Tensor((M, 2 * K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA1: T.Tensor((M_pad * k_blocks, words), T.uint32), SFA2: T.Tensor((M_pad * k_blocks, words), T.uint32),
        SFB: T.Tensor((N_pad * k_blocks, words), T.uint32),
        alpha: T.Tensor((1,), T.float32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_s = T.alloc_shared((block_M, words), T.uint32)
            SFB_s = T.alloc_shared((block_N, words), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_l)
            for it in T.Pipelined(2 * k_blocks, num_stages=num_stages):
                kb = it // 2
                p = it % 2
                T.copy(A_cat[by * block_M, p * K + kb * block_K], A_s)
                T.copy(B[bx * block_N, kb * block_K], B_s)
                # scale tiles via element loops: T.copy of a scale slice mis-lowers on the bulk-TMA path (upstream note)
                for r, w in T.Parallel(block_M, words):
                    SFA_s[r, w] = T.if_then_else(p == 0, SFA1[(by * k_blocks + kb) * block_M + r, w],
                                                 SFA2[(by * k_blocks + kb) * block_M + r, w])
                for r, w in T.Parallel(block_N, words):
                    SFB_s[r, w] = SFB[(bx * k_blocks + kb) * block_N + r, w]
                T.mma_gemm_blockscaled(A_s, B_s, C_l, SFA_s, SFB_s, transpose_B=True, clear_accum=False,
                                       k_start=kb * block_K, sf_a_granularity_k=SF_GRANULARITY_K,
                                       sf_b_granularity_k=SF_GRANULARITY_K, sf_layout="blockscaled_chunk_kmajor")
            for i, j in T.Parallel(block_M, block_N):
                C_l[i, j] = C_l[i, j] * alpha[0]
            T.copy(C_l, C[by * block_M, bx * block_N])

    return main
