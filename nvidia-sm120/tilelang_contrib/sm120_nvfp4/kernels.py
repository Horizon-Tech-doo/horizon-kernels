# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""SM120 NVFP4 kernels with a runtime token count (dynamic M) and a residual multi-pass activation mode.

``passes=1`` is plain W4A4: activations quantised once to NVFP4.
``passes=2`` is residual two-pass: q1 = nvfp4(x), q2 = nvfp4(x - dq(q1)); the GEMM accumulates W*q1 + W*q2 with the
native block-scaled FP4 MMA (``mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X``). Two 4-bit passes behave
like a ~7-8 bit activation (about 1% relative GEMM error on real LLM layers vs about 10% for one pass) while the
NVFP4 weights stay untouched.

Layouts (shared by both kernels, M must be a multiple of 128, K a multiple of 256):
  A    int8 (M, passes*K/2)   packed FP4; pass p occupies bytes [p*K/2, (p+1)*K/2)
  SFA  uint32 (passes, M, K/64)  four ue4m3 scale bytes per word, SM120 blockscaled_chunk_kmajor order
  B    int8 (N, K/2) packed FP4 weights, SFB uint32 (N, K/64) same scale layout
"""

import tilelang
import tilelang.language as T

from ._device import (encode_ue4m3_ceil, decode_ue4m3, ue4m3_inverse, scale_source_coords, FP4_E2M1_MAX,
                      CHUNK_WORDS)

SF_GRANULARITY_K = 16


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def nvfp4_quant_kernel(x, rs, passes: int = 2, rows_per_cta: int = 32, num_threads: int = 512):
    """BF16 (M, K) -> residual NVFP4 activations of rs[m] * x[m, :], one read of x, all passes in registers.

    rs is a per-row scale chosen by the caller so each row's largest value lands near the top of the NVFP4 range
    (block scale ue4m3 <= 448, FP4 <= 6): the ue4m3 block scales then stay in their normal range and the small
    second-pass residual does not underflow. The GEMM divides it back out per row (exact, it is a row scaling).
    Private: M must be a multiple of 128 (use quantize()/linear(), which pad)."""
    M = T.dynamic("M")
    K = T.const("K")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    f32 = T.float32
    x: T.Tensor[(M, K), T.bfloat16]
    rs: T.Tensor[(M,), T.float32]
    A = T.empty((M, passes * (K // 2)), T.int8)
    SFA = T.empty((passes, M, K // 64), T.uint32)

    with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        x_s = T.alloc_shared((rows_per_cta, 256), T.bfloat16)
        codes = T.alloc_shared((passes, rows_per_cta, 16), T.int32)
        T.copy(x[pid_m * rows_per_cta, pid_k256 * 256], x_s, prefer_instruction="tma")
        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            grow = pid_m * rows_per_cta + row
            for k64 in T.serial(CHUNK_WORDS):
                elem = k64 * 64 + lane * 2
                byte = pid_k256 * 128 + k64 * 32 + lane
                group = k64 * 4 + (lane // 8)
                glane = lane % 8
                r0 = T.alloc_var(f32, init=T.cast(x_s[row, elem], f32) * rs[grow])
                r1 = T.alloc_var(f32, init=T.cast(x_s[row, elem + 1], f32) * rs[grow])
                for p in T.unroll(passes):
                    # quantise the current residual over this 16-value group (8 lanes x 2 values)
                    am = T.alloc_var(f32, init=T.max(T.abs(r0), T.abs(r1)))
                    am = T.max(am, T.shfl_xor(am, 4, width=8))
                    am = T.max(am, T.shfl_xor(am, 2, width=8))
                    am = T.max(am, T.shfl_xor(am, 1, width=8))
                    code = T.alloc_var(T.int32, init=0)
                    sc = T.alloc_var(f32, init=0.0)
                    inv = T.alloc_var(f32, init=0.0)
                    if glane == 0:
                        code = encode_ue4m3_ceil(am / FP4_E2M1_MAX)
                        sc = decode_ue4m3(code)
                        inv = ue4m3_inverse(code)
                        codes[p, row, group] = code
                    sc = T.shfl_sync(sc, 0, width=8)
                    inv = T.shfl_sync(inv, 0, width=8)
                    f0 = T.cast(r0 * inv, T.float4_e2m1fn)
                    f1 = T.cast(r1 * inv, T.float4_e2m1fn)
                    A[grow, p * (K // 2) + byte] = T.cast(T.reinterpret(f0, T.uint8) | (T.reinterpret(f1, T.uint8) << 4), T.int8)
                    # what the tensor core will miss from this pass becomes the next pass's input
                    r0 = r0 - T.cast(f0, f32) * sc
                    r1 = r1 - T.cast(f1, f32) * sc
        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            if lane < CHUNK_WORDS:
                prow, pword = scale_source_coords(pid_m * rows_per_cta + row, pid_k256 * CHUNK_WORDS + lane, K // 64)
                for p in T.unroll(passes):
                    SFA[p, prow, pword] = (T.cast(codes[p, row, lane * 4], T.uint32)
                                           | (T.cast(codes[p, row, lane * 4 + 1], T.uint32) << 8)
                                           | (T.cast(codes[p, row, lane * 4 + 2], T.uint32) << 16)
                                           | (T.cast(codes[p, row, lane * 4 + 3], T.uint32) << 24))

    return A, SFA


@tilelang.jit
def nvfp4_gemm_kernel(A, B, SFA, SFB, alpha, rsinv, passes: int = 2, block_M: int = 128, block_N: int = 128,
                      block_K: int = 256, num_stages: int = 2, out_dtype=T.bfloat16):
    """C (M, N) = alpha[n] * sum_p dq(A_p) @ dq(B)^T, native SM120 block-scaled FP4 MMA, runtime M.

    alpha is per output column, so fused layers whose shards carry different NVFP4 global scales
    (e.g. q|k|v or qkv|z) are dequantised exactly per shard. rsinv is per row (undoes the quantizer's row scale).
    Private: M must be a multiple of 128, N of 128 and K of 256 (use linear())."""
    assert block_M == 128 and block_N == 128 and block_K % 64 == 0
    M = T.dynamic("M")
    N = T.const("N")
    K = T.const("K")
    words = block_K // 64
    k_blocks = K // block_K
    fp4 = T.float4_e2m1fn
    A: T.Tensor[(M, passes * K), fp4]
    B: T.Tensor[(N, K), fp4]
    SFA: T.Tensor[(passes, M * k_blocks, words), T.uint32]
    SFB: T.Tensor[(N * k_blocks, words), T.uint32]
    alpha: T.Tensor[(N,), T.float32]
    rsinv: T.Tensor[(M,), T.float32]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_s = T.alloc_shared((block_M, block_K), fp4)
        B_s = T.alloc_shared((block_N, block_K), fp4)
        SFA_s = T.alloc_shared((block_M, words), T.uint32)
        SFB_s = T.alloc_shared((block_N, words), T.uint32)
        C_l = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_l)
        # passes interleaved per K block: each weight tile is used by every pass back to back (second read hits L2)
        # and only one activation tile needs shared memory per stage
        for it in T.Pipelined(passes * k_blocks, num_stages=num_stages):
            kb = it // passes
            p = it % passes
            T.copy(A[by * block_M, p * K + kb * block_K], A_s)
            T.copy(B[bx * block_N, kb * block_K], B_s)
            for r, w in T.Parallel(block_M, words):
                SFA_s[r, w] = SFA[p, (by * k_blocks + kb) * block_M + r, w]
            for r, w in T.Parallel(block_N, words):
                SFB_s[r, w] = SFB[(bx * k_blocks + kb) * block_N + r, w]
            T.mma_gemm_blockscaled(A_s, B_s, C_l, SFA_s, SFB_s, transpose_B=True, clear_accum=False,
                                   k_start=kb * block_K, sf_a_granularity_k=SF_GRANULARITY_K,
                                   sf_b_granularity_k=SF_GRANULARITY_K, sf_layout="blockscaled_chunk_kmajor")
        for i, j in T.Parallel(block_M, block_N):
            C_l[i, j] = C_l[i, j] * (alpha[bx * block_N + j] * rsinv[by * block_M + i])
        T.copy(C_l, C[by * block_M, bx * block_N])

    return C
