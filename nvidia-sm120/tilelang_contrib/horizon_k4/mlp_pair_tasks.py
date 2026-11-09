# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Paired gate/up GEMM: two N tiles per warp share each loaded A fragment.

Gate and up weight tiles are gathered into one staged arena; each is rounded to
BF16 before activation. Single-split GU only. No output arithmetic reassociation.
"""
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter
from tilelang.contrib.horizon_k4.tasks import THREADS, BLOCK_K, WARPS_N


def gemm_pair_alloc(wm):
    a = T.alloc_shared((wm, BLOCK_K), T.bfloat16)
    w = T.alloc_shared((8, 2, 128), T.int32)
    s = T.alloc_shared((8, 2, 16), T.int32)
    ticket = T.alloc_shared((1,), T.int32)
    T.annotate_layout({a: make_mma_swizzle_layout(a)})
    return a, w, s, ticket


def gemm_pair_task_body(I: int, K: int, WM: int, num_stages: int, direct=False):
    """One (bx = 128-column tile, bk = split) task of Y = x @ W^T for a regfrag-prepared NVFP4 weight (Wg, Sg, alpha).

    Arithmetic identical to decode_regfrag / regfrag_xl with the same split_k (module docstring). Rows: M <= WM (one m
    tile; padding rows wrap to row (i % M) and are never stored).
    split_k > 1: fp32 partials P[bk] (alpha applied), a ticket per tile at tickets[toff + bx] (self-resetting); the last task of
    a tile sums P[0..split_k-1] in order into Y. done_s[0] = 1 on the task that completed the tile (the only one that
    may signal the tile's consumers), else 0.
    Shared buffers come from the caller (gemm_alloc), so every GEMM op of a K4 kernel uses the same stage set."""
    N, NB, BLOCK_N, split_k = 2 * I, 2, 256, 1
    nblk = BLOCK_N // 32
    kchunks = BLOCK_K // 32
    k_tiles = K // BLOCK_K
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and k_tiles % split_k == 0, (N, K, split_k)
    assert WM % 16 == 0
    kt_per_split = k_tiles // split_k
    mt = WM // 16
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=1, block_col_warps=WARPS_N,
                                      warp_row_tiles=WM, warp_col_tiles=32 * NB, chunk=BLOCK_K)

    def _partial_sum(P, i, col):
        acc = P[0, i, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, i, col]
        return acc

    @T.macro
    def body(x, Y, Wg, Sg, alpha, P, tickets, toff, M, bx, bk, tid, done_s, A_s, Wq_s, S_s, ticket_s, activated):
        q_l = T.alloc_local((NB * 4,), i32)
        s_l = T.alloc_local((NB * 2,), i32)
        B_l = T.alloc_local((NB * 32,), bf16)
        A_l = T.alloc_local((mt * 8,), bf16)
        C_l = T.alloc_local((mt * NB * 2 * 8,), f32)
        warp = tid // 32
        lane = tid % 32
        wn = warp
        T.clear(C_l)
        for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
            kc0 = (bk * kt_per_split + kt) * kchunks
            for i, kk in T.Parallel(WM, BLOCK_K):
                A_s[i, kk] = x[i % M, kc0 * 32 + kk]
            T.copy(Wg[bx * 4, kc0, 0], Wq_s[0:4, 0:kchunks, 0:128])
            T.copy(Wg[I // 32 + bx * 4, kc0, 0], Wq_s[4:8, 0:kchunks, 0:128])
            T.copy(Sg[bx * 4, kc0, 0], S_s[0:4, 0:kchunks, 0:16])
            T.copy(Sg[I // 32 + bx * 4, kc0, 0], S_s[4:8, 0:kchunks, 0:16])
            for c in T.unroll(kchunks):
                for b in T.unroll(NB):
                    for v in T.vectorized(4):
                        q_l[b * 4 + v] = Wq_s[wn + b * 4, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[b * 2 + v] = S_s[wn + b * 4, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                  T.access_ptr(s_l[b * 2], "r", extent=2),
                                  T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                    for i in T.unroll(mt):
                        for b in T.unroll(NB):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8 + 4, T.bool(False))
        for i in T.unroll(mt):
            for b in T.unroll(NB):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = bx * 128 + wn * 32 + b * I + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M:
                            if split_k == 1:
                                for v in T.vectorized(2):
                                    Y[grow, gcol + v] = T.cast(
                                        C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                if direct and b == 0:
                                    for v in T.unroll(2):
                                        gate = T.cast(C_l[((i * NB) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                        up = T.cast(C_l[((i * NB + 1) * 2 + j) * 8 + r2 * 2 + v] * alpha[I + gcol + v], bf16)
                                        activated[grow, gcol + v] = T.cast(T.call_extern(f32, "hz_silu_mul_ind",
                                            T.cast(gate, f32), T.cast(up, f32)), bf16)
                            else:
                                for v in T.vectorized(2):
                                    P[bk, grow, gcol + v] = C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]
        if split_k == 1:
            if tid == 0:
                done_s[0] = 1
        else:
            T.call_extern("handle", "__threadfence")
            T.sync_threads()
            if tid == 0:
                ticket_s[0] = T.atomic_add(tickets[toff + bx], 1, return_prev=True)
            T.sync_threads()
            if tid == 0:
                done_s[0] = 0
            if ticket_s[0] == split_k - 1:
                T.call_extern("handle", "__threadfence")
                for i, j in T.Parallel(WM, BLOCK_N):
                    if i < M:
                        Y[i, bx * BLOCK_N + j] = T.cast(_partial_sum(P, i, bx * BLOCK_N + j), bf16)
                if tid == 0:
                    tickets[toff + bx] = 0
                    done_s[0] = 1
        T.sync_threads()

    return body
