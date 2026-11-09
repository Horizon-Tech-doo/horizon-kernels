# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-split up GEMM with register-owned activation epilogue.

The body acquires the matching gate tile after GEMM and before its epilogue. Gate loads bypass L1;
up values retain the BF16 rounding boundary before SiLU. No serving hook.
"""
from tilelang.contrib.horizon_k4.tasks import *
from tilelang.contrib.horizon_k4.tasks import _hw_loader_src
GATE_SRC = r"""
#ifndef HZ_GATE_F32_DEFINED
#define HZ_GATE_F32_DEFINED
__device__ __forceinline__ float hz_gate_f32(const void* p) {
  unsigned short bits;
  asm volatile("ld.global.cg.b16 %0, [%1];" : "=h"(bits) : "l"(p) : "memory");
  return __uint_as_float(static_cast<unsigned>(bits) << 16);
}
#endif
"""

def gemm_task_hw_activate(tag: str, N: int, K: int, WM: int, split_k: int, S: int, block_k: int = BLOCK_K, timeout_ns: int = 2_000_000_000):
    """(src, pre, body) for one GEMM op: `src` goes into the kernel prelude (after hw_common_src()); `pre(Wg, Sg, bx,
    bk, sb, tid)` issues the weight half of stages 0..S-2 (call before the dependency wait); `body(...)` issues the
    activation half, runs the pipeline and the (split-K) epilogue. Arithmetic identical to gemm_task_body."""
    nblk, kch = BLOCK_N // 32, block_k // 32
    k_tiles = K // block_k
    assert N % BLOCK_N == 0 and K % block_k == 0 and k_tiles % split_k == 0, (N, K, split_k)
    kps = k_tiles // split_k
    assert kps >= S - 1 and S >= 3, (kps, S)
    mt = WM // 16
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=1, block_col_warps=WARPS_N,
                                      warp_row_tiles=WM, warp_col_tiles=32 * NB, chunk=block_k)
    assert split_k == 1
    src = _hw_loader_src(tag, WM, K, block_k) + GATE_SRC
    ldw, lda = f"k4_ldw_{tag}", f"k4_lda_{tag}"

    def _partial_sum(P, i, col):
        acc = P[0, i, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, i, col]
        return acc

    @T.macro
    def pre(Wg, Sg, bx, bk, sb, tid):
        for p in T.unroll(S - 1):
            T.call_extern("handle", ldw, sb[1], sb[2], p, T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), bx,
                          (bk * kps + p) * kch, tid)
            T.call_extern("handle", "xl_commit")

    @T.macro
    def body(x, Y, Wg, Sg, alpha, P, tickets, toff, M, bx, bk, tid, done_s, A_s, Wq_s, S_s, ticket_s, sb, Aout, counters, tile, err, errh, ok):
        q_l = T.alloc_local((NB * 4,), i32)
        s_l = T.alloc_local((NB * 2,), i32)
        B_l = T.alloc_local((NB * 32,), bf16)
        A_l = T.alloc_local((mt * 8,), bf16)
        C_l = T.alloc_local((mt * NB * 2 * 8,), f32)
        stg = T.alloc_local((2,), i32)
        warp = tid // 32
        lane = tid % 32
        wn = warp
        T.clear(C_l)
        for p in T.unroll(S - 1):
            T.call_extern("handle", lda, sb[0], p, T.access_ptr(x, "r"), M, (bk * kps + p) * kch, tid)
        T.call_extern("handle", "xl_commit")
        stg[0] = 0
        stg[1] = S - 1
        for kt in T.serial(kps):
            if kt == 0:
                T.call_extern("handle", "xl_wait<0>")
            else:
                T.call_extern("handle", f"xl_wait<{S - 2}>")
            T.sync_threads()
            if kt + S - 1 < kps:
                T.call_extern("handle", lda, sb[0], stg[1], T.access_ptr(x, "r"), M, (bk * kps + kt + S - 1) * kch,
                              tid)
                T.call_extern("handle", ldw, sb[1], sb[2], stg[1], T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"),
                              bx, (bk * kps + kt + S - 1) * kch, tid)
            T.call_extern("handle", "xl_commit")
            st = stg[0]
            for c in T.unroll(kch):
                for b in T.unroll(NB):
                    for v in T.vectorized(4):
                        q_l[b * 4 + v] = Wq_s[st, wn * NB + b, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[b * 2 + v] = S_s[st, wn * NB + b, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                  T.access_ptr(s_l[b * 2], "r", extent=2), T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    emitter.ldmatrix_a(A_l, A_s[st, 0:WM, 0:block_k], c * 2 + s)
                    for i in T.unroll(mt):
                        for b in T.unroll(NB):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8 + 4, T.bool(False))
            stg[0] = T.if_then_else(stg[0] == S - 1, 0, stg[0] + 1)
            stg[1] = T.if_then_else(stg[1] == S - 1, 0, stg[1] + 1)
        T.call_extern("handle", "xl_wait<0>")
        T.sync_threads()
        if tid == 0:
            ok[0] = 1 - T.call_extern(i32, 'hz_wait_ge', T.access_ptr(counters[tile], 'r'), 1,
                                     T.access_ptr(err[0], 'rw'), T.access_ptr(errh[0], 'rw'), T.int64(timeout_ns))
        T.sync_threads()
        for i in T.unroll(mt):
            for b in T.unroll(NB):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = bx * BLOCK_N + wn * 32 * NB + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M and ok[0] == 1:
                            if split_k == 1:
                                for v in T.vectorized(2):
                                    up = T.cast(C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                    Y[grow, gcol + v] = up
                                    Aout[grow, gcol + v - N // 2] = T.cast(T.call_extern(f32, 'hz_silu_mul_ind',
                                        T.call_extern(f32, 'hz_gate_f32', T.access_ptr(Y[grow, gcol + v - N // 2], 'r')),
                                        T.cast(up, f32)), bf16)
                            else:
                                for v in T.vectorized(2):
                                    P[bk, grow, gcol + v] = C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]
        if split_k == 1:
            if tid == 0:
                done_s[0] = ok[0]
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
                    done_s[0] = ok[0]
        T.sync_threads()

    return src, pre, body

