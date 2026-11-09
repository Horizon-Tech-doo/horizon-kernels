# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""P-ATTN v1: causal prefill attention over vLLM's paged FP8 KV cache (Qwen3.8 full attention: 24 q heads, 4 kv heads,
head_dim 256, page 1600), FlashAttention-2 structure, SM120 bf16 tensor cores.

One CTA = one GQA group (kv head h, its G = 6 query heads) x 16 query tokens of one request: 96 rows, warp w owns the
16 rows of query head h*G + w (FA2: every warp owns whole rows, the softmax never leaves the warp). The KV tile (32
tokens, fp8) is converted to bf16 in shared memory once and serves all 6 heads (GQA reuse). Causal: query token t of
a request with q_len tokens and kv_len keys sees keys j <= kv_len - q_len + t. FP8 -> bf16 is exact; scores, softmax
and the output accumulator are fp32; P is rounded to bf16 for the P @ V product (as FlashInfer / FlashAttention).
Deterministic (no atomics, fixed KV order).

Inputs follow FlashInfer's paged prefill plan: qo_indptr (B+1), kv page lists (kv_indptr / kv_indices / last page len)
are turned into a dense block table + kv lengths by the wrapper; tiles (request, first query token) come from a
host-built tile list (prefill runs eagerly, outside CUDA graphs).
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
LOG2E = 1.4426950408889634


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def attn_prefill_kernel(KVH: int = 4, G: int = 6, D: int = 256, PS: int = 1600, TQ: int = 16, BN: int = 32,
                        share: int = 0, SP: int = 0, ST: int = 0, SH: int = 0):
    """share=1: K and V use ONE shared tile (K for Q K^T, then V for P V), so BN = 64 fits 99 KB"""
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    B = T.dynamic("B")
    MP = T.dynamic("MP")
    NTILE = T.dynamic("NTILE")
    QO = T.dynamic("QO")
    HQ = KVH * G
    R = G * TQ
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    fp8 = T.float8_e4m3fn
    NEG = -T.infinity(f32)
    threads = 32 * G

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16), kc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             vc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             bt: T.Tensor((B, MP), i32), qo: T.Tensor((QO,), i32), kvlen: T.Tensor((B,), i32),
             tiles: T.Tensor((NTILE, 2), i32), out: T.Tensor((NT, HQ, D), bf16), scale_log2: f32, v_scale: f32):
        with T.Kernel(NTILE, KVH, threads=threads) as (ti, h):
            Q_s = T.alloc_shared((R, D), bf16)
            K_s = T.alloc_shared((BN, D), bf16)
            V_s = K_s if share else T.alloc_shared((BN, D), bf16)
            P_s = T.alloc_shared((R, BN), bf16)
            acc_s = T.alloc_fragment((R, BN), f32)
            acc_o = T.alloc_fragment((R, D), f32)
            m_i = T.alloc_fragment((R,), f32)
            m_prev = T.alloc_fragment((R,), f32)
            alpha = T.alloc_fragment((R,), f32)
            l_i = T.alloc_fragment((R,), f32)
            rsum = T.alloc_fragment((R,), f32)
            b = T.alloc_var(i32)
            t0 = T.alloc_var(i32)
            q0 = T.alloc_var(i32)
            ql = T.alloc_var(i32)
            L = T.alloc_var(i32)
            b = tiles[ti, 0]
            t0 = tiles[ti, 1]                          # first query token of the tile inside its request
            q0 = qo[b]
            ql = qo[b + 1] - q0
            L = kvlen[b]
            for i, j in T.Parallel(R, D):
                Q_s[i, j] = T.if_then_else(t0 + i % TQ < ql, q[q0 + T.min(t0 + i % TQ, ql - 1), h * G + i // TQ, j],
                                           T.cast(0.0, bf16))
            T.fill(acc_o, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, NEG)
            # keys needed by the last query token of the tile: j <= L - ql + min(t0 + TQ, ql) - 1
            kend = T.min(L, L - ql + T.min(t0 + TQ, ql))
            for kb in T.serial(T.ceildiv(kend, BN)):
                p0 = kb * BN
                for i, j in T.Parallel(BN, D):
                    K_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(kc[bt[b, T.min(p0 + i, kend - 1) // PS],
                                                                         T.min(p0 + i, kend - 1) % PS, h, j], bf16),
                                               T.cast(0.0, bf16))
                if not share:
                    for i, j in T.Parallel(BN, D):
                        V_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(vc[bt[b, T.min(p0 + i, kend - 1) // PS],
                                                                             T.min(p0 + i, kend - 1) % PS, h, j], bf16),
                                                   T.cast(0.0, bf16))
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(T.And(p0 + j < kend, p0 + j <= L - ql + t0 + i % TQ),
                                                 acc_s[i, j] * scale_log2, NEG)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(R):
                    alpha[i] = T.if_then_else(m_i[i] == NEG, T.float32(1.0), T.exp2(m_prev[i] - m_i[i]))
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(m_i[i] == NEG, T.float32(0.0), T.exp2(acc_s[i, j] - m_i[i]))
                T.reduce_sum(acc_s, rsum, dim=1)
                for i in T.Parallel(R):
                    l_i[i] = l_i[i] * alpha[i] + rsum[i]
                T.copy(acc_s, P_s)
                for i, j in T.Parallel(R, D):
                    acc_o[i, j] *= alpha[i]
                if share:
                    T.sync_threads()
                    for i, j in T.Parallel(BN, D):
                        V_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(vc[bt[b, T.min(p0 + i, kend - 1) // PS],
                                                                             T.min(p0 + i, kend - 1) % PS, h, j], bf16),
                                                   T.cast(0.0, bf16))
                T.gemm(P_s, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if share:
                    T.sync_threads()
            for i, j in T.Parallel(R, D):
                if t0 + i % TQ < ql:
                    out[q0 + t0 + i % TQ, h * G + i // TQ, j] = T.cast(
                        T.if_then_else(l_i[i] > 0, acc_o[i, j] / l_i[i], T.float32(0.0)) * v_scale, bf16)

    return main


def prefill(query, k_cache, v_cache, qo_indptr, block_table, kv_lens, sm_scale_k, v_scale, out=None, TQ=16, BN=32,
            share=0, kv_layout="NHD"):
    """query (NT, 24, 256) bf16 packed by request (qo_indptr); k/v cache (pages, PS, KVH, D) fp8; block_table (B, MP)
    int32 pages per request; kv_lens (B,) int32 (keys incl. this chunk). sm_scale_k = softmax scale * k_scale."""
    import torch
    NT, HQ, D = query.shape
    if kv_layout == "HND":
        k_cache, v_cache = k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)
    _, PS, KVH, _ = k_cache.shape
    SP, ST, SH, _ = k_cache.stride()
    G = HQ // KVH
    qo_cpu = qo_indptr.cpu().tolist()
    tl = [(b, t) for b in range(len(qo_cpu) - 1) for t in range(0, qo_cpu[b + 1] - qo_cpu[b], TQ)]
    tiles = torch.tensor(tl, dtype=torch.int32, device=query.device).view(-1, 2)
    if out is None:
        out = torch.empty_like(query)
    key = (KVH, G, D, PS, TQ, BN, share, SP, ST, SH)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_prefill_kernel(KVH, G, D, PS, TQ, BN, share, SP, ST, SH)
    k(query, k_cache, v_cache, block_table, qo_indptr, kv_lens, tiles, out, float(sm_scale_k) * LOG2E,
      float(v_scale))
    return out


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def attn_prefill_kernel_v2(KVH: int = 4, G: int = 6, D: int = 256, PS: int = 1600, TQ: int = 8, BN: int = 32,
                           SP: int = 0, ST: int = 0, SH: int = 0):
    """v2: the FP8 K / V tiles are prefetched with cp.async into a 2-stage ring (a tile of BN keys lies inside ONE page:
    PS % BN == 0) while the current tile computes; converted to bf16 in shared memory. Rows = G heads x TQ tokens
    (head-major: warp w owns heads 2w, 2w+1 x TQ tokens at TQ = 8), one warp per 16 rows (FA2 row ownership).
    Shared: Q 24 KB + fp8 ring 32 KB + bf16 K, V 32 KB + P 3 KB at TQ 8, BN 32."""
    from tilelang.contrib.horizon_ops.attn_decode import CP_SRC
    assert PS % BN == 0
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    B = T.dynamic("B")
    MP = T.dynamic("MP")
    NTILE = T.dynamic("NTILE")
    QO = T.dynamic("QO")
    HQ = KVH * G
    R = G * TQ
    assert R % 16 == 0
    threads = 32 * (R // 16)
    CH16 = D // 16
    NCP = -(-(BN * CH16) // threads)            # cp.async per thread per tensor (guarded: 512 chunks / 96 threads)
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    fp8 = T.float8_e4m3fn
    NEG = -T.infinity(f32)

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16), kc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             vc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             bt: T.Tensor((B, MP), i32), qo: T.Tensor((QO,), i32), kvlen: T.Tensor((B,), i32),
             tiles: T.Tensor((NTILE, 2), i32), out: T.Tensor((NT, HQ, D), bf16), scale_log2: f32, v_scale: f32):
        with T.Kernel(NTILE, KVH, threads=threads, prelude=CP_SRC) as (ti, h):
            tid = T.get_thread_binding()
            Q_s = T.alloc_shared((R, D), bf16)
            K8_s = T.alloc_shared((2, BN, D), fp8)
            V8_s = T.alloc_shared((2, BN, D), fp8)
            K_s = T.alloc_shared((BN, D), bf16)
            V_s = T.alloc_shared((BN, D), bf16)
            P_s = T.alloc_shared((R, BN), bf16)
            acc_s = T.alloc_fragment((R, BN), f32)
            acc_o = T.alloc_fragment((R, D), f32)
            m_i = T.alloc_fragment((R,), f32)
            m_prev = T.alloc_fragment((R,), f32)
            alpha = T.alloc_fragment((R,), f32)
            l_i = T.alloc_fragment((R,), f32)
            rsum = T.alloc_fragment((R,), f32)
            b = T.alloc_var(i32)
            t0 = T.alloc_var(i32)
            q0 = T.alloc_var(i32)
            ql = T.alloc_var(i32)
            L = T.alloc_var(i32)
            b = tiles[ti, 0]
            t0 = tiles[ti, 1]
            q0 = qo[b]
            ql = qo[b + 1] - q0
            L = kvlen[b]
            for i, j in T.Parallel(R, D):
                Q_s[i, j] = T.if_then_else(t0 + i % TQ < ql, q[q0 + T.min(t0 + i % TQ, ql - 1), h * G + i // TQ, j],
                                           T.cast(0.0, bf16))
            T.fill(acc_o, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, NEG)
            kend = T.min(L, L - ql + T.min(t0 + TQ, ql))
            nblk = T.ceildiv(kend, BN)
            if nblk > 0:
                for c in T.unroll(NCP):
                    e = c * threads + tid
                    if e < BN * CH16:
                        T.call_extern("handle", "hz_cp16", T.address_of(K8_s[0, e // CH16, (e % CH16) * 16]),
                                      T.address_of(kc[bt[b, 0], e // CH16, h, (e % CH16) * 16]))
                        T.call_extern("handle", "hz_cp16", T.address_of(V8_s[0, e // CH16, (e % CH16) * 16]),
                                      T.address_of(vc[bt[b, 0], e // CH16, h, (e % CH16) * 16]))
                T.call_extern("handle", "hz_cp_commit")
            for kb in T.serial(nblk):
                p0 = kb * BN
                if kb + 1 < nblk:
                    for c in T.unroll(NCP):
                        e = c * threads + tid
                        if e < BN * CH16:
                            T.call_extern("handle", "hz_cp16",
                                          T.address_of(K8_s[(kb + 1) % 2, e // CH16, (e % CH16) * 16]),
                                          T.address_of(kc[bt[b, (p0 + BN) // PS], (p0 + BN) % PS + e // CH16, h,
                                                          (e % CH16) * 16]))
                            T.call_extern("handle", "hz_cp16",
                                          T.address_of(V8_s[(kb + 1) % 2, e // CH16, (e % CH16) * 16]),
                                          T.address_of(vc[bt[b, (p0 + BN) // PS], (p0 + BN) % PS + e // CH16, h,
                                                          (e % CH16) * 16]))
                    T.call_extern("handle", "hz_cp_commit")
                    T.call_extern("handle", "hz_cp_wait1")
                else:
                    T.call_extern("handle", "hz_cp_wait0")
                T.sync_threads()
                for i, j in T.Parallel(BN, D):
                    K_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(K8_s[kb % 2, i, j], bf16), T.cast(0.0, bf16))
                for i, j in T.Parallel(BN, D):
                    V_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(V8_s[kb % 2, i, j], bf16), T.cast(0.0, bf16))
                T.sync_threads()
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(T.And(p0 + j < kend, p0 + j <= L - ql + t0 + i % TQ),
                                                 acc_s[i, j] * scale_log2, NEG)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(R):
                    alpha[i] = T.if_then_else(m_i[i] == NEG, T.float32(1.0), T.exp2(m_prev[i] - m_i[i]))
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(m_i[i] == NEG, T.float32(0.0), T.exp2(acc_s[i, j] - m_i[i]))
                T.reduce_sum(acc_s, rsum, dim=1)
                for i in T.Parallel(R):
                    l_i[i] = l_i[i] * alpha[i] + rsum[i]
                T.copy(acc_s, P_s)
                for i, j in T.Parallel(R, D):
                    acc_o[i, j] *= alpha[i]
                T.gemm(P_s, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(R, D):
                if t0 + i % TQ < ql:
                    out[q0 + t0 + i % TQ, h * G + i // TQ, j] = T.cast(
                        T.if_then_else(l_i[i] > 0, acc_o[i, j] / l_i[i], T.float32(0.0)) * v_scale, bf16)

    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def attn_prefill_kernel_v3(KVH: int = 4, G: int = 6, D: int = 256, PS: int = 1600, TQ: int = 16, BN: int = 32,
                           SP: int = 0, ST: int = 0, SH: int = 0):
    """v3: 96 rows (6 heads x 16 tokens, 6 warps) like v1, ONE fp8 stage (16 KB) and ONE bf16 K/V tile (K, then V):
    the next tile's fp8 prefetch is issued right after V is converted and overlaps the P V product. 86 KB shared.
    (v2 docstring follows) v2: the FP8 K / V tiles are prefetched with cp.async into a 2-stage ring (a tile of BN keys lies inside ONE page:
    PS % BN == 0) while the current tile computes; converted to bf16 in shared memory. Rows = G heads x TQ tokens
    (head-major: warp w owns heads 2w, 2w+1 x TQ tokens at TQ = 8), one warp per 16 rows (FA2 row ownership).
    Shared: Q 24 KB + fp8 ring 32 KB + bf16 K, V 32 KB + P 3 KB at TQ 8, BN 32."""
    from tilelang.contrib.horizon_ops.attn_decode import CP_SRC
    assert PS % BN == 0
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    B = T.dynamic("B")
    MP = T.dynamic("MP")
    NTILE = T.dynamic("NTILE")
    QO = T.dynamic("QO")
    HQ = KVH * G
    R = G * TQ
    assert R % 16 == 0
    threads = 32 * (R // 16)
    CH16 = D // 16
    NCP = -(-(BN * CH16) // threads)            # cp.async per thread per tensor (guarded: 512 chunks / 96 threads)
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    fp8 = T.float8_e4m3fn
    NEG = -T.infinity(f32)

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16), kc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             vc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             bt: T.Tensor((B, MP), i32), qo: T.Tensor((QO,), i32), kvlen: T.Tensor((B,), i32),
             tiles: T.Tensor((NTILE, 2), i32), out: T.Tensor((NT, HQ, D), bf16), scale_log2: f32, v_scale: f32):
        with T.Kernel(NTILE, KVH, threads=threads, prelude=CP_SRC) as (ti, h):
            tid = T.get_thread_binding()
            Q_s = T.alloc_shared((R, D), bf16)
            K8_s = T.alloc_shared((1, BN, D), fp8)
            V8_s = T.alloc_shared((1, BN, D), fp8)
            K_s = T.alloc_shared((BN, D), bf16)
            V_s = K_s
            P_s = T.alloc_shared((R, BN), bf16)
            acc_s = T.alloc_fragment((R, BN), f32)
            acc_o = T.alloc_fragment((R, D), f32)
            m_i = T.alloc_fragment((R,), f32)
            m_prev = T.alloc_fragment((R,), f32)
            alpha = T.alloc_fragment((R,), f32)
            l_i = T.alloc_fragment((R,), f32)
            rsum = T.alloc_fragment((R,), f32)
            b = T.alloc_var(i32)
            t0 = T.alloc_var(i32)
            q0 = T.alloc_var(i32)
            ql = T.alloc_var(i32)
            L = T.alloc_var(i32)
            b = tiles[ti, 0]
            t0 = tiles[ti, 1]
            q0 = qo[b]
            ql = qo[b + 1] - q0
            L = kvlen[b]
            for i, j in T.Parallel(R, D):
                Q_s[i, j] = T.if_then_else(t0 + i % TQ < ql, q[q0 + T.min(t0 + i % TQ, ql - 1), h * G + i // TQ, j],
                                           T.cast(0.0, bf16))
            T.fill(acc_o, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, NEG)
            kend = T.min(L, L - ql + T.min(t0 + TQ, ql))
            nblk = T.ceildiv(kend, BN)
            if nblk > 0:
                for c in T.unroll(NCP):
                    e = c * threads + tid
                    if e < BN * CH16:
                        T.call_extern("handle", "hz_cp16", T.address_of(K8_s[0, e // CH16, (e % CH16) * 16]),
                                      T.address_of(kc[bt[b, 0], e // CH16, h, (e % CH16) * 16]))
                        T.call_extern("handle", "hz_cp16", T.address_of(V8_s[0, e // CH16, (e % CH16) * 16]),
                                      T.address_of(vc[bt[b, 0], e // CH16, h, (e % CH16) * 16]))
                T.call_extern("handle", "hz_cp_commit")
            for kb in T.serial(nblk):
                p0 = kb * BN
                T.call_extern("handle", "hz_cp_wait0")
                T.sync_threads()
                for i, j in T.Parallel(BN, D):
                    K_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(K8_s[0, i, j], bf16), T.cast(0.0, bf16))
                T.sync_threads()
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(T.And(p0 + j < kend, p0 + j <= L - ql + t0 + i % TQ),
                                                 acc_s[i, j] * scale_log2, NEG)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(R):
                    alpha[i] = T.if_then_else(m_i[i] == NEG, T.float32(1.0), T.exp2(m_prev[i] - m_i[i]))
                for i, j in T.Parallel(R, BN):
                    acc_s[i, j] = T.if_then_else(m_i[i] == NEG, T.float32(0.0), T.exp2(acc_s[i, j] - m_i[i]))
                T.reduce_sum(acc_s, rsum, dim=1)
                for i in T.Parallel(R):
                    l_i[i] = l_i[i] * alpha[i] + rsum[i]
                T.copy(acc_s, P_s)
                for i, j in T.Parallel(R, D):
                    acc_o[i, j] *= alpha[i]
                T.sync_threads()                                  # every warp finished reading K from K_s
                for i, j in T.Parallel(BN, D):
                    V_s[i, j] = T.if_then_else(p0 + i < kend, T.cast(V8_s[0, i, j], bf16), T.cast(0.0, bf16))
                T.sync_threads()                                  # the fp8 stage is free: prefetch the next tile
                if kb + 1 < nblk:
                    for c in T.unroll(NCP):
                        e = c * threads + tid
                        if e < BN * CH16:
                            T.call_extern("handle", "hz_cp16", T.address_of(K8_s[0, e // CH16, (e % CH16) * 16]),
                                          T.address_of(kc[bt[b, (p0 + BN) // PS], (p0 + BN) % PS + e // CH16, h,
                                                          (e % CH16) * 16]))
                            T.call_extern("handle", "hz_cp16", T.address_of(V8_s[0, e // CH16, (e % CH16) * 16]),
                                          T.address_of(vc[bt[b, (p0 + BN) // PS], (p0 + BN) % PS + e // CH16, h,
                                                          (e % CH16) * 16]))
                    T.call_extern("handle", "hz_cp_commit")
                T.gemm(P_s, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.sync_threads()
            for i, j in T.Parallel(R, D):
                if t0 + i % TQ < ql:
                    out[q0 + t0 + i % TQ, h * G + i // TQ, j] = T.cast(
                        T.if_then_else(l_i[i] > 0, acc_o[i, j] / l_i[i], T.float32(0.0)) * v_scale, bf16)

    return main


def prefill_v2(query, k_cache, v_cache, qo_indptr, block_table, kv_lens, sm_scale_k, v_scale, out=None, TQ=8, BN=32,
               kv_layout="NHD"):
    """same contract as prefill(); cp.async double-buffered fp8 tiles (v2)"""
    import torch
    NT, HQ, D = query.shape
    if kv_layout == "HND":
        k_cache, v_cache = k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)
    _, PS, KVH, _ = k_cache.shape
    SP, ST, SH, _ = k_cache.stride()
    G = HQ // KVH
    qo_cpu = qo_indptr.cpu().tolist()
    tl = [(b, t) for b in range(len(qo_cpu) - 1) for t in range(0, qo_cpu[b + 1] - qo_cpu[b], TQ)]
    tiles = torch.tensor(tl, dtype=torch.int32, device=query.device).view(-1, 2)
    if out is None:
        out = torch.empty_like(query)
    key = ("v2", KVH, G, D, PS, TQ, BN, SP, ST, SH)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_prefill_kernel_v2(KVH, G, D, PS, TQ, BN, SP, ST, SH)
    k(query, k_cache, v_cache, block_table, qo_indptr, kv_lens, tiles, out, float(sm_scale_k) * LOG2E,
      float(v_scale))
    return out


def prefill_v3(query, k_cache, v_cache, qo_indptr, block_table, kv_lens, sm_scale_k, v_scale, out=None, TQ=16, BN=32,
               kv_layout="NHD"):
    """same contract as prefill(); v3 (96 rows, one fp8 stage, prefetch overlapping P V)"""
    import torch
    NT, HQ, D = query.shape
    if kv_layout == "HND":
        k_cache, v_cache = k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)
    _, PS, KVH, _ = k_cache.shape
    SP, ST, SH, _ = k_cache.stride()
    G = HQ // KVH
    qo_cpu = qo_indptr.cpu().tolist()
    tl = [(b, t) for b in range(len(qo_cpu) - 1) for t in range(0, qo_cpu[b + 1] - qo_cpu[b], TQ)]
    tiles = torch.tensor(tl, dtype=torch.int32, device=query.device).view(-1, 2)
    if out is None:
        out = torch.empty_like(query)
    key = ("v3", KVH, G, D, PS, TQ, BN, SP, ST, SH)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_prefill_kernel_v3(KVH, G, D, PS, TQ, BN, SP, ST, SH)
    k(query, k_cache, v_cache, block_table, qo_indptr, kv_lens, tiles, out, float(sm_scale_k) * LOG2E,
      float(v_scale))
    return out
