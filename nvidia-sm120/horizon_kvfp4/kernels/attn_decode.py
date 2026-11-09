# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""A1: split-KV decode attention over vLLM's paged FP8 KV cache (Qwen3.8 full-attention layers), SM120.

Drop-in for flashinfer.decode.xqa_batch_decode_with_kv_cache as vLLM 0.29 calls it for decode batches (NHD layout,
k/v caches (pages, page_size, kv_heads, head_dim) float8_e4m3fn, uniform q_len_per_req, causal draft mask):
    out[b*QL + t, h*G + g] = softmax_j( q . k_j * bmm1_scale ) @ v_j * bmm2_scale,  j < seq_len[b] - QL + t + 1
FP8 K/V are converted to bf16 exactly (every e4m3 value is a bf16 value); scores and the softmax are fp32, P is bf16
for the P @ V tensor-core product (as in FlashAttention); the split-KV combine runs in a fixed order, so the result is
deterministic and batch-invariant (a request's output does not depend on the other requests in the batch).

Two layers (K4 reuse, contract of horizon_ops):
  attn_split_body(...)(q, kc, vc, bt, sl, opart, lse, b, h, s, chunk, scale_log2, tid, Q_s, K_s, V_s)
      one task = (request b, kv head h, KV split s): the G*QL query rows of the GQA group against KV positions
      [s*chunk, (s+1)*chunk) -> normalised partial output (fp32) + log2-sum-exp per row.
  attn_combine_body(...)(opart, lse, out, b, h, S, v_scale, tid)
      one task = (request b, kv head h): fixed-order merge of the S partials -> bf16 output rows.
DEPS: attn_split(b, h, s): q rows of request b (q/k-norm + RoPE output), K/V pages of request b written for all
      positions < seq_len (the step's KV-cache write); attn_combine(b, h): attn_split(b, h, 0..S-1).
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
LOG2E = 1.4426950408889634

CP_SRC = r"""
#ifndef HZ_CP_SRC_DEFINED
#define HZ_CP_SRC_DEFINED
__device__ __forceinline__ void hz_cp16(void* s, const void* g) {
  unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g) : "memory");
}
__device__ __forceinline__ void hz_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void hz_cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void hz_cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
#endif
"""

PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def attn_split_body(KVH: int, G: int, QL: int, D: int, PS: int, block_N: int = 64, BH: int = 32, min_chunk: int = 256,
                    pipe: int = 0, stages: int = 2):
    """pipe=1: the fp8 K / V tiles (block_N positions inside ONE page: PS % block_N == 0) are staged with cp.async in a
    `stages`-deep pipeline (extra shared K8_s, V8_s passed in) and converted to bf16 in shared memory; pipe=0: direct
    loads with the conversion (the first version)."""
    assert not pipe or PS % block_N == 0
    """T.macro: one task = (request b, kv head h, split s of S). Shared: Q_s (BH, D) bf16, K_s / V_s (block_N, D) bf16,
    P_s (BH, block_N) bf16 = 80 KB at block_N 64. Writes opart[b, h, s] (normalised partial) and lse[b, h, s]."""
    R = G * QL
    assert R <= BH
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    NEG = -T.infinity(f32)

    @T.macro
    def body(q, kc, vc, bt, sl, opart, lse, b, h, s, S, scale_log2, tid, Q_s, K_s, V_s, P_s, K8_s=None, V8_s=None):
        acc_s = T.alloc_fragment((BH, block_N), f32)
        acc_o = T.alloc_fragment((BH, D), f32)
        m_i = T.alloc_fragment((BH,), f32)
        m_prev = T.alloc_fragment((BH,), f32)
        alpha = T.alloc_fragment((BH,), f32)
        l_i = T.alloc_fragment((BH,), f32)
        rsum = T.alloc_fragment((BH,), f32)
        L = T.alloc_var(i32)
        st = T.alloc_var(i32)
        en = T.alloc_var(i32)
        chunk = T.alloc_var(i32)
        L = sl[b]
        chunk = T.max(min_chunk, T.ceildiv(T.ceildiv(L, S), block_N) * block_N)
        st = s * chunk
        en = T.min(L, st + chunk)
        for i, j in T.Parallel(BH, D):
            Q_s[i, j] = T.if_then_else(i < R, q[b * QL + i // G, h * G + i % G, j], T.cast(0.0, bf16))
        T.fill(acc_o, 0.0)
        T.fill(l_i, 0.0)
        T.fill(m_i, NEG)
        nblk = T.ceildiv(T.max(en - st, 0), block_N)
        CH16 = D // 16                                   # 16-byte chunks per fp8 row
        NCP = block_N * CH16 // 128                      # cp.async per thread per tensor (128 threads)
        if pipe:
            if nblk > 0:                                 # prologue: tile 0 -> stage 0
                for c in T.unroll(NCP):
                    e = c * 128 + tid
                    T.call_extern("handle", "hz_cp16", T.address_of(K8_s[0, e // CH16, (e % CH16) * 16]),
                                  T.address_of(kc[bt[b, st // PS], st % PS + e // CH16, h, (e % CH16) * 16]))
                    T.call_extern("handle", "hz_cp16", T.address_of(V8_s[0, e // CH16, (e % CH16) * 16]),
                                  T.address_of(vc[bt[b, st // PS], st % PS + e // CH16, h, (e % CH16) * 16]))
                T.call_extern("handle", "hz_cp_commit")
        for kb in T.serial(nblk):
            p0 = st + kb * block_N
            if pipe:
                if kb + 1 < nblk:                        # prefetch tile kb + 1 into the other stage
                    for c in T.unroll(NCP):
                        e = c * 128 + tid
                        T.call_extern("handle", "hz_cp16",
                                      T.address_of(K8_s[(kb + 1) % 2, e // CH16, (e % CH16) * 16]),
                                      T.address_of(kc[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e // CH16, h,
                                                      (e % CH16) * 16]))
                        T.call_extern("handle", "hz_cp16",
                                      T.address_of(V8_s[(kb + 1) % 2, e // CH16, (e % CH16) * 16]),
                                      T.address_of(vc[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e // CH16, h,
                                                      (e % CH16) * 16]))
                    T.call_extern("handle", "hz_cp_commit")
                    T.call_extern("handle", "hz_cp_wait1")
                else:
                    T.call_extern("handle", "hz_cp_wait0")
                T.sync_threads()
                for i, j in T.Parallel(block_N, D):
                    K_s[i, j] = T.if_then_else(p0 + i < en, T.cast(K8_s[kb % 2, i, j], bf16), T.cast(0.0, bf16))
                for i, j in T.Parallel(block_N, D):
                    V_s[i, j] = T.if_then_else(p0 + i < en, T.cast(V8_s[kb % 2, i, j], bf16), T.cast(0.0, bf16))
                T.sync_threads()
            else:
                for i, j in T.Parallel(block_N, D):
                    K_s[i, j] = T.if_then_else(p0 + i < en,
                                               T.cast(kc[bt[b, T.min(p0 + i, en - 1) // PS],
                                                         T.min(p0 + i, en - 1) % PS, h, j], bf16),
                                               T.cast(0.0, bf16))
                for i, j in T.Parallel(block_N, D):
                    V_s[i, j] = T.if_then_else(p0 + i < en,
                                               T.cast(vc[bt[b, T.min(p0 + i, en - 1) // PS],
                                                         T.min(p0 + i, en - 1) % PS, h, j], bf16),
                                               T.cast(0.0, bf16))
            T.clear(acc_s)
            T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(BH, block_N):
                # causal among the QL window tokens: row token t sees positions < L - QL + t + 1
                acc_s[i, j] = T.if_then_else(
                    T.And(i < R, T.And(p0 + j < en, p0 + j < L - QL + i // G + 1)),
                    acc_s[i, j] * scale_log2, NEG)
            T.copy(m_i, m_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for i in T.Parallel(BH):
                alpha[i] = T.if_then_else(m_i[i] == NEG, T.float32(1.0), T.exp2(m_prev[i] - m_i[i]))
            for i, j in T.Parallel(BH, block_N):
                acc_s[i, j] = T.if_then_else(m_i[i] == NEG, T.float32(0.0), T.exp2(acc_s[i, j] - m_i[i]))
            T.reduce_sum(acc_s, rsum, dim=1)
            for i in T.Parallel(BH):
                l_i[i] = l_i[i] * alpha[i] + rsum[i]
            T.copy(acc_s, P_s)
            for i, j in T.Parallel(BH, D):
                acc_o[i, j] *= alpha[i]
            T.gemm(P_s, V_s, acc_o, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BH, D):
            if i < R:
                opart[b, h, s, i, j] = T.if_then_else(l_i[i] > 0, acc_o[i, j] / l_i[i], T.float32(0.0))
        for i in T.Parallel(BH):
            if i < R:
                lse[b, h, s, i] = T.if_then_else(l_i[i] > 0, T.log2(l_i[i]) + m_i[i], NEG)

    return body


@tilelang.jit(pass_configs=PASS_CONFIGS)
def attn_split_kernel(KVH: int, G: int, QL: int, D: int, PS: int, block_N: int = 64, BH: int = 32,
                      threads: int = 128, min_chunk: int = 256, SP: int = 0, ST: int = 0, SH: int = 0,
                      pipe: int = 0):
    """grid (B, KVH, S). Each request splits its OWN length: chunk = max(min_chunk, ceil(L / S)) rounded to block_N,
    so a CUDA graph captured for one S serves every context length (splits past the end exit at once).
    k/v cache views are indexed [page, position, kv head, d] with strides (SP, ST, SH, 1): NHD tensors as they are,
    vLLM's HND cache (K | V interleaved per token) through a permuted view (see decode())."""
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    NP = T.dynamic("NP")
    MP = T.dynamic("MP")
    S = T.dynamic("S")
    HQ = KVH * G
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    fp8 = T.float8_e4m3fn
    body = attn_split_body(KVH, G, QL, D, PS, block_N, BH, min_chunk, pipe)

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16),
             kc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             vc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), fp8),
             bt: T.Tensor((B, MP), i32),
             sl: T.Tensor((B,), i32),
             opart: T.Tensor((B, KVH, S, BH, D), f32),
             lse: T.Tensor((B, KVH, S, BH), f32),
             scale_log2: f32):
        with T.Kernel(B, KVH, S, threads=threads, prelude=CP_SRC) as (b, h, s):
            Q_s = T.alloc_shared((BH, D), bf16)
            K_s = T.alloc_shared((block_N, D), bf16)
            V_s = T.alloc_shared((block_N, D), bf16)
            P_s = T.alloc_shared((BH, block_N), bf16)
            if pipe:
                K8_s = T.alloc_shared((2, block_N, D), fp8)
                V8_s = T.alloc_shared((2, block_N, D), fp8)
                body(q, kc, vc, bt, sl, opart, lse, b, h, s, S, scale_log2, T.get_thread_binding(), Q_s, K_s, V_s, P_s,
                     K8_s, V8_s)
            else:
                body(q, kc, vc, bt, sl, opart, lse, b, h, s, S, scale_log2, T.get_thread_binding(), Q_s, K_s, V_s, P_s)

    return main


def attn_combine_body(KVH: int, G: int, QL: int, D: int, BH: int = 32, threads: int = 128):
    """T.macro: one task = (request b, kv head h): the G * QL rows of the group, fixed-order merge of the S split
    partials (max, then sum of weights, then the weighted sum, all in split order) -> bf16 output. No shared memory."""
    R = G * QL
    per = D // threads
    assert D % threads == 0
    bf16, f32 = T.bfloat16, T.float32
    NEG = -T.infinity(f32)

    @T.macro
    def body(opart, lse, out, b, h, S, v_scale, tid):
        mx = T.alloc_var(f32)
        tot = T.alloc_var(f32)
        w = T.alloc_var(f32)
        acc = T.alloc_local((per,), f32)
        for r in T.serial(R):
            mx = NEG
            for k in T.serial(S):
                mx = T.max(mx, lse[b, h, k, r])
            tot = 0.0
            for k in T.serial(S):
                tot = tot + T.if_then_else(lse[b, h, k, r] == NEG, T.float32(0.0), T.exp2(lse[b, h, k, r] - mx))
            for e in T.unroll(per):
                acc[e] = 0.0
            for k in T.serial(S):
                w = T.if_then_else(lse[b, h, k, r] == NEG, T.float32(0.0), T.exp2(lse[b, h, k, r] - mx) / tot)
                for e in T.unroll(per):
                    acc[e] = acc[e] + opart[b, h, k, r, e * threads + tid] * w
            for e in T.unroll(per):
                out[b * QL + r // G, h * G + r % G, e * threads + tid] = T.cast(acc[e] * v_scale, bf16)

    return body


@tilelang.jit(pass_configs=PASS_CONFIGS)
def attn_combine_kernel(KVH: int, G: int, QL: int, D: int, BH: int = 32, threads: int = 256):
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    S = T.dynamic("S")
    HQ = KVH * G
    R = G * QL
    bf16, f32 = T.bfloat16, T.float32
    NEG = -T.infinity(f32)
    assert D % threads == 0 or threads % D == 0

    @T.prim_func
    def main(opart: T.Tensor((B, KVH, S, BH, D), f32), lse: T.Tensor((B, KVH, S, BH), f32),
             out: T.Tensor((NT, HQ, D), bf16), v_scale: f32):
        with T.Kernel(B * KVH, R, threads=threads) as (bh, r):
            b = bh // KVH
            h = bh % KVH
            tid = T.get_thread_binding()
            mx = T.alloc_var(f32)
            tot = T.alloc_var(f32)
            w = T.alloc_var(f32)
            acc = T.alloc_local((D // threads if D >= threads else 1,), f32)
            mx = NEG
            for k in T.serial(S):
                mx = T.max(mx, lse[b, h, k, r])
            tot = 0.0
            for k in T.serial(S):
                tot = tot + T.if_then_else(lse[b, h, k, r] == NEG, T.float32(0.0), T.exp2(lse[b, h, k, r] - mx))
            for e in T.unroll(D // threads if D >= threads else 1):
                acc[e] = 0.0
            for k in T.serial(S):
                w = T.if_then_else(lse[b, h, k, r] == NEG, T.float32(0.0), T.exp2(lse[b, h, k, r] - mx) / tot)
                for e in T.unroll(D // threads if D >= threads else 1):
                    if e * threads + tid < D:
                        acc[e] = acc[e] + opart[b, h, k, r, e * threads + tid] * w
            for e in T.unroll(D // threads if D >= threads else 1):
                if e * threads + tid < D:
                    out[b * QL + r // G, h * G + r % G, e * threads + tid] = T.cast(acc[e] * v_scale, bf16)

    return main


def get_split(KVH, G, QL, D, PS, block_N=64, BH=32, threads=128, min_chunk=256, SP=0, ST=0, SH=0, pipe=0):
    key = ("s", KVH, G, QL, D, PS, block_N, BH, threads, min_chunk, SP, ST, SH, pipe)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_split_kernel(KVH, G, QL, D, PS, block_N, BH, threads, min_chunk, SP, ST, SH, pipe)
    return k


def get_combine(KVH, G, QL, D, BH=32, threads=256):
    key = ("c", KVH, G, QL, D, BH, threads)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_combine_kernel(KVH, G, QL, D, BH, threads)
    return k


_WS = {}


def _workspace(device, B, KVH, S, BH, D):
    import torch
    need = B * KVH * S * BH * (D + 1)
    ws = _WS.get(device)
    if ws is None or ws.numel() < need:
        # 64 MiB covers every batch up to 256 requests (splits shrink as the batch grows): allocated once, outside
        # CUDA-graph capture, so captured graphs never own the workspace
        ws = _WS[device] = torch.empty(max(need, 1 << 24), dtype=torch.float32, device=device)
    op = ws[: B * KVH * S * BH * D].view(B, KVH, S, BH, D)
    ls = ws[B * KVH * S * BH * D: need].view(B, KVH, S, BH)
    return op, ls


def plan_splits(B, KVH, target_ctas=188, max_splits=64):
    """splits per (request, kv head): one wave of CTAs (the kernel sizes each request's chunks from its own length)"""
    return max(1, min(max_splits, target_ctas // max(1, B * KVH)))


def decode(query, k_cache, v_cache, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale, q_len_per_req,
           out=None, block_N=64, splits=None, kv_layout="NHD", pipe=0):
    """query (B*QL, HQ, D) bf16; k/v_cache (pages, PS, KVH, D) fp8 e4m3fn (NHD); returns out (B*QL, HQ, D) bf16.
    max_seq_len is unused (kept for the XQA signature): splits are sized per request inside the kernel."""
    import torch
    NT, HQ, D = query.shape
    if kv_layout == "HND":             # (pages, kv heads, page size, D) -> a [page, position, head, d] view
        k_cache, v_cache = k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(3) == 1
    _, PS, KVH, _ = k_cache.shape
    SP, ST, SH, _ = k_cache.stride()
    QL = int(q_len_per_req or 1)
    B = NT // QL
    G = HQ // KVH
    if out is None:
        out = torch.empty_like(query)
    S = splits or plan_splits(B, KVH)
    op, ls = _workspace(query.device, B, KVH, S, 32, D)
    if pipe and PS % block_N:
        pipe = 0
    get_split(KVH, G, QL, D, PS, block_N, SP=SP, ST=ST, SH=SH, pipe=pipe)(query, k_cache, v_cache, block_tables, seq_lens, op, ls,
                                          float(bmm1_scale) * LOG2E)
    get_combine(KVH, G, QL, D)(op, ls, out, float(bmm2_scale))
    return out
