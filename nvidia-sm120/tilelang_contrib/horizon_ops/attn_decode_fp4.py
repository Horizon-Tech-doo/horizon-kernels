# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode attention over the Horizon NVFP4 paged KV cache (hz-kvfp4 v1, kv_fp4_ref.py), SM120.

Same structure and contract as attn_decode.py (the FP8 version), with the cache read changed:
    out[b*QL + t, h*G + g] = softmax_j( q . k_j * bmm1_scale * gk[h] ) @ v_j * gv[h] * bmm2_scale,
    j < seq_len[b] - QL + t + 1                      (QL = 1 .. 4: MTP's verify tokens, causal among themselves)
k_j / v_j = e2m1(code) * e4m3(block scale): the product has at most 5 significant bits, so the bf16 tile the tensor
cores see is EXACT; the per-(kv head) global scales gk / gv are folded into the fp32 score scale and the output.
Scores, the online softmax and the accumulator are fp32, P is bf16 for the P @ V product (FlashAttention), the split
combine is fixed-order (attn_decode.attn_combine_kernel is reused): deterministic and batch-invariant.
If the keys were written rotated (kv_write_fp4 with signs), pass q' = rotate_q(q, signs) (same FWHT, one bf16
rounding; kv_fp4_ref.rotate_q is its bit-exact reference).

Bytes read per (token, kv head): 2 * (D/2 + D/16) = 288 at D = 256 (FP8: 512): the decode is memory bound at long
context, so the ideal speed-up over our FP8 kernel is 512 / 288 = 1.78x at equal efficiency (not measured).

Two layers (K4 reuse, contract of horizon_ops):
  attn_fp4_split_body(...)(q, kd, ksf, vd, vsf, bt, sl, gk, gv, opart, lse, b, h, s, S, scale_log2, tid, smem...)
      one task = (request b, kv head h, KV split s) -> normalised partial output (fp32, times gv[h]) + log2-sum-exp
  attn_decode.attn_combine_body / attn_combine_kernel: unchanged (v_scale = bmm2_scale)
  rotate_q_kernel: q (NT, HQ, D) bf16 -> FWHT(q * signs) * D^-0.5 rounded to bf16
DEPS: as attn_decode.py; rotate_q(t) before attn_fp4_split(b, .) when keys are rotated.
"""

import tilelang
import tilelang.language as T

from .attn_decode import CP_SRC, LOG2E, get_combine, plan_splits, _workspace
from .kv_write_fp4 import FP4_SRC, fwht_body, fwht_scale

_KERNELS = {}
BLOCK = 16
PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _fp4_val(byte, j, sf):
    """TIR: value of element j (low nibble if j even) of a packed byte times its e4m3 block scale (exact in fp32):
    2 * |e2m1| = {0,1,2,3,4,6,8,12}[code & 7] = m (e == 0) else (2 + m) << (e - 1), e = code >> 1 & 3, m = code & 1"""
    i32, f32 = T.int32, T.float32
    c = T.bitwise_and(T.shift_right(T.cast(byte, i32), (j % 2) * 4), 15)
    e = T.bitwise_and(T.shift_right(c, 1), 3)
    m = T.bitwise_and(c, 1)
    two_v = T.if_then_else(e == 0, m, T.shift_left(2 + m, T.max(e - 1, 0)))
    sgn = T.if_then_else(T.bitwise_and(c, 8) != 0, T.float32(-0.5), T.float32(0.5))
    return T.cast(two_v, f32) * sgn * T.cast(sf, f32)


def attn_fp4_split_body(KVH: int, G: int, QL: int, D: int, PS: int, block_N: int = 64, BH: int = 32,
                        min_chunk: int = 256, pipe: int = 0, threads: int = 128):
    """pipe=1: the raw fp4 bytes + scales of a K / V tile (block_N positions inside ONE page: PS % block_N == 0) are
    staged with cp.async in a 2-deep pipeline (extra shared K4_s, KS_s, V4_s, VS_s) and expanded to bf16 in shared
    memory; pipe=0: direct global loads with the expansion. Shared: Q_s (BH, D) bf16, K_s / V_s (block_N, D) bf16,
    P_s (BH, block_N) bf16 (+ 2 * 2 * block_N * (D/2 + D/16) bytes with pipe)."""
    assert not pipe or (PS % block_N == 0 and D % 256 == 0)
    R = G * QL
    assert R <= BH
    DH, DS = D // 2, D // BLOCK
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    NEG = -T.infinity(f32)
    CHD = DH // 16                                   # 16-byte chunks per data row
    NCD = block_N * CHD                              # data chunks per tile per tensor
    NCS = block_N * DS // 16                         # scale chunks per tile per tensor (1 per row at D = 256)
    RD = (NCD + threads - 1) // threads
    RS = (NCS + threads - 1) // threads

    @T.macro
    def body(q, kd, ksf, vd, vsf, bt, sl, gk, gv, opart, lse, b, h, s, S, scale_log2, tid, Q_s, K_s, V_s, P_s,
             K4_s=None, KS_s=None, V4_s=None, VS_s=None):
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
        sc = T.alloc_var(f32)
        L = sl[b]
        sc = scale_log2 * gk[h]
        chunk = T.max(min_chunk, T.ceildiv(T.ceildiv(L, S), block_N) * block_N)
        st = s * chunk
        en = T.min(L, st + chunk)
        for i, j in T.Parallel(BH, D):
            Q_s[i, j] = T.if_then_else(i < R, q[b * QL + i // G, h * G + i % G, j], T.cast(0.0, bf16))
        T.fill(acc_o, 0.0)
        T.fill(l_i, 0.0)
        T.fill(m_i, NEG)
        nblk = T.ceildiv(T.max(en - st, 0), block_N)
        if pipe:
            if nblk > 0:                                 # prologue: tile 0 -> stage 0
                for c in T.unroll(RD):
                    e = c * threads + tid
                    if e < NCD:
                        T.call_extern("handle", "hz_cp16", T.address_of(K4_s[0, e // CHD, (e % CHD) * 16]),
                                      T.address_of(kd[bt[b, st // PS], st % PS + e // CHD, h, (e % CHD) * 16]))
                        T.call_extern("handle", "hz_cp16", T.address_of(V4_s[0, e // CHD, (e % CHD) * 16]),
                                      T.address_of(vd[bt[b, st // PS], st % PS + e // CHD, h, (e % CHD) * 16]))
                for c in T.unroll(RS):
                    e = c * threads + tid
                    if e < NCS:
                        T.call_extern("handle", "hz_cp16", T.address_of(KS_s[0, e, 0]),
                                      T.address_of(ksf[bt[b, st // PS], st % PS + e, h, 0]))
                        T.call_extern("handle", "hz_cp16", T.address_of(VS_s[0, e, 0]),
                                      T.address_of(vsf[bt[b, st // PS], st % PS + e, h, 0]))
                T.call_extern("handle", "hz_cp_commit")
        for kb in T.serial(nblk):
            p0 = st + kb * block_N
            if pipe:
                if kb + 1 < nblk:                        # prefetch tile kb + 1 into the other stage
                    for c in T.unroll(RD):
                        e = c * threads + tid
                        if e < NCD:
                            T.call_extern("handle", "hz_cp16",
                                          T.address_of(K4_s[(kb + 1) % 2, e // CHD, (e % CHD) * 16]),
                                          T.address_of(kd[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e // CHD,
                                                          h, (e % CHD) * 16]))
                            T.call_extern("handle", "hz_cp16",
                                          T.address_of(V4_s[(kb + 1) % 2, e // CHD, (e % CHD) * 16]),
                                          T.address_of(vd[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e // CHD,
                                                          h, (e % CHD) * 16]))
                    for c in T.unroll(RS):
                        e = c * threads + tid
                        if e < NCS:
                            T.call_extern("handle", "hz_cp16", T.address_of(KS_s[(kb + 1) % 2, e, 0]),
                                          T.address_of(ksf[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e, h, 0]))
                            T.call_extern("handle", "hz_cp16", T.address_of(VS_s[(kb + 1) % 2, e, 0]),
                                          T.address_of(vsf[bt[b, (p0 + block_N) // PS], (p0 + block_N) % PS + e, h, 0]))
                    T.call_extern("handle", "hz_cp_commit")
                    T.call_extern("handle", "hz_cp_wait1")
                else:
                    T.call_extern("handle", "hz_cp_wait0")
                T.sync_threads()
                for i, j in T.Parallel(block_N, D):
                    K_s[i, j] = T.if_then_else(p0 + i < en,
                                               T.cast(_fp4_val(K4_s[kb % 2, i, j // 2], j, KS_s[kb % 2, i, j // BLOCK]),
                                                      bf16), T.cast(0.0, bf16))
                for i, j in T.Parallel(block_N, D):
                    V_s[i, j] = T.if_then_else(p0 + i < en,
                                               T.cast(_fp4_val(V4_s[kb % 2, i, j // 2], j, VS_s[kb % 2, i, j // BLOCK]),
                                                      bf16), T.cast(0.0, bf16))
                T.sync_threads()
            else:
                for i, j in T.Parallel(block_N, D):
                    K_s[i, j] = T.if_then_else(
                        p0 + i < en,
                        T.cast(_fp4_val(kd[bt[b, T.min(p0 + i, en - 1) // PS], T.min(p0 + i, en - 1) % PS, h, j // 2],
                                        j,
                                        ksf[bt[b, T.min(p0 + i, en - 1) // PS], T.min(p0 + i, en - 1) % PS, h,
                                            j // BLOCK]), bf16),
                        T.cast(0.0, bf16))
                for i, j in T.Parallel(block_N, D):
                    V_s[i, j] = T.if_then_else(
                        p0 + i < en,
                        T.cast(_fp4_val(vd[bt[b, T.min(p0 + i, en - 1) // PS], T.min(p0 + i, en - 1) % PS, h, j // 2],
                                        j,
                                        vsf[bt[b, T.min(p0 + i, en - 1) // PS], T.min(p0 + i, en - 1) % PS, h,
                                            j // BLOCK]), bf16),
                        T.cast(0.0, bf16))
            T.clear(acc_s)
            T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(BH, block_N):
                # causal among the QL window tokens: row token t sees positions < L - QL + t + 1
                acc_s[i, j] = T.if_then_else(
                    T.And(i < R, T.And(p0 + j < en, p0 + j < L - QL + i // G + 1)),
                    acc_s[i, j] * sc, NEG)
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
                opart[b, h, s, i, j] = T.if_then_else(l_i[i] > 0, acc_o[i, j] / l_i[i] * gv[h], T.float32(0.0))
        for i in T.Parallel(BH):
            if i < R:
                lse[b, h, s, i] = T.if_then_else(l_i[i] > 0, T.log2(l_i[i]) + m_i[i], NEG)

    return body


@tilelang.jit(pass_configs=PASS_CONFIGS)
def attn_fp4_split_kernel(KVH: int, G: int, QL: int, D: int, PS: int, block_N: int = 64, BH: int = 32,
                          threads: int = 128, min_chunk: int = 256, DSP: int = 0, DST: int = 0, DSH: int = 0,
                          SSP: int = 0, SST: int = 0, SSH: int = 0, pipe: int = 0):
    """grid (B, KVH, S); each request splits its OWN length (one CUDA graph per S serves every context length).
    Cache parts are [page, position, kv head, byte] views: data (NP, PS, KVH, D/2) strides (DSP, DST, DSH, 1),
    scales (NP, PS, KVH, D/16) e4m3 strides (SSP, SST, SSH, 1) (kv_fp4_ref.split_views)."""
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    NP = T.dynamic("NP")
    MP = T.dynamic("MP")
    S = T.dynamic("S")
    HQ = KVH * G
    DH, DS = D // 2, D // BLOCK
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    fp8 = T.float8_e4m3fn
    body = attn_fp4_split_body(KVH, G, QL, D, PS, block_N, BH, min_chunk, pipe, threads)

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16),
             kd: T.StridedTensor((NP, PS, KVH, DH), (DSP, DST, DSH, 1), T.uint8),
             ksf: T.StridedTensor((NP, PS, KVH, DS), (SSP, SST, SSH, 1), fp8),
             vd: T.StridedTensor((NP, PS, KVH, DH), (DSP, DST, DSH, 1), T.uint8),
             vsf: T.StridedTensor((NP, PS, KVH, DS), (SSP, SST, SSH, 1), fp8),
             bt: T.Tensor((B, MP), i32),
             sl: T.Tensor((B,), i32),
             gk: T.Tensor((KVH,), f32),
             gv: T.Tensor((KVH,), f32),
             opart: T.Tensor((B, KVH, S, BH, D), f32),
             lse: T.Tensor((B, KVH, S, BH), f32),
             scale_log2: f32):
        with T.Kernel(B, KVH, S, threads=threads, prelude=CP_SRC) as (b, h, s):
            Q_s = T.alloc_shared((BH, D), bf16)
            K_s = T.alloc_shared((block_N, D), bf16)
            V_s = T.alloc_shared((block_N, D), bf16)
            P_s = T.alloc_shared((BH, block_N), bf16)
            if pipe:
                K4_s = T.alloc_shared((2, block_N, DH), T.uint8)
                V4_s = T.alloc_shared((2, block_N, DH), T.uint8)
                KS_s = T.alloc_shared((2, block_N, DS), fp8)
                VS_s = T.alloc_shared((2, block_N, DS), fp8)
                body(q, kd, ksf, vd, vsf, bt, sl, gk, gv, opart, lse, b, h, s, S, scale_log2, T.get_thread_binding(),
                     Q_s, K_s, V_s, P_s, K4_s, KS_s, V4_s, VS_s)
            else:
                body(q, kd, ksf, vd, vsf, bt, sl, gk, gv, opart, lse, b, h, s, S, scale_log2, T.get_thread_binding(),
                     Q_s, K_s, V_s, P_s)

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def rotate_q_kernel(HQ: int, D: int, threads: int = 128, F16: int = 0):
    """q (NT, HQ, D) bf16 -> out (NT, HQ, D) bf16 (F16 = 1: fp16, for the fp16 prefill) = FWHT(q * sg) * D^-0.5 (fp32,
    kv_fp4_ref.fwht order), one rounding. One CTA per token; shared (HQ, D) fp32 (24 KB at HQ 24, D 256)."""
    NT = T.dynamic("NT")
    nel = HQ * D
    per = (nel + threads - 1) // threads
    csc = fwht_scale(D)
    f32, bf16 = T.float32, T.bfloat16
    odt = T.float16 if F16 else bf16
    fwht = fwht_body(HQ, D, threads)

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), bf16), sg: T.Tensor((D,), f32), out: T.Tensor((NT, HQ, D), odt)):
        with T.Kernel(NT, threads=threads, prelude=FP4_SRC) as t:
            tid = T.get_thread_binding()
            X_s = T.alloc_shared((HQ, D), f32)
            for e in T.unroll(per):
                idx = e * threads + tid
                if idx < nel:
                    X_s[idx // D, idx % D] = T.call_extern(f32, "hz_fmul", T.cast(q[t, idx // D, idx % D], f32),
                                                           sg[idx % D])
            T.sync_threads()
            fwht(X_s, tid)
            for e in T.unroll(per):
                idx = e * threads + tid
                if idx < nel:
                    out[t, idx // D, idx % D] = T.cast(
                        T.call_extern(f32, "hz_fmul", X_s[idx // D, idx % D], T.float32(csc)), odt)

    return main


def get_split(KVH, G, QL, D, PS, block_N=64, BH=32, threads=128, min_chunk=256, dstr=(0, 0, 0), sstr=(0, 0, 0),
              pipe=0):
    key = ("s4", KVH, G, QL, D, PS, block_N, BH, threads, min_chunk, tuple(dstr), tuple(sstr), pipe)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_fp4_split_kernel(KVH, G, QL, D, PS, block_N, BH, threads, min_chunk, *dstr, *sstr,
                                                  pipe)
    return k


def get_rotate(HQ, D, threads=128, f16=0):
    key = ("rq", HQ, D, threads, int(bool(f16)))
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = rotate_q_kernel(HQ, D, threads, int(bool(f16)))
    return k


def rotate_q(q, signs, out=None, dtype=None):
    """q (NT, HQ, D) bf16 contiguous -> rotated q in out / a new tensor of dtype (bf16 default, or fp16)"""
    import torch
    if out is None:
        out = torch.empty(q.shape, dtype=dtype or torch.bfloat16, device=q.device)
    get_rotate(q.shape[1], q.shape[2], 128, out.dtype == torch.float16)(q, signs.float().contiguous(), out)
    return out


def decode(query, k_data, k_sf, v_data, v_sf, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale,
           q_len_per_req, gk, gv, out=None, block_N=None, splits=None, pipe=1):
    """query (B*QL, HQ, D) bf16 (ROTATED with rotate_q when the keys were written with signs); cache parts as
    [page, position, kv head, byte] views (kv_fp4_ref.split_views); gk / gv (KVH,) fp32 global scales; returns out
    (B*QL, HQ, D) bf16. max_seq_len unused (XQA signature). pipe=1 needs PS % block_N == 0 (block_N 32 default)."""
    import torch
    NT, HQ, D = query.shape
    _, PS, KVH, DH = k_data.shape
    assert DH * 2 == D and k_data.stride() == v_data.stride() and k_sf.stride() == v_sf.stride()
    QL = int(q_len_per_req or 1)
    B = NT // QL
    G = HQ // KVH
    if block_N is None:
        block_N = 32 if pipe else 64
    if pipe and (PS % block_N or D % 256):
        pipe, block_N = 0, 64
    if out is None:
        out = torch.empty_like(query)
    S = splits or plan_splits(B, KVH)
    op, ls = _workspace(query.device, B, KVH, S, 32, D)
    f8 = torch.float8_e4m3fn
    get_split(KVH, G, QL, D, PS, block_N, dstr=k_data.stride()[:3], sstr=k_sf.stride()[:3], pipe=pipe)(
        query, k_data, k_sf.view(f8), v_data, v_sf.view(f8), block_tables, seq_lens, gk.float().contiguous(),
        gv.float().contiguous(), op, ls, float(bmm1_scale) * LOG2E)
    get_combine(KVH, G, QL, D)(op, ls, out, float(bmm2_scale))
    return out
