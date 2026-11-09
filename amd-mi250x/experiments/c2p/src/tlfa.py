"""Prefill flash attention written only in TileLang: no edits on generated source. Used to port gqa_flash_fast (N5) onto upstream main plus the C2P patches.
Layout: Q [B, L, H, D], K/V [B, L, Hkv, D], H = Hkv * G, fp16 in/out, fp32 accumulate.
Options, string of letters:
  S register-staged K and V: 2 rows per thread, 16 bytes per lane, loaded before the V GEMM, stored at the top of the next step
  J same thread layout but loaded in the step itself (joint load, no overlap)
  D skip the O rescale when sf == 1
  F T.__exp in place of T.exp
  B row sum kept per fragment element (l_p) and reduced once after the K loop
  R row sum in a reducer, T.reducer_scale per step, finalized once after the K loop
  P<n> n halves of padding per LDS row of K and V (K8)
  L<n> T.annotate_min_blocks_per_sm(n): n blocks per CU, so HIP emits __launch_bounds__(th, ceil(n * ceil(th / 64) / 4))
  W<n> diagnosis only, not a TileLang form: text edit on generated source to __launch_bounds__(th, n), used where L cannot express n
"""
import re

import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy


def make(B, L, Hkv, G, D, causal, P, bN, threads, opts="", k_pack=2, edge=True, rev=True, recip=True, swap=True, rows=0):
    H = Hkv * G
    bM = rows if rows else P * G
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32
    NB = -(-L * G // rows) if rows else -(-L // P)
    d0, d1 = (B * Hkv, NB) if swap else (NB, B * Hkv)
    exp = T.__exp if "F" in opts else T.exp
    staged = "S" in opts or "J" in opts
    cpr = D // 8
    m_pad = re.search(r"P(\d+)", opts)
    pad = int(m_pad.group(1)) if m_pad else 0
    m_lb = re.search(r"L(\d+)", opts)

    def stage_layout():
        if bN // 2 * cpr > threads:
            raise ValueError("one load per thread needs (bN/2)*(D/8) <= threads")
        return T.Fragment((bN, D), forward_fn=lambda i, j: ((i // 2) * cpr + j // 8, (i % 2) * 8 + j % 8))

    @T.prim_func
    def main(Q: T.Tensor((B, L, H, D), dtype), K: T.Tensor((B, L, Hkv, D), dtype), V: T.Tensor((B, L, Hkv, D), dtype),
             O: T.Tensor((B, L, H, D), dtype)):
        with T.Kernel(d0, d1, threads=threads) as (b0, b1):
            if m_lb:
                T.annotate_min_blocks_per_sm(int(m_lb.group(1)))
            bx = b1 if swap else b0
            byz = b0 if swap else b1
            b = byz // Hkv
            hk = byz % Hkv
            if rows:
                r0 = (NB - 1 - bx) * bM if rev else bx * bM
            else:
                r0 = (NB - 1 - bx) * P * G if rev else bx * P * G
            K_s = T.alloc_shared((bN, D + pad), dtype)
            V_s = T.alloc_shared((bN, D + pad), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            acc_s = T.alloc_fragment((bM, bN), acc)
            p_cast = T.alloc_fragment((bM, bN), dtype)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            row_sum = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            rl = T.alloc_fragment((bM,), acc)
            Q_x = T.alloc_fragment((bM, D), dtype)
            if staged:
                K_r = T.alloc_fragment((bN, D), dtype)
                V_r = T.alloc_fragment((bN, D), dtype)
                T.annotate_layout({K_r: stage_layout(), V_r: stage_layout()})
            for i, d in T.Parallel(bM, D):
                Q_x[i, d] = T.if_then_else((r0 + i) // G < L, Q[b, (r0 + i) // G, hk * G + (r0 + i) % G, d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            if "B" in opts:
                l_p = T.alloc_fragment((bM, bN), acc)
                T.fill(l_p, 0)
            if "R" in opts:
                l_r = T.alloc_reducer((bM,), acc, op="sum")
                T.reducer_init(l_r)
            T.clear(acc_s)
            n_kv = T.ceildiv(T.min((r0 + bM - 1) // G + 1, L), bN) if causal else T.ceildiv(L, bN)
            if "S" in opts:
                T.copy(K[b, 0:bN, hk, :], K_r)
                T.copy(V[b, 0:bN, hk, :], V_r)
            for k in T.serial(n_kv):
                kv0 = k * bN
                if "J" in opts:
                    T.copy(K[b, kv0:kv0 + bN, hk, :], K_r)
                    T.copy(V[b, kv0:kv0 + bN, hk, :], V_r)
                if staged:
                    T.copy(K_r, K_s[0:bN, 0:D])
                    T.copy(V_r, V_s[0:bN, 0:D])
                else:
                    T.copy(K[b, kv0:kv0 + bN, hk, :], K_s[0:bN, 0:D])
                    T.copy(V[b, kv0:kv0 + bN, hk, :], V_s[0:bN, 0:D])
                T.gemm(Q_x, K_s[0:bN, 0:D], acc_s, transpose_B=True, clear_accum=True, k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
                if causal:
                    if kv0 + bN - 1 > r0 // G:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else(((r0 + i) // G >= kv0 + j) & (kv0 + j < L), acc_s[i, j], -T.infinity(acc))
                elif L % bN:
                    if kv0 + bN > L:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else(kv0 + j < L, acc_s[i, j], -T.infinity(acc))
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = exp(m_prev[i] * scale - m_i[i] * scale)
                if "R" in opts:
                    for i in T.Parallel(bM):
                        T.reducer_scale(l_r[i], sf[i])
                elif "B" not in opts:
                    for i in T.Parallel(bM):
                        l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    if "D" in opts:
                        if sf[i] != 1.0:
                            acc_o[i, d] *= sf[i]
                    else:
                        acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, bN):
                    acc_s[i, j] = exp(acc_s[i, j] * scale - m_i[i] * scale)
                if "R" in opts:
                    for i, j in T.Parallel(bM, bN):
                        T.reducer_update(l_r[i], acc_s[i, j])
                elif "B" in opts:
                    for i, j in T.Parallel(bM, bN):
                        l_p[i, j] = l_p[i, j] * sf[i] + acc_s[i, j]
                else:
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(bM):
                        l_i[i] += row_sum[i]
                T.copy(acc_s, p_cast)
                if "S" in opts:
                    if k + 1 < n_kv:
                        T.copy(K[b, kv0 + bN:kv0 + 2 * bN, hk, :], K_r)
                        T.copy(V[b, kv0 + bN:kv0 + 2 * bN, hk, :], V_r)
                T.gemm(p_cast, V_s[0:bN, 0:D], acc_o, policy=GemmWarpPolicy.FullRow)
            if "B" in opts:
                T.reduce_sum(l_p, l_i, dim=1)
            if "R" in opts:
                T.finalize_reducer(l_r, l_i)
            if recip:
                for i in T.Parallel(bM):
                    rl[i] = 1.0 / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
                for i, d in T.Parallel(bM, D):
                    if (r0 + i) // G < L:
                        O[b, (r0 + i) // G, hk * G + (r0 + i) % G, d] = acc_o[i, d] * rl[i]
            else:
                for i, d in T.Parallel(bM, D):
                    if (r0 + i) // G < L:
                        O[b, (r0 + i) // G, hk * G + (r0 + i) % G, d] = acc_o[i, d] / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)

    return main


@tilelang.jit(out_idx=[3])
def fwd(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows):
    return make(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows)


# Diagnosis only, not a TileLang form: option H applies N5's row-sum edit (B) to the generated source, to size what a reducer-based row sum could gain.
_H = [False]


def _block_end(s, i):
    d, j = 0, s.index("{", i)
    while True:
        d += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
        if d == 0:
            return j


def _h_edit(code, target):
    import re
    if not _H[0]:
        return code
    m = re.search(r"row_sum\[0\] = tl::AllReduce<tl::SumOp, (\d+), (\d+), 0>::run\(row_sum\[0\][^;]*;\n", code)
    if not m:
        return code
    code = code[:m.start()] + code[m.end():]
    k = code.index("for (int k = 0;")
    e = _block_end(code, k)
    return code[:e] + f"\n  l_i[0] = tl::AllReduce<tl::SumOp, {m.group(1)}, {m.group(2)}, 0>::run(l_i[0]);\n" + code[e:]


def fwd_h(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows):
    import tvm_ffi
    tvm_ffi.register_global_func("tilelang_callback_hip_postproc", _h_edit, override=True)
    _H[0] = True
    try:
        return tilelang.compile(make(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows), out_idx=[3], compile_flags=["-DTLFA_H_V1"])
    finally:
        _H[0] = False


def fwd_w(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows):
    import tvm_ffi
    w = int(re.search(r"W(\d+)", opts).group(1))

    def edit(code, target):
        return code.replace(f"__launch_bounds__({threads})", f"__launch_bounds__({threads}, {w})")
    tvm_ffi.register_global_func("tilelang_callback_hip_postproc", edit, override=True)
    try:
        return tilelang.compile(make(B, L, Hkv, G, D, causal, P, bN, threads, opts, k_pack, edge, rev, recip, swap, rows), out_idx=[3], compile_flags=[f"-DTLFA_W_{w}"])
    finally:
        tvm_ffi.register_global_func("tilelang_callback_hip_postproc", lambda code, target: code, override=True)
