"""E3 second kernel set: finer ablation and V-layout changes. 'full' with defaults equals gqa_e3.py full.
Blocks can be made dead at run time (scalar never != 0, always called with 0), keeping layouts comparable.
"""
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy

MODES = {
    # name: load_in_loop, load_once, xsync, mask, g1, red, rsc, exp, g2, fake_p
    "full":   (1, 0, 0, 1, 1, 1, 1, 1, 1, 0),
    "comp_b": (0, 1, 1, 1, 1, 1, 1, 1, 1, 0),
    "c_nrsc": (0, 1, 1, 1, 1, 1, 0, 1, 1, 0),  # no O rescale
    "c_nred": (0, 1, 1, 1, 1, 0, 1, 1, 1, 0),  # no max reduce (keeps rescale code)
    "c_nexp": (0, 1, 1, 1, 1, 1, 1, 0, 1, 1),  # no exp/sum/cast
    "c_nmsk": (0, 1, 1, 0, 1, 1, 1, 1, 1, 0),  # no mask (acc_s accumulates, timing only)
    "mfma_b": (0, 1, 1, 0, 1, 0, 0, 0, 1, 1),
    "mfma1":  (0, 1, 1, 0, 1, 0, 0, 0, 0, 1),
    "mfma2":  (0, 1, 1, 0, 0, 0, 0, 0, 1, 1),
    "empty":  (0, 1, 1, 0, 0, 0, 0, 0, 0, 1),  # barriers only per K step
    "load":   (1, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    "l_gem":  (1, 0, 0, 0, 1, 0, 0, 0, 1, 1),
}


def make(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, recip, mode, vt, v_pack, rev=False):
    H = Hkv * G
    bM = P * G
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32
    NB = -(-L // P)
    lloop, lonce, xs, r_mask, r_g1, r_red, r_rsc, r_exp, r_g2, fake_p = (bool(x) for x in MODES[mode])
    has_edge = causal or L % bN != 0

    @T.prim_func
    def main(Q: T.Tensor((B, L, H, D), dtype), K: T.Tensor((B, L, Hkv, D), dtype), V: T.Tensor((B, L, Hkv, D), dtype),
             O: T.Tensor((B, L, H, D), dtype), never: T.int32):
        with T.Kernel(NB, B * Hkv, threads=threads) as (bx, byz):
            b = byz // Hkv
            hk = byz % Hkv
            if rev:
                p0 = (NB - 1 - bx) * P
            else:
                p0 = bx * P
            K_s = T.alloc_shared((bN, D), dtype)
            if vt:
                V_t = T.alloc_shared((D, bN), dtype)
            else:
                V_s = T.alloc_shared((bN, D), dtype)
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
            for i, d in T.Parallel(bM, D):
                Q_x[i, d] = T.if_then_else(p0 + i // G < L, Q[b, p0 + i // G, hk * G + i % G, d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            if fake_p:
                T.fill(p_cast, 0.0078125)
            if not r_mask or not (r_mask and not edge):
                T.clear(acc_s)
            if lonce:
                T.copy(K[b, 0:bN, hk, :], K_s)
                if vt:
                    for d, kk in T.Parallel(D, bN):
                        V_t[d, kk] = V[b, kk, hk, d]
                else:
                    T.copy(V[b, 0:bN, hk, :], V_s)
            n_kv = T.ceildiv(T.min(p0 + P, L), bN) if causal else T.ceildiv(L, bN)
            for k in T.serial(n_kv):
                kv0 = k * bN
                if lloop:
                    T.copy(K[b, kv0:kv0 + bN, hk, :], K_s)
                    if vt:
                        for d, kk in T.Parallel(D, bN):
                            V_t[d, kk] = V[b, kv0 + kk, hk, d]
                    else:
                        T.copy(V[b, kv0:kv0 + bN, hk, :], V_s)
                if xs:
                    T.sync_threads()
                if r_mask and not edge:
                    if causal:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else((p0 + i // G >= kv0 + j) & (kv0 + j < L), 0, -T.infinity(acc))
                    else:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else(kv0 + j < L, 0, -T.infinity(acc))
                if r_g1:
                    T.gemm(Q_x, K_s, acc_s, transpose_B=True, clear_accum=(edge and r_mask), k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
                else:
                    if never != 0:
                        T.gemm(Q_x, K_s, acc_s, transpose_B=True, k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
                if edge and r_mask and has_edge:
                    if causal:
                        if kv0 + bN - 1 > p0:
                            for i, j in T.Parallel(bM, bN):
                                acc_s[i, j] = T.if_then_else((p0 + i // G >= kv0 + j) & (kv0 + j < L), acc_s[i, j], -T.infinity(acc))
                    else:
                        if kv0 + bN > L:
                            for i, j in T.Parallel(bM, bN):
                                acc_s[i, j] = T.if_then_else(kv0 + j < L, acc_s[i, j], -T.infinity(acc))
                if r_red or fake_p or not r_exp:
                    if r_red:
                        T.copy(m_i, m_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for i in T.Parallel(bM):
                            m_i[i] = T.max(m_i[i], m_prev[i])
                    else:
                        if never != 0:
                            T.copy(m_i, m_prev)
                            T.reduce_max(acc_s, m_i, dim=1, clear=False)
                            for i in T.Parallel(bM):
                                m_i[i] = T.max(m_i[i], m_prev[i])
                else:
                    # c_nred: reduce truly removed (no dead branch), rest of the row bookkeeping kept
                    T.copy(m_i, m_prev)
                    for i in T.Parallel(bM):
                        m_i[i] = T.max(m_i[i], m_prev[i])
                if r_rsc:
                    for i in T.Parallel(bM):
                        sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                        l_i[i] *= sf[i]
                    for i, d in T.Parallel(bM, D):
                        acc_o[i, d] *= sf[i]
                elif fake_p or not r_exp:
                    if never != 0:
                        for i in T.Parallel(bM):
                            sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                            l_i[i] *= sf[i]
                        for i, d in T.Parallel(bM, D):
                            acc_o[i, d] *= sf[i]
                else:
                    # c_nrsc: only the acc_o rescale truly removed; sf and l_i kept
                    for i in T.Parallel(bM):
                        sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                        l_i[i] *= sf[i]
                if r_exp:
                    for i, j in T.Parallel(bM, bN):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(bM):
                        l_i[i] += row_sum[i]
                    T.copy(acc_s, p_cast)
                else:
                    if never != 0:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                        T.reduce_sum(acc_s, row_sum, dim=1)
                        for i in T.Parallel(bM):
                            l_i[i] += row_sum[i]
                        T.copy(acc_s, p_cast)
                if r_g2:
                    if vt:
                        T.gemm(p_cast, V_t, acc_o, transpose_B=True, k_pack=v_pack, policy=GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(p_cast, V_s, acc_o, k_pack=v_pack, policy=GemmWarpPolicy.FullRow)
                else:
                    if never != 0:
                        if vt:
                            T.gemm(p_cast, V_t, acc_o, transpose_B=True, k_pack=v_pack, policy=GemmWarpPolicy.FullRow)
                        else:
                            T.gemm(p_cast, V_s, acc_o, k_pack=v_pack, policy=GemmWarpPolicy.FullRow)
                if xs:
                    T.sync_threads()
            if recip:
                for i in T.Parallel(bM):
                    rl[i] = 1.0 / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
                for i, d in T.Parallel(bM, D):
                    if p0 + i // G < L:
                        O[b, p0 + i // G, hk * G + i % G, d] = acc_o[i, d] * rl[i]
            else:
                for i, d in T.Parallel(bM, D):
                    if p0 + i // G < L:
                        O[b, p0 + i // G, hk * G + i % G, d] = acc_o[i, d] / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)

    return main


@tilelang.jit(out_idx=[3])
def fwd(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, recip, mode, vt, v_pack, rev=False):
    return make(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, recip, mode, vt, v_pack, rev)
