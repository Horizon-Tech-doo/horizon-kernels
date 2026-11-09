# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GDN chunked prefill (gated delta rule over whole prompt chunks), a drop-in for vLLM 0.29's FLA
`chunk_gated_delta_rule` (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens) on SM120.

FLA runs six Triton kernels per layer (chunk-local cumsum, beta K K^T, triangular solve, WY recompute, the
sequential state pass that WRITES the state of every 64-token chunk, and the output pass that READS it back). Ours:
  wy kernel   (parallel over chunks x value heads): cumsum of g, A = strict_lower(beta_i k_i.k_j e^(G_i - G_j)),
              X = (I + A)^-1 by forward substitution (fp32, in shared memory), W = X (beta k e^G), U = X (beta v)
  pass kernel (per sequence x value head x 32-wide value slice, sequential over the chunks): with the state h kept on
              chip for the whole prompt: V_new = U - W h^T, O = scale (Q h^T e^G + mask(Q K^T e^(G_i - G_j)) V_new),
              h = e^(G_last) h + (V_new e^(G_last - G))^T K. The per-chunk states never touch memory (FLA writes and
              re-reads 128 x 128 x bf16 per chunk and head: ~0.8 GB per layer for an 8k-token chunk).
Operand precisions follow FLA (bf16 MMA inputs: W, U, the state copy h used in products, V_new, P; fp32 accumulate and
fp32 state); the summation orders differ, so the gate is the long-prompt log-prob comparison (not bit-identity).
Shapes (Qwen3.8): q, k (T, 16, 128) bf16, v (T, 48, 128) bf16, g, beta (T, 48) fp32, states (N, 48, 128, 128) fp32.
"""

import tilelang
import tilelang.language as T

BT = 64
_KERNELS = {}
PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=PASS)
def wy_kernel(H: int, HV: int, K: int = 128, V: int = 128, threads: int = 128):
    TT = T.dynamic("TT")
    NC = T.dynamic("NC")
    NQ = T.dynamic("NQ")
    G = HV // H
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    @T.prim_func
    def main(k: T.Tensor((TT, H, K), bf16), v: T.Tensor((TT, HV, V), bf16), g: T.Tensor((TT, HV), f32),
             beta: T.Tensor((TT, HV), f32), cu: T.Tensor((NQ,), i32), ci: T.Tensor((NC, 2), i32),
             W: T.Tensor((TT, HV, K), bf16), U: T.Tensor((TT, HV, V), bf16), GC: T.Tensor((TT, HV), f32)):
        with T.Kernel(NC, HV, threads=threads) as (c, hv):
            tid = T.get_thread_binding()
            K_s = T.alloc_shared((BT, K), bf16)
            V_s = T.alloc_shared((BT, V), bf16)
            A_s = T.alloc_shared((BT, BT), f32)
            X_s = T.alloc_shared((BT, BT), f32)
            Xb_s = T.alloc_shared((BT, BT), bf16)
            gc_s = T.alloc_shared((BT,), f32)
            bt_s = T.alloc_shared((BT,), f32)
            A_f = T.alloc_fragment((BT, BT), f32)
            O_f = T.alloc_fragment((BT, K), f32)
            n = T.alloc_var(i32)
            t0 = T.alloc_var(i32)
            bos = T.alloc_var(i32)
            nv = T.alloc_var(i32)
            acc = T.alloc_var(f32)
            s = T.alloc_var(f32)
            n = ci[c, 0]
            bos = cu[n]
            t0 = ci[c, 1] * BT
            nv = T.min(BT, cu[n + 1] - bos - t0)          # valid rows of this chunk
            kh = hv // G
            for i, j in T.Parallel(BT, K):
                K_s[i, j] = T.if_then_else(i < nv, k[bos + t0 + T.min(i, nv - 1), kh, j], T.cast(0.0, bf16))
            for i, j in T.Parallel(BT, V):
                V_s[i, j] = T.if_then_else(i < nv, v[bos + t0 + T.min(i, nv - 1), hv, j], T.cast(0.0, bf16))
            if tid == 0:
                acc = 0.0
                for i in T.serial(BT):
                    if i < nv:
                        acc = acc + g[bos + t0 + i, hv]
                    gc_s[i] = acc
            for i in T.Parallel(BT):
                bt_s[i] = T.if_then_else(i < nv, beta[bos + t0 + T.min(i, nv - 1), hv], T.float32(0.0))
            T.sync_threads()
            T.clear(A_f)
            T.gemm(K_s, K_s, A_f, transpose_B=True)
            for i, j in T.Parallel(BT, BT):
                A_s[i, j] = T.if_then_else(T.And(j < i, i < nv), A_f[i, j] * bt_s[i] * T.exp(gc_s[i] - gc_s[j]),
                                           T.float32(0.0))
            for i, j in T.Parallel(BT, BT):
                X_s[i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
            T.sync_threads()
            # forward substitution, one COLUMN per thread (a column only depends on itself): X[i, c] =
            # -sum_{c <= j < i} A[i, j] X[j, c]  (X = (I + A)^-1, unit lower); no barrier inside
            if tid < BT:
                for i in T.serial(BT - 1 - tid):
                    r = tid + 1 + i
                    s = 0.0
                    for jj in T.serial(r - tid):
                        s = s + A_s[r, tid + jj] * X_s[tid + jj, tid]
                    X_s[r, tid] = -s
            T.sync_threads()
            for i, j in T.Parallel(BT, BT):
                Xb_s[i, j] = T.cast(X_s[i, j], bf16)
            # U = X (beta v)
            for i, j in T.Parallel(BT, V):
                V_s[i, j] = T.cast(T.cast(V_s[i, j], f32) * bt_s[i], bf16)
            T.clear(O_f)
            T.gemm(Xb_s, V_s, O_f)
            for i, j in T.Parallel(BT, V):
                if i < nv:
                    U[bos + t0 + i, hv, j] = T.cast(O_f[i, j], bf16)
            # W = X (beta k e^G)
            for i, j in T.Parallel(BT, K):
                K_s[i, j] = T.cast(T.cast(K_s[i, j], f32) * (bt_s[i] * T.exp(gc_s[i])), bf16)
            T.clear(O_f)
            T.gemm(Xb_s, K_s, O_f)
            for i, j in T.Parallel(BT, K):
                if i < nv:
                    W[bos + t0 + i, hv, j] = T.cast(O_f[i, j], bf16)
            for i in T.Parallel(BT):
                if i < nv:
                    GC[bos + t0 + i, hv] = gc_s[i]

    return main


@tilelang.jit(pass_configs=PASS)
def pass_kernel(H: int, HV: int, K: int = 128, V: int = 128, BV: int = 32, threads: int = 128):
    TT = T.dynamic("TT")
    NQ = T.dynamic("NQ")
    NS = T.dynamic("NS")
    G = HV // H
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    @T.prim_func
    def main(q: T.Tensor((TT, H, K), bf16), k: T.Tensor((TT, H, K), bf16), W: T.Tensor((TT, HV, K), bf16),
             U: T.Tensor((TT, HV, V), bf16), GC: T.Tensor((TT, HV), f32), cu: T.Tensor((NQ,), i32),
             h0: T.Tensor((NS, HV, V, K), f32), ht: T.Tensor((NS, HV, V, K), f32), o: T.Tensor((TT, HV, V), bf16),
             scale: f32):
        with T.Kernel(V // BV, HV, NQ - 1, threads=threads) as (vb, hv, n):
            W_s = T.alloc_shared((BT, K), bf16)
            Q_s = T.alloc_shared((BT, K), bf16)
            K_s = T.alloc_shared((BT, K), bf16)
            U_s = T.alloc_shared((BT, BV), bf16)
            hb_s = T.alloc_shared((BV, K), bf16)
            P_s = T.alloc_shared((BT, BT), bf16)
            VN_s = T.alloc_shared((BT, BV), bf16)
            VS_s = T.alloc_shared((BT, BV), bf16)
            gc_s = T.alloc_shared((BT,), f32)
            h_f = T.alloc_fragment((BV, K), f32)
            vn_f = T.alloc_fragment((BT, BV), f32)
            o_f = T.alloc_fragment((BT, BV), f32)
            s_f = T.alloc_fragment((BT, BT), f32)
            bos = T.alloc_var(i32)
            L = T.alloc_var(i32)
            gl = T.alloc_var(f32)
            bos = cu[n]
            L = cu[n + 1] - bos
            kh = hv // G
            T.copy(h0[n, hv, vb * BV:(vb + 1) * BV, :], h_f)
            for it in T.serial(T.ceildiv(L, BT)):
                t0 = it * BT
                nv = T.min(BT, L - t0)
                for i, j in T.Parallel(BT, K):
                    W_s[i, j] = T.if_then_else(i < nv, W[bos + t0 + T.min(i, nv - 1), hv, j], T.cast(0.0, bf16))
                for i, j in T.Parallel(BT, K):
                    Q_s[i, j] = T.if_then_else(i < nv, q[bos + t0 + T.min(i, nv - 1), kh, j], T.cast(0.0, bf16))
                for i, j in T.Parallel(BT, K):
                    K_s[i, j] = T.if_then_else(i < nv, k[bos + t0 + T.min(i, nv - 1), kh, j], T.cast(0.0, bf16))
                for i, j in T.Parallel(BT, BV):
                    U_s[i, j] = T.if_then_else(i < nv, U[bos + t0 + T.min(i, nv - 1), hv, vb * BV + j],
                                               T.cast(0.0, bf16))
                for i in T.Parallel(BT):
                    gc_s[i] = T.if_then_else(i < nv, GC[bos + t0 + T.min(i, nv - 1), hv], T.float32(0.0))
                T.copy(h_f, hb_s)
                T.sync_threads()
                gl = gc_s[nv - 1]
                # V_new = U - W h^T
                T.clear(vn_f)
                T.gemm(W_s, hb_s, vn_f, transpose_B=True)
                for i, j in T.Parallel(BT, BV):
                    vn_f[i, j] = T.cast(U_s[i, j], f32) - vn_f[i, j]
                    VN_s[i, j] = T.cast(vn_f[i, j], bf16)
                    VS_s[i, j] = T.cast(T.if_then_else(i < nv, vn_f[i, j] * T.exp(gl - gc_s[i]), T.float32(0.0)),
                                        bf16)
                # O = scale (Q h^T e^G + mask(Q K^T e^(G_i - G_j)) V_new)
                T.clear(o_f)
                T.gemm(Q_s, hb_s, o_f, transpose_B=True)
                T.clear(s_f)
                T.gemm(Q_s, K_s, s_f, transpose_B=True)
                for i, j in T.Parallel(BT, BT):
                    P_s[i, j] = T.cast(T.if_then_else(T.And(j <= i, i < nv), s_f[i, j] * T.exp(gc_s[i] - gc_s[j]),
                                                      T.float32(0.0)), bf16)
                for i, j in T.Parallel(BT, BV):
                    o_f[i, j] = o_f[i, j] * T.exp(gc_s[i])
                T.sync_threads()
                T.gemm(P_s, VN_s, o_f)
                for i, j in T.Parallel(BT, BV):
                    if i < nv:
                        o[bos + t0 + i, hv, vb * BV + j] = T.cast(o_f[i, j] * scale, bf16)
                # h = e^(G_last) h + (V_new e^(G_last - G))^T K
                for i, j in T.Parallel(BV, K):
                    h_f[i, j] = h_f[i, j] * T.exp(gl)
                T.gemm(VS_s, K_s, h_f, transpose_A=True)
                T.sync_threads()
            T.copy(h_f, ht[n, hv, vb * BV:(vb + 1) * BV, :])

    return main


def chunk_gated_delta_rule(q, k, v, g, beta, initial_state, cu_seqlens, scale=None, BV=32, chunk_indices=None):
    """q, k (1, T, H, K) bf16, v (1, T, HV, V) bf16, g, beta (1, T, HV) fp32, initial_state (N, HV, V, K) fp32
    (zeros where there is none), cu_seqlens (N + 1,) int32 -> (o (1, T, HV, V) bf16, final_state (N, HV, V, K))"""
    import torch
    _, TT, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    scale = K ** -0.5 if scale is None else scale
    q2, k2, v2 = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    g2, b2 = g[0].contiguous().float(), beta[0].contiguous().float()
    cu = cu_seqlens.to(torch.int32).contiguous()
    if chunk_indices is not None:           # vLLM's precomputed FLA chunk list: (sequence, chunk) pairs, 64 tokens
        ci = chunk_indices.to(torch.int32).contiguous().view(-1, 2)
    else:
        lens = (cu[1:] - cu[:-1]).tolist()
        ci = torch.tensor([(n, c) for n, L in enumerate(lens) for c in range(-(-L // BT))], dtype=torch.int32,
                          device=q.device).view(-1, 2)
    W = torch.empty(TT, HV, K, dtype=torch.bfloat16, device=q.device)
    U = torch.empty(TT, HV, V, dtype=torch.bfloat16, device=q.device)
    GC = torch.empty(TT, HV, dtype=torch.float32, device=q.device)
    key = ("wy", H, HV, K, V)
    kw = _KERNELS.get(key) or _KERNELS.setdefault(key, wy_kernel(H, HV, K, V))
    kw(k2, v2, g2, b2, cu, ci, W, U, GC)
    o = torch.empty(TT, HV, V, dtype=torch.bfloat16, device=q.device)
    ht = torch.empty_like(initial_state)
    key = ("pass", H, HV, K, V, BV)
    kp = _KERNELS.get(key) or _KERNELS.setdefault(key, pass_kernel(H, HV, K, V, BV))
    kp(q2, k2, W, U, GC, cu, initial_state.contiguous(), ht, o, float(scale))
    return o.unsqueeze(0), ht
