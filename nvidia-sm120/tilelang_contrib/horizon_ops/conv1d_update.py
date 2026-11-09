# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""G1b: GDN causal conv1d state update for speculative decode (4 taps, SiLU), bit-identical to vLLM 0.29's Triton
`_causal_conv1d_update_kernel` (IS_SPEC_DECODING, IS_VARLEN, no bias, width 4) as its PTX computes it:
    window (per channel, fp32 of the bf16 state): c0, c1, c2 = state[slot, ch, off + 0..2], off = n_acc - 1
    new state[slot, ch, i] = state[slot, ch, off + i + 1] for i < S - L, else x[i - (S - L)]   (S = 3 + (Lmax - 1))
    per token t: acc = ((((0 + bf16(c0 w0)) + bf16(c1 w1)) + bf16(c2 w2)) + bf16(x_t w3))  (bf16 products, fp32 adds)
                 out = bf16(acc / (1 + ex2.approx((0 - acc) * log2e)))   with div.full.f32; then c0, c1, c2 = c1, c2, x_t
    (the new window is written BEFORE the outputs are computed, from values loaded first: in-place x is safe).
Two layers: conv_update_body(...)(x, w, state, sidx, nacc, cu, out, r, cb, tid) (one task = request r, a block of
CH channels; no shared memory) and the standalone kernel (grid requests x channel blocks, 128 threads).
DEPS: conv_update(r, cb): in_proj_qkvz rows of request r, columns [cb*CH, +CH) of the qkv part; the previous step's
      conv_update of the same request (state slot).
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
LOG2E_BITS = 0x3FB8AA3B

CONV_SRC = r"""
#ifndef HZ_CONV_SRC_DEFINED
#define HZ_CONV_SRC_DEFINED
__device__ __forceinline__ float hz_cv_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float hz_cv_silu(float acc) {
  float t = __fsub_rn(0.0f, acc);
  t = __fmul_rn(t, __int_as_float(0x3fb8aa3b));
  float e; asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  float d = __fadd_rn(e, 1.0f);
  float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(acc), "f"(d));
  return r;
}
#endif
"""


def conv_update_body(C: int, CH: int, threads: int = 128, width: int = 4):
    assert width == 4 and CH % threads == 0
    per = CH // threads
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32

    def pm(a, b):   # bf16 product (mul.bf16 == round-to-nearest of the exact fp32 product of two bf16 values)
        return T.cast(T.cast(T.cast(a, f32) * T.cast(b, f32), bf16), f32)

    @T.macro
    def body(x, w, state, sidx, nacc, cu, out, r, cb, tid, lmax):
        bos = T.alloc_var(i32)
        L = T.alloc_var(i32)
        slot = T.alloc_var(i32)
        off = T.alloc_var(i32)
        S = T.alloc_var(i32)
        bos = cu[r]
        L = cu[r + 1] - bos
        slot = sidx[r, 0]
        off = nacc[r] - 1
        S = 3 + lmax - 1 - (lmax - L)          # Triton: state_len = (width-1 + (seqlen-1)) - (seqlen - qlen)
        if T.And(L > 0, slot > 0):
            cs = T.alloc_local((per * 3,), f32)
            wv = T.alloc_local((per * 4,), f32)
            xs = T.alloc_local((per * 8,), f32)
            nw = T.alloc_local((8,), bf16)
            for e in T.unroll(per):
                ch = cb * CH + e * threads + tid
                for j in T.unroll(3):
                    cs[e * 3 + j] = T.cast(state[slot, ch, off + j], f32)
                for j in T.unroll(4):
                    wv[e * 4 + j] = T.cast(w[ch, j], f32)
                for t in T.unroll(8):
                    xs[e * 8 + t] = T.if_then_else(t < L, T.cast(x[bos + T.min(t, L - 1), ch], f32), T.float32(0.0))
                # new window (values loaded above / from the state before any store)
                for i in T.unroll(8):
                    if i < S:
                        nw[i] = T.if_then_else(i < S - L, state[slot, ch, T.min(off + i + 1, off + 7)],
                                               T.cast(xs[e * 8 + T.max(i - (S - L), 0)], bf16))
                for i in T.unroll(8):
                    if i < S:
                        state[slot, ch, i] = nw[i]
            for e in T.unroll(per):
                ch = cb * CH + e * threads + tid
                c0 = T.alloc_var(f32)
                c1 = T.alloc_var(f32)
                c2 = T.alloc_var(f32)
                acc = T.alloc_var(f32)
                c0 = cs[e * 3]
                c1 = cs[e * 3 + 1]
                c2 = cs[e * 3 + 2]
                for t in T.serial(L):
                    acc = T.call_extern(f32, "hz_cv_add", T.float32(0.0), pm(c0, wv[e * 4]))
                    acc = T.call_extern(f32, "hz_cv_add", acc, pm(c1, wv[e * 4 + 1]))
                    acc = T.call_extern(f32, "hz_cv_add", acc, pm(c2, wv[e * 4 + 2]))
                    acc = T.call_extern(f32, "hz_cv_add", acc, pm(xs[e * 8 + t], wv[e * 4 + 3]))
                    out[bos + t, ch] = T.cast(T.call_extern(f32, "hz_cv_silu", acc), bf16)
                    c0 = c1
                    c1 = c2
                    c2 = xs[e * 8 + t]

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def conv_update_kernel(C: int, XS: int, SS: int, SD: int, ST: int, OS: int, CH: int = 256, threads: int = 128,
                       SL: int = 6):
    """x (L, C) row stride XS; state (slots, C, SL) strides (SS, SD, ST); out (L, C) row stride OS (may alias x)."""
    L = T.dynamic("L")
    NR = T.dynamic("NR")
    NC = T.dynamic("NC")
    NS = T.dynamic("NS")
    SW = T.dynamic("SW")
    bf16, i32 = T.bfloat16, T.int32
    body = conv_update_body(C, CH, threads)

    @T.prim_func
    def main(x: T.StridedTensor((L, C), (XS, 1), bf16), w: T.Tensor((C, 4), bf16),
             state: T.StridedTensor((NS, C, SL), (SS, SD, ST), bf16), sidx: T.Tensor((NR, SW), i32),
             nacc: T.Tensor((NR,), i32), cu: T.Tensor((NC,), i32), out: T.StridedTensor((L, C), (OS, 1), bf16),
             lmax: i32):
        with T.Kernel(NR, T.ceildiv(C, CH), threads=threads, prelude=CONV_SRC) as (r, cb):
            tid = T.get_thread_binding()
            body(x, w, state, sidx, nacc, cu, out, r, cb, tid, lmax)

    return main


def conv_update(x, state, weight, state_indices, num_accepted_tokens, query_start_loc, max_query_len, out=None):
    """spec-decode causal_conv1d_update (vLLM contract with conv_state_indices = state_indices[:, 0], no bias, SiLU)"""
    C = x.shape[1]
    if out is None:
        out = x
    if state_indices.dim() == 1:
        state_indices = state_indices.contiguous().view(-1, 1)
    key = (C, x.stride(0), state.stride(), out.stride(0), state.shape[2])
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = conv_update_kernel(C, x.stride(0), state.stride(0), state.stride(1), state.stride(2),
                                               out.stride(0), SL=state.shape[2])
    k(x, weight, state, state_indices, num_accepted_tokens, query_start_loc, out, int(max_query_len))
    return out
