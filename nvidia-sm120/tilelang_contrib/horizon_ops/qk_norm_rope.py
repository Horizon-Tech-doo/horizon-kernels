# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""A2: q/k Gemma-RMSNorm + partial interleaved-mRoPE + gate split for Qwen3.5/3.8 full attention, bit-identical to
vLLM 0.29's Triton kernel `_fused_qk_rmsnorm_rope_gate_kernel` (fused_qk_norm_rope.py, 4 warps, Triton 3.7).

Reference semantics (per token, per head of the 24 q heads + 4 k heads, head_dim 256, rotary_dim 64):
    x      = bf16 row (q part of [q | gate] for q heads, k for k heads)
    ss     = sum(x^2) in Triton's order: thread t (of 128) owns x[2t], x[2t+1]: v_t = fma(x[2t], x[2t], x[2t+1]^2);
             xor butterfly 16..1 inside each warp; the 4 warp sums w0..w3 combine as (w0 + w2) + (w1 + w3)
    inv    = rsqrt.approx.ftz(div.full(ss, 256) + eps)
    y[i]   = bf16((inv * x[i]) * (w[i] + 1))                         i >= 64 (pass-through)
    rotary (i < 32): x1 = float(bf16((inv * x[i]) * (w[i] + 1))), x2 = same for i + 32,
             pos = t/h/w position by i (interleaved mRoPE: i % 3 == 1 -> h, i % 3 == 2 and i < 30 -> w, else t)
             o1 = fma(x1, cos, -(x2 * sin)), o2 = fma(x2, cos, x1 * sin) -> bf16 at i and i + 32
    gate   = copied verbatim (q heads)
We emulate the same reduction tree with ONE warp per head (lane l carries the virtual threads l, l+32, l+64, l+96):
the order of every addition is identical, so the result is bit-identical while a CTA covers 4 heads.

Two layers: qk_norm_rope_body(...)(qg, k, qo, ko, go, qw, kw, cs, pos, tok, hd, lane) (one warp = one (token, head))
and the standalone kernel (grid tokens x 7, 128 threads).
DEPS: qk_norm_rope(tok, head): qkv+gate projection rows `tok` (q|gate cols of head, or k cols).
"""

import tilelang
import tilelang.language as T

_KERNELS = {}

QK_SRC = r"""
#ifndef HZ_QK_SRC_DEFINED
#define HZ_QK_SRC_DEFINED
__device__ __forceinline__ float hz_qk_inv(float ss, float eps) {
  float v, r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(v) : "f"(ss), "f"(256.0f));
  v = v + eps;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ float hz_mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float hz_fma_rn(float a, float b, float c) { return __fmaf_rn(a, b, c); }
#endif
"""


def qk_norm_rope_body(HQ: int, HK: int, D: int = 256, ROT: int = 64, sec_h: int = 11, sec_w: int = 10):
    assert D == 256 and ROT == 64
    half = ROT // 2
    f32, bf16, i64 = T.float32, T.bfloat16, T.int64

    @T.macro
    def body(qg, k, qo, ko, go, qw, kw, cs, pos, tok, hd, lane, eps):
        is_k = hd >= HQ
        xr = T.alloc_local((8,), f32)          # virtual thread j (0..3): elements 2(lane+32j), 2(lane+32j)+1
        ws = T.alloc_local((4,), f32)
        inv = T.alloc_var(f32)
        for j in T.unroll(4):
            e = 2 * (lane + 32 * j)
            for u in T.unroll(2):
                xr[2 * j + u] = T.if_then_else(is_k, T.cast(k[tok, (hd - HQ) * D + e + u], f32),
                                               T.cast(qg[tok, hd * 2 * D + e + u], f32))
            ws[j] = T.call_extern(f32, "hz_fma_rn", xr[2 * j], xr[2 * j],
                                  T.call_extern(f32, "hz_mul_rn", xr[2 * j + 1], xr[2 * j + 1]))
        for j in T.unroll(4):
            ws[j] = T.warp_reduce_sum(ws[j])
        inv = T.call_extern(f32, "hz_qk_inv", (ws[0] + ws[2]) + (ws[1] + ws[3]), eps)
        # pass-through tail (elements >= ROT): y = bf16((inv * x) * (w + 1))
        for j in T.unroll(4):
            e = 2 * (lane + 32 * j)
            for u in T.unroll(2):
                if e + u >= ROT:
                    if is_k:
                        ko[tok, (hd - HQ) * D + e + u] = T.cast(
                            T.call_extern(f32, "hz_mul_rn", T.call_extern(f32, "hz_mul_rn", inv, xr[2 * j + u]),
                                          T.cast(kw[e + u], f32) + T.float32(1.0)), bf16)
                    else:
                        qo[tok, hd * D + e + u] = T.cast(
                            T.call_extern(f32, "hz_mul_rn", T.call_extern(f32, "hz_mul_rn", inv, xr[2 * j + u]),
                                          T.cast(qw[e + u], f32) + T.float32(1.0)), bf16)
        # rotary pair (lane, lane + 32) for lane < 32 = half
        x1 = T.alloc_var(f32)
        x2 = T.alloc_var(f32)
        c = T.alloc_var(f32)
        sn = T.alloc_var(f32)
        p = T.alloc_var(i64)
        r = lane
        x1 = T.if_then_else(is_k, T.cast(k[tok, (hd - HQ) * D + r], f32), T.cast(qg[tok, hd * 2 * D + r], f32))
        x2 = T.if_then_else(is_k, T.cast(k[tok, (hd - HQ) * D + r + half], f32),
                            T.cast(qg[tok, hd * 2 * D + r + half], f32))
        x1 = T.cast(T.cast(T.call_extern(f32, "hz_mul_rn", T.call_extern(f32, "hz_mul_rn", inv, x1),
                                         T.if_then_else(is_k, T.cast(kw[r], f32), T.cast(qw[r], f32))
                                         + T.float32(1.0)), bf16), f32)
        x2 = T.cast(T.cast(T.call_extern(f32, "hz_mul_rn", T.call_extern(f32, "hz_mul_rn", inv, x2),
                                         T.if_then_else(is_k, T.cast(kw[r + half], f32), T.cast(qw[r + half], f32))
                                         + T.float32(1.0)), bf16), f32)
        p = T.if_then_else(T.And(r % 3 == 1, r < 3 * sec_h), T.cast(pos[1, tok], i64),
                           T.if_then_else(T.And(r % 3 == 2, r < 3 * sec_w), T.cast(pos[2, tok], i64),
                                          T.cast(pos[0, tok], i64)))
        c = T.cast(cs[p, r], f32)
        sn = T.cast(cs[p, r + half], f32)
        o1 = T.call_extern(f32, "hz_fma_rn", x1, c, -T.call_extern(f32, "hz_mul_rn", x2, sn))
        o2 = T.call_extern(f32, "hz_fma_rn", x2, c, T.call_extern(f32, "hz_mul_rn", x1, sn))
        if is_k:
            ko[tok, (hd - HQ) * D + r] = T.cast(o1, bf16)
            ko[tok, (hd - HQ) * D + r + half] = T.cast(o2, bf16)
        else:
            qo[tok, hd * D + r] = T.cast(o1, bf16)
            qo[tok, hd * D + r + half] = T.cast(o2, bf16)
            for j in T.unroll(D // 32):
                go[tok, hd * D + lane + 32 * j] = qg[tok, hd * 2 * D + D + lane + 32 * j]

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def qk_norm_rope_kernel(HQ: int, HK: int, D: int = 256, ROT: int = 64, QGS: int = 0, KS: int = 0, CSD: str = "bfloat16",
                        sec_h: int = 11, sec_w: int = 10, W: int = 4):
    """grid (tokens, ceil((HQ + HK) / W)), one warp per head. QGS / KS: row strides of the q|gate and k views."""
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    PT = T.dynamic("PT")
    HT = HQ + HK
    bf16, f32 = T.bfloat16, T.float32
    body = qk_norm_rope_body(HQ, HK, D, ROT, sec_h, sec_w)

    @T.prim_func
    def main(qg: T.StridedTensor((NT, HQ * 2 * D), (QGS, 1), bf16), k: T.StridedTensor((NT, HK * D), (KS, 1), bf16),
             qo: T.Tensor((NT, HQ * D), bf16), ko: T.Tensor((NT, HK * D), bf16), go: T.Tensor((NT, HQ * D), bf16),
             qw: T.Tensor((D,), bf16), kw: T.Tensor((D,), bf16), cs: T.Tensor((NP, ROT), CSD),
             pos: T.StridedTensor((3, NT), (PT, 1), T.int64), eps: f32):
        with T.Kernel(NT, T.ceildiv(HT, W), threads=32 * W, prelude=QK_SRC) as (tok, hb):
            tid = T.get_thread_binding()
            hd = hb * W + tid // 32
            if hd < HT:
                body(qg, k, qo, ko, go, qw, kw, cs, pos, tok, hd, tid % 32, eps)

    return main
