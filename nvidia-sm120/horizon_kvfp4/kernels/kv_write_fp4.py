# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""KV-cache write for the Horizon NVFP4 paged cache (hz-kvfp4 v1, format and exact rules in kv_fp4_ref.py).

Per (token, kv head, side): D/2 bytes of E2M1 codes (element 2j = low nibble of byte j) + D/16 E4M3 block scales (one
per 16 values) in vLLM 0.29's NVFP4 page geometry; a static fp32 global scale per (layer, kv head, side) (gk, gv).
Keys may be rotated first (HAD=1): k' = FWHT(k * signs) * D^-0.5 in fp32, butterfly order as in kv_fp4_ref.fwht; the
decode kernel then consumes q' = rotate_q(q) (attn_decode_fp4), and q' . k' = q . k.
Every arithmetic step is an explicit IEEE fp32 intrinsic (__fdiv_rn / __fmul_rn / __fadd_rn / __fsub_rn, no FMA
contraction), so the cache bytes are bit-identical to kv_fp4_ref.write_reference whatever the compiler flags.

Body: kv_write_fp4_body(...)(key, value, kd, ksf, vd, vsf, slots, gk, gv, sg, t, tid, X_s)
      one task = one token (all kv heads, K and V): 2 * KVH * D / 16 block tasks spread over the CTA's threads
      (KVH 4, D 256: 128 blocks = one per thread of a 128-thread CTA). X_s (KVH, D) fp32 shared, only with HAD.
DEPS: kv_write_fp4(t): the token's k (after q/k-norm + RoPE) and v columns of the qkv projection.
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
BLOCK = 16

FP4_SRC = r"""
#ifndef HZ_FP4_SRC_DEFINED
#define HZ_FP4_SRC_DEFINED
#include <cuda_fp8.h>
// e4m3fn encode, round to nearest even, saturate to 448 (x >= 0, finite)
__device__ __forceinline__ unsigned hz_e4m3_enc(float x) {
  return (unsigned)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
// e4m3fn decode of a non-negative finite code (0 .. 126), exact
__device__ __forceinline__ float hz_e4m3_dec(unsigned c) {
  unsigned e = (c >> 3) & 15u, m = c & 7u;
  return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : __fmul_rn((float)m, 0.001953125f);
}
// e2m1 magnitude code of y >= 0: nearest even on {0,.5,1,1.5,2,3,4,6}, saturating
__device__ __forceinline__ unsigned hz_e2m1_mag(float y) {
  return y <= 0.25f ? 0u : y < 0.75f ? 1u : y <= 1.25f ? 2u : y < 1.75f ? 3u
       : y <= 2.5f ? 4u : y < 3.5f ? 5u : y <= 5.0f ? 6u : 7u;
}
__device__ __forceinline__ float hz_e2m1_val(unsigned m) {
  unsigned e = (m >> 1) & 3u, l = m & 1u;
  return e ? __uint_as_float(((e + 126u) << 23) | (l << 22)) : (l ? 0.5f : 0.0f);
}
// effective block scale s = e4m3(c) * g
__device__ __forceinline__ float hz_fp4_scale(unsigned c, float g) { return __fmul_rn(hz_e4m3_dec(c), g); }
// candidate block scale codes: amax / (g * 6) and amax / (g * 4)
__device__ __forceinline__ unsigned hz_fp4_c6(float amax, float g) { return hz_e4m3_enc(__fdiv_rn(amax, __fmul_rn(g, 6.0f))); }
__device__ __forceinline__ unsigned hz_fp4_c4(float amax, float g) { return hz_e4m3_enc(__fdiv_rn(amax, __fmul_rn(g, 4.0f))); }
// squared error of one value under effective scale s (0 when s == 0 -> x * x)
__device__ __forceinline__ float hz_fp4_sqerr(float x, float s) {
  float ax = fabsf(x);
  unsigned m = (s > 0.0f) ? hz_e2m1_mag(__fdiv_rn(ax, s)) : 0u;
  float d = __fsub_rn(__fmul_rn(hz_e2m1_val(m), s), ax);
  return __fmul_rn(d, d);
}
// signed 4-bit code of one value (bit 3 = x < 0)
__device__ __forceinline__ unsigned hz_fp4_code(float x, float s) {
  unsigned m = (s > 0.0f) ? hz_e2m1_mag(__fdiv_rn(fabsf(x), s)) : 0u;
  return m | (x < 0.0f ? 8u : 0u);
}
__device__ __forceinline__ float hz_fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float hz_fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float hz_fmul(float a, float b) { return __fmul_rn(a, b); }
#endif
"""

PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _log2(n: int) -> int:
    assert n > 0 and n & (n - 1) == 0, n
    return n.bit_length() - 1


def fwht_scale(D: int) -> float:
    """D^-0.5 rounded to fp32 (same constant as kv_fp4_ref.fwht_scale)"""
    import struct
    return struct.unpack("f", struct.pack("f", D ** -0.5))[0]


def fwht_body(ROWS: int, D: int, threads: int):
    """T.macro: in-place unnormalised FWHT of every row of X_s (ROWS, D) fp32 in shared memory, stage h = 1, 2, ..,
    D/2, butterfly (x[i], x[i+h]) <- (x[i] + x[i+h], x[i] - x[i+h]) for (i & h) == 0 (kv_fp4_ref.fwht order). The
    caller has filled X_s and synchronised; returns synchronised."""
    LOG = _log2(D)
    npair = ROWS * D // 2
    per = (npair + threads - 1) // threads
    f32, i32 = T.float32, T.int32

    @T.macro
    def body(X_s, tid):
        a = T.alloc_var(f32)
        c = T.alloc_var(f32)
        for st in T.unroll(LOG):
            for e in T.unroll(per):
                p = e * threads + tid
                if p < npair:
                    r = p % (D // 2)
                    row = p // (D // 2)
                    i = T.shift_left(T.shift_right(r, st), st + 1) + T.bitwise_and(r, T.shift_left(1, st) - 1)
                    a = X_s[row, i]
                    c = X_s[row, i + T.shift_left(1, st)]
                    X_s[row, i] = T.call_extern(f32, "hz_fadd", a, c)
                    X_s[row, i + T.shift_left(1, st)] = T.call_extern(f32, "hz_fsub", a, c)
            T.sync_threads()

    return body


def quant_block_body(search: int = 1):
    """T.macro: quantise the 16 fp32 values in xs (local) with global scale g -> writes 8 code bytes to
    dst[p, o, h, blk * 8 + j] and the scale byte to sdst[p, o, h, blk]. Scale search as in kv_fp4_ref."""
    f32, i32 = T.float32, T.int32

    @T.macro
    def body(xs, g, dst, sdst, p, o, h, blk):
        amax = T.alloc_var(f32)
        best = T.alloc_var(f32)
        err = T.alloc_var(f32)
        s = T.alloc_var(f32)
        c0 = T.alloc_var(i32)
        cb = T.alloc_var(i32)
        cc = T.alloc_local((4,), i32)
        amax = 0.0
        for i in T.unroll(BLOCK):
            amax = T.max(amax, T.abs(xs[i]))
        c0 = T.call_extern(i32, "hz_fp4_c6", amax, g)
        cb = c0
        if search:
            cc[0] = c0
            cc[1] = T.min(c0 + 1, 126)
            cc[2] = T.max(c0 - 1, 0)
            cc[3] = T.call_extern(i32, "hz_fp4_c4", amax, g)
            for k in T.unroll(4):
                s = T.call_extern(f32, "hz_fp4_scale", cc[k], g)
                err = 0.0
                for i in T.unroll(BLOCK):
                    err = T.call_extern(f32, "hz_fadd", err, T.call_extern(f32, "hz_fp4_sqerr", xs[i], s))
                if k == 0:
                    best = err
                elif err < best:
                    best = err
                    cb = cc[k]
        s = T.call_extern(f32, "hz_fp4_scale", cb, g)
        for j in T.unroll(BLOCK // 2):
            dst[p, o, h, blk * (BLOCK // 2) + j] = T.cast(
                T.bitwise_or(T.call_extern(i32, "hz_fp4_code", xs[2 * j], s),
                             T.shift_left(T.call_extern(i32, "hz_fp4_code", xs[2 * j + 1], s), 4)), T.uint8)
        sdst[p, o, h, blk] = T.cast(cb, T.uint8)

    return body


def kv_write_fp4_body(KVH: int, D: int, PS: int, threads: int = 128, had: int = 1, search: int = 1):
    """T.macro: one task = one token. had=1 rotates keys (FWHT in X_s), values are never rotated."""
    NB = KVH * D // BLOCK                   # blocks per side
    rounds = (2 * NB + threads - 1) // threads
    nel = KVH * D
    per = (nel + threads - 1) // threads
    csc = fwht_scale(D)
    f32, i64 = T.float32, T.int64
    fwht = fwht_body(KVH, D, threads)
    qb = quant_block_body(search)

    @T.macro
    def body(key, value, kd, ksf, vd, vsf, slots, gk, gv, sg, t, tid, X_s):
        sl = T.alloc_var(i64)
        xs = T.alloc_local((BLOCK,), f32)
        sl = slots[t]
        if sl >= 0:
            if had:
                for e in T.unroll(per):
                    idx = e * threads + tid
                    if idx < nel:
                        X_s[idx // D, idx % D] = T.call_extern(f32, "hz_fmul", T.cast(key[t, idx // D, idx % D], f32),
                                                               sg[idx % D])
                T.sync_threads()
                fwht(X_s, tid)
            for r in T.unroll(rounds):
                bi = r * threads + tid
                if bi < NB:                                      # key block
                    h = bi // (D // BLOCK)
                    blk = bi % (D // BLOCK)
                    for i in T.unroll(BLOCK):
                        if had:
                            xs[i] = T.call_extern(f32, "hz_fmul", X_s[h, blk * BLOCK + i], T.float32(csc))
                        else:
                            xs[i] = T.cast(key[t, h, blk * BLOCK + i], f32)
                    qb(xs, gk[h], kd, ksf, sl // PS, sl % PS, h, blk)
                elif bi < 2 * NB:                                # value block
                    h = (bi - NB) // (D // BLOCK)
                    blk = (bi - NB) % (D // BLOCK)
                    for i in T.unroll(BLOCK):
                        xs[i] = T.cast(value[t, h, blk * BLOCK + i], f32)
                    qb(xs, gv[h], vd, vsf, sl // PS, sl % PS, h, blk)

    return body


@tilelang.jit(pass_configs=PASS_CONFIGS)
def kv_write_fp4_kernel(KVH: int, D: int, PS: int, KS0: int, KS1: int, VS0: int, VS1: int,
                        DSP: int, DST: int, DSH: int, SSP: int, SST: int, SSH: int,
                        threads: int = 128, had: int = 1, search: int = 1):
    """key (NT, KVH, D) strides (KS0, KS1, 1), value strides (VS0, VS1, 1). Cache parts as uint8 views indexed
    [page, position, head, byte]: data (NP, PS, KVH, D/2) strides (DSP, DST, DSH, 1), scales (NP, PS, KVH, D/16)
    strides (SSP, SST, SSH, 1) (K and V parts share strides: kv_fp4_ref.split_views)."""
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    NSL = T.dynamic("NSL")
    body = kv_write_fp4_body(KVH, D, PS, threads, had, search)
    DH, SH_ = D // 2, D // BLOCK

    @T.prim_func
    def main(key: T.StridedTensor((NT, KVH, D), (KS0, KS1, 1), T.bfloat16),
             value: T.StridedTensor((NT, KVH, D), (VS0, VS1, 1), T.bfloat16),
             kd: T.StridedTensor((NP, PS, KVH, DH), (DSP, DST, DSH, 1), T.uint8),
             ksf: T.StridedTensor((NP, PS, KVH, SH_), (SSP, SST, SSH, 1), T.uint8),
             vd: T.StridedTensor((NP, PS, KVH, DH), (DSP, DST, DSH, 1), T.uint8),
             vsf: T.StridedTensor((NP, PS, KVH, SH_), (SSP, SST, SSH, 1), T.uint8),
             slots: T.Tensor((NSL,), T.int64), gk: T.Tensor((KVH,), T.float32), gv: T.Tensor((KVH,), T.float32),
             sg: T.Tensor((D,), T.float32)):
        with T.Kernel(NSL, threads=threads, prelude=FP4_SRC) as t:
            tid = T.get_thread_binding()
            X_s = T.alloc_shared((KVH, D), T.float32)
            body(key, value, kd, ksf, vd, vsf, slots, gk, gv, sg, t, tid, X_s)

    return main


def get_write(KVH, D, PS, key_strides, val_strides, d_strides, s_strides, threads=128, had=1, search=1):
    k = (KVH, D, PS, tuple(key_strides), tuple(val_strides), tuple(d_strides), tuple(s_strides), threads, had, search)
    ker = _KERNELS.get(k)
    if ker is None:
        ker = _KERNELS[k] = kv_write_fp4_kernel(KVH, D, PS, *key_strides, *val_strides, *d_strides, *s_strides,
                                                threads, had, search)
    return ker


_ONES = {}


def write(key, value, k_data, k_sf, v_data, v_sf, slot_mapping, gk, gv, signs=None, search=True):
    """key / value (NT, KVH, D) or (NT, KVH * D) bf16; k_data / v_data (NP, PS, KVH, D/2) and k_sf / v_sf
    (NP, PS, KVH, D/16) uint8 [page, position, head, byte] views (kv_fp4_ref.split_views, or the vLLM NVFP4 cache
    through nvfp4_split_data_scale + permute); slot_mapping (NSL,) int64, slot < 0 skipped; gk / gv (KVH,) fp32;
    signs (D,) fp32 +-1 -> keys rotated (the decode must then get rotate_q(q))."""
    import torch
    NT = slot_mapping.shape[0]
    key3 = key.view(key.shape[0], -1, k_data.shape[-1] * 2) if key.dim() == 2 else key
    val3 = value.view(value.shape[0], -1, v_data.shape[-1] * 2) if value.dim() == 2 else value
    _, PS, KVH, DH = k_data.shape
    D = DH * 2
    assert k_data.stride() == v_data.stride() and k_sf.stride() == v_sf.stride()
    assert k_data.dtype == torch.uint8 and k_sf.dtype in (torch.uint8, torch.float8_e4m3fn)
    had = 1 if signs is not None else 0
    if signs is None:
        signs = _ONES.get((D, key.device))
        if signs is None:
            signs = _ONES[(D, key.device)] = torch.ones(D, dtype=torch.float32, device=key.device)
    ker = get_write(KVH, D, PS, key3.stride()[:2], val3.stride()[:2], k_data.stride()[:3], k_sf.stride()[:3],
                    had=had, search=int(bool(search)))
    ker(key3[:NT], val3[:NT], k_data, k_sf.view(torch.uint8), v_data, v_sf.view(torch.uint8), slot_mapping,
        gk.float().contiguous(), gv.float().contiguous(), signs.float().contiguous())
