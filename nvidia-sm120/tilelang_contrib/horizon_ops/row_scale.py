# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-row activation scale for the row-scaled NVFP4 prefill (sm120_nvfp4.prefill_rs.quantize_rs), in ONE kernel.

prefill_rs.quantize_rs computes, before its quantisation kernel, with five torch ops (abs -> a full |x| copy, amax
reduce, clamp, two where):
    amax  = max_k |x[m, k]|                                (bf16, exact)
    rs    = (1 / max(amax, 1e-30)) * 2600 if amax > 0 else 1   (Tensor.__rtruediv__: IEEE reciprocal, one multiply)
    rsinv = amax * (1 / 2600)         if amax > 0 else 1   (PyTorch divides by a CPU scalar as a * (1/b), both fp32)
This kernel reads x once and writes rs and rsinv (bit-identical; the |x| temporary, 4 of the 5 launches and 2 of the
4 passes over x disappear). One CTA per row, 256 threads, 16-byte loads, fixed-order max (max is exact anyway).

quantize_rs also drops quantize_rs's torch pad (a full copy + fill of x whenever the row count is not a multiple of
128, i.e. almost every prefill chunk: 0.46 s of a 64k-token prompt): rows past the end are read as zeros inside a
copy of prefill_rs.nvfp4_quant_rs_kernel whose grid comes from the padded row count (zero rows give zero codes and
scale codes, exactly what the padded zeros gave).
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
ROW_TARGET = 2600.0
VEC = 8

RS_SRC = r"""
#ifndef HZ_RS_SRC_DEFINED
#define HZ_RS_SRC_DEFINED
__device__ __forceinline__ float hz_rs_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float hz_rs_mul(float a, float b) { return __fmul_rn(a, b); }
#endif
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def row_scale_kernel(K: int, threads: int = 256):
    """grid = rs rows (Mp); rows >= x rows (Mx) get rs = rsinv = 1 (the padded zero rows of the torch path)"""
    M = T.dynamic("M")
    MX = T.dynamic("MX")
    assert K % (VEC * threads) == 0 or K % VEC == 0
    per = T.ceildiv(K, VEC * threads)
    bf16, f32 = T.bfloat16, T.float32
    warps = threads // 32

    @T.prim_func
    def main(x: T.Tensor((MX, K), bf16), rs: T.Tensor((M,), f32), rsinv: T.Tensor((M,), f32)):
        with T.Kernel(M, threads=threads, prelude=RS_SRC) as row:
            tid = T.get_thread_binding()
            xb = T.alloc_local((VEC,), bf16)
            red = T.alloc_shared((warps,), f32)
            am = T.alloc_var(f32)
            tot = T.alloc_var(f32)
            am = 0.0
            for i in T.serial(per):
                k0 = (i * threads + tid) * VEC
                if T.And(k0 < K, row < MX):
                    for v in T.vectorized(VEC):
                        xb[v] = x[row, k0 + v]
                    for v in T.unroll(VEC):
                        am = T.max(am, T.abs(T.cast(xb[v], f32)))
            am = T.max(am, T.shfl_xor(am, 16))
            am = T.max(am, T.shfl_xor(am, 8))
            am = T.max(am, T.shfl_xor(am, 4))
            am = T.max(am, T.shfl_xor(am, 2))
            am = T.max(am, T.shfl_xor(am, 1))
            if tid % 32 == 0:
                red[tid // 32] = am
            T.sync_threads()
            if tid == 0:
                tot = red[0]
                for w in T.unroll(warps - 1):
                    tot = T.max(tot, red[w + 1])
                # PyTorch: 2600 / t is Tensor.__rtruediv__ = t.reciprocal() * 2600 (IEEE 1/t, then one multiply)
                rs[row] = T.if_then_else(tot > 0, T.call_extern(f32, "hz_rs_mul", T.call_extern(
                    f32, "hz_rs_div", T.float32(1.0), T.max(tot, T.float32(1e-30))), T.float32(ROW_TARGET)),
                    T.float32(1.0))
                rsinv[row] = T.if_then_else(tot > 0, T.call_extern(f32, "hz_rs_mul", tot, T.call_extern(
                    f32, "hz_rs_div", T.float32(1.0), T.float32(ROW_TARGET))),
                                            T.float32(1.0))

    return main


def row_scales(x, mp=None):
    """x (M, K) bf16 contiguous -> (rs, rsinv) fp32 (mp,), bit-identical to prefill_rs.quantize_rs's torch ops (on x
    padded with zero rows to mp)."""
    import torch
    M, K = x.shape
    mp = mp or M
    k = _KERNELS.get(K)
    if k is None:
        k = _KERNELS[K] = row_scale_kernel(K)
    rs = torch.empty(mp, dtype=torch.float32, device=x.device)
    rsinv = torch.empty(mp, dtype=torch.float32, device=x.device)
    k(x, rs, rsinv)
    return rs, rsinv


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def quant_rs_nopad_kernel(K: int, passes: int = 2, rows_per_cta: int = 32, num_threads: int = 512):
    """prefill_rs.nvfp4_quant_rs_kernel (same code, same outputs) with x holding MX <= M rows: the (M, ...) outputs are
    sized by the padded row count M and rows >= MX are quantised from zeros (guarded load instead of the TMA copy)."""
    from tilelang.contrib.sm120_nvfp4._device import (encode_ue4m3_ceil, decode_ue4m3, ue4m3_inverse, FP4_E2M1_MAX,
                                                       CHUNK_WORDS)
    M = T.dynamic("M")
    MX = T.dynamic("MX")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    kbn = K // 256
    f32 = T.float32

    @T.prim_func
    def main(x: T.Tensor((MX, K), T.bfloat16), rs: T.Tensor((M,), f32),
             A: T.Tensor((M, passes * (K // 2)), T.int8), SX: T.Tensor((passes, M * kbn, 4), T.uint32)):
        with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
            lane = T.get_lane_idx()
            warp = T.get_warp_idx_sync()
            x_s = T.alloc_shared((rows_per_cta, 256), T.bfloat16)
            codes = T.alloc_shared((passes, rows_per_cta, 16), T.int32)
            for i, j in T.Parallel(rows_per_cta, 256):
                x_s[i, j] = T.if_then_else(pid_m * rows_per_cta + i < MX,
                                           x[T.min(pid_m * rows_per_cta + i, MX - 1), pid_k256 * 256 + j],
                                           T.cast(0.0, T.bfloat16))
            T.sync_threads()
            glane = lane % 8
            st = (lane // 8) % 2
            rel0 = 2 * (glane // 4) + (glane % 2) + 4 * (((glane // 2) % 2) ^ st)
            for row_iter in T.serial(rows_per_warp):
                row = warp * rows_per_warp + row_iter
                grow = pid_m * rows_per_cta + row
                for k64 in T.serial(CHUNK_WORDS):
                    gbase = k64 * 64 + (lane // 8) * 16
                    byte = pid_k256 * 128 + k64 * 32 + lane
                    group = k64 * 4 + (lane // 8)
                    r0 = T.alloc_var(f32, init=T.cast(x_s[row, gbase + rel0], f32) * rs[grow])
                    r1 = T.alloc_var(f32, init=T.cast(x_s[row, gbase + rel0 + 8], f32) * rs[grow])
                    for p in T.unroll(passes):
                        am = T.alloc_var(f32, init=T.max(T.abs(r0), T.abs(r1)))
                        am = T.max(am, T.shfl_xor(am, 4, width=8))
                        am = T.max(am, T.shfl_xor(am, 2, width=8))
                        am = T.max(am, T.shfl_xor(am, 1, width=8))
                        code = T.alloc_var(T.int32, init=0)
                        sc = T.alloc_var(f32, init=0.0)
                        inv = T.alloc_var(f32, init=0.0)
                        if glane == 0:
                            code = encode_ue4m3_ceil(am / FP4_E2M1_MAX)
                            sc = decode_ue4m3(code)
                            inv = ue4m3_inverse(code)
                            codes[p, row, group] = code
                        sc = T.shfl_sync(sc, 0, width=8)
                        inv = T.shfl_sync(inv, 0, width=8)
                        f0 = T.cast(r0 * inv, T.float4_e2m1fn)
                        f1 = T.cast(r1 * inv, T.float4_e2m1fn)
                        A[grow, p * (K // 2) + byte] = T.cast(T.reinterpret(f0, T.uint8) | (T.reinterpret(f1, T.uint8) << 4), T.int8)
                        r0 = r0 - T.cast(f0, f32) * sc
                        r1 = r1 - T.cast(f1, f32) * sc
            T.sync_threads()
            for row_iter in T.serial(rows_per_warp):
                row = warp * rows_per_warp + row_iter
                grow = pid_m * rows_per_cta + row
                if lane < CHUNK_WORDS:
                    for p in T.unroll(passes):
                        SX[p, ((grow // 128) * kbn + pid_k256) * 128 + grow % 128, lane] = (
                            T.cast(codes[p, row, lane * 4], T.uint32)
                            | (T.cast(codes[p, row, lane * 4 + 1], T.uint32) << 8)
                            | (T.cast(codes[p, row, lane * 4 + 2], T.uint32) << 16)
                            | (T.cast(codes[p, row, lane * 4 + 3], T.uint32) << 24))

    return main


def quantize_rs(x, passes: int = 2):
    """drop-in for sm120_nvfp4.prefill_rs.quantize_rs with the fused row-scale kernel"""
    import torch
    from tilelang.contrib.sm120_nvfp4 import _ROWS
    assert x.dim() == 2 and x.shape[1] % 256 == 0, x.shape
    m, K = x.shape
    mp = (m + _ROWS - 1) // _ROWS * _ROWS
    x = x.contiguous()
    rs, rsinv = row_scales(x, mp)
    key = ("q", K, passes)
    qk = _KERNELS.get(key)
    if qk is None:
        qk = _KERNELS[key] = quant_rs_nopad_kernel(K, passes)
    X = torch.empty(mp, passes * (K // 2), dtype=torch.int8, device=x.device)
    SX = torch.empty(passes, mp * (K // 256), 4, dtype=torch.uint32, device=x.device)
    qk(x, rs, X, SX)
    return X, SX, rsinv
