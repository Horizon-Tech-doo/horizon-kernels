# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""SiLU(gate) * up for decode rows (the MLP activation between gate_up and down), K4-foundations helper (27-09-2026).

y[m, j] = bf16(silu(float(x[m, j])) * float(x[m, I + j])), silu(a) = a / (1 + exp(-a)); x = gate_up output (M, 2I).
mode 0: the formula above with IEEE division (the K4-foundations chain benchmark).
mode 1 / 2 (27-09-2026, for serving: HORIZON_K5_SILU): the exact instruction sequence of the Triton kernel Inductor
generates for vLLM's SiluAndMul.forward_native (torch 2.13, Triton 3.7): torch/_inductor/decomposition.py rewrites
silu(a) as a / (1 + exp(-a)) in fp32, exp is the CUDA math library's expf, and Triton lowers the fp32 "/" to div.full.f32 (the
approximate full-range division, Inductor's eager_numerics.division_rounding is off); the bf16 round trip between silu
and "* up" is elided (emulate_precision_casts off), so there is one rounding to bf16 at the end. Mode 2 additionally
forbids contracting expf's last multiply with the "+ 1" into an FMA (Triton may or may not do that; the exhaustive
bit test k4/silu_bits.py decides which mode matches).
One CTA covers VEC * threads columns of one row; deterministic, no workspace, CUDA-graph capturable.
"""

import tilelang
import tilelang.language as T

VEC = 8
_KERNELS = {}

SILU_SRC = r"""
#ifndef HZ_SILU_SRC_DEFINED
#define HZ_SILU_SRC_DEFINED
// silu(a) * u as Inductor's Triton kernel computes it: a / (1 + expf(-a)) with div.full.f32, then * u (fp32)
__device__ __forceinline__ float hz_silu_mul_ind(float a, float u) {
  float e = expf(-a);
  float d = 1.0f + e;
  float s;
  asm("div.full.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(d));
  return s * u;
}
// the same, with an optimisation barrier so expf's last multiply is never fused with the + 1
__device__ __forceinline__ float hz_silu_mul_ind2(float a, float u) {
  float e = expf(-a);
  asm volatile("" : "+f"(e));
  float d = 1.0f + e;
  float s;
  asm("div.full.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(d));
  return s * u;
}
#endif
"""


def silu_mul_body(I: int, threads: int, mode: int = 0):
    """mode 0: IEEE division (benchmark formula); 1 / 2: Inductor's exact Triton sequence (needs SILU_SRC in the
    kernel prelude)."""
    bf16, f32 = T.bfloat16, T.float32
    fn = {1: "hz_silu_mul_ind", 2: "hz_silu_mul_ind2"}.get(mode)

    @T.macro
    def body(x, y, row, c0, tid):
        xb = T.alloc_local((VEC,), bf16)
        ub = T.alloc_local((VEC,), bf16)
        ob = T.alloc_local((VEC,), bf16)
        j0 = c0 + tid * VEC
        if j0 < I:
            for v in T.vectorized(VEC):
                xb[v] = x[row, j0 + v]
            for v in T.vectorized(VEC):
                ub[v] = x[row, I + j0 + v]
            if fn is None:
                for v in T.unroll(VEC):
                    ob[v] = T.cast(T.cast(xb[v], f32) / (T.float32(1.0) + T.exp(-T.cast(xb[v], f32)))
                                   * T.cast(ub[v], f32), bf16)
            else:
                for v in T.unroll(VEC):
                    ob[v] = T.cast(T.call_extern(f32, fn, T.cast(xb[v], f32), T.cast(ub[v], f32)), bf16)
            for v in T.vectorized(VEC):
                y[row, j0 + v] = ob[v]

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def silu_mul_kernel(I: int, threads: int = 128, pdl: int = 0, stamps: int = 0, mode: int = 0):
    from tilelang.contrib.horizon_k4 import pdl as _pdl, hkt as _hkt
    pdl = _pdl.check(pdl)
    assert I % VEC == 0
    M = T.dynamic("M")
    TSN = T.dynamic("TSN")
    per_cta = VEC * threads
    nc = (I + per_cta - 1) // per_cta
    body = silu_mul_body(I, threads, mode)
    src = (_pdl.PDL_SRC if pdl else "") + (_hkt.HKT_SRC if stamps else "") + (SILU_SRC if mode else "")

    @T.macro
    def _kernel(x, y, TS):
        with T.Kernel(nc, M, threads=threads, prelude=src) as (bc, row):
            tid = T.get_thread_binding()
            if stamps:
                _hkt.stamp(TS, row * nc + bc, _hkt.ENTRY, tid)
                _hkt.stamp_smid(TS, row * nc + bc, tid)
            if _pdl.trigger_early(pdl):
                T.pdl_trigger()
            if pdl:
                T.pdl_sync()
            if stamps:
                _hkt.stamp(TS, row * nc + bc, _hkt.SYNC, tid)
            body(x, y, row, bc * per_cta, tid)
            if _pdl.trigger_late(pdl):
                T.pdl_trigger()
            if stamps:
                _hkt.stamp(TS, row * nc + bc, _hkt.LOOP, tid)
                _hkt.stamp_exit(TS, row * nc + bc, tid)

    if stamps:
        @T.prim_func
        def main_s(x: T.Tensor((M, 2 * I), T.bfloat16), y: T.Tensor((M, I), T.bfloat16),
                   TS: T.Tensor((TSN, _hkt.SLOTS), T.int64)):
            _kernel(x, y, TS)
        return main_s

    @T.prim_func
    def main(x: T.Tensor((M, 2 * I), T.bfloat16), y: T.Tensor((M, I), T.bfloat16)):
        _kernel(x, y, None)

    return main


def get_kernel(I, threads=128, pdl=0, stamps=0, mode=0):
    key = (I, threads, pdl, stamps, mode)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = (silu_mul_kernel(I, threads, pdl, stamps, mode) if mode
                             else silu_mul_kernel(I, threads, pdl, stamps))
    return k


def silu_mul(x, pdl=None, ts=None, out=None, mode=0):
    """x (M, 2I) bf16 contiguous -> (M, I) bf16. mode: see the module docstring (1/2 = bit-identical to Inductor)."""
    import torch
    from tilelang.contrib.horizon_k4 import pdl as _pdl
    M, I2 = x.shape
    I = I2 // 2
    pdl = _pdl.resolve(pdl, M)
    y = out if out is not None else torch.empty(M, I, dtype=torch.bfloat16, device=x.device)
    k = get_kernel(I, 128, pdl, 1 if ts is not None else 0, mode)
    if ts is not None:
        k(x, y, ts)
    else:
        k(x, y)
    return y
