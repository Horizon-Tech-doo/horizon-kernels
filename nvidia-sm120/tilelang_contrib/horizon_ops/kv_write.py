# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""KV-cache write for the FP8 paged cache, bit-identical to vLLM 0.29's reshape_and_cache_flash_kernel<bf16, uint8,
kFp8E4M3> (csrc/libtorch_stable/cache_kernels.cu, built WITHOUT fast-math): per element
    cache = __nv_cvt_float_to_fp8(float(x) / scale, __NV_SATFINITE, __NV_E4M3)      (IEEE division, one rounding)
at cache[slot // page, slot % page, head, d] through arbitrary strides (vLLM's FlashInfer SM120 cache is HND with K and
V interleaved per token). slot < 0 (padding) writes nothing.
Body: kv_write_body(...)(key, value, kc, vc, slots, ks, vs, t, tid): one task = one token (all kv heads).
DEPS: kv_write(t): the token's k (after q/k-norm + RoPE) and v columns of the qkv projection.
"""

import tilelang
import tilelang.language as T

_KERNELS = {}

KV_SRC = r"""
#ifndef HZ_KV_SRC_DEFINED
#define HZ_KV_SRC_DEFINED
#include <cuda_fp8.h>
__device__ __forceinline__ unsigned char hz_kv_fp8(float x, float scale) {
  return (unsigned char)__nv_cvt_float_to_fp8(__fdiv_rn(x, scale), __NV_SATFINITE, __NV_E4M3);
}
#endif
"""


def kv_write_body(KVH: int, D: int, PS: int, threads: int = 128):
    n = KVH * D
    assert n % threads == 0
    per = n // threads
    f32, i64 = T.float32, T.int64

    @T.macro
    def body(key, value, kc, vc, slots, ks, vs, t, tid):
        sl = T.alloc_var(i64)
        sl = slots[t]
        if sl >= 0:
            for e in T.unroll(per):
                idx = e * threads + tid
                h = idx // D
                d = idx % D
                kc[sl // PS, sl % PS, h, d] = T.call_extern(T.uint8, "hz_kv_fp8", T.cast(key[t, h, d], f32), ks[0])
                vc[sl // PS, sl % PS, h, d] = T.call_extern(T.uint8, "hz_kv_fp8", T.cast(value[t, h, d], f32), vs[0])

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def kv_write_kernel(KVH: int, D: int, PS: int, KS0: int, KS1: int, VS0: int, VS1: int, SP: int, ST: int, SH: int,
                    threads: int = 128):
    """key (NT, KVH, D) strides (KS0, KS1, 1), value strides (VS0, VS1, 1); caches as uint8 views indexed
    [page, position, head, d] with strides (SP, ST, SH, 1)."""
    NT = T.dynamic("NT")
    NP = T.dynamic("NP")
    NSL = T.dynamic("NSL")
    body = kv_write_body(KVH, D, PS, threads)

    @T.prim_func
    def main(key: T.StridedTensor((NT, KVH, D), (KS0, KS1, 1), T.bfloat16),
             value: T.StridedTensor((NT, KVH, D), (VS0, VS1, 1), T.bfloat16),
             kc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), T.uint8),
             vc: T.StridedTensor((NP, PS, KVH, D), (SP, ST, SH, 1), T.uint8),
             slots: T.Tensor((NSL,), T.int64), ks: T.Tensor((1,), T.float32), vs: T.Tensor((1,), T.float32)):
        with T.Kernel(NSL, threads=threads, prelude=KV_SRC) as t:
            tid = T.get_thread_binding()
            body(key, value, kc, vc, slots, ks, vs, t, tid)

    return main


def write(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale):
    """vLLM reshape_and_cache_flash contract for an NHD-indexed view (pages, page, heads, D) of the cache (any
    strides; pass cache.transpose / permute views), fp8 e4m3 cache, per-tensor scales, bf16 key/value."""
    import torch
    NT = slot_mapping.shape[0]
    key3 = key.view(key.shape[0], -1, key.shape[-1]) if key.dim() == 2 else key
    val3 = value.view(value.shape[0], -1, value.shape[-1]) if value.dim() == 2 else value
    kc, vc = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    _, PS, KVH, D = kc.shape
    k = _KERNELS.get((KVH, D, PS, key3.stride()[:2], val3.stride()[:2], kc.stride()[:3]))
    if k is None:
        k = _KERNELS[(KVH, D, PS, key3.stride()[:2], val3.stride()[:2], kc.stride()[:3])] = kv_write_kernel(
            KVH, D, PS, key3.stride(0), key3.stride(1), val3.stride(0), val3.stride(1), kc.stride(0), kc.stride(1),
            kc.stride(2))
    k(key3[:NT], val3[:NT], kc, vc, slot_mapping, k_scale.reshape(1).float(), v_scale.reshape(1).float())
