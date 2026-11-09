# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K5 "add_rmsnorm": residual add + Gemma-style RMSNorm (y = x * rsqrt(mean(x^2) + eps) * (1 + w)) for decode rows.

Semantics (vLLM's ir.ops.fused_add_rms_norm / rms_norm with weight = w.float() + 1, GemmaRMSNorm):
    s = float(x) + float(r)              (only with a residual; else s = float(x))
    r_out = bf16(s)
    rstd = rsqrt(sum(s^2) / H + eps)     (the variance of the UNROUNDED fp32 sum, as vLLM)
    y = bf16((s * rstd) * (float(w) + 1))
Everything is fp32 and elementwise-identical to vLLM's native code; only sum(s^2) is added in a different (fixed) order
than Inductor's Triton reduction, so rstd can differ in its last bit and y by 1 bf16 ulp on rare elements.

Two layers (K4 reuse):
  add_rmsnorm_body(H, eps, has_res, threads) -> T.macro body(x, r, w, y, r_out, row, tid, red, tot)
      Normalises ONE row, executed by `threads` threads of one CTA (thread index tid in [0, threads)).
      Inputs  x, r (M, H) bf16 global (r ignored without residual), w (H,) bf16 global (Gemma weight, 1 + w applied).
      Outputs y[row], r_out[row] (H,) bf16 global (r_out only with a residual). In-place use (y is x, r_out is r) is
      safe: each thread reads its elements before it writes the same elements.
      Shared  red: (threads // 32,) fp32, tot: (1,) fp32 -> 4 * (threads / 32 + 1) bytes (84 B at 640 threads).
      Registers: H / threads fp32 row values + 4 x 16-byte vectors; 32 regs at H 5120 / 640 threads (ptxas sm_120a).
      Synchronisation: two CTA barriers inside; the second publishes rstd. A task that follows immediately may reuse
      red/tot only after its own first barrier (the body writes red before its first barrier and tot between the two,
      so back-to-back bodies on the same buffers need one T.sync_threads() between them; the wrapper launches one row
      per CTA and needs none).
  add_rmsnorm_kernel(H, eps, has_res, threads): the standalone wrapper K5 launches (grid M, one CTA per row).
"""

import tilelang
import tilelang.language as T

VEC = 8
_KERNELS = {}


def add_rmsnorm_body(H: int, eps: float, has_res: bool, threads: int):
    assert H % (VEC * threads) == 0 and threads % 32 == 0, (H, threads)
    per = H // (VEC * threads)
    warps = threads // 32
    bf16, f32 = T.bfloat16, T.float32

    @T.macro
    def body(x, r, w, y, r_out, row, tid, red, tot):
        s = T.alloc_local((per * VEC,), f32)
        xb = T.alloc_local((VEC,), bf16)
        rb = T.alloc_local((VEC,), bf16)
        ob = T.alloc_local((VEC,), bf16)
        wb = T.alloc_local((VEC,), bf16)
        ss = T.alloc_var(f32)
        t = T.alloc_var(f32)
        rstd = T.alloc_var(f32)
        ss = 0.0
        for i in T.unroll(per):
            k0 = (i * threads + tid) * VEC
            for v in T.vectorized(VEC):
                xb[v] = x[row, k0 + v]
            if has_res:
                for v in T.vectorized(VEC):
                    rb[v] = r[row, k0 + v]
                for v in T.unroll(VEC):
                    s[i * VEC + v] = T.cast(xb[v], f32) + T.cast(rb[v], f32)
                    ob[v] = T.cast(s[i * VEC + v], bf16)
                for v in T.vectorized(VEC):
                    r_out[row, k0 + v] = ob[v]
            else:
                for v in T.unroll(VEC):
                    s[i * VEC + v] = T.cast(xb[v], f32)
            for v in T.unroll(VEC):
                ss = ss + s[i * VEC + v] * s[i * VEC + v]
        ss = T.warp_reduce_sum(ss)
        if tid % 32 == 0:
            red[tid // 32] = ss
        T.sync_threads()
        if tid == 0:
            t = red[0]
            for j in T.unroll(warps - 1):
                t = t + red[j + 1]
            tot[0] = T.rsqrt(t / T.float32(H) + T.float32(eps))
        T.sync_threads()
        rstd = tot[0]
        for i in T.unroll(per):
            k0 = (i * threads + tid) * VEC
            for v in T.vectorized(VEC):
                wb[v] = w[k0 + v]
            for v in T.unroll(VEC):
                ob[v] = T.cast((s[i * VEC + v] * rstd) * (T.cast(wb[v], f32) + T.float32(1.0)), bf16)
            for v in T.vectorized(VEC):
                y[row, k0 + v] = ob[v]

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def add_rmsnorm_kernel(H: int, eps: float, has_res: bool, threads: int = 640):
    M = T.dynamic("M")
    bf16 = T.bfloat16
    body = add_rmsnorm_body(H, eps, has_res, threads)

    @T.prim_func
    def main(x: T.Tensor((M, H), bf16), r: T.Tensor((M, H), bf16), w: T.Tensor((H,), bf16),
             y: T.Tensor((M, H), bf16), r_out: T.Tensor((M, H), bf16)):
        with T.Kernel(M, threads=threads) as row:
            red = T.alloc_shared((threads // 32,), T.float32)
            tot = T.alloc_shared((1,), T.float32)
            tid = T.get_thread_binding()
            body(x, r, w, y, r_out, row, tid, red, tot)

    return main


def get_kernel(H, eps, has_res, threads=640):
    key = (H, float(eps), bool(has_res), threads)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = add_rmsnorm_kernel(H, float(eps), bool(has_res), threads)
    return k


def _threads(H):
    return 640 if H % (VEC * 640) == 0 else 128


def supported(H):
    return H % (VEC * 128) == 0


def add_rmsnorm(x, r, w, eps):
    """(y, r_out) for x, r (M, H) bf16 contiguous, w (H,) bf16 (Gemma weight: the kernel applies 1 + w)."""
    import torch
    y = torch.empty_like(x)
    r_out = torch.empty_like(x)
    get_kernel(x.shape[1], eps, True, _threads(x.shape[1]))(x, r, w, y, r_out)
    return y, r_out


def rmsnorm(x, w, eps):
    import torch
    y = torch.empty_like(x)
    get_kernel(x.shape[1], eps, False, _threads(x.shape[1]))(x, x, w, y, y)
    return y
