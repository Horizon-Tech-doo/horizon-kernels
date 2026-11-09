# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K5 "skinny": BF16 x BF16 -> BF16 GEMM y = x @ W^T for few output columns (N <= 512) and decode batches (M <= 48).

Why: cuBLAS picks a 16x16-tile tensor-core kernel for Qwen3.5/3.8's GDN in_proj_ba (N = 96, K = 5120, M = 4..128)
with grid (8, 1, 1) x 32 threads: 8 single-warp CTAs walk the whole K = 5120, ~31 us for 1 MB of weights (0.6 us of
DRAM time), 48 calls per decode step.

Design: CUDA-core dot products; a task computes an (MB rows x BN columns) output tile; `threads` threads split K in
16-byte vectors (thread t owns k vectors t, t + threads, ...). Each thread accumulates its MB x BN partial sums in fp32 in
a FIXED order, the warp reduces with a butterfly shuffle (fixed), and the warps' results are added in warp order
0..W-1 (fixed). Deterministic, CUDA-graph capturable (no workspace, no atomics) and batch-invariant: row m's result
does not depend on M or on the other rows (cuBLAS changes kernels, hence summation order, with M).
Rounding: fp32 accumulation, one bf16 rounding at the end: the same rounding class as cuBLAS (fp32 accumulate, bf16
out); results differ from cuBLAS only by summation order.

Two layers (K4 reuse):
  skinny_body(K, MB, BN, threads) -> T.macro body(x, W, Y, M, m0, n0, tid, red)
      One output tile per call, executed by `threads` threads of one CTA (thread index tid in [0, threads)).
      Inputs  x (M, K) bf16 global, W (N, K) bf16 global; output Y[m0:m0+MB, n0:n0+BN] (rows < M only).
      Shared  red: (threads // 32, MB * BN) fp32 = 4 * threads / 32 * MB * BN bytes (e.g. 4 warps, 8 x 4: 512 B).
      Registers (ptxas, sm_120a, 128 threads): 48 at MB 4 / BN 1, 96 at 8 x 1, 125 at 8 x 2, 181 at 8 x 4.
      Synchronisation: two CTA barriers (T.sync_threads) inside, the last one at the end, so `red` can be reused by the
      next task of a persistent kernel. No kernel-exit or grid-level synchronisation is assumed.
  skinny_bf16_kernel(N, K, MB, BN, threads): the standalone wrapper K5 launches (grid N/BN x ceil(M/MB)).

Padding rows (m0 + m >= M) read row M - 1 (always a valid address) and are never stored.
"""

import tilelang
import tilelang.language as T

VEC = 8                         # bf16 per 16-byte vector
SUPPORTS_M = 48                 # measured: faster than cuBLAS up to M = 48 (5.7 vs 5.9 us), slower above (8.5 vs 5.7 at 96)
_KERNELS = {}


def red_shape(MB, BN, threads):
    return (threads // 32, MB * BN)


def skinny_body(K: int, MB: int, BN: int, threads: int):
    """T.macro computing one (MB x BN) tile of Y = x @ W^T; see the module docstring for the contract."""
    assert K % (VEC * threads) == 0 and threads % 32 == 0, (K, threads)
    per = K // (VEC * threads)
    warps = threads // 32
    R = MB * BN
    bf16, f32 = T.bfloat16, T.float32

    @T.macro
    def body(x, W, Y, M, m0, n0, tid, red):
        acc = T.alloc_local((R,), f32)
        xv = T.alloc_local((MB * VEC,), bf16)
        wv = T.alloc_local((VEC,), bf16)
        s = T.alloc_var(f32)
        T.clear(acc)
        for i in T.unroll(per):
            k0 = (i * threads + tid) * VEC
            for m in T.unroll(MB):
                for v in T.vectorized(VEC):
                    xv[m * VEC + v] = x[T.min(m0 + m, M - 1), k0 + v]
            for c in T.unroll(BN):
                for v in T.vectorized(VEC):
                    wv[v] = W[n0 + c, k0 + v]
                for m in T.unroll(MB):
                    for v in T.unroll(VEC):
                        acc[m * BN + c] = acc[m * BN + c] + T.cast(xv[m * VEC + v], f32) * T.cast(wv[v], f32)
        for j in T.unroll(R):
            acc[j] = T.warp_reduce_sum(acc[j])
        if tid % 32 == 0:
            for j in T.unroll(R):
                red[tid // 32, j] = acc[j]
        T.sync_threads()
        for jj in T.serial((R + threads - 1) // threads):
            e = jj * threads + tid
            if e < R:
                s = red[0, e]
                for w in T.unroll(warps - 1):
                    s = s + red[w + 1, e]
                if m0 + e // BN < M:
                    Y[m0 + e // BN, n0 + e % BN] = T.cast(s, bf16)
        T.sync_threads()        # red is free again (persistent-kernel reuse)

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def skinny_bf16_kernel(N: int, K: int, MB: int, BN: int, threads: int = 128):
    assert N % BN == 0, (N, BN)
    M = T.dynamic("M")
    body = skinny_body(K, MB, BN, threads)

    @T.prim_func
    def main(x: T.Tensor((M, K), T.bfloat16), W: T.Tensor((N, K), T.bfloat16), Y: T.Tensor((M, N), T.bfloat16)):
        with T.Kernel(N // BN, T.ceildiv(M, MB), threads=threads) as (bn, bm):
            T.assume(M >= 1)  # v0.1.15 (#3294): M >= 1 is the launch contract; keeps the unguarded loads
            red = T.alloc_shared(red_shape(MB, BN, threads), T.float32)
            tid = T.get_thread_binding()
            body(x, W, Y, M, bm * MB, bn * BN, tid, red)

    return main


def config(N, K, M):
    """(MB, BN, threads) per batch size. Few rows: one column per CTA (N CTAs, most parallel); more rows: x is re-read
    by every CTA from L2, so group columns (BN) to cut that traffic. Tuned on the RTX PRO 6000 (K5 race)."""
    if M <= 8:
        return (M if M in (1, 2, 4) else 8, 1, 128)
    if M <= 16:
        return (8, 2, 128)
    return (8, 4, 128)


def all_configs(N, K):
    return sorted({config(N, K, m) for m in range(1, SUPPORTS_M + 1)})


def get_kernel(N, K, cfg):
    key = (N, K) + tuple(cfg)
    k = _KERNELS.get(key)
    if k is None:
        MB, BN, threads = cfg
        k = _KERNELS[key] = skinny_bf16_kernel(N, K, MB, BN, threads)
    return k


def supported(N, K):
    return N <= 512 and N % 4 == 0 and K % (VEC * 128) == 0


def linear(x, W, cfg=None, out=None):
    """y = x @ W^T for x (M, K) bf16 contiguous, W (N, K) bf16 contiguous, 1 <= M <= SUPPORTS_M."""
    import torch
    M, K = x.shape
    N = W.shape[0]
    if cfg is None:
        cfg = config(N, K, M)
    y = out if out is not None else torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    get_kernel(N, K, cfg)(x, W, y)
    return y
