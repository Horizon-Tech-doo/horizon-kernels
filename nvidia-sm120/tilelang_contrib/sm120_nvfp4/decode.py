# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""SM120 NVFP4 W4A16 decode GEMM (small M): memory-bound, exact dequantisation, split-K.

For decode batches (1-64 tokens) a GEMM is limited by reading the weights, not by math. This kernel keeps the
activations in BF16 (W4A16, same numerics class as Marlin) and dequantises each FP4 weight with its FP8 block
scale into BF16 *exactly*: an E2M1 value has at most 2 significant bits and a UE4M3 scale at most 4, so the
product fits BF16's 8-bit mantissa. The per-shard NVFP4 global scale is applied per output column (alpha[n]).

Split-K spreads narrow layers (small N) over all SMs: grid = (N / block_N, split_k); each CTA reduces a K slice
into an fp32 partial tile, and partials are summed with fp32 atomics into a workspace.

Layouts: x (M, K) bf16; B int8 (N, K/2) packed FP4 (low nibble = even k); S fp8 e4m3 (N, K/16) row-major
(the checkpoint's own weight_scale layout, no swizzle); alpha fp32 (N,).
"""

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_decode_gemm_kernel(x, B, S, alpha, C, block_N: int = 64, block_K: int = 256, split_k: int = 4,
                             num_stages: int = 3, threads: int = 128, block_M: int = 16):
    M = T.dynamic("M")
    N = T.const("N")
    K = T.const("K")
    assert block_K % 16 == 0
    fp4 = T.float4_e2m1fn
    x: T.Tensor[(M, K), T.bfloat16]
    B: T.Tensor[(N, K), fp4]
    S: T.Tensor[(N, K // 16), T.float8_e4m3fn]
    alpha: T.Tensor[(N,), T.float32]
    C: T.Tensor[(M, N), T.float32]   # zero-initialised by the caller: split-K partials are added with fp32 atomics

    k_per_split = K // split_k
    with T.Kernel(T.ceildiv(N, block_N), split_k, T.ceildiv(M, block_M), threads=threads) as (bx, bk, bm):
        A_s = T.alloc_shared((block_M, block_K), T.bfloat16)
        W_s = T.alloc_shared((block_N, block_K), fp4)
        S_s = T.alloc_shared((block_N, block_K // 16), T.float8_e4m3fn)
        Wb = T.alloc_shared((block_N, block_K), T.bfloat16)
        C_l = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_l)
        for ko in T.Pipelined(k_per_split // block_K, num_stages=num_stages):
            k0 = bk * k_per_split + ko * block_K
            T.copy(x[bm * block_M, k0], A_s)
            T.copy(B[bx * block_N, k0], W_s)
            T.copy(S[bx * block_N, k0 // 16], S_s)
            for i, j in T.Parallel(block_N, block_K):
                Wb[i, j] = T.cast(T.cast(W_s[i, j], T.float32) * T.cast(S_s[i, j // 16], T.float32), T.bfloat16)
            T.gemm(A_s, Wb, C_l, transpose_B=True)
        for i, j in T.Parallel(block_M, block_N):
            if bm * block_M + i < M:
                T.atomic_add(C[bm * block_M + i, bx * block_N + j], C_l[i, j] * alpha[bx * block_N + j])


def decode_linear(x, packed, scale_fp8, alpha, **cfg):
    """y = x @ W^T for small M. packed int8 (N, K/2), scale_fp8 (N, K/16) e4m3 (checkpoint layout), alpha fp32 (N,)."""
    import torch
    K = packed.shape[1] * 2
    split_k, block_K = cfg.get("split_k", 4), cfg.get("block_K", 256)
    assert K % (split_k * block_K) == 0, f"K={K} must be a multiple of split_k*block_K={split_k * block_K}"
    C = torch.zeros(x.shape[0], packed.shape[0], device=x.device, dtype=torch.float32)
    nvfp4_decode_gemm_kernel(x.contiguous(), packed, scale_fp8, alpha, C, **cfg)
    return C.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------
# K2 v1: all token rows (M <= block_M) in one CTA, so every weight tile is read and dequantised ONCE and reused for
# all rows (v0 re-read the whole weight per 16-row tile). Deterministic: split_k == 1 writes C directly; split_k > 1
# writes per-split fp32 partials to a workspace that is reduced in a fixed order (no atomics).
# block_M in {16, 32, 64} is chosen by the caller from M; block_K=128 keeps 3 stages under the 99 KB smem limit.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_decode_gemm_v1_kernel(x, B, S, alpha, P, block_M: int = 64, block_N: int = 64, block_K: int = 128,
                                split_k: int = 1, num_stages: int = 3, threads: int = 128):
    M = T.dynamic("M")
    N = T.const("N")
    K = T.const("K")
    fp4 = T.float4_e2m1fn
    x: T.Tensor[(M, K), T.bfloat16]
    B: T.Tensor[(N, K), fp4]
    S: T.Tensor[(N, K // 16), T.float8_e4m3fn]
    alpha: T.Tensor[(N,), T.float32]
    P: T.Tensor[(split_k, M, N), T.float32]     # split_k == 1: P[0] is the (fp32) output
    k_per_split = K // split_k

    with T.Kernel(T.ceildiv(N, block_N), split_k, threads=threads) as (bx, bk):
        A_s = T.alloc_shared((block_M, block_K), T.bfloat16)
        W_s = T.alloc_shared((block_N, block_K), fp4)
        S_s = T.alloc_shared((block_N, block_K // 16), T.float8_e4m3fn)
        Wb = T.alloc_shared((block_N, block_K), T.bfloat16)
        C_l = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_l)
        for ko in T.Pipelined(k_per_split // block_K, num_stages=num_stages):
            k0 = bk * k_per_split + ko * block_K
            T.copy(x[0, k0], A_s)                  # rows >= M are zero-filled by the bounded copy
            T.copy(B[bx * block_N, k0], W_s)
            T.copy(S[bx * block_N, k0 // 16], S_s)
            for i, j in T.Parallel(block_N, block_K):
                Wb[i, j] = T.cast(T.cast(W_s[i, j], T.float32) * T.cast(S_s[i, j // 16], T.float32), T.bfloat16)
            T.gemm(A_s, Wb, C_l, transpose_B=True)
        for i, j in T.Parallel(block_M, block_N):
            if i < M:
                P[bk, i, bx * block_N + j] = C_l[i, j] * alpha[bx * block_N + j]


def decode_linear_v1(x, packed, scale_fp8, alpha, block_N: int = 64, split_k: int | None = None, **cfg):
    """Deterministic W4A16 decode GEMM for M <= 64. split_k defaults to 1 for wide layers and 4 for narrow ones."""
    import torch
    M, K = x.shape
    N = packed.shape[0]
    assert M <= 64, "decode kernel covers M <= 64 (larger batches go to Marlin/FP4 via the adaptive table)"
    block_M = 16 if M <= 16 else (32 if M <= 32 else 64)
    if split_k is None:
        split_k = 1 if N // block_N >= 2 * 188 else 4
    block_K = cfg.get("block_K", 128)
    assert K % (split_k * block_K) == 0, (K, split_k, block_K)
    P = torch.empty(split_k, M, N, device=x.device, dtype=torch.float32)
    nvfp4_decode_gemm_v1_kernel(x.contiguous(), packed, scale_fp8, alpha, P, block_M=block_M, block_N=block_N,
                                split_k=split_k, **cfg)
    y = P[0] if split_k == 1 else P.sum(0)      # fixed-order reduction: deterministic
    return y.to(torch.bfloat16)
