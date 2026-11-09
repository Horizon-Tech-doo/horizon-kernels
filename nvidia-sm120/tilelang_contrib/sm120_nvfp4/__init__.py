# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""SM120 (RTX Blackwell) NVFP4 linear layers on the native block-scaled FP4 tensor cores.

Typical use with an NVFP4 checkpoint (compressed-tensors layout: weight_packed uint8 (N, K/2), weight_scale fp8
(N, K/16), weight_global_scale fp32):

    w = prepare_weight(weight_packed, weight_scale, weight_global_scale)   # once, at load time
    y = linear(x, w, passes=2)                                             # x: (M, K) bf16, any M

``passes=2`` (residual two-pass activations, ~1% GEMM error) is the accuracy-preserving default; ``passes=1`` is
plain W4A4 (~10% GEMM error, about 2x faster at large M).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernels import nvfp4_gemm_kernel, nvfp4_quant_kernel

__all__ = ["NVFP4Weight", "prepare_weight", "swizzle_scale_words", "quantize", "linear"]

_ROWS = 128


def _torch():
    import torch
    return torch


def swizzle_scale_words(words):
    """(rows, K/64) uint32 semantic scale words -> SM120 blockscaled_chunk_kmajor storage (rows padded to 128)."""
    torch = _torch()
    rows, cols = words.shape
    assert words.dtype == torch.uint32 and cols % 4 == 0
    if rows % _ROWS:
        pad = torch.zeros(((rows + _ROWS - 1) // _ROWS * _ROWS, cols), dtype=words.dtype, device=words.device)
        pad[:rows] = words
        words, rows = pad, pad.shape[0]
    src = words.contiguous().reshape(rows // _ROWS, 4, 32, cols)
    return src.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


def _pack_scale_bytes(scale_bytes):
    torch = _torch()
    b = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    return (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)).to(torch.uint32).contiguous()


@dataclass
class NVFP4Weight:
    packed: object      # int8 (N, K/2)
    sfb: object         # uint32 (N * K/256, 4), swizzled
    alpha: object       # fp32 (N,) = 1 / weight_global_scale of the shard each output column belongs to
    n: int
    k: int


def prepare_weight(weight_packed, weight_scale, weight_global_scale, shard_sizes=None) -> NVFP4Weight:
    """compressed-tensors NVFP4 tensors -> kernel layout. Weights are not changed, only re-laid-out.

    weight_global_scale: one value, or one per shard of a fused layer (compressed-tensors stores divisors).
    shard_sizes: output rows per shard when several global scales are given (default: equal split).
    """
    torch = _torch()
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    assert n % 128 == 0 and k % 256 == 0, (n, k)
    sfb = swizzle_scale_words(_pack_scale_bytes(weight_scale.view(torch.uint8))).reshape(-1, 4)
    gs = weight_global_scale.float().reshape(-1)
    if gs.numel() == 1:
        alpha = (1.0 / gs).expand(n).contiguous()
    else:
        sizes = list(shard_sizes) if shard_sizes is not None else [n // gs.numel()] * gs.numel()
        assert sum(sizes) == n and len(sizes) == gs.numel(), (sizes, n, gs.numel())
        alpha = torch.cat([(1.0 / g).expand(sz) for g, sz in zip(gs, sizes)]).contiguous()
    return NVFP4Weight(weight_packed.view(torch.int8).contiguous(), sfb, alpha.to(weight_packed.device), n, k)


# per-row target: row max -> ~2600 (< 448 * 6 = 2688, so the ceil-rounded ue4m3 block scale never clips)
_ROW_TARGET = 2600.0


def quantize(x, passes: int = 2):
    """(M, K) bf16 -> (A int8 (Mp, passes*K/2), SFA uint32 (passes, Mp, K/64), rsinv fp32 (Mp,)).

    M is padded to a multiple of 128. Each row is scaled by rs = 2600 / max|row| before quantisation so the NVFP4
    block scales use their full range; rsinv = 1 / rs is returned for the GEMM epilogue."""
    torch = _torch()
    assert x.dim() == 2 and x.shape[1] % 256 == 0, x.shape
    m = x.shape[0]
    mp = (m + _ROWS - 1) // _ROWS * _ROWS
    if mp != m:
        x = torch.nn.functional.pad(x, (0, 0, 0, mp - m))
    x = x.contiguous()
    amax = x.abs().amax(dim=1).float()
    ok = amax > 0
    rs = torch.where(ok, _ROW_TARGET / amax.clamp_min(1e-30), torch.ones_like(amax))
    rsinv = torch.where(ok, amax / _ROW_TARGET, torch.ones_like(amax))
    A, SFA = nvfp4_quant_kernel(x, rs, passes=passes)
    return A, SFA, rsinv


def linear(x, w: NVFP4Weight, passes: int = 2, out_dtype=None):
    """y = x @ W^T with W in NVFP4 and x quantised to `passes` residual NVFP4 passes. Returns (M, N)."""
    torch = _torch()
    import tilelang.language as T
    assert x.dim() == 2 and x.shape[1] == w.k, (tuple(x.shape), w.k)
    m = x.shape[0]
    A, SFA, rsinv = quantize(x, passes)
    mp = A.shape[0]
    tl_out = T.float32 if out_dtype == torch.float32 else T.bfloat16
    C = nvfp4_gemm_kernel(A, w.packed, SFA.reshape(passes, mp * (w.k // 256), 4), w.sfb, w.alpha, rsinv,
                          passes=passes, out_dtype=tl_out)
    return C[:m]
