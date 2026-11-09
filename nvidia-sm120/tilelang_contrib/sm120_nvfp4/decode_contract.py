# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Contract for K2 decode-GEMM candidates (W4A16, NVFP4 weights, small M), so they can be raced uniformly.

Every candidate module ``decode_<name>.py`` in this package exports:

    prepare(packed_int8, scale_fp8, alpha_fp32) -> prepared      # once at load time (NOT timed); may re-layout
    linear(x_bf16, prepared) -> y_bf16                            # timed; x (M, K) with 1 <= M <= 64, y (M, N)
    SUPPORTS_M = 64                                               # largest M the candidate handles (>= 1)

Inputs (the checkpoint's own compressed-tensors layout):
    packed_int8  int8   (N, K/2)   two E2M1 codes per byte, LOW nibble = even k, HIGH nibble = odd k
    scale_fp8    e4m3   (N, K/16)  one UE4M3 block scale per 16 consecutive k
    alpha_fp32   fp32   (N,)       1 / weight_global_scale of the shard each output column belongs to
Mathematically  y[m, n] = alpha[n] * sum_k x[m, k] * fp4(packed[n, k]) * scale[n, k // 16].

Rules:
- Exact class: dequantise to BF16 (fp4 * ue4m3 fits BF16 exactly), accumulate in fp32, round once to bf16.
  Reference relative error on real layers is ~0.00166 (same as Marlin); a candidate must be within 1.5x of that.
- Deterministic: same inputs -> bit-identical output (no fp32 atomics across CTAs).
- Shapes to support: (N, K) in {(16384,5120), (14336,5120), (5120,6144), (34816,5120), (5120,17408)}.
  All have N % 128 == 0, K % 256 == 0. Do not assume K % 1024 == 0 (6144 = 24*256).
- M is dynamic at run time (T.dynamic), or the candidate may compile a few M-buckets (16/32/64) and pick one.
- Under CUDA-graph capture: no allocations that depend on data, no host syncs. Workspaces are fine if
  pre-allocated in prepare() or allocated with torch.empty inside linear() (the caching allocator is capture-safe).
- Shared memory per block <= 99 KB on SM120.
"""

import torch

SHAPES = {"gdn.in_proj_qkvz": (16384, 5120), "attn.qkv_proj": (14336, 5120), "o/out_proj": (5120, 6144),
          "mlp.gate_up": (34816, 5120), "mlp.down": (5120, 17408)}

_LUT = None


def fp4_lut(device):
    global _LUT
    if _LUT is None or _LUT.device != torch.device(device):
        _LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=device)
    return _LUT


def dequantize_reference(packed_int8, scale_fp8, alpha_fp32):
    """Exact fp32 dequantised weight (N, K): alpha[n] * fp4 * scale."""
    u = packed_int8.view(torch.uint8)
    lut = fp4_lut(u.device)
    w = torch.empty(u.shape[0], u.shape[1] * 2, device=u.device, dtype=torch.float32)
    w[:, 0::2] = lut[(u & 0xF).long()]
    w[:, 1::2] = lut[(u >> 4).long()]
    return w * scale_fp8.float().repeat_interleave(16, 1) * alpha_fp32.float()[:, None]


def synthetic_layer(N, K, seed=0, device="cuda"):
    """Random NVFP4 layer with realistic scale magnitudes (checkpoint-like: global scale ~2700)."""
    g = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=device, generator=g)
    scale = (torch.rand(N, K // 16, device=device, generator=g) * 3 + 0.25).to(torch.float8_e4m3fn)
    alpha = torch.full((N,), 1.0 / 2688.0, device=device)
    return packed, scale, alpha
