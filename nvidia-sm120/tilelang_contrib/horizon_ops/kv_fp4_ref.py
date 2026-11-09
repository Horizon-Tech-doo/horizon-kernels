# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exact PyTorch reference of the Horizon NVFP4 KV-cache format (hz-kvfp4 v1). Pure torch, runs on CPU or GPU, no
TileLang / vLLM imports, so the format can be tested anywhere and the TileLang kernels (kv_write_fp4, attn_decode_fp4)
can be compared against it bit for bit.

Format, per (token, kv head, side K or V), head_dim D (a power of two, D % 16 == 0):
    data : D/2 bytes, E2M1 codes, element 2j in the LOW nibble of byte j, element 2j+1 in the high nibble
    sf   : D/16 bytes, one E4M3 block scale per 16 consecutive elements (block b = elements 16b .. 16b+15)
    g    : fp32 global scale per (layer, kv head, side), static (not stored per token; 1.0 when the checkpoint has no
           kv_cache_scheme, the same value as the FP8 cache's k_scale / v_scale)
    value = e2m1(code) * e4m3(sf) * g
At D = 256: 128 + 16 = 144 bytes per side, 288 bytes for K + V (FP8: 512 bytes) -> 512 / 288 = 16 / 9 = 1.778x.

Quantisation of one block of 16 fp32 values x (all steps are single IEEE fp32 operations, round to nearest even,
no fused multiply-add):
    amax  = max |x_i|
    c0    = e4m3_satfinite(amax / (g * 6))                 (0 when amax / (6 g) rounds to 0)
    search = False: s8 = c0
    search = True (default): s8 = the candidate with the smallest block error among, in this order,
             c0, min(c0 + 1, 126), max(c0 - 1, 0), e4m3_satfinite(amax / (g * 4))
             error(c) = sum_i (e2m1(code_i) * s - |x_i|)^2, summed sequentially i = 0..15, strict < keeps the earlier
             candidate on a tie (codes 0..126 are the non-negative finite e4m3 values in increasing order)
    s     = float(s8) * g                                  (effective block scale)
    code  = e2m1_rne(|x_i| / s) | (x_i < 0 ? 8 : 0)        (0 when s == 0; e2m1 saturates at 6)
The search only changes the writer; readers see the same format. On synthetic Gaussian / Laplace / Student-t data it
lowers the relative L2 error from 0.095 to 0.083 (MSE -24%), see nvidia-sm120/docs/kv-fp4-design.md.
e2m1_rne on y >= 0 (grid 0, .5, 1, 1.5, 2, 3, 4, 6, ties to the even code):
    y <= .25 -> 0 | < .75 -> .5 | <= 1.25 -> 1 | < 1.75 -> 1.5 | <= 2.5 -> 2 | < 3.5 -> 3 | <= 5 -> 4 | else 6

Optional key rotation (Hadamard): k' = FWHT(k * signs) * D^-0.5 in fp32, with a fixed random +-1 sign per channel.
The same rotation is applied to q (q' = FWHT(q * signs) * D^-0.5, rounded once to bf16), and q' . k' = q . k because
the rotation is orthogonal, so the model's math is unchanged; only the quantisation error is spread over all channels.
FWHT butterfly order (the kernels use exactly this order, so the rotation is bit-identical):
    for h in 1, 2, 4, ..., D/2: for every i with (i & h) == 0: (x[i], x[i+h]) = (x[i] + x[i+h], x[i] - x[i+h])

Paged layout: vLLM 0.29's NVFP4 page geometry (FlashInferBackend.customize_spec: 2 * KVH head slots of D/2 + D/16
bytes). One page of PS tokens, HND, holds [K data (KVH, PS, D/2) | K sf (KVH, PS, D/16) | V data | V sf]. The kernels
index every part as a [page, position, head, byte] view with its own strides (split_views()).
"""

import math

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 16
SIGN_SEED = 0x6B7634  # "kv4"


def fwht_scale(D: int) -> float:
    """D^-0.5 rounded to fp32 (exact for D = 4^k, e.g. 256 -> 1/16)"""
    return float(torch.tensor(1.0 / math.sqrt(D), dtype=torch.float32))


def hadamard_signs(D: int, seed: int = SIGN_SEED) -> torch.Tensor:
    """fixed random +-1 per channel (fp32, CPU); the same vector must be used for q and k of a layer"""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (D,), generator=g) * 2 - 1).to(torch.float32)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """unnormalised FWHT over the last dim of an fp32 tensor, butterfly order as in the module docstring"""
    assert x.dtype == torch.float32
    D = x.shape[-1]
    assert D & (D - 1) == 0
    y = x.reshape(-1, D).clone()
    h = 1
    while h < D:
        v = y.view(-1, D // (2 * h), 2, h)
        a, b = v[:, :, 0, :].clone(), v[:, :, 1, :].clone()
        v[:, :, 0, :] = a + b
        v[:, :, 1, :] = a - b
        h *= 2
    return y.view(x.shape)


def rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """x (..., D) any float dtype -> fp32 FWHT(x * signs) * D^-0.5 (what kv_write_fp4 quantises for K)"""
    D = x.shape[-1]
    return fwht(x.float() * signs.to(x.device, torch.float32)) * fwht_scale(D)


def rotate_q(q: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """the bf16 query the decode kernel consumes: rotate() rounded once to bf16"""
    return rotate(q, signs).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------- E4M3


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """fp32 >= 0 -> e4m3fn byte, round to nearest even, saturate to 448 (== __nv_cvt_float_to_fp8(x, SATFINITE,
    E4M3) for finite non-negative x)"""
    return x.float().clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).float()


def e4m3_table() -> torch.Tensor:
    """all 256 e4m3fn byte values as fp64 (NaN for 0x7f / 0xff); independent of torch's cast, used by the tests"""
    out = []
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            v = float("nan")
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        out.append(-v if s else v)
    return torch.tensor(out, dtype=torch.float64)


# ---------------------------------------------------------------------------------------------------------------- E2M1


def e2m1_code(y: torch.Tensor) -> torch.Tensor:
    """y fp32 >= 0 (already divided by the block scale) -> 3-bit magnitude code, RNE on the E2M1 grid, sat. at 6"""
    c = torch.full(y.shape, 7, dtype=torch.uint8, device=y.device)
    for thr, code, le in ((5.0, 6, True), (3.5, 5, False), (2.5, 4, True), (1.75, 3, False), (1.25, 2, True),
                          (0.75, 1, False), (0.25, 0, True)):
        hit = y <= thr if le else y < thr
        c = torch.where(hit, torch.tensor(code, dtype=torch.uint8, device=y.device), c)
    return c


def e2m1_value(code: torch.Tensor) -> torch.Tensor:
    """4-bit code (sign in bit 3) -> fp32"""
    lut = torch.tensor(E2M1_GRID, dtype=torch.float32, device=code.device)
    v = lut[(code & 7).long()]
    return torch.where((code & 8) != 0, -v, v)


# ------------------------------------------------------------------------------------------------------ rows (tokens)


def _block_codes(xb, s):
    """xb (..., 16) fp32, s (..., 1) fp32 effective scale -> magnitude codes (..., 16) uint8"""
    y = xb.abs() / torch.where(s > 0, s, torch.ones_like(s))
    return torch.where(s > 0, e2m1_code(y), torch.zeros_like(y, dtype=torch.uint8))


def _block_sse(xb, s):
    """sequential fp32 sum over the 16 elements of (e2m1(code) * s - |x|)^2 (the kernel's exact order)"""
    mag = _block_codes(xb, s)
    d = e2m1_value(mag) * s - xb.abs()
    sq = d * d
    acc = sq[..., 0]
    for i in range(1, BLOCK):
        acc = acc + sq[..., i]
    return acc


def quantize_rows(x: torch.Tensor, g, search: bool = True) -> tuple:
    """x (..., D) fp32 (bf16 inputs are converted exactly), g fp32 scalar or tensor broadcastable to x[..., 0]
    -> (data uint8 (..., D/2), sf uint8 (..., D/16))"""
    x = x.float()
    D = x.shape[-1]
    assert D % BLOCK == 0
    xb = x.reshape(*x.shape[:-1], D // BLOCK, BLOCK)
    g = torch.as_tensor(g, dtype=torch.float32, device=x.device)
    g = g.reshape(*g.shape, 1) if g.dim() else g                  # broadcast over the block axis
    amax = xb.abs().amax(dim=-1)
    sf = e4m3_encode(amax / (g * FP4_MAX))
    if search:
        c0 = sf.to(torch.int16)
        cands = [sf, (c0 + 1).clamp(max=126).to(torch.uint8), (c0 - 1).clamp(min=0).to(torch.uint8),
                 e4m3_encode(amax / (g * 4.0))]
        best = _block_sse(xb, (e4m3_decode(sf) * g).unsqueeze(-1))
        for c in cands[1:]:
            e = _block_sse(xb, (e4m3_decode(c) * g).unsqueeze(-1))
            take = e < best
            best = torch.where(take, e, best)
            sf = torch.where(take, c, sf)
    s = (e4m3_decode(sf) * g).unsqueeze(-1)
    mag = _block_codes(xb, s)
    code = mag | torch.where(xb < 0, torch.tensor(8, dtype=torch.uint8, device=x.device),
                             torch.tensor(0, dtype=torch.uint8, device=x.device))
    code = code.reshape(*x.shape[:-1], D)
    data = code[..., 0::2] | (code[..., 1::2] << 4)
    return data.contiguous(), sf.contiguous()


def dequantize_rows(data: torch.Tensor, sf: torch.Tensor, g=None, dtype=torch.float32) -> torch.Tensor:
    """(..., D/2), (..., D/16) -> (..., D). g None: code * sf only (exact in bf16 and fp32; the kernels fold g into
    the softmax / output scale); else times g."""
    D = data.shape[-1] * 2
    code = torch.stack((data & 15, data >> 4), dim=-1).reshape(*data.shape[:-1], D)
    v = e2m1_value(code).reshape(*data.shape[:-1], D // BLOCK, BLOCK) * e4m3_decode(sf).unsqueeze(-1)
    v = v.reshape(*data.shape[:-1], D).to(dtype)
    if g is not None:
        g = torch.as_tensor(g, dtype=dtype, device=v.device)
        v = v * (g.reshape(*g.shape, 1) if g.dim() else g)
    return v


# ---------------------------------------------------------------------------------------------------------- paged cache


def page_bytes(PS: int, KVH: int, D: int) -> int:
    return 2 * KVH * PS * (D // 2 + D // BLOCK)


def alloc_cache(num_pages: int, PS: int, KVH: int, D: int, device="cpu", fill: int = 0) -> torch.Tensor:
    """vLLM-shaped NVFP4 cache of one layer: uint8 (num_pages, 2 * KVH, PS, D/2 + D/16), HND (head slots outermost
    inside a page: K heads, then V heads)"""
    return torch.full((num_pages, 2 * KVH, PS, D // 2 + D // BLOCK), fill, dtype=torch.uint8, device=device)


def split_views(cache: torch.Tensor, KVH: int) -> tuple:
    """(num_pages, 2 KVH, PS, full) HND cache -> (k_data, k_sf, v_data, v_sf), each a [page, position, head, byte]
    view. Same byte placement as vllm.utils.torch_utils.nvfp4_split_data_scale on kv_cache.split(KVH, dim=1)."""
    NP, H2, PS, full = cache.shape
    assert H2 == 2 * KVH and cache.is_contiguous()
    dd = full * 8 // 9
    sd = full - dd
    PB = cache.stride(0)
    side = KVH * PS * full
    base = cache.storage_offset()
    views = []
    for s in range(2):
        b0 = base + s * side
        views.append(torch.as_strided(cache, (NP, PS, KVH, dd), (PB, dd, PS * dd, 1), b0))
        views.append(torch.as_strided(cache, (NP, PS, KVH, sd), (PB, sd, PS * sd, 1), b0 + KVH * PS * dd))
    return tuple(views)


def write_reference(key, value, views, slot_mapping, gk, gv, signs=None, search=True):
    """reference of kv_write_fp4.write: key / value (NT, KVH, D) bf16; views from split_views; slot < 0 skipped;
    gk / gv (KVH,) fp32; signs (D,) fp32 -> rotate keys first; search = block-scale search"""
    kd, ksf, vd, vsf = views
    PS = kd.shape[1]
    k = rotate(key, signs) if signs is not None else key.float()
    dk, sk = quantize_rows(k, gk.float().to(k.device), search)
    dv, sv = quantize_rows(value.float(), gv.float().to(value.device), search)
    for t, sl in enumerate(slot_mapping.tolist()):
        if sl < 0:
            continue
        p, o = sl // PS, sl % PS
        kd[p, o], ksf[p, o], vd[p, o], vsf[p, o] = dk[t], sk[t], dv[t], sv[t]


def gather(views, block_table_row, L: int):
    """dequantised (code * sf, no g) K and V of one request: (L, KVH, D) fp64 each"""
    kd, ksf, vd, vsf = views
    PS = kd.shape[1]
    pages = block_table_row[: math.ceil(L / PS)].long()

    def g_(d, s):
        x = dequantize_rows(d[pages], s[pages], dtype=torch.float64)          # (np, PS, KVH, D)
        return x.reshape(-1, x.shape[2], x.shape[3])[:L]
    return g_(kd, ksf), g_(vd, vsf)


def attention_reference(q, views, block_tables, seq_lens, QL: int, sm_scale: float, gk, gv, v_scale: float = 1.0):
    """fp64 reference of attn_decode_fp4.decode over the SAME dequantised cache. q (B*QL, HQ, D) is taken as given
    (pass the rotated bf16 q' when keys were written with a rotation). Causal window: token t of a request with
    seq_len L sees positions < L - QL + t + 1. Returns (B*QL, HQ, D) fp64."""
    B = seq_lens.shape[0]
    NT, HQ, D = q.shape
    KVH = views[0].shape[2]
    G = HQ // KVH
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    gk = gk.double().to(q.device).view(KVH, 1, 1)
    gv = gv.double().to(q.device).view(1, KVH, 1)
    for b in range(B):
        L = int(seq_lens[b])
        K, V = gather(views, block_tables[b], L)
        for t in range(QL):
            lim = L - QL + t + 1
            qq = q[b * QL + t].double().view(KVH, G, D)
            s = torch.einsum("hgd,lhd->hgl", qq, K[:lim]) * gk * sm_scale
            p = torch.softmax(s, dim=-1)
            out[b * QL + t] = (torch.einsum("hgl,lhd->hgd", p, V[:lim] * gv) * v_scale).reshape(HQ, D)
    return out


# ------------------------------------------------------------------------------------------ K8V4 (FP8 keys, FP4 values)
# hz-kv K8V4 (30-09-2026): one slot of 400 bytes per (token, kv head) = [256 B K e4m3 | 128 B V E2M1 | 16 B V E4M3
# block scales]. Keys are vLLM's fp8 KV bytes (e4m3(k / k_scale), round to nearest even, saturating to +-448); values
# are the hz-kvfp4 v1 format with global scale gv. Views: split_views_k8 -> (kd, vd, vsf) [page, position, head, byte].
K8_SLOT = 400


def split_views_k8(cache: torch.Tensor):
    """cache (pages, KVH, PS, 400) uint8 (vLLM's logical LBHNC block) -> kd, vd, vsf as [page, position, head, byte]"""
    t = cache.permute(0, 2, 1, 3)
    return t[..., :256], t[..., 256:384], t[..., 384:400]


def fp8_encode_signed(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3fn byte, round to nearest even, saturating to +-448 (== __nv_cvt_float_to_fp8(x, SATFINITE, E4M3))"""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def write_reference_k8(key, value, views, slot_mapping, gk, gv, search=True):
    """reference of the K8V4 write: keys e4m3(k / gk) (IEEE fp32 division, as vLLM), values quantize_rows(v, gv)"""
    kd, vd, vsf = views
    PS = kd.shape[1]
    k8 = fp8_encode_signed(key.float() / gk.float().to(key.device).view(1, -1, 1))
    dv, sv = quantize_rows(value.float(), gv.float().to(value.device), search)
    for t, sl in enumerate(slot_mapping.tolist()):
        if sl < 0:
            continue
        p, o = sl // PS, sl % PS
        kd[p, o], vd[p, o], vsf[p, o] = k8[t], dv[t], sv[t]


def gather_k8(views, block_table_row, L: int):
    """dequantised K (e4m3 value, no k_scale) and V (code * sf, no g) of one request: (L, KVH, D) fp64 each"""
    kd, vd, vsf = views
    PS = kd.shape[1]
    pages = block_table_row[: math.ceil(L / PS)].long()
    k = kd[pages].view(torch.float8_e4m3fn).double()
    v = dequantize_rows(vd[pages], vsf[pages], dtype=torch.float64)
    return k.reshape(-1, k.shape[2], k.shape[3])[:L], v.reshape(-1, v.shape[2], v.shape[3])[:L]
