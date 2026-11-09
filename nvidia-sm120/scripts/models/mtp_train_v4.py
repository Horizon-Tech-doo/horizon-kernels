#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MTP draft-head retrain v4 (30-09-2026): retrain for chunk-invariant prefill (PI), with a teacher and a
# student that compute what the SERVED draft path computes. Builds on mtp_train_v2.py (imported, unchanged).
#
# Gap analysis and runbook: mtp-retrain-after-pi (not in this release). What v4 adds over v2 (each switchable, defaults = production
# with PI):
#   teacher rows   SERVE_ROWS=pi: every prompt row (prefill in serving) goes through an emulation of the fp4x2 prefill
#                  GEMM (row-scaled two-pass NVFP4 activations: prefill_rs.quantize_rs, bit-for-bit in torch); generated
#                  rows (decode in serving) stay W4A16 (regfrag = exact class). GEN_FP4X2_FRAC puts a share of generated
#                  rows on fp4x2 too (decode rows that share a step with a prefill, under load). prepi = approximation of
#                  production before PI (prompt chunks of >= PREPI_MIN_ROWS rows on fp4x2); w4a16 = v2's teacher.
#   shard scales   SHARD_SCALES=served: GDN in_proj_qkv and in_proj_z use the larger of their two NVFP4 global scales, as
#                  vLLM does with HORIZON_EXACT_SHARD_SCALES=0 (production). v2 used the checkpoint's exact per-shard
#                  scales, i.e. a slightly different main model in 48 layers.
#   KV cache       KV_FP8=1: K and V rounded to FP8 e4m3 (scale 1.0) in every full-attention layer, main model and MTP
#                  layer, as production's fp8 KV cache stores them.
#   MTP layer      MTP_QUANT=nvfp4: quantisation-aware training. Each MTP linear runs on the exact NVFP4 RTN weights that
#                  horizon_kernels.mtp_quant builds at load (merged qkv / gate_up global scales), straight-through
#                  gradients to the BF16 master; y = (x @ unit^T) * gs like regfrag_xl's epilogue.
#   draft head     HEAD_QUANT=fp8: K1a v1 numerics (per-row FP8 weights AND per-row FP8 activations, production
#                  HORIZON_DRAFT_HEAD_ACT=fp8): exact fp32 argmax in evaluation, straight-through fake quant in training.
#   schedule       NSTEPS=3 (production drafts at most 3; 1 above 10 running requests), STEP_W 1.0,0.7,0.5; evaluation
#                  still reports EVAL_NSTEPS=4. v3 spread its learning over 5 positions and lost at positions 1-3.
#   init           INIT=prod: start from the live head (horizon_mtp: MTP layer + trained 48k draft head) instead of Qwen's
#                  original MTP layer.
#   save gate      beat the production head, emulated the same way, at 1 AND 3 drafts overall, lose no more than
#                  DOMAIN_FLOOR in any of prose / maths / code at 3 drafts, and no more than LOAD_FLOOR on the load
#                  path (above 16 running requests vLLM drafts through the full BF16 lm_head, not the trained head:
#                  the MTP layer must stay compatible with it). Output only to a NEW OUTDIR.
#   diagnostics    --eval-only: the production head under the post-PI (pi) and pre-PI (prepi, w4a16) teachers, per
#                  domain (how much acceptance PI itself cost, on identical text). FIDELITY_N > 0: before vLLM stops,
#                  fetch served prompt log-probs of held-out docs; after loading, compare each teacher mode with them.
#
# Run (under the GPU lock; stop your vLLM service first and restart it afterwards): see
# scripts/kernels/mtp4/run_mtp4_train.sh (not in this release). Every option is an env var (NAME) or a flag (--name).
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import types
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_train_v2 as V2  # noqa: E402
from mtp_train_v2 import GIB, log, now  # noqa: E402

PROD_DIR_DEFAULT = V2.MODEL_DEFAULT + "/horizon_mtp"
OUTDIR_DEFAULT = V2.MODEL_DEFAULT + "/horizon_mtp_v4"
SERVED_URL_DEFAULT = "http://localhost:8001"
ROW_TARGET = 2600.0          # tilelang.contrib.sm120_nvfp4._ROW_TARGET (row-scaled prefill)
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GATE_DOMAINS = ("prose", "maths", "code")

# =====================================================================================================================
# Served numerics, pure torch (CPU-testable in test_mtp_train_v4.py)
# =====================================================================================================================


def e2m1_rtne(a: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest FP4 e2m1 value, round-to-nearest-even, saturating at +-6 (cvt.rn.satfinite.e2m1; the same rule
    as mtp_quant.quantize_nvfp4_rtn's bucketize + tie-to-even-code)."""
    pos = torch.tensor(E2M1_POS, dtype=torch.float32, device=a.device)
    mids = (pos[1:] + pos[:-1]) / 2
    m = a.abs().clamp(max=6.0)
    idx = torch.bucketize(m, mids)                                  # m == mids[i] -> i (the lower neighbour)
    tie = (idx < 7) & (m == mids[idx.clamp(max=6)])
    idx = idx + (tie & (idx % 2 == 1)).long()                       # ties go to the even code
    v = pos[idx]
    return torch.where(a < 0, -v, v)


def ue4m3_ceil(x: torch.Tensor) -> torch.Tensor:
    """fp32 >= 0 -> the scale value the fp4x2 quantiser uses: _nvfp4_tl.encode_ue4m3_ceil (the smallest unsigned E4M3
    value >= x, saturating at 448, subnormals from 2**-9) followed by decode_ue4m3."""
    x = x.float().clamp(0.0, 448.0)
    sub_m = torch.ceil(x * 512.0).to(torch.int32)
    sub_code = torch.where(sub_m > 7, torch.full_like(sub_m, 8), sub_m.clamp(0, 7))
    bits = x.contiguous().view(torch.int32)
    e = (((bits >> 23) & 0xFF) - 127).clamp(-6, 8)
    mb = bits & 0x7FFFFF
    man = (mb >> 20) + ((mb & 0xFFFFF) != 0).to(torch.int32)
    be = e + 7
    carry = man >= 8
    be = torch.where(carry, be + 1, be).clamp(1, 15)
    man = torch.where(carry, torch.zeros_like(man), man).clamp(0, 7)
    code = torch.where(x < 2.0 ** -6, sub_code, (be << 3) | man)
    code = torch.where(x <= 0, torch.zeros_like(code), code)
    ex, mn = (code >> 3) & 0xF, (code & 7).float()
    normal = torch.ldexp(1.0 + mn / 8.0, (ex - 7).to(torch.int32))
    sub = mn * (0.125 * 2.0 ** -6)
    return torch.where(ex == 0, sub, normal).float()


def fp4x2_act_qdq(x: torch.Tensor, passes: int = 2) -> torch.Tensor:
    """(R, K) activations -> (R, K) fp32: exactly the activation values the row-scaled fp4x2 prefill GEMM multiplies
    (tilelang sm120_nvfp4.prefill_rs.quantize_rs + nvfp4_quant_rs_kernel): per-row scale 2600/amax, then per 16-value
    group `passes` rounds of (ceil-E4M3 block scale of amax/6, RTNE e2m1, residual). Row-invariant by construction."""
    R, K = x.shape
    assert K % 16 == 0, K
    xf = x.float()
    amax = xf.abs().amax(-1)
    ok = amax > 0
    rs = torch.where(ok, ROW_TARGET / amax.clamp_min(1e-30), torch.ones_like(amax))
    rsinv = torch.where(ok, amax / ROW_TARGET, torch.ones_like(amax))
    r = (xf * rs[:, None]).view(R, K // 16, 16)
    acc = torch.zeros_like(r)
    for _ in range(passes):
        sc = ue4m3_ceil(r.abs().amax(-1, keepdim=True) / 6.0)
        inv = torch.where(sc > 0, 1.0 / sc.clamp_min(1e-30), torch.zeros_like(sc))
        f = e2m1_rtne(r * inv) * sc
        acc = acc + f
        r = r - f
    return acc.view(R, K) * rsinv[:, None]


def linear_fp32_via_bf16(xf: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x (fp32) @ w^T with w in bf16, at ~fp32 input precision: x = hi + lo (two bf16 parts), two bf16 GEMMs."""
    hi = xf.to(w.dtype)
    lo = (xf - hi.float()).to(w.dtype)
    return F.linear(hi, w).float() + F.linear(lo, w).float()


def fp4x2_linear(x: torch.Tensor, w: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """fp4x2 prefill emulation of one NVFP4 linear: (R, K) x, (N, K) bf16 weight -> (R, N) in x.dtype."""
    out = []
    for c in range(0, x.shape[0], chunk):
        out.append(linear_fp32_via_bf16(fp4x2_act_qdq(x[c:c + chunk]), w).to(x.dtype))
    return torch.cat(out) if out else x.new_empty((0, w.shape[0]))


def fp8_qdq(t: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 round trip with scale 1.0 (vLLM fp8 KV cache, k_scale = v_scale = 1)."""
    return t.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(t.dtype)


def fp8_rows(x: torch.Tensor):
    """horizon_kernels.draft_head.quant_rows_fp8: per-row amax/448 scale, e4m3 values. -> (q fp8, s fp32)."""
    s = x.float().abs().amax(-1).clamp(min=1e-8) / 448.0
    return (x.float() / s[:, None]).to(torch.float8_e4m3fn), s.float()


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """straight-through: forward xq, gradient of identity."""
    return x + (xq.to(x.dtype) - x).detach()


def fp8_rows_ste(x: torch.Tensor) -> torch.Tensor:
    q, s = fp8_rows(x.detach().reshape(-1, x.shape[-1]))
    return ste(x, (q.float() * s[:, None]).reshape(x.shape))


def fp8_head_argmax(x: torch.Tensor, head, chunk: int = 1024) -> torch.Tensor:
    """K1a v1 exactly: argmax_n (fp8(x) . fp8(W_n)) * sa * sw_n in fp32. head = (Wf = fp8 codes as fp32 (V, K), sw)."""
    Wf, sw = head
    out = []
    for c in range(0, x.shape[0], chunk):
        xq, sa = fp8_rows(x[c:c + chunk])
        out.append(((xq.float() @ Wf.t()) * sa[:, None] * sw[None, :]).argmax(-1))
    return torch.cat(out)


def nvfp4_group_gs(weights) -> torch.Tensor:
    """global multiplier of a merged (fused) MTP matrix, exactly as mtp_quant.quantize_nvfp4_rtn computes it on the
    BF16 weight vLLM holds: max|w| / (6 * 448), floored so the smallest block denominator stays a normal fp32."""
    amax = torch.stack([w.detach().to(torch.bfloat16).float().abs().max() for w in weights]).max()
    return (amax / (6.0 * 448.0)).clamp_min(torch.finfo(torch.float32).tiny * (2 ** 9))


def nvfp4_rtn_unit(w: torch.Tensor, gs: torch.Tensor, rows: int = 2048) -> torch.Tensor:
    """(N, K) weight -> (N, K) fp32 'unit' = e2m1 value * e4m3 block scale of mtp_quant.quantize_nvfp4_rtn(bf16(w)) with
    global multiplier gs; the served dequantised weight is unit * gs (exact: unit has <= 6 significant bits)."""
    N, K = w.shape
    out = torch.empty(N, K, dtype=torch.float32, device=w.device)
    pos = torch.tensor(E2M1_POS, device=w.device)
    mids = (pos[1:] + pos[:-1]) / 2
    for r0 in range(0, N, rows):
        wf = w[r0:r0 + rows].detach().to(torch.bfloat16).float()
        blk = wf.reshape(wf.shape[0], K // 16, 16)
        s = (blk.abs().amax(-1, keepdim=True) / 6.0 / gs).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
        sf = s.float() * gs
        a = (blk / sf).abs().clamp(max=6.0)
        idx = torch.bucketize(a, mids)
        tie = (idx < 7) & (a == mids[idx.clamp(max=6)])
        idx = idx + (tie & (idx % 2 == 1)).long()
        v = pos[idx]
        v = torch.where((blk < 0) & (idx > 0), -v, v)
        out[r0:r0 + rows] = (v * s.float()).reshape(-1, K)
    return out


# MTP linears as vLLM fuses them (one global scale per fused layer): mtp_quant logs mtp.fc, qkv_proj, o_proj,
# gate_up_proj, down_proj on production.
MTP_QUANT_GROUPS = (("fc",), ("layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj"),
                    ("layers.0.self_attn.o_proj",), ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"),
                    ("layers.0.mlp.down_proj",))


def _qat_forward(self, x):
    """served MTP linear: y = (x @ unit^T) * gs with straight-through gradients to the master weight."""
    gs = self._hz_gs
    wu = self.weight / gs
    return F.linear(x, wu + (self._hz_unit.to(wu.dtype) - wu).detach()) * gs


def install_mtp_qat(mtp):
    """Route every MTP linear through _qat_forward. Plain attributes only: state_dict keys stay the original ones."""
    mods = dict(mtp.named_modules())
    for group in MTP_QUANT_GROUPS:
        for name in group:
            m = mods[name]
            N, K = m.weight.shape
            if N % 128 or K % 256:
                raise RuntimeError(f"MTP {name} {N}x{K}: mtp_quant would leave it in BF16; update MTP_QUANT_GROUPS")
            m._hz_group = group
            m.forward = types.MethodType(_qat_forward, m)
    refresh_mtp_qat(mtp)


@torch.no_grad()
def refresh_mtp_qat(mtp):
    """Recompute the NVFP4 units from the current master weights (call after every optimizer step)."""
    mods = dict(mtp.named_modules())
    for group in MTP_QUANT_GROUPS:
        ms = [mods[n] for n in group]
        if not hasattr(ms[0], "_hz_group"):
            return
        gs = nvfp4_group_gs([m.weight for m in ms])
        for m in ms:
            m._hz_gs = gs
            m._hz_unit = nvfp4_rtn_unit(m.weight, gs).to(torch.bfloat16)   # exact in bf16


def dequant_nvfp4(packed, scale, gscale, dtype=torch.bfloat16):
    """compressed-tensors nvfp4-pack-quantized -> dense (trainctx.dequant_nvfp4_torch): w = fp4 * (scale / gscale)."""
    N, K2 = packed.shape
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8).to(torch.int32)
    vals = torch.stack((lut[p & 0x0F], lut[p >> 4]), dim=-1).view(N, K2 * 2 // 16, 16)
    s = scale.to(torch.float32) / gscale.to(torch.float32).reshape(-1)[0]
    return (vals * s.unsqueeze(-1)).view(N, K2 * 2).to(dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Row masks: which rows run the fp4x2 prefill path in serving
# ---------------------------------------------------------------------------------------------------------------------

def segments(mask):
    """maximal runs of True in a 1-D bool list/tensor -> [(start, end)]"""
    m = [bool(v) for v in (mask.tolist() if isinstance(mask, torch.Tensor) else mask)]
    out, s = [], None
    for i, v in enumerate(m + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    return out


def fp4_row_mask(resp, mode, gen_frac=0.0, seed=0, prepi_min_rows=768, chunk=1600):
    """resp: 1 = generated token (decode in serving), 0 = prompt (prefill). -> (T,) bool: rows on the fp4x2 prefill
    path, or None (all rows W4A16).
      w4a16   v2's teacher: no fp4x2 row.
      pi      production with PI: every prompt row.
      pi_all  every row (what a served prompt_logprobs request computes: the whole text is one prefill).
      prepi   approximation of production before PI: prompt runs cut into `chunk`-row prefill chunks
              (--long-prefill-token-threshold 1600); a chunk runs fp4x2 iff it has >= prepi_min_rows rows (the
              calibration table switches from regfrag to fp4x2 between 513 and 1025 rows, per shape).
    gen_frac > 0: that share of generated rows also on fp4x2 (decode rows that share a step with a prefill)."""
    r = torch.as_tensor(resp, dtype=torch.bool)
    T = r.numel()
    if mode == "w4a16":
        m = torch.zeros(T, dtype=torch.bool)
    elif mode == "pi":
        m = ~r
    elif mode == "pi_all":
        m = torch.ones(T, dtype=torch.bool)
    elif mode == "prepi":
        m = torch.zeros(T, dtype=torch.bool)
        for s, e in segments(~r):
            for c in range(s, e, chunk):
                if min(e, c + chunk) - c >= prepi_min_rows:
                    m[c:min(e, c + chunk)] = True
    else:
        raise ValueError(f"unknown row mode {mode!r}")
    if gen_frac > 0 and mode != "pi_all":
        g = torch.Generator().manual_seed(seed)
        m |= r & (torch.rand(T, generator=g) < gen_frac)
    return m if bool(m.any()) else None


class Emu:
    """Global switches read by the patched main-model modules during one forward."""
    rows = None          # (T,) bool on the model device, or None
    kv_fp8 = False


def _emu_linear_forward(self, x):
    rows = Emu.rows
    if rows is None or not self._hz_fp4:
        return F.linear(x, self.weight, self.bias)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    if x2.shape[0] != rows.numel():
        raise RuntimeError(f"row mask of {rows.numel()} rows for an input of {x2.shape[0]} rows")
    idx_f = rows.nonzero().squeeze(1)
    idx_w = (~rows).nonzero().squeeze(1)
    y = x2.new_empty((x2.shape[0], self.weight.shape[0]))
    if idx_w.numel():
        y.index_copy_(0, idx_w, F.linear(x2.index_select(0, idx_w), self.weight))
    y.index_copy_(0, idx_f, fp4x2_linear(x2.index_select(0, idx_f), self.weight))
    if self.bias is not None:
        y = y + self.bias
    return y.reshape(*shp[:-1], -1)


def install_row_emulation(lm, select=None, n_mult=128, k_mult=256):
    """Patch every NVFP4 linear of the main model (compressed-tensors modules: `quantization_scheme` set) so rows in
    Emu.rows run the fp4x2 emulation. Shapes the fp4x2 kernel cannot take (N % 128, K % 256) stay W4A16, as in
    vLLM (layer.horizon_nvfp4 is None there). Returns (patched, fp4x2-eligible)."""
    if select is None:
        def select(name, m):   # noqa: ARG001
            return getattr(m, "quantization_scheme", None) is not None
    n = n_fp4 = 0
    for name, m in lm.named_modules():
        if isinstance(m, nn.Linear) and select(name, m):
            N, K = m.weight.shape
            m._hz_fp4 = (N % n_mult == 0) and (K % k_mult == 0)
            m.forward = types.MethodType(_emu_linear_forward, m)
            n += 1
            n_fp4 += int(m._hz_fp4)
    return n, n_fp4


def install_kv_fp8():
    """Round K and V to FP8 e4m3 in every sdpa attention call while Emu.kv_fp8 is set (straight-through for the MTP
    layer's gradients). Patches the `sdpa` entry of transformers' attention registry (instance-local override)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if getattr(ALL_ATTENTION_FUNCTIONS["sdpa"], "_hz_fp8kv", False):
        return
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_fp8kv(module, query, key, value, attention_mask, **kw):
        if Emu.kv_fp8:
            key, value = ste(key, fp8_qdq(key.detach())), ste(value, fp8_qdq(value.detach()))
        return orig(module, query, key, value, attention_mask, **kw)

    sdpa_fp8kv._hz_fp8kv = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_fp8kv


@torch.no_grad()
def apply_served_shard_scales(lm, model_dir, device, log_fn=log):
    """vLLM (HORIZON_EXACT_SHARD_SCALES=0) fuses GDN in_proj_qkv + in_proj_z and keeps the LARGER of their global
    scales for both (trainctx.load_weights, shard_scales='served'). Re-dequantise both shards with it."""
    from safetensors import safe_open
    pre = "model.language_model.layers."
    n = 0
    worst = 0.0
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for i, layer in enumerate(lm.layers):
            la = getattr(layer, "linear_attn", None)
            if la is None:
                continue
            names = ("in_proj_qkv", "in_proj_z")
            gsk = [f"{pre}{i}.linear_attn.{s}.weight_global_scale" for s in names]
            if not all(k in keys for k in gsk):
                raise RuntimeError(f"layer {i}: global scales not found ({gsk[0]}): checkpoint layout changed?")
            gmax = torch.stack([f.get_tensor(k).float().reshape(-1)[0] for k in gsk]).max()
            for s, k in zip(names, gsk):
                mod = getattr(la, s)
                packed = f.get_tensor(f"{pre}{i}.linear_attn.{s}.weight_packed").to(device)
                scale = f.get_tensor(f"{pre}{i}.linear_attn.{s}.weight_scale").to(device)
                own = f.get_tensor(k).to(device)
                exact = dequant_nvfp4(packed, scale, own, mod.weight.dtype)
                rel = float((exact.float() - mod.weight.float()).norm() / mod.weight.float().norm().clamp_min(1e-30))
                worst = max(worst, rel)
                if rel > 1e-2:
                    raise RuntimeError(f"layer {i} {s}: loaded weight differs from the checkpoint's own dequantisation "
                                       f"(rel {rel:.3g}): naming/layout mismatch, refusing to rescale")
                mod.weight.copy_(dequant_nvfp4(packed, scale, gmax.to(device), mod.weight.dtype))
                n += 1
    log_fn(f"served shard scales: {n} GDN shard weights re-dequantised with the fused max global scale "
           f"(loaded vs checkpoint dequant worst rel {worst:.2e})")
    return n


@torch.no_grad()
def kernel_check(model_dir, tl_prefix, device="cuda", rows=192, log_fn=log):
    """GPU check of the fp4x2 ACTIVATION emulation against the production kernel itself (prefill_rs.linear_native, the
    PI prefill path) on two real checkpoint layers (layer 3 MLP down_proj, layer 2 GDN out_proj). Gated metrics (03-10-2026,
    option B): the emulated activations x the exact (fp32) dequantised weights in fp64 vs the kernel with fp32
    output (emu_vs_kernel), against W4A16 (bf16 x bf16 weights) vs the same fp32 kernel output (w4a16_vs_kernel); the
    emulation should sit far closer to the kernel than W4A16 does. Report only: the trainer's own path with its BF16
    weight copy (trainer_bf16w_vs_kernel; inherited from v2, the 03-10 diagnostic put it at ~0.0032) and the bf16-output
    comparison the check used before (emu_bf16_vs_kernel_bf16, w4a16_vs_kernel_bf16). Needs the production TileLang
    prefix (TL_PREFIX) and the CUDA 13 toolkit (CUDA_HOME) TileLang compiles with."""
    from safetensors import safe_open
    if tl_prefix and tl_prefix not in sys.path:
        sys.path.insert(0, tl_prefix)
    from tilelang.contrib.sm120_nvfp4 import prefill_rs
    out = {}
    g = torch.Generator(device=device).manual_seed(0)
    pre = "model.language_model.layers."
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))  # noqa: E731
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for name in ("3.mlp.down_proj", "2.linear_attn.out_proj"):   # layer 3 is full attention; 2 is a GDN layer
            packed = f.get_tensor(f"{pre}{name}.weight_packed").to(device)
            scale = f.get_tensor(f"{pre}{name}.weight_scale").to(device)
            gsc = f.get_tensor(f"{pre}{name}.weight_global_scale").to(device)
            K = packed.shape[1] * 2
            x = (torch.randn(rows, K, device=device, generator=g) * 0.5).to(torch.bfloat16)
            x[:, :8] *= 20.0                                              # a few outlier channels, like real rows
            h = prefill_rs.prepare_weight_native(packed, scale, gsc)
            yk32 = prefill_rs.linear_native(x, h, passes=2, out_dtype=torch.float32).float()
            yk16 = prefill_rs.linear_native(x, h, passes=2).float()
            w32 = dequant_nvfp4(packed, scale, gsc, torch.float32)
            w16 = dequant_nvfp4(packed, scale, gsc, torch.bfloat16)
            ye = fp4x2_act_qdq(x).double() @ w32.double().t()
            yw = F.linear(x, w16).float()
            out[name] = {"emu_vs_kernel": round(rel(ye, yk32), 7), "w4a16_vs_kernel": round(rel(yw, yk32), 6),
                         "trainer_bf16w_vs_kernel": round(rel(linear_fp32_via_bf16(fp4x2_act_qdq(x), w16), yk32), 6),
                         "emu_bf16_vs_kernel_bf16": round(rel(fp4x2_linear(x, w16).float(), yk16), 6),
                         "w4a16_vs_kernel_bf16": round(rel(yw, yk16), 6)}
            del packed, scale, h, yk32, yk16, ye, yw, w32, w16
    log_fn(f"kernel check (fp4x2 emulation vs prefill_rs.linear_native): {json.dumps(out)}")
    return out


# =====================================================================================================================
# Data
# =====================================================================================================================

CODE_KINDS = {"code", "code_gen", "code_edit", "long_code"}


def domain_of(row) -> str:
    """kind -> gate domain. prose/maths/code are the hz acceptance domains."""
    k = str(row.get("kind") or "")
    if k == "math":
        return "maths"
    if k in CODE_KINDS:
        return "code"
    if k in ("chat", "long_prose"):
        return "prose"
    if k == "long":                                   # data v3: long code review or news summaries
        return "code" if "```" in (row.get("prompt") or row.get("prompt_text") or "") else "prose"
    if k in ("multi", "tools", "traffic"):
        return k
    return "other"


def parse_excludes(spec):
    """'chat,gen_v3.jsonl:code_gen' -> [(file or None, kind)]"""
    out = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        f, _, k = tok.rpartition(":")
        out.append((f or None, k))
    return out


def excluded(src, kind, ex):
    return any((f is None or f == src) and k == kind for f, k in ex)


def load_rows_multi(paths, excludes):
    """rows of every file in `paths` (comma list), minus excluded (file, kind); kind becomes 'domain:kind'."""
    rows, st = [], {}
    ex = parse_excludes(excludes)
    for p in [p.strip() for p in paths.split(",") if p.strip()]:
        r, bad = V2.load_rows(p)
        src = os.path.basename(p)
        keep = [x for x in r if not excluded(src, str(x.get("kind")), ex)]
        for x in keep:
            x["kind"] = f"{domain_of(x)}:{x.get('kind')}"
        st[src] = {"rows": len(r), "bad": bad, "kept": len(keep)}
        rows += keep
    return rows, st


def domain_key(kind):
    return kind.split(":", 1)[0]


# =====================================================================================================================
# Teacher passes, evaluation, gates
# =====================================================================================================================

def main_pass_v4(ctx, ids, rows):
    Emu.rows = None if rows is None else rows.to(ctx.device)
    try:
        return V2.main_pass(ctx, ids)
    finally:
        Emu.rows = None


@torch.no_grad()
def cache_eval_docs_v4(ctx, docs, mode, a):
    cache, ds = [], {"n": 0, "oov": 0, "greedy": 0, "mass": 0.0, "nomass": 0}
    for j, d in enumerate(docs):
        rows = fp4_row_mask(d.resp, mode, a.gen_fp4x2_frac, seed=j, prepi_min_rows=a.prepi_min_rows)
        mp = main_pass_v4(ctx, d.ids, rows)
        resp = torch.tensor(d.resp, dtype=torch.bool, device=ctx.device)
        c = V2.data_stat_counts(mp["x"][0], resp, mp["targ"], ctx.t2d, mp["mass"], mp["valid"])
        for k in ds:
            ds[k] += c[k]
        cache.append({"kind": d.kind, "ids": mp["x"][0].to("cpu", torch.int32), "resp": resp.cpu(),
                      "h": mp["h"][0].cpu(), "targ": mp["targ"].to("cpu", torch.int32)})
        del mp
    return cache, V2.finalize_data_stats(ds)


def head_for_eval(head_w, quant):
    w = head_w.detach().to(torch.bfloat16)                     # what draft_head.safetensors holds
    if quant == "fp8":
        q, s = fp8_rows(w)                                     # DraftHeadFP8: quant_rows_fp8(weight_bf16)
        return ("fp8", (q.float(), s))
    return ("bf16", w)


@torch.no_grad()
def evaluate_v4(ctx, cache, nsteps, head_quant, chunk=V2.LOGIT_CHUNK):
    """Exact greedy chain acceptance with the served draft path: MTP layer as installed (QAT = NVFP4 units), K1a v1
    FP8 head (or BF16), argmax mapped through d2t. head_quant='full': the full-vocabulary BF16 lm_head, which vLLM uses
    for a draft step with more than 16 rows (the K1a kernel's BM), i.e. above 16 running requests. Per kind and domain."""
    was = ctx.mtp.training
    ctx.mtp.eval()
    dev = ctx.device
    if head_quant == "full":          # > 16 rows in a draft step: vLLM falls back to the full 248k BF16 lm_head
        kind, head = "full", ctx.lm_head_w
    else:
        kind, head = head_for_eval(ctx.head_w, head_quant)
    tot, by_kind = V2.new_counts(nsteps), {}
    for d in cache:
        ids = d["ids"].to(dev, torch.long)
        n_last = V2.n_slots(ids.numel(), nsteps - 1)
        if n_last < 1:
            continue
        ridx = V2.root_mask(d["resp"].to(dev), nsteps, n_last).nonzero(as_tuple=True)[0]
        if ridx.numel() == 0:
            continue
        h = d["h"].to(dev)[None]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            outs = V2.run_chain(ctx.mtp, h, ctx.embed(ids[None]), V2.rope_for(ctx.lm, h), nsteps)
        targ = d["targ"].to(dev, torch.long)
        rows = []
        for k, out in enumerate(outs):
            x = out[0].index_select(0, ridx).to(torch.bfloat16)
            if kind == "fp8":
                pred = fp8_head_argmax(x, head, chunk)
            else:
                pred = torch.cat([F.linear(x[c:c + chunk], head).float().argmax(-1) for c in range(0, x.shape[0], chunk)])
            if kind != "full":
                pred = ctx.d2t[pred]
            rows.append(pred == targ[V2.teacher_index(ridx, k)])
        c = V2.chain_counts(torch.stack(rows))
        V2.add_counts(tot, c)
        V2.add_counts(by_kind.setdefault(d["kind"], V2.new_counts(nsteps)), c)
    if was:
        ctx.mtp.train()
    res = V2.finalize_counts(tot, nsteps)
    by_dom = {}
    for k, v in by_kind.items():
        V2.add_counts(by_dom.setdefault(domain_key(k), V2.new_counts(nsteps)), v)
    res["by_kind"] = {k: {"n_roots": v["n"], "acc_len": V2.finalize_counts(v, nsteps)["acc_len"]}
                      for k, v in sorted(by_kind.items())}
    res["by_domain"] = {k: {"n_roots": v["n"], "acc_len": V2.finalize_counts(v, nsteps)["acc_len"]}
                        for k, v in sorted(by_dom.items())}
    return res


def fmt_eval_v4(tag, ev, gates):
    s = f"{tag}: acc_len {[ev['acc_len'][str(m)] for m in range(1, len(ev['acc_len']) + 1)]} per_pos {ev['per_pos']}"
    s += f" roots {ev['n_roots']}"
    for d, v in ev.get("by_domain", {}).items():
        s += f" | {d}: " + " ".join(f"@{m} {v['acc_len'][str(m)]}" for m in gates)
    return s


def gate_v4(ev, ref, steps=(1, 3), domains=GATE_DOMAINS, floor=0.01, ev_full=None, ref_full=None, load_floor=0.005):
    """The save rule: better than the production head (same emulation) at every gate step overall, no gate domain worse
    by more than `floor` at the largest gate step, and (when given) the load path (1 draft through the full BF16
    lm_head, > 16 running requests) no worse than `load_floor`. -> (ok, details)"""
    top = str(max(steps))
    det = {"overall": {str(m): round(ev["acc_len"][str(m)] - ref["acc_len"][str(m)], 4) for m in steps}, "domains": {}}
    ok = all(v > 0 for v in det["overall"].values())
    if ev_full is not None and ref_full is not None:
        det["load_path_1"] = round(ev_full["acc_len"]["1"] - ref_full["acc_len"]["1"], 4)
        ok = ok and det["load_path_1"] >= -load_floor
    for d in domains:
        a, b = ev.get("by_domain", {}).get(d), ref.get("by_domain", {}).get(d)
        if a is None or b is None:
            continue
        delta = round(a["acc_len"][top] - b["acc_len"][top], 4)
        det["domains"][d] = delta
        ok = ok and delta >= -floor
    return ok, det


def better_v4(ev, best, steps=(1, 3)):
    key = lambda e: tuple(e["acc_len"][str(m)] for m in sorted(steps, reverse=True))  # noqa: E731
    return best is None or key(ev) > key(best)


def diag_table(evs, steps=(1, 3)):
    """{mode: ev} -> {domain: {mode: acc@max(steps)}} plus 'overall'."""
    top = str(max(steps))
    out = {"overall": {m: e["acc_len"][top] for m, e in evs.items()}}
    for m, e in evs.items():
        for d, v in e.get("by_domain", {}).items():
            out.setdefault(d, {})[m] = v["acc_len"][top]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fidelity: teacher modes vs the served stack's own prompt log-probs
# ---------------------------------------------------------------------------------------------------------------------

def served_prompt_logprobs(url, ids, model="qwen3.8", timeout=900):
    body = {"model": model, "prompt": list(map(int, ids)), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        pl = json.load(r)["choices"][0]["prompt_logprobs"]
    return parse_prompt_logprobs(pl, ids)


def parse_prompt_logprobs(pl, ids):
    """vLLM prompt_logprobs (list, None at position 0) -> (lp of the actual token, model top-1 id) for positions 1..T-1."""
    lp, top1 = [], []
    for pos, d in enumerate(pl):
        if d is None:
            continue
        lp.append(float(d[str(ids[pos])]["logprob"]))
        top1.append(int(min(d.items(), key=lambda kv: kv[1].get("rank", 99))[0]))
    return lp, top1


@torch.no_grad()
def teacher_prompt_logprobs(ctx, ids, rows):
    Emu.rows = None if rows is None else rows.to(ctx.device)
    try:
        x = torch.tensor([ids], device=ctx.device, dtype=torch.long)
        h = ctx.lm(input_ids=x, use_cache=False).last_hidden_state[0]
    finally:
        Emu.rows = None
    lp, top1 = [], []
    for c in range(0, h.shape[0] - 1, V2.LOGIT_CHUNK):
        lg = F.linear(h[c:min(c + V2.LOGIT_CHUNK, h.shape[0] - 1)], ctx.lm_head_w).float()
        ls = lg.log_softmax(-1)
        nxt = x[0, c + 1:c + 1 + ls.shape[0]]
        lp.append(ls.gather(-1, nxt[:, None]).squeeze(1))
        top1.append(lg.argmax(-1))
    return torch.cat(lp).cpu().tolist(), torch.cat(top1).cpu().tolist()


def compare_lp(ref, got):
    """ref/got: [(lp list, top1 list)] per doc -> mean |dlogp|, dNLL (got - ref) and top-1 agreement over all tokens."""
    n = absd = dn = same = 0.0
    for (a, ta), (b, tb) in zip(ref, got):
        m = min(len(a), len(b))
        n += m
        absd += sum(abs(x - y) for x, y in zip(a[:m], b[:m]))
        dn += sum(-(y - x) for x, y in zip(a[:m], b[:m]))
        same += sum(int(x == y) for x, y in zip(ta[:m], tb[:m]))
    n = max(1.0, n)
    return {"tokens": int(n), "mean_abs_dlogp": round(absd / n, 5), "dnll": round(dn / n, 5),
            "top1_agree": round(same / n, 5)}


# =====================================================================================================================
# Main
# =====================================================================================================================

def parse_args(argv=None):
    e = os.environ.get
    ap = argparse.ArgumentParser(description="Retrain the MTP draft head on the served (post-PI) numerics.")
    ap.add_argument("--model", default=e("MODEL", V2.MODEL_DEFAULT))
    ap.add_argument("--data", default=e("DATA", "$HK_DATA/models/mtp-data/gen_v4.jsonl"),
                    help="comma list of mtp_gen jsonl files")
    ap.add_argument("--exclude", default=e("EXCLUDE_KINDS", ""), help="kinds to drop: 'kind' or 'file.jsonl:kind', comma list")
    ap.add_argument("--eval-data", default=e("EVAL_DATA", ""))
    ap.add_argument("--vocab", default=e("VOCAB", PROD_DIR_DEFAULT + "/draft_vocab.pt"))
    ap.add_argument("--prod-dir", default=e("PROD_DIR", PROD_DIR_DEFAULT), help="live head: baseline and INIT=prod")
    ap.add_argument("--init", default=e("INIT", "prod"), choices=["prod", "orig"])
    ap.add_argument("--outdir", default=e("OUTDIR", OUTDIR_DEFAULT))
    ap.add_argument("--serve-rows", default=e("SERVE_ROWS", "pi"), choices=["pi", "prepi", "w4a16"])
    ap.add_argument("--gen-fp4x2-frac", type=float, default=float(e("GEN_FP4X2_FRAC", "0")))
    ap.add_argument("--prepi-min-rows", type=int, default=int(e("PREPI_MIN_ROWS", "768")))
    ap.add_argument("--shard-scales", default=e("SHARD_SCALES", "served"), choices=["served", "exact"])
    ap.add_argument("--kv-fp8", type=int, default=int(e("KV_FP8", "1")))
    ap.add_argument("--mtp-quant", default=e("MTP_QUANT", "nvfp4"), choices=["nvfp4", "none"])
    ap.add_argument("--head-quant", default=e("HEAD_QUANT", "fp8"), choices=["fp8", "bf16"])
    ap.add_argument("--nsteps", type=int, default=int(e("NSTEPS", "3")))
    ap.add_argument("--eval-nsteps", type=int, default=int(e("EVAL_NSTEPS", "4")))
    ap.add_argument("--step-w", default=e("STEP_W", "1.0,0.7,0.5,0.3,0.2"))
    ap.add_argument("--gate-steps", default=e("GATE_STEPS", "1,3"))
    ap.add_argument("--domain-floor", type=float, default=float(e("DOMAIN_FLOOR", "0.01")))
    ap.add_argument("--load-floor", type=float, default=float(e("LOAD_FLOOR", "0.005")),
                    help="max drop of 1-draft acceptance through the full BF16 lm_head (> 16 running requests)")
    ap.add_argument("--lr", type=float, default=float(e("LR", "3e-5")))
    ap.add_argument("--lr-head", type=float, default=float(e("LR_HEAD")) if e("LR_HEAD") else None)
    ap.add_argument("--epochs", type=int, default=int(e("EPOCHS", "1")))
    ap.add_argument("--max-packs", type=int, default=int(e("MAX_PACKS", "2200")), help="cap on training steps (0 = all)")
    ap.add_argument("--maxlen", type=int, default=int(e("MAXLEN", "4096")))
    ap.add_argument("--topk", type=int, default=int(e("TOPK", "64")))
    ap.add_argument("--heldout", type=int, default=int(e("HELDOUT", "240")))
    ap.add_argument("--eval-every", type=int, default=int(e("EVAL_EVERY", "300")))
    ap.add_argument("--log-every", type=int, default=int(e("LOG_EVERY", "25")))
    ap.add_argument("--chat-template", default=e("CHAT_TEMPLATE", V2.TEMPLATE_DEFAULT))
    ap.add_argument("--fidelity-n", type=int, default=int(e("FIDELITY_N", "8")), help="held-out docs for the check (0 = off)")
    ap.add_argument("--fidelity-maxlen", type=int, default=int(e("FIDELITY_MAXLEN", "4096")))
    ap.add_argument("--fidelity-require", type=int, default=int(e("FIDELITY_REQUIRE", "1")),
                    help="abort before training if the pi emulation is not closer to the served log-probs than v2's")
    ap.add_argument("--served-url", default=e("SERVED_URL", SERVED_URL_DEFAULT))
    ap.add_argument("--tl-prefix", default=e("TL_PREFIX", ""),
                    help="production TileLang prefix: check the fp4x2 emulation against the real kernel (off if empty)")
    ap.add_argument("--min-greedy-frac", type=float, default=float(e("MIN_GREEDY_FRAC", "0.5")),
                    help="abort if the teacher reproduces fewer of the data's own tokens greedily (broken load)")
    ap.add_argument("--min-avail-gb", type=float, default=float(e("MIN_AVAIL_GB", "4")))
    ap.add_argument("--load-rss-gb", type=float, default=float(e("LOAD_RSS_GB", "30")))
    ap.add_argument("--grad-ckpt", action="store_true", default=e("GRAD_CKPT", "0") == "1")
    ap.add_argument("--smoke", action="store_true", help="load, fidelity, 2 train steps, tiny eval, save nothing")
    ap.add_argument("--eval-only", action="store_true", help="fidelity + the production head under pi / prepi / w4a16")
    a = ap.parse_args(argv)
    w = [float(v) for v in a.step_w.split(",") if v.strip()]
    if a.nsteps < 1 or len(w) < a.nsteps:
        ap.error(f"STEP_W has {len(w)} weights, NSTEPS={a.nsteps} needs at least that many")
    a.step_w = w[:a.nsteps]
    a.gate_steps = tuple(sorted({int(v) for v in a.gate_steps.split(",") if v.strip()}))
    a.eval_nsteps = max(a.eval_nsteps, a.nsteps, max(a.gate_steps))
    if a.lr_head is None:
        a.lr_head = 0.1 * a.lr
    if a.smoke and a.eval_only:
        ap.error("--smoke and --eval-only are exclusive")
    return a


def load_mtp_weights(mtp, head_w, d2t, prod_dir, model_dir, which):
    """which = prod: the live MTP layer + trained draft head; orig: Qwen's MTP layer + lm_head rows (head_w as given)."""
    from safetensors.torch import load_file
    src = os.path.join(prod_dir if which == "prod" else model_dir, "model_mtp.safetensors")
    sd = load_file(src)
    mtp.load_state_dict({k[4:]: v.to(mtp.fc.weight.device, torch.float32) for k, v in sd.items()}, strict=True)
    if which == "prod":
        pv = torch.load(os.path.join(prod_dir, "draft_vocab.pt"), map_location="cpu", weights_only=True).long()
        if not torch.equal(pv, d2t.cpu()):
            raise RuntimeError("VOCAB differs from the production head's draft_vocab.pt: the trained head rows would "
                               "not line up (use the production vocabulary, or INIT=orig)")
        hw = load_file(os.path.join(prod_dir, "draft_head.safetensors"))["weight"]
        with torch.no_grad():
            head_w.copy_(hw.to(head_w.device, torch.float32))
    return src


def run(args):
    t_run = time.time()
    mode = "smoke" if args.smoke else "eval-only" if args.eval_only else "train"
    log(f"mtp_train_v4 start, mode {mode}, pid {os.getpid()}")
    log("config", json.dumps(vars(args)))
    torch.manual_seed(0)
    random.seed(0)

    # ---- 0. cheap checks while vLLM keeps serving
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    vocab_size = cfg.get("text_config", cfg)["vocab_size"]
    d2t, t2d = V2.build_vocab_maps(torch.load(args.vocab, map_location="cpu", weights_only=True), vocab_size)
    log(f"draft vocab {args.vocab}: {d2t.numel()} ids")
    for p in (args.outdir,):
        if os.path.realpath(p) in (os.path.realpath(args.model), os.path.realpath(args.prod_dir)):
            raise RuntimeError(f"OUTDIR {p} must be a new directory (never the model dir or the live head)")
    if not (args.smoke or args.eval_only):
        os.makedirs(args.outdir, exist_ok=True)
        if os.path.exists(os.path.join(args.outdir, "meta.json")):
            raise RuntimeError(f"{args.outdir} already holds a saved head; choose a new OUTDIR")
        st = os.statvfs(args.outdir)
        if st.f_bavail * st.f_frsize / GIB < 3:
            raise RuntimeError(f"less than 3 GiB free under {args.outdir}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(args.chat_template).read() if os.path.exists(args.chat_template) else None
    if template is None:
        raise RuntimeError(f"served chat template {args.chat_template} not found")

    rows, src_stats = load_rows_multi(args.data, args.exclude)
    random.Random(0).shuffle(rows)
    if args.eval_data:
        erows, est = load_rows_multi(args.eval_data, "")
        eval_docs, st_e = V2.build_docs(erows, tok, template, args.maxlen)
        eval_docs = eval_docs[:args.heldout]
        train_docs, st = V2.build_docs(rows, tok, template, args.maxlen)
    else:
        docs, st = V2.build_docs(rows, tok, template, args.maxlen)
        eval_docs, train_docs, st_e = docs[-args.heldout:], docs[:-args.heldout], None
    if len(eval_docs) < 8:
        raise RuntimeError(f"only {len(eval_docs)} held-out docs")
    packs = V2.pack_docs(train_docs, args.maxlen)
    random.Random(1).shuffle(packs)
    if args.max_packs and len(packs) > args.max_packs:
        packs = packs[:args.max_packs]
    data_info = {"sources": src_stats, "build": st, "eval_build": st_e, "train_docs": len(train_docs),
                 "train_packs": len(packs), "train_tokens": sum(len(p[0]) for p in packs),
                 "train_generated_tokens": sum(sum(p[1]) for p in packs), "heldout_docs": len(eval_docs),
                 "heldout_domains": _count(domain_key(d.kind) for d in eval_docs),
                 "train_domains": _count(domain_key(d.kind) for d in train_docs)}
    log("data", json.dumps(data_info))
    if args.smoke:
        eval_docs = eval_docs[:6]
        packs = sorted(packs, key=lambda p: -len(p[0]))[:2]
    total = 2 if args.smoke else len(packs) * args.epochs

    fid_docs = [d for d in eval_docs if len(d.ids) <= args.fidelity_maxlen][:args.fidelity_n]
    served_lp = []
    if fid_docs:
        t0 = time.time()
        served_lp = [served_prompt_logprobs(args.served_url, d.ids) for d in fid_docs]
        log(f"fidelity: served prompt log-probs of {len(fid_docs)} held-out docs "
            f"({sum(len(x[0]) for x in served_lp)} tokens) in {time.time() - t0:.0f}s from {args.served_url}")

    # ---- 1. stop vLLM, load the teacher
    log("before stop:", V2.fmt_mem(V2.mem_snapshot()))
    V2.stop_vllm()
    s = V2.mem_snapshot()
    if s["avail_gb"] < 2 * args.min_avail_gb:
        raise RuntimeError(f"only {s['avail_gb']:.1f} GiB host RAM available after stopping vLLM")
    t0 = time.time()
    V2.WATCHDOG.phase = "load"
    model, n_qdq, freed = V2.load_main_model(args.model)
    V2.WATCHDOG.phase = "run"
    t_load = time.time() - t0
    lm = model.model.language_model
    try:
        model.model.visual = None
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    _gpu("empty_cache")
    log(f"main model loaded in {t_load / 60:.1f} min, QDQ off on {n_qdq} modules | GPU "
        f"{_gpu('memory_allocated') / GIB:.1f} GiB | " + V2.fmt_mem(V2.mem_snapshot()))
    n_lin, n_fp4 = install_row_emulation(lm)
    log(f"row emulation on {n_lin} NVFP4 linears ({n_fp4} fp4x2-eligible)")
    if n_lin == 0:
        raise RuntimeError("no compressed-tensors linears found: is this the NVFP4 checkpoint?")
    install_kv_fp8()
    kcheck = None
    if args.tl_prefix:
        try:
            kcheck = kernel_check(args.model, args.tl_prefix)
        except Exception as ex:  # noqa: BLE001
            kcheck = {"error": f"{type(ex).__name__}: {ex}"[:300]}
            log("kernel check could not run:", kcheck["error"])
        bad = [k for k, v in kcheck.items() if isinstance(v, dict) and not v["emu_vs_kernel"] < 0.2 * v["w4a16_vs_kernel"]]
        if args.fidelity_require and ("error" in kcheck or bad):
            raise RuntimeError(f"kernel check failed ({bad or kcheck.get('error')}): the fp4x2 emulation does not "
                               "reproduce the production prefill kernel; not training on it")
    tcfg = model.config.text_config
    lm_head_w = model.lm_head.weight
    ctx = V2.Ctx(lm, lm_head_w, None, None, d2t, t2d, args.topk, args.grad_ckpt)

    # fidelity keys: v2 = v2's teacher (W4A16 rows, exact shard scales, BF16 KV); then one change at a time towards
    # serving: shard (the SHARD_SCALES setting), kv (FP8 KV), pi_all (every row on the fp4x2 emulation, which is what a
    # served prompt_logprobs request computes). The last one must be closer to the served log-probs than v2.
    fidelity = {}

    def fid(tag, rows_mode, kv):
        Emu.kv_fp8 = bool(kv)
        got = [teacher_prompt_logprobs(ctx, d.ids, fp4_row_mask(d.resp, rows_mode)) for d in fid_docs]
        fidelity[tag] = compare_lp(served_lp, got)
        log(f"fidelity {tag}: {json.dumps(fidelity[tag])}")

    if served_lp:
        fid("v2", "w4a16", 0)
    if args.shard_scales == "served":
        apply_served_shard_scales(lm, args.model, ctx.device)
    if served_lp:
        fid(f"w4a16+{args.shard_scales}_shard", "w4a16", 0)
        fid(f"w4a16+{args.shard_scales}_shard+fp8kv", "w4a16", 1)
        fid(f"pi_all+{args.shard_scales}_shard+fp8kv", "pi_all", 1)
        v2t, emu = fidelity["v2"]["mean_abs_dlogp"], fidelity[f"pi_all+{args.shard_scales}_shard+fp8kv"]["mean_abs_dlogp"]
        if args.fidelity_require and not emu < v2t:
            raise RuntimeError(f"fidelity check failed: the served-numerics teacher ({emu}) is not closer to the served "
                               f"log-probs than v2's teacher ({v2t}); emulation is wrong, not training on it")
    Emu.kv_fp8 = bool(args.kv_fp8)

    # ---- 2. MTP (production or original weights), QAT, draft head
    dev = lm_head_w.device
    with torch.device(dev):
        mtp = V2.make_mtp(tcfg).float()
    head_w = nn.Parameter(lm_head_w.detach()[d2t.to(dev)].float().clone())
    load_mtp_weights(mtp, head_w, d2t, args.prod_dir, args.model, "prod")
    if args.mtp_quant == "nvfp4":
        install_mtp_qat(mtp)
    ctx.mtp, ctx.head_w = mtp, head_w
    from safetensors.torch import load_file
    orig = load_file(os.path.join(args.model, "model_mtp.safetensors"))   # key set / shapes / dtypes to save with
    orig_meta = {k: (v.dtype, tuple(v.shape)) for k, v in orig.items()}
    del orig

    # ---- 3. held-out caches, production-head baselines (the PI diagnostic)
    t0 = time.time()
    caches, dstats = {}, {}
    modes = [args.serve_rows] + ([m for m in ("prepi", "w4a16") if m != args.serve_rows] if args.eval_only else [])
    for m in modes:
        caches[m], dstats[m] = cache_eval_docs_v4(ctx, eval_docs, m, args)
        log(f"held-out cache ({m}): {len(caches[m])} docs | data stats {json.dumps(dstats[m])}")
    t_cache = time.time() - t0
    if dstats[args.serve_rows]["data_greedy_frac"] < args.min_greedy_frac:
        raise RuntimeError(f"teacher reproduces < {args.min_greedy_frac:.0%} of its own generated tokens greedily: "
                           "loading looks broken")
    t0 = time.time()
    base = {m: evaluate_v4(ctx, caches[m], args.eval_nsteps, args.head_quant) for m in modes}
    t_eval = (time.time() - t0) / len(modes)
    for m in modes:
        log(fmt_eval_v4(f"(p) production head, teacher {m}", base[m], args.gate_steps))
    base_full = evaluate_v4(ctx, caches[args.serve_rows], 1, "full")
    log(fmt_eval_v4("(p) production MTP layer + full BF16 lm_head (load path, 1 draft)", base_full, (1,)))
    result = {"kernel_check": kcheck, "fidelity": fidelity, "data_stats": dstats, "baseline_prod": base,
              "baseline_prod_load_path": base_full,
              "pi_diagnostic": diag_table(base, args.gate_steps)}
    if args.eval_only:
        log("PI DIAGNOSTIC (acc_len at %d drafts, production head, identical text):" % max(args.gate_steps),
            json.dumps(result["pi_diagnostic"]))
        log("RESULT", json.dumps(result))
        return
    ref = base[args.serve_rows]
    cache = caches[args.serve_rows]
    if args.init == "orig":
        load_mtp_weights(mtp, head_w, d2t, args.prod_dir, args.model, "orig")
        with torch.no_grad():
            head_w.copy_(lm_head_w.detach()[d2t.to(dev)].float())
        if args.mtp_quant == "nvfp4":
            refresh_mtp_qat(mtp)
    mtp.train()

    # ---- 4. training
    params = list(mtp.parameters()) + [head_w]
    opt = torch.optim.AdamW([{"params": list(mtp.parameters()), "lr": args.lr, "base_lr": args.lr},
                             {"params": [head_w], "lr": args.lr_head, "base_lr": args.lr_head}],
                            betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    warm = min(50, max(1, total // 10))
    best, best_ev, best_step, history = None, None, None, []
    win = _new_win(args.nsteps)
    step, step_times = 0, []
    _gpu("reset_peak_memory_stats")
    t_train = time.time()
    log(f"training: {total} steps, warmup {warm}, LR {args.lr}/{args.lr_head}, NSTEPS {args.nsteps} STEP_W "
        f"{args.step_w}, rows {args.serve_rows} (+gen frac {args.gen_fp4x2_frac}), MTP {args.mtp_quant}, head "
        f"{args.head_quant}, KV fp8 {args.kv_fp8}, gate at {args.gate_steps} drafts vs the production head")
    for ep in range(1 if args.smoke else args.epochs):
        order = list(range(len(packs)))
        if not args.smoke:
            random.Random(100 + ep).shuffle(order)
        for pi in order:
            if step >= total:
                break
            ids, resp_l = packs[pi]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * V2.lr_scale(step, total, warm)
            t0 = time.time()
            rows = fp4_row_mask(resp_l, args.serve_rows, args.gen_fp4x2_frac, seed=10_000 + step,
                                prepi_min_rows=args.prepi_min_rows)
            mp = main_pass_v4(ctx, ids, rows)
            resp = torch.tensor(resp_l, dtype=torch.bool, device=dev)
            outs = V2.run_chain(mtp, mp["h"], mp["e"], mp["pe"], args.nsteps, args.grad_ckpt)
            if args.head_quant == "fp8":
                outs_h = [fp8_rows_ste(o) for o in outs]
                w_c = fp8_rows_ste(head_w.to(torch.bfloat16))
            else:
                outs_h, w_c = outs, head_w.to(torch.bfloat16)
            loss, st_k = V2.mtp_losses(outs_h, w_c, ctx.d2t, mp["q"], mp["di"], mp["valid"], mp["targ"], resp,
                                       args.step_w)
            opt.zero_grad(set_to_none=True)
            if loss is None or not torch.isfinite(loss):
                win["skipped"] += 1
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                    if args.mtp_quant == "nvfp4":
                        refresh_mtp_qat(mtp)
                    win["gn"] += float(gn)
                else:
                    win["skipped"] += 1
                win["loss"] += float(loss.detach())
            _gpu("synchronize")
            dt = time.time() - t0
            step += 1
            step_times.append(dt)
            win["t"] += dt; win["tok"] += len(ids); win["n"] += 1
            for k in range(args.nsteps):
                if k < len(st_k["kl"]) and st_k["count"][k]:
                    win["kl"][k].append(st_k["kl"][k]); win["agree"][k].append(st_k["agree"][k])
            V2.add_counts(win["chain"], V2.train_chain_counts(st_k["matches"], resp, args.nsteps))
            del mp, outs, outs_h, loss, st_k
            if step % args.log_every == 0 or args.smoke or step == total:
                n = max(1, win["n"])
                ch = V2.finalize_counts(win["chain"], args.nsteps)
                mean = lambda v: round(sum(v) / len(v), 4) if v else float("nan")  # noqa: E731
                log(f"step {step}/{total} | {win['t'] / n:.2f} s/step | loss {win['loss'] / n:.4f} | kl "
                    f"{[mean(v) for v in win['kl']]} | agree {[mean(v) for v in win['agree']]} | train acc_len "
                    f"{[ch['acc_len'][str(m)] for m in range(1, args.nsteps + 1)]} | lr {opt.param_groups[0]['lr']:.2e}"
                    f" | gnorm {win['gn'] / n:.3f} | skipped {win['skipped']} | GPU peak "
                    f"{_gpu('max_memory_allocated') / GIB:.1f} GiB | RAM avail {V2.mem_snapshot()['avail_gb']:.1f}"
                    f" GiB | eta {(total - step) * win['t'] / n / 60:.0f} min")
                win = _new_win(args.nsteps)
            if (not args.smoke) and (step % args.eval_every == 0 or step == total):
                ev = evaluate_v4(ctx, cache, args.eval_nsteps, args.head_quant)
                ev["load_path"] = evaluate_v4(ctx, cache, 1, "full")
                ok, det = gate_v4(ev, ref, args.gate_steps, floor=args.domain_floor, ev_full=ev["load_path"],
                                  ref_full=base_full, load_floor=args.load_floor)
                history.append({"step": step, "eval": ev, "gate": det, "passes": ok})
                log(fmt_eval_v4(f"EVAL step {step}", ev, args.gate_steps) + f" | vs production {json.dumps(det)} "
                    f"| passes {ok}")
                if ok and better_v4(ev, best_ev, args.gate_steps):
                    best, best_ev, best_step = V2.snapshot(ctx), ev, step
                    log(f"new best at step {step}")
    t_train = time.time() - t_train

    if args.smoke:
        ev = evaluate_v4(ctx, cache, args.eval_nsteps, args.head_quant)
        log(fmt_eval_v4("(c) after 2 smoke steps", ev, args.gate_steps))
        md, hd = V2.build_save_dicts(V2.snapshot(ctx), orig_meta)
        log(f"save dry run OK: {len(md)} MTP tensors, draft head {tuple(hd['weight'].shape)}; nothing written")
        full = data_info["train_packs"] * args.epochs
        per = args.heldout / max(1, len(cache))
        n_ev = full // args.eval_every + 1
        log(f"SMOKE OK | step times {[round(t, 2) for t in step_times]} s | peak GPU "
            f"{_gpu('max_memory_allocated') / GIB:.1f} GiB | min MemAvailable {V2.WATCHDOG.min_avail_seen:.1f} GiB"
            f" | projected full run: load {t_load / 60:.1f} min + fidelity + cache/baseline "
            f"~{(t_cache * per + t_eval * per) / 60:.0f} min + {full} steps x {step_times[-1]:.2f} s = "
            f"{full * step_times[-1] / 60:.0f} min + {n_ev} evals ~{n_ev * t_eval * per / 60:.0f} min")
        return

    # ---- 5. gate and save
    result.update({"history": history, "best_step": best_step, "trained": best_ev, "gate_steps": args.gate_steps})
    if best is None:
        log(f"NO IMPROVEMENT: no evaluation beat the production head at {args.gate_steps} drafts within the domain "
            f"floor {args.domain_floor}. Nothing saved.")
        log("RESULT", json.dumps(result))
        return
    md, hd = V2.build_save_dicts(best, orig_meta)
    meta = {"date": now("%d-%m-%Y %H:%M"), "script": os.path.abspath(__file__), "config": vars(args),
            "model": args.model, "vocab": {"path": args.vocab, "size": int(d2t.numel())}, "data": data_info,
            "timing_min": {"load": round(t_load / 60, 1), "train": round(t_train / 60, 1),
                           "total": round((time.time() - t_run) / 60, 1)},
            "mtp_keys": sorted(md), **result}
    paths = V2.save_outputs(args.outdir, args.model, md, hd, d2t, meta)
    final = gate_v4(best_ev, ref, args.gate_steps, floor=args.domain_floor, ev_full=best_ev["load_path"],
                    ref_full=base_full, load_floor=args.load_floor)[1]
    log(f"SAVED (best step {best_step}) | vs production {json.dumps(final)} | "
        f"files {json.dumps(paths)}")
    log("RESULT", json.dumps(result))


def _gpu(fn):
    """torch.cuda.<fn>() on a GPU run, 0 on a CPU test run."""
    return getattr(torch.cuda, fn)() if torch.cuda.is_available() else 0


def _count(it):
    out = {}
    for k in it:
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _new_win(K):
    return {"t": 0.0, "tok": 0, "n": 0, "loss": 0.0, "kl": [[] for _ in range(K)], "agree": [[] for _ in range(K)],
            "chain": V2.new_counts(K), "gn": 0.0, "skipped": 0}


def main(argv=None):
    import signal
    import traceback
    args = parse_args(argv)
    signal.signal(signal.SIGUSR1, V2._on_usr1)
    signal.signal(signal.SIGTERM, V2._on_term)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, V2._on_term)
    V2.WATCHDOG = V2.RamWatchdog(args.min_avail_gb, args.load_rss_gb)
    V2.WATCHDOG.start()
    code, why = 1, "error"
    try:
        run(args)
        code, why = 0, "finished"
    except V2.RamAbort as ex:
        code, why = 3, "RAM watchdog abort"
        log("ABORTED by the RAM watchdog:", ex)
    except KeyboardInterrupt:
        code, why = 130, "interrupted"
    except SystemExit as ex:
        code, why = 143, "terminated"
        log("terminated:", ex)
    except BaseException:  # noqa: BLE001
        log("FAILED")
        traceback.print_exc()
    finally:
        V2.STATE["shutdown"] = True
        V2.WATCHDOG.stop()
        if V2.STATE["vllm_stopped"]:
            V2.schedule_vllm_restart(why)
        log(f"exit {code} ({why})")
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
