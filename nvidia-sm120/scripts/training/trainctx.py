#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Training context-length ladder for the Qwen3.8-27B text model on one RTX PRO 6000 (SM120, 96 GB) - 27-09-2026.

A MEASUREMENT, not a training run: LoRA (rank 16, alpha 32) on every text-model linear projection, frozen base,
microbatch 1, one sequence per step, AdamW on the LoRA parameters only (fp32 master copy, BF16 compute).
No adapter is kept.

Two frozen base representations, both built from our NVFP4 checkpoint (the BF16 original is not on disk):
  r1  BF16-dequantized base resident on the GPU (~54 GB)
  r2  NVFP4-packed base resident (~19 GB incl. BF16 embeddings/lm_head); every quantized linear dequantizes its
      weight to BF16 on the fly in a custom autograd Function, again in backward (no BF16 weight copies are kept,
      and no activation is saved for the frozen weight).

Memory techniques (always on):
  * per-decoder-layer recompute (a reentrant checkpoint Function: only each layer's input is stored; optionally the
    first N layer inputs go to pinned host memory: --offload-gb)
  * chunked lm_head + cross-entropy with recompute (--ce-chunk tokens; full logits at 128k would be ~127 GB)
  * FLA chunk_gated_delta_rule (fwd+bwd) and causal_conv1d for the 48 Gated-DeltaNet layers
  * SDPA with a memory-efficient backend for the 16 full-attention layers (head_dim 256; backend chosen by --probe)
  * optional chunked MLP inside the layer recompute (--mlp-chunk tokens)

Modes:
  --probe            GPU capability probe (SDPA backends at head_dim 256, FLA + causal_conv1d vs torch references,
                     Triton NVFP4 dequant vs torch, expandable_segments allocator) -> JSON
  --ref-logprobs F   ask the SERVED model (vLLM, :8001) for prompt log-probs of the fixed 512-token prompt -> F
                     (CPU only; needs the served model up)
  --tokens F         tokenize the fixed public text (CNN/DailyMail test articles) once -> F (CPU only)
  default            load the model (--rep r1|r2), optional --smoke, then the ladder over --lengths
"""
import argparse
import gc
import json
import math
import os
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

GIB = 2 ** 30
MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
PREFIX = "model.language_model."
QUANT_LINEARS = {  # module path inside a decoder layer -> kind
    "linear_attention": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"],
    "full_attention": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}
BF16_LINEARS = {"linear_attention": ["linear_attn.in_proj_a", "linear_attn.in_proj_b"], "full_attention": []}


def now():
    return time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())


def log(*a):
    print(f"[{now()}]", *a, flush=True)


# =====================================================================================================================
# Host memory
# =====================================================================================================================

def _kb_table(path):
    out = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.partition(":")
            p = v.split()
            if p and p[0].isdigit():
                out[k.strip()] = int(p[0])
    return out


def host_mem():
    mi, st = _kb_table("/proc/meminfo"), _kb_table("/proc/self/status")
    return {"avail_gb": mi.get("MemAvailable", 0) / 2 ** 20, "rss_gb": st.get("VmRSS", 0) / 2 ** 20,
            "hwm_gb": st.get("VmHWM", 0) / 2 ** 20}


class HostMonitor(threading.Thread):
    """Samples MemAvailable / RSS; aborts the process (SIGKILL is the kernel's job otherwise) if host RAM runs out."""

    def __init__(self, min_avail_gb=3.0, period=0.5):
        super().__init__(daemon=True)
        self.min_avail_gb, self.period = min_avail_gb, period
        self.min_avail, self.max_rss = float("inf"), 0.0
        self.tripped = None

    def reset(self):
        self.min_avail, self.max_rss = float("inf"), 0.0

    def run(self):
        while True:
            try:
                m = host_mem()
                self.min_avail = min(self.min_avail, m["avail_gb"])
                self.max_rss = max(self.max_rss, m["rss_gb"])
                if m["avail_gb"] < self.min_avail_gb and self.tripped is None:
                    self.tripped = f"host MemAvailable {m['avail_gb']:.1f} GiB < {self.min_avail_gb}"
                    log("HOST RAM WATCHDOG:", self.tripped, "- exiting")
                    sys.stdout.flush()
                    os._exit(7)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.period)


# =====================================================================================================================
# NVFP4 dequantization
# =====================================================================================================================

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_nvfp4_torch(packed, scale, gscale, dtype=torch.bfloat16):
    """compressed-tensors nvfp4-pack-quantized -> dense. packed (N, K/2) uint8 (low nibble = even column),
    scale (N, K/16) float8_e4m3fn, gscale (1,) fp32. w = fp4 * (scale / gscale), computed in fp32, cast once."""
    N, K2 = packed.shape
    K = K2 * 2
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    p = packed.to(torch.int32)
    vals = torch.stack((lut[p & 0x0F], lut[p >> 4]), dim=-1).view(N, K // 16, 16)
    s = scale.to(torch.float32) / gscale.to(torch.float32)
    return (vals * s.unsqueeze(-1)).view(N, K).to(dtype)


_TRITON = {"kernel": None, "ok": None}


def _build_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dq(p_ptr, s_ptr, g_ptr, o_ptr, N, K2, BLOCK_N: tl.constexpr, BLOCK_K2: tl.constexpr):
        # packed column j holds output columns 2j (low nibble) and 2j+1 (high nibble); both in scale group j // 8
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_k * BLOCK_K2 + tl.arange(0, BLOCK_K2)
        mask = (rn[:, None] < N) & (rk[None, :] < K2)
        p = tl.load(p_ptr + rn[:, None] * K2 + rk[None, :], mask=mask, other=0).to(tl.int32)
        # e4m3fn scale, decoded from its bits (exact; no fp8 dtype needed, works in the interpreter too)
        sb = tl.load(s_ptr + rn[:, None] * (K2 // 8) + (rk[None, :] // 8), mask=mask, other=0).to(tl.int32)
        se = (sb >> 3) & 0xF
        sm = (sb & 0x7).to(tl.float32)
        pw = ((se + 120) << 23).to(tl.float32, bitcast=True)                  # 2^(se-7), exact
        smag = tl.where(se == 0, sm * 0.001953125, (1.0 + sm * 0.125) * pw)    # subnormal: m/8 * 2^-6
        sval = tl.where((sb & 0x80) != 0, -smag, smag)
        g = tl.load(g_ptr)
        s = tl.math.div_rn(sval, g)                                            # IEEE division like torch
        lo = p & 0xF
        hi = (p >> 4) & 0xF
        # e2m1: sign bit 3, exponent bits 2-1, mantissa bit 0 -> 0, 0.5, 1, 1.5, 2, 3, 4, 6
        le = (lo >> 1) & 3
        lm = (lo & 1).to(tl.float32)
        lmag = tl.where(le == 0, lm * 0.5, (1.0 + lm * 0.5) * tl.where(le == 1, 1.0, tl.where(le == 2, 2.0, 4.0)))
        lv = tl.where((lo & 8) != 0, -lmag, lmag)
        he = (hi >> 1) & 3
        hm = (hi & 1).to(tl.float32)
        hmag = tl.where(he == 0, hm * 0.5, (1.0 + hm * 0.5) * tl.where(he == 1, 1.0, tl.where(he == 2, 2.0, 4.0)))
        hv = tl.where((hi & 8) != 0, -hmag, hmag)
        base = o_ptr + rn[:, None] * (2 * K2) + 2 * rk[None, :]
        tl.store(base, (lv * s).to(o_ptr.dtype.element_ty), mask=mask)
        tl.store(base + 1, (hv * s).to(o_ptr.dtype.element_ty), mask=mask)

    return _dq


def dequant_nvfp4_triton(packed, scale, gscale, dtype=torch.bfloat16):
    if _TRITON["kernel"] is None:
        _TRITON["kernel"] = _build_triton_kernel()
    N, K2 = packed.shape
    out = torch.empty((N, 2 * K2), dtype=dtype, device=packed.device)
    BN, BK2 = 32, 64
    grid = ((N + BN - 1) // BN, (K2 + BK2 - 1) // BK2)
    _TRITON["kernel"][grid](packed, scale.view(torch.uint8), gscale.to(torch.float32).reshape(1), out, N, K2,
                            BLOCK_N=BN, BLOCK_K2=BK2)
    return out


def dequant_nvfp4(packed, scale, gscale, dtype=torch.bfloat16):
    if packed.is_cuda and _TRITON["ok"]:
        return dequant_nvfp4_triton(packed, scale, gscale, dtype)
    return dequant_nvfp4_torch(packed, scale, gscale, dtype)


class NVFP4LinearFn(torch.autograd.Function):
    """y = x @ dequant(W)^T with a frozen NVFP4 weight. Nothing but the packed weight is kept for backward: the BF16
    weight is rebuilt in backward (grad_x = grad_y @ W); no weight gradient."""

    @staticmethod
    def forward(ctx, x, packed, scale, gscale):
        w = dequant_nvfp4(packed, scale, gscale, x.dtype)
        ctx.save_for_backward(packed, scale, gscale)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, gy):
        packed, scale, gscale = ctx.saved_tensors
        w = dequant_nvfp4(packed, scale, gscale, gy.dtype)
        return gy @ w, None, None, None


# =====================================================================================================================
# LoRA linear over a frozen base
# =====================================================================================================================

class LoRALinear(nn.Module):
    """Frozen base (bf16 weight | nvfp4 packed) + LoRA A (r, in), B (out, r) in fp32 (master), BF16 compute."""

    def __init__(self, in_f, out_f, rank=16, alpha=32, lora=True):
        super().__init__()
        self.in_features, self.out_features = in_f, out_f
        self.weight = None                       # bf16 frozen (r1, or the BF16 in_proj_a/b)
        self.packed = self.scale = self.gscale = None
        self.rank, self.scaling = rank, alpha / rank
        if lora and rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = self.lora_B = None

    def set_dense(self, w):
        assert tuple(w.shape) == (self.out_features, self.in_features), (tuple(w.shape), self.out_features)
        self.weight = w.requires_grad_(False)
        self.packed = self.scale = self.gscale = None

    def set_packed(self, packed, scale, gscale):
        assert tuple(packed.shape) == (self.out_features, self.in_features // 2), tuple(packed.shape)
        assert tuple(scale.shape) == (self.out_features, self.in_features // 16), tuple(scale.shape)
        self.packed, self.scale, self.gscale = packed, scale, gscale.reshape(1).float()
        self.weight = None

    def base(self, x):
        if self.weight is not None:
            return F.linear(x, self.weight)
        return NVFP4LinearFn.apply(x, self.packed, self.scale, self.gscale)

    def forward(self, x):
        y = self.base(x)
        if self.lora_A is not None:
            y = y + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        return y


# =====================================================================================================================
# Model (text-only Qwen3.8 / qwen3_5: 48 Gated-DeltaNet + 16 gated full-attention layers)
# =====================================================================================================================

class RMSNorm(nn.Module):
    """Qwen3_5RMSNorm: fp32 normalisation, weight stored as (w) and applied as (1 + w)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """Qwen3_5RMSNormGated: norm before gate, weight applied as (w), silu(gate) in fp32."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x, gate):
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dt)
        h = h * F.silu(gate.float())
        return h.to(dt)


class Impl:
    """Kernel selection (set once by build/probe)."""
    gdn = "fla"            # fla | torch
    conv = "cuda"          # cuda (causal_conv1d) | torch
    sdpa_backends = None   # list of torch.nn.attention.SDPBackend or None (default dispatch)
    mlp_chunk = 0          # tokens per MLP chunk inside the layer recompute (0: whole sequence)


def _torch_gdn():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    f = m.torch_chunk_gated_delta_rule
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    return f


def gated_delta_rule(q, k, v, g, beta):
    """q, k: (1, T, H, K) already expanded to the value heads, v: (1, T, HV, V), g fp32 (1, T, HV), beta (1, T, HV)."""
    if Impl.gdn == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)
        return o
    o, _ = _torch_gdn()(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=True)
    return o


def causal_conv_silu(x, w):
    """x (1, C, T) (channel-last strides allowed), w (C, width) -> silu(causal depthwise conv)."""
    if Impl.conv == "cuda":
        from causal_conv1d import causal_conv1d_fn
        return causal_conv1d_fn(x, w, None, activation="silu")
    T = x.shape[-1]
    out = F.conv1d(x.to(w.dtype), w.unsqueeze(1), None, padding=w.shape[-1] - 1, groups=x.shape[1])[..., :T]
    return F.silu(out).to(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, c, rank, alpha):
        super().__init__()
        H = c["hidden_size"]
        self.nk, self.nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        self.dk, self.dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_w = nn.Parameter(torch.zeros(conv_dim, c["linear_conv_kernel_dim"]), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.ones(self.nv), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(self.nv), requires_grad=False)
        self.norm = RMSNormGated(self.dv, c["rms_norm_eps"])
        self.in_proj_qkv = LoRALinear(H, conv_dim, rank, alpha)
        self.in_proj_z = LoRALinear(H, self.value_dim, rank, alpha)
        self.in_proj_b = LoRALinear(H, self.nv, rank, alpha)
        self.in_proj_a = LoRALinear(H, self.nv, rank, alpha)
        self.out_proj = LoRALinear(self.value_dim, H, rank, alpha)

    def forward(self, x):                                   # x: (1, T, H)
        T = x.shape[1]
        qkv = self.in_proj_qkv(x).transpose(1, 2)            # (1, C, T), channel-last strides
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        qkv = causal_conv_silu(qkv, self.conv_w).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, T, self.nk, self.dk)
        k = k.reshape(1, T, self.nk, self.dk)
        v = v.reshape(1, T, self.nv, self.dv)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        rep = self.nv // self.nk
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        o = gated_delta_rule(q, k, v, g, beta)                # (1, T, HV, V)
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(1, T, -1)
        return self.out_proj(o)


def rope_cos_sin(c, T, device, dtype):
    """Text-only positions: the interleaved mrope of Qwen3.5 reduces to plain RoPE on the rotary dims (all three
    position rows are equal). cos/sin in `dtype` like HF (computed in fp32)."""
    hd = c["head_dim"]
    rp = c.get("rope_parameters") or {}
    theta = rp.get("rope_theta", c.get("rope_theta", 10000.0))
    prf = rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    dim = int(hd * prf)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    fr = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([fr, fr], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x, cos, sin):                                # x: (1, heads, T, hd); cos/sin: (T, rot)
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + _rot_half(xr) * sin, xp], dim=-1)


class Attention(nn.Module):
    def __init__(self, c, rank, alpha):
        super().__init__()
        H = c["hidden_size"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.q_proj = LoRALinear(H, self.nh * self.hd * 2, rank, alpha)
        self.k_proj = LoRALinear(H, self.nkv * self.hd, rank, alpha)
        self.v_proj = LoRALinear(H, self.nkv * self.hd, rank, alpha)
        self.o_proj = LoRALinear(self.nh * self.hd, H, rank, alpha)
        self.q_norm = RMSNorm(self.hd, c["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, c["rms_norm_eps"])

    def forward(self, x, rope):
        T = x.shape[1]
        qg = self.q_proj(x).view(1, T, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        gate = gate.reshape(1, T, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(1, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(1, T, self.nkv, self.hd).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if Impl.sdpa_backends:
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(Impl.sdpa_backends):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.hd ** -0.5)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.hd ** -0.5)
        o = o.transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c, rank, alpha):
        super().__init__()
        H, I = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = LoRALinear(H, I, rank, alpha)
        self.up_proj = LoRALinear(H, I, rank, alpha)
        self.down_proj = LoRALinear(I, H, rank, alpha)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Recompute(torch.autograd.Function):
    """Reentrant activation checkpoint: forward runs fn under no_grad and keeps only x (on the GPU, or in a pinned
    host buffer); backward re-runs fn with grad and back-propagates into the LoRA parameters and x."""

    @staticmethod
    def forward(ctx, fn, host_buf, x):
        ctx.fn = fn
        ctx.host_buf = host_buf
        if host_buf is not None:
            host_buf.copy_(x, non_blocking=True)
            ctx.x_meta = (x.shape, x.dtype, x.device)
        else:
            ctx.save_for_backward(x)
        with torch.no_grad():
            return fn(x)

    @staticmethod
    def backward(ctx, gy):
        if ctx.host_buf is not None:
            shape, dtype, dev = ctx.x_meta
            x = ctx.host_buf.to(dev, non_blocking=True).view(shape)
        else:
            (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = ctx.fn(x)
        torch.autograd.backward(y, gy)
        return None, None, x.grad


class DecoderLayer(nn.Module):
    def __init__(self, c, idx, rank, alpha):
        super().__init__()
        self.kind = c["layer_types"][idx]
        if self.kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(c, rank, alpha)
        else:
            self.self_attn = Attention(c, rank, alpha)
        self.mlp = MLP(c, rank, alpha)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def forward(self, h, rope):
        x = self.input_layernorm(h)
        h = h + (self.linear_attn(x) if self.kind == "linear_attention" else self.self_attn(x, rope))
        m = self.post_attention_layernorm(h)
        ch = Impl.mlp_chunk
        if ch and m.shape[1] > ch:
            if torch.is_grad_enabled():
                outs = [Recompute.apply(self.mlp, None, m[:, s:s + ch]) for s in range(0, m.shape[1], ch)]
            else:
                outs = [self.mlp(m[:, s:s + ch]) for s in range(0, m.shape[1], ch)]
            return h + torch.cat(outs, dim=1)
        return h + self.mlp(m)


class TextModel(nn.Module):
    def __init__(self, c, rank=16, alpha=32):
        super().__init__()
        self.c = c
        self.embed = None      # (V, H) bf16, frozen
        self.lm_head = None    # (V, H) bf16, frozen
        self.layers = nn.ModuleList([DecoderLayer(c, i, rank, alpha) for i in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def hidden(self, ids, ckpt=False, host_bufs=None):
        """ids (T,) long -> final-norm hidden (1, T, H)."""
        T = ids.shape[0]
        h = F.embedding(ids, self.embed).unsqueeze(0)
        rope = rope_cos_sin(self.c, T, h.device, h.dtype)
        if ckpt and torch.is_grad_enabled():
            h = h.detach().requires_grad_(True)           # so every Recompute output requires grad
            for i, layer in enumerate(self.layers):
                buf = host_bufs[i] if host_bufs is not None else None
                h = Recompute.apply(lambda x, _l=layer: _l(x, rope), buf, h)
        else:
            for layer in self.layers:
                h = layer(h, rope)
        return self.norm(h)

    def logits(self, h):
        return F.linear(h, self.lm_head)


class CEChunk(torch.autograd.Function):
    """sum over a chunk of CE(lm_head(h_c), y_c) in fp32; logits recomputed in backward (never stored), softmax
    built in place (peak ~2 fp32 chunk-logit buffers)."""

    @staticmethod
    def forward(ctx, h, w, y):
        ctx.save_for_backward(h, y)
        ctx.w = w
        logits = F.linear(h, w).float()
        lse = torch.logsumexp(logits, dim=-1)
        tgt = logits.gather(1, y[:, None]).squeeze(1)
        return (lse - tgt).sum()

    @staticmethod
    def backward(ctx, g):
        h, y = ctx.saved_tensors
        w = ctx.w
        p = F.linear(h, w).float()
        lse = torch.logsumexp(p, dim=-1, keepdim=True)
        p.sub_(lse).exp_()                                       # softmax, in place
        p[torch.arange(y.shape[0], device=y.device), y] -= 1.0
        p.mul_(g)
        gh = p.to(h.dtype) @ w
        return gh, None, None


def chunked_ce(h, w, y, chunk):
    """h (T, H) with grad, w (V, H) frozen, y (T,) -> mean CE."""
    total = None
    for s in range(0, h.shape[0], chunk):
        part = CEChunk.apply(h[s:s + chunk], w, y[s:s + chunk])
        total = part if total is None else total + part
    return total / h.shape[0]


# =====================================================================================================================
# Loading (streamed from safetensors, straight to the GPU)
# =====================================================================================================================

def text_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def expected_tensors(c):
    """name -> (kind, expected shape) for every text-model tensor we load (no vision, no MTP)."""
    H, V = c["hidden_size"], c["vocab_size"]
    exp = {"lm_head.weight": ("bf16", (V, H)), PREFIX + "embed_tokens.weight": ("bf16", (V, H)),
           PREFIX + "norm.weight": ("bf16", (H,))}
    for i, kind in enumerate(c["layer_types"]):
        p = f"{PREFIX}layers.{i}."
        exp[p + "input_layernorm.weight"] = ("bf16", (H,))
        exp[p + "post_attention_layernorm.weight"] = ("bf16", (H,))
        lin = QUANT_LINEARS[kind] + QUANT_LINEARS["mlp"]
        shapes = _linear_shapes(c, kind)
        for name in lin:
            o, k = shapes[name]
            exp[p + name + ".weight_packed"] = ("u8", (o, k // 2))
            exp[p + name + ".weight_scale"] = ("f8", (o, k // 16))
            exp[p + name + ".weight_global_scale"] = ("f32", (1,))
        for name in BF16_LINEARS[kind]:
            exp[p + name + ".weight"] = ("bf16", shapes[name])
        if kind == "linear_attention":
            nv = c["linear_num_value_heads"]
            conv = 2 * c["linear_num_key_heads"] * c["linear_key_head_dim"] + nv * c["linear_value_head_dim"]
            exp[p + "linear_attn.A_log"] = ("bf16", (nv,))
            exp[p + "linear_attn.dt_bias"] = ("bf16", (nv,))
            exp[p + "linear_attn.conv1d.weight"] = ("bf16", (conv, 1, c["linear_conv_kernel_dim"]))
            exp[p + "linear_attn.norm.weight"] = ("bf16", (c["linear_value_head_dim"],))
        else:
            exp[p + "self_attn.q_norm.weight"] = ("bf16", (c["head_dim"],))
            exp[p + "self_attn.k_norm.weight"] = ("bf16", (c["head_dim"],))
    return exp


def _linear_shapes(c, kind):
    H, I = c["hidden_size"], c["intermediate_size"]
    s = {"mlp.gate_proj": (I, H), "mlp.up_proj": (I, H), "mlp.down_proj": (H, I)}
    if kind == "linear_attention":
        kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
        vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
        nv = c["linear_num_value_heads"]
        s.update({"linear_attn.in_proj_qkv": (2 * kd + vd, H), "linear_attn.in_proj_z": (vd, H),
                  "linear_attn.out_proj": (H, vd), "linear_attn.in_proj_a": (nv, H), "linear_attn.in_proj_b": (nv, H)})
    else:
        nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        s.update({"self_attn.q_proj": (2 * nh * hd, H), "self_attn.k_proj": (nkv * hd, H),
                  "self_attn.v_proj": (nkv * hd, H), "self_attn.o_proj": (H, nh * hd)})
    return s


def setp(mod, name, t):
    """Replace a (meta/placeholder) frozen parameter with the loaded tensor."""
    old = getattr(mod, name)
    assert tuple(old.shape) == tuple(t.shape), (name, tuple(old.shape), tuple(t.shape))
    setattr(mod, name, nn.Parameter(t, requires_grad=False))


def _get(mod, path):
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


FUSED_SHARDS = {"linear_attention": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]}


def load_weights(model, model_dir, rep, device, log_every=8, shard_scales="served"):
    """Fill `model` from the checkpoint. rep r1: quantized linears dequantized to BF16 on the GPU; r2: packed kept.
    shard_scales "exact": every linear with its own NVFP4 global scale (compressed-tensors / transformers semantics).
    "served": like vLLM, which fuses in_proj_qkv + in_proj_z into one layer and keeps only the LARGER of the two
    global scales for both (compressed_tensors_w4a4_nvfp4.py: weight_global_scale.max()). In our checkpoint the two
    differ by up to 2.1x, so the served model runs one of the shards 0.47-0.97x smaller than its true size
    (our run notes, 25/26-09-2026, the shard-scale puzzle). q/k/v and gate/up share one scale."""
    from safetensors import safe_open

    c = model.c
    path = os.path.join(model_dir, "model.safetensors")
    t0 = time.time()
    with safe_open(path, framework="pt", device="cpu") as f:
        def g(name, dtype=None):
            t = f.get_tensor(name).to(device, non_blocking=False)
            return t if dtype is None else t.to(dtype)

        model.embed = g(PREFIX + "embed_tokens.weight").contiguous()
        model.lm_head = g("lm_head.weight").contiguous()
        setp(model.norm, "weight", g(PREFIX + "norm.weight"))
        for i, layer in enumerate(model.layers):
            p = f"{PREFIX}layers.{i}."
            setp(layer.input_layernorm, "weight", g(p + "input_layernorm.weight"))
            setp(layer.post_attention_layernorm, "weight", g(p + "post_attention_layernorm.weight"))
            kind = layer.kind
            fused = FUSED_SHARDS.get(kind, []) if shard_scales == "served" else []
            gmax = max(g(p + n + ".weight_global_scale", torch.float32) for n in fused) if fused else None
            for name in QUANT_LINEARS[kind] + QUANT_LINEARS["mlp"]:
                mod = _get(layer, name)
                packed, scale = g(p + name + ".weight_packed"), g(p + name + ".weight_scale")
                gs = g(p + name + ".weight_global_scale", torch.float32)
                if name in fused:
                    gs = gmax.clone()
                if rep == "r1":
                    mod.set_dense(dequant_nvfp4(packed, scale, gs, torch.bfloat16))
                    del packed, scale
                else:
                    mod.set_packed(packed, scale, gs)
            for name in BF16_LINEARS[kind]:
                _get(layer, name).set_dense(g(p + name + ".weight"))
            if kind == "linear_attention":
                la = layer.linear_attn
                setp(la, "A_log", g(p + "linear_attn.A_log"))
                setp(la, "dt_bias", g(p + "linear_attn.dt_bias"))
                setp(la, "conv_w", g(p + "linear_attn.conv1d.weight").squeeze(1).contiguous())
                setp(la.norm, "weight", g(p + "linear_attn.norm.weight"))
            else:
                sa = layer.self_attn
                setp(sa.q_norm, "weight", g(p + "self_attn.q_norm.weight"))
                setp(sa.k_norm, "weight", g(p + "self_attn.k_norm.weight"))
            if device != "cpu" and (i % log_every == log_every - 1):
                log(f"  loaded {i + 1}/{len(model.layers)} layers, GPU {torch.cuda.memory_allocated() / GIB:.1f} GiB,"
                    f" host RSS {host_mem()['rss_gb']:.1f} GiB, {time.time() - t0:.0f}s")
    return time.time() - t0


def build_model(model_dir, rep, device, rank=16, alpha=32, seed=0, shard_scales="served"):
    c = text_config(model_dir)
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = TextModel(c, rank, alpha)
    # LoRA params and small frozen params materialised on the device (fp32 LoRA, bf16 frozen); base weights loaded
    model = model.to_empty(device=device)
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.lora_A is not None:
            nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
            nn.init.zeros_(m.lora_B)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    t = load_weights(model, model_dir, rep, device, shard_scales=shard_scales)
    return model, t


# =====================================================================================================================
# Probe (GPU): kernels / backends that the ladder depends on
# =====================================================================================================================

def probe(out_path):
    import torch.nn.attention as tna

    dev = "cuda"
    res = {"time": now(), "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0))}
    log("probe:", res)
    # --- SDPA at head_dim 256, 24 query heads (after GQA expansion), causal, fwd + bwd
    torch.manual_seed(0)
    Tq, nh, hd = 256, 24, 256
    q, k, v = (torch.randn(1, nh, Tq, hd, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    go = torch.randn(1, nh, Tq, hd, device=dev, dtype=torch.bfloat16)
    with tna.sdpa_kernel([tna.SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True, scale=hd ** -0.5)
        ref.backward(go.float())
    rg = [t.grad.clone() for t in (q, k, v)]
    backends = {}
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"):
        be = getattr(tna.SDPBackend, name, None)
        if be is None:
            backends[name] = {"ok": False, "err": "missing"}
            continue
        try:
            for t in (q, k, v):
                t.grad = None
            with tna.sdpa_kernel([be]):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd ** -0.5)
                o.backward(go)
            err = max((o.float() - ref).abs().max().item(),
                      *[(t.grad.float() - r).abs().max().item() / (r.abs().max().item() + 1e-9)
                        for t, r in zip((q, k, v), rg)])
            # speed at 8k
            T8 = 8192
            q8, k8, v8 = (torch.randn(1, nh, T8, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
                          for _ in range(3))
            g8 = torch.randn_like(q8)
            with tna.sdpa_kernel([be]):
                for it in range(2):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    o8 = F.scaled_dot_product_attention(q8, k8, v8, is_causal=True, scale=hd ** -0.5)
                    o8.backward(g8)
                    torch.cuda.synchronize()
                    dt = time.time() - t0
            backends[name] = {"ok": err < 0.05, "max_err": round(err, 5), "fwd_bwd_8k_s": round(dt, 4)}
            del q8, k8, v8, g8, o8
        except Exception as ex:  # noqa: BLE001
            backends[name] = {"ok": False, "err": repr(ex)[:300]}
        torch.cuda.empty_cache()
    res["sdpa"] = backends
    ok = [n for n, r in backends.items() if r.get("ok")]
    res["sdpa_choice"] = min(ok, key=lambda n: backends[n]["fwd_bwd_8k_s"]) if ok else None
    log("sdpa:", json.dumps(backends), "->", res["sdpa_choice"])

    # --- causal_conv1d (cuda) vs torch conv, fwd + bwd, channel-last input like the model
    try:
        C, T = 10240, 4096
        x0 = torch.randn(1, T, C, device=dev, dtype=torch.bfloat16)
        w = torch.randn(C, 4, device=dev, dtype=torch.bfloat16) * 0.3
        gy = torch.randn(1, C, T, device=dev, dtype=torch.bfloat16)
        outs = {}
        for impl in ("torch", "cuda"):
            Impl.conv = impl
            x = x0.clone().requires_grad_(True)
            y = causal_conv_silu(x.transpose(1, 2), w)
            y.backward(gy)
            outs[impl] = (y.float(), x.grad.float())
        e = max((outs["cuda"][0] - outs["torch"][0]).abs().max().item(),
                (outs["cuda"][1] - outs["torch"][1]).abs().max().item() / (outs["torch"][1].abs().max().item()))
        res["causal_conv1d"] = {"ok": e < 0.05, "max_err": round(e, 5)}
    except Exception as ex:  # noqa: BLE001
        res["causal_conv1d"] = {"ok": False, "err": repr(ex)[:300]}
    Impl.conv = "cuda" if res["causal_conv1d"].get("ok") else "torch"
    log("causal_conv1d:", res["causal_conv1d"])

    # --- FLA chunk_gated_delta_rule vs the transformers torch reference (fp32), fwd + bwd
    try:
        torch.manual_seed(1)
        T, Hh, K = 256, 6, 128
        base = [torch.randn(1, T, Hh, K, device=dev) for _ in range(3)]
        a0 = torch.randn(1, T, Hh, device=dev)
        b0 = torch.randn(1, T, Hh, device=dev)
        go = torch.randn(1, T, Hh, K, device=dev)
        outs = {}
        for impl in ("torch", "fla"):
            Impl.gdn = impl
            q, k, v = (t.clone().to(torch.bfloat16).requires_grad_(True) for t in base)
            g = -F.softplus(a0) * 0.5
            beta = b0.sigmoid().to(torch.bfloat16)
            o = gated_delta_rule(q, k, v, g, beta)
            o.float().backward(go)
            outs[impl] = [o.float()] + [t.grad.float() for t in (q, k, v)]
        e = max(((a - b).abs().max() / (b.abs().max() + 1e-6)).item() for a, b in zip(outs["fla"], outs["torch"]))
        res["fla_gdn"] = {"ok": e < 0.05, "max_rel_err": round(e, 5)}
    except Exception as ex:  # noqa: BLE001
        res["fla_gdn"] = {"ok": False, "err": repr(ex)[:300]}
    log("fla gdn:", res["fla_gdn"])

    # --- Triton NVFP4 dequant vs torch reference on real tensors
    try:
        from safetensors import safe_open
        with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu") as f:
            errs = []
            for name in ("layers.0.mlp.down_proj", "layers.3.self_attn.q_proj", "layers.0.linear_attn.in_proj_qkv"):
                p = PREFIX + name
                pk = f.get_tensor(p + ".weight_packed").to(dev)
                sc = f.get_tensor(p + ".weight_scale").to(dev)
                gs = f.get_tensor(p + ".weight_global_scale").to(dev).float()
                a = dequant_nvfp4_torch(pk, sc, gs)
                b = dequant_nvfp4_triton(pk, sc, gs)
                errs.append(int((a != b).sum().item()))          # by value (+0.0 / -0.0 are the same weight)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    dequant_nvfp4_triton(pk, sc, gs)
                torch.cuda.synchronize()
                tt = (time.time() - t0) / 5
                t0 = time.time()
                for _ in range(5):
                    dequant_nvfp4_torch(pk, sc, gs)
                torch.cuda.synchronize()
                res.setdefault("dequant_ms", {})[name] = {"triton": round(tt * 1e3, 3),
                                                          "torch": round((time.time() - t0) / 5 * 1e3, 3),
                                                          "shape": list(a.shape)}
        res["triton_dequant"] = {"ok": sum(errs) == 0, "mismatched_elements": errs}
    except Exception as ex:  # noqa: BLE001
        res["triton_dequant"] = {"ok": False, "err": repr(ex)[:300]}
    log("triton dequant:", res["triton_dequant"], res.get("dequant_ms"))
    res["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    log("probe written:", out_path)


# =====================================================================================================================
# Data
# =====================================================================================================================

CORPUS = ("$HK_DATA/huggingface/hub/datasets--abisee--cnn_dailymail/snapshots/"
          "96df5e686bee6baa90b8bee7c28b81fa3fa6223d/3.0.0/test-00000-of-00001.parquet")


def make_tokens(out_path, max_articles=400):
    """Fixed public text: the first 400 articles of the CNN/DailyMail 3.0.0 test split (news), joined by blank
    lines and tokenized once. Memory and speed do not depend on content; real tokens keep the loss meaningful."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    arts = pq.read_table(CORPUS, columns=["article"]).column("article").to_pylist()[:max_articles]
    ids = tok("\n\n".join(arts), add_special_tokens=False)["input_ids"]
    t = torch.tensor(ids, dtype=torch.int32)
    torch.save({"ids": t, "source": f"CNN/DailyMail 3.0.0 test split, first {len(arts)} articles (public), "
                                    "tokenizer of Qwen3.8-27B-NVFP4"}, out_path)
    log(f"tokens: {t.numel()} ids from {len(arts)} articles -> {out_path}")


def seq_for(ids, T, step, seed=1234):
    """T + 1 real tokens (inputs + next-token labels) from a fixed-seed offset, wrapping around the corpus."""
    g = torch.Generator().manual_seed(seed * 1000003 + T * 31 + step)
    n = ids.numel()
    off = int(torch.randint(0, n, (1,), generator=g).item())
    idx = (torch.arange(T + 1) + off) % n
    return ids[idx].long()


REF_PROMPT_LEN = 512
REF_OFFSET = 1000


def ref_prompt(ids):
    return ids[REF_OFFSET:REF_OFFSET + REF_PROMPT_LEN].long()


def ref_logprobs(tokens_path, out_path, url="http://localhost:8001/v1/completions"):
    import urllib.request

    ids = torch.load(tokens_path)["ids"]
    prompt = ref_prompt(ids).tolist()
    body = {"model": "qwen3.8", "prompt": prompt, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    pl = r["choices"][0]["prompt_logprobs"]
    lp, top1, top1_lp = [], [], []
    for pos, d in enumerate(pl):
        if d is None:
            lp.append(None); top1.append(None); top1_lp.append(None)
            continue
        lp.append(d[str(prompt[pos])]["logprob"])
        best = min(d.items(), key=lambda kv: kv[1].get("rank", 99))
        top1.append(int(best[0])); top1_lp.append(best[1]["logprob"])
    out = {"time": now(), "url": url, "model": r.get("model"), "prompt": prompt, "logprob": lp, "top1": top1,
           "top1_logprob": top1_lp}
    with open(out_path, "w") as f:
        json.dump(out, f)
    log(f"reference prompt log-probs: {sum(x is not None for x in lp)} positions -> {out_path}")


# =====================================================================================================================
# Training steps
# =====================================================================================================================

def check_logits(a):
    """Forward the fixed 512-token prompt through the base model (LoRA B = 0) and compare with the served model's
    prompt log-probs. Works on the CPU too (torch reference kernels, no autograd), e.g. while the GPU is busy."""
    dev = a.device
    if dev == "cpu":
        Impl.gdn, Impl.conv, Impl.sdpa_backends = "torch", "torch", None
        torch.set_num_threads(a.threads)
    else:
        with open(a.probe_file) as f:
            pr = json.load(f)
        import torch.nn.attention as tna
        Impl.sdpa_backends = [getattr(tna.SDPBackend, pr["sdpa_choice"])]
        Impl.conv = "cuda" if pr.get("causal_conv1d", {}).get("ok") else "torch"
        _TRITON["ok"] = bool(pr.get("triton_dequant", {}).get("ok"))
    ids = torch.load(a.token_file)["ids"]
    with open(a.ref_file) as f:
        ref = json.load(f)
    t0 = time.time()
    model, t_load = build_model(MODEL, a.rep, dev, shard_scales=a.shard_scales)
    prompt = ref_prompt(ids).to(dev)
    assert ref["prompt"] == prompt.tolist(), "reference prompt differs"
    with torch.no_grad():
        lp = torch.log_softmax(model.logits(model.hidden(prompt)[0]).float(), dim=-1).cpu()
    ours = lp[:-1].gather(1, prompt[1:, None].cpu()).squeeze(1)
    top1 = lp[:-1].argmax(-1)
    rl = torch.tensor([x if x is not None else float("nan") for x in ref["logprob"][1:]])
    rt = torch.tensor([x if x is not None else -1 for x in ref["top1"][1:]])
    m = ~torch.isnan(rl)
    d = (ours[m] - rl[m]).abs()
    res = {"time": now(), "rep": a.rep, "device": dev, "shard_scales": a.shard_scales, "positions": int(m.sum()),
           "top1_agree": round((top1[m] == rt[m]).float().mean().item(), 4),
           "mean_abs_dlogprob": round(d.mean().item(), 4), "median_abs_dlogprob": round(d.median().item(), 4),
           "p99_abs_dlogprob": round(d.quantile(0.99).item(), 4), "max_abs_dlogprob": round(d.max().item(), 4),
           "mean_logprob_ours": round(ours[m].mean().item(), 4), "mean_logprob_vllm": round(rl[m].mean().item(), 4),
           "seconds": round(time.time() - t0, 1)}
    torch.save({"logprob": ours, "top1": top1}, os.path.join(a.save_dir, f"logits-{a.rep}-{dev}-{a.shard_scales}.pt"))
    log("check-logits:", json.dumps(res))
    with open(a.check_logits, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


class Phase:
    name = "idle"


def is_oom(ex):
    return isinstance(ex, torch.cuda.OutOfMemoryError) or "out of memory" in str(ex).lower()


def train_step(model, opt, params, seq, ce_chunk, host_bufs):
    dev = model.embed.device
    x = seq[:-1].to(dev, non_blocking=True)
    y = seq[1:].to(dev, non_blocking=True)
    Phase.name = "forward"
    h = model.hidden(x, ckpt=True, host_bufs=host_bufs)[0]
    Phase.name = "loss"
    loss = chunked_ce(h, model.lm_head, y, ce_chunk)
    Phase.name = "backward"
    loss.backward()
    del h
    Phase.name = "optimizer"
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    Phase.name = "done"
    return loss.detach(), gn.detach()


def make_host_bufs(model, T, budget_gb):
    """Host buffers for the first N layer inputs (N limited by budget_gb), pinned EXACTLY: one pageable allocation
    registered with cudaHostRegister. (torch's pin_memory=True goes through the caching host allocator, which rounds
    every block up to a power of two: 1.34 GB -> 2 GB at 128k, 0.67 -> 1 GB at 64k. On 27-09-2026 that turned a 30 GB
    budget into ~45 GB and the host-RAM watchdog stopped both offload attempts.)"""
    if budget_gb <= 0:
        return None, 0, None
    H = model.c["hidden_size"]
    per = T * H
    n = min(len(model.layers), int(budget_gb * GIB // (per * 2)))
    if n <= 0:
        return None, 0, None
    flat = torch.empty(n * per, dtype=torch.bfloat16)
    flat.fill_(0)                                            # fault the pages in now (measured, not lazily)
    pinned = None
    try:
        rc = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), flat.numel() * 2, 0)
        pinned = flat if int(rc) == 0 else None
        if pinned is None:
            log(f"  cudaHostRegister rc={rc}: offload buffers stay pageable (slower copies)")
    except Exception as ex:  # noqa: BLE001
        log("  cudaHostRegister failed:", repr(ex)[:200], "- pageable offload buffers")
    bufs = [flat[i * per:(i + 1) * per].view(1, T, H) for i in range(n)]
    bufs += [None] * (len(model.layers) - n)
    return bufs, n, (flat, pinned)


def free_host_bufs(handle):
    if handle is None:
        return
    flat, pinned = handle
    if pinned is not None:
        try:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(pinned.data_ptr())
        except Exception:  # noqa: BLE001
            pass


def gpu_mem():
    return {"alloc_gb": round(torch.cuda.max_memory_allocated() / GIB, 2),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / GIB, 2)}


def run_length(model, opt, params, ids, T, steps, ce_chunk, setting, hostmon, emit, deadline):
    """3 full steps (the first is the warm-up, not timed). Returns a result dict."""
    Impl.mlp_chunk = setting.get("mlp_chunk", 0)
    host_bufs, n_off, host_handle = make_host_bufs(model, T, setting.get("offload_gb", 0))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hostmon.reset()
    base_alloc = torch.cuda.memory_allocated() / GIB
    r = {"T": T, "setting": setting, "offloaded_layers": n_off, "status": "ok", "steps": [],
         "static_alloc_gb": round(base_alloc, 2)}
    try:
        for s in range(steps):
            if time.time() > deadline:
                r["status"] = f"stopped: deadline after {s} steps"
                break
            seq = seq_for(ids, T, s)
            torch.cuda.synchronize()
            t0 = time.time()
            loss, gn = train_step(model, opt, params, seq, ce_chunk, host_bufs)
            torch.cuda.synchronize()
            dt = time.time() - t0
            st = {"step": s, "s": round(dt, 3), "loss": round(loss.item(), 4), "grad_norm": round(gn.item(), 5),
                  **gpu_mem()}
            r["steps"].append(st)
            log(f"  T={T} step {s}: {dt:.2f}s loss {st['loss']} gnorm {st['grad_norm']} "
                f"peak alloc {st['alloc_gb']} / reserved {st['reserved_gb']} GiB")
            if not (math.isfinite(st["loss"]) and math.isfinite(st["grad_norm"])):
                r["status"] = "non-finite"
                break
    except Exception as ex:  # noqa: BLE001
        if is_oom(ex):
            r["status"] = f"OOM in {Phase.name}"
        else:
            r["status"] = f"error in {Phase.name}: {repr(ex)[:300]}"
        log(f"  T={T} {r['status']}")
        opt.zero_grad(set_to_none=True)
    finally:
        del host_bufs
        free_host_bufs(host_handle)
        host_handle = None
    timed = [x["s"] for x in r["steps"][1:]]
    r.update(gpu_mem())
    r["host_min_avail_gb"] = round(hostmon.min_avail, 1)
    r["host_rss_peak_gb"] = round(max(hostmon.max_rss, 0), 1)
    r["host_hwm_gb"] = round(host_mem()["hwm_gb"], 1)
    if timed:
        r["step_s"] = round(sum(timed) / len(timed), 3)
        r["tok_s"] = round(T / r["step_s"], 1)
    if r["steps"]:
        r["loss"] = r["steps"][-1]["loss"]
        r["grad_norm"] = r["steps"][-1]["grad_norm"]
    gc.collect()
    torch.cuda.empty_cache()
    emit(r)
    return r


def smoke(model, opt, params, ids, ref, rep, ce_chunk, emit, save_dir):
    """1k smoke: loss finite, LoRA grads finite and non-zero (B at step 1, A and B at step 2), params change,
    correctness vs the served model's prompt log-probs, save/reload equivalence of the LoRA state."""
    dev = model.embed.device
    out = {"kind": "smoke", "rep": rep}
    # ---- correctness (base model: LoRA B = 0) on the fixed 512-token prompt
    prompt = ref_prompt(ids).to(dev)
    with torch.no_grad():
        h = model.hidden(prompt)[0]
        lp = torch.log_softmax(model.logits(h).float(), dim=-1)          # (512, V): row t predicts token t+1
    ours_lp = lp[:-1].gather(1, prompt[1:, None]).squeeze(1).cpu()
    ours_top1 = lp[:-1].argmax(-1).cpu()
    out["ours_logprob_head"] = [round(x, 4) for x in ours_lp[:8].tolist()]
    torch.save({"logprob": ours_lp, "top1": ours_top1}, os.path.join(save_dir, f"logits-{rep}.pt"))
    if ref is not None:
        assert ref["prompt"] == prompt.tolist(), "reference prompt differs"
        rl = torch.tensor([x if x is not None else float("nan") for x in ref["logprob"][1:]])
        rt = torch.tensor([x if x is not None else -1 for x in ref["top1"][1:]])
        m = ~torch.isnan(rl)
        d = (ours_lp[m] - rl[m]).abs()
        out["correctness"] = {"positions": int(m.sum()), "top1_agree": round((ours_top1[m] == rt[m]).float().mean().item(), 4),
                              "mean_abs_dlogprob": round(d.mean().item(), 4),
                              "median_abs_dlogprob": round(d.median().item(), 4),
                              "p99_abs_dlogprob": round(d.quantile(0.99).item(), 4),
                              "max_abs_dlogprob": round(d.max().item(), 4),
                              "mean_logprob_ours": round(ours_lp[m].mean().item(), 4),
                              "mean_logprob_vllm": round(rl[m].mean().item(), 4)}
        log("correctness vs served model:", out["correctness"])
    del h, lp
    # ---- two training steps at 1k
    T = 1024
    before = {id(p): p.detach().clone() for p in params}
    checks = []
    for s in range(2):
        seq = seq_for(ids, T, 100 + s).to(dev)
        x, y = seq[:-1], seq[1:]
        hh = model.hidden(x, ckpt=True)[0]
        loss = chunked_ce(hh, model.lm_head, y, 512)
        loss.backward()
        names = [n for n, _ in model.named_parameters() if "lora_" in n]
        grads = {n: p.grad for n, p in model.named_parameters() if "lora_" in n}
        fin = all(g is not None and torch.isfinite(g).all().item() for g in grads.values())
        nzA = sum(1 for n in names if n.endswith("lora_A") and grads[n] is not None and grads[n].abs().sum().item() > 0)
        nzB = sum(1 for n in names if n.endswith("lora_B") and grads[n] is not None and grads[n].abs().sum().item() > 0)
        nA = sum(1 for n in names if n.endswith("lora_A"))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        changed = sum(1 for p in params if not torch.equal(p.detach(), before[id(p)]))
        checks.append({"step": s, "loss": round(loss.item(), 4), "loss_finite": math.isfinite(loss.item()),
                       "grads_finite": fin, "grad_norm": round(gn, 5), "nonzero_grad_A": f"{nzA}/{nA}",
                       "nonzero_grad_B": f"{nzB}/{nA}", "params_changed": f"{changed}/{len(params)}"})
        log("smoke step", checks[-1])
        before = {id(p): p.detach().clone() for p in params}
    out["steps"] = checks
    last = checks[-1]
    out["pass"] = bool(all(c["loss_finite"] and c["grads_finite"] for c in checks)
                       and checks[0]["nonzero_grad_B"].split("/")[0] == checks[0]["nonzero_grad_B"].split("/")[1]
                       and last["nonzero_grad_A"].split("/")[0] == last["nonzero_grad_A"].split("/")[1]
                       and last["params_changed"].split("/")[0] == last["params_changed"].split("/")[1])
    # ---- save / reload equivalence of the LoRA state
    try:
        seq = seq_for(ids, T, 200).to(dev)

        def eval_loss():
            with torch.no_grad():
                return chunked_ce(model.hidden(seq[:-1])[0], model.lm_head, seq[1:], 512).item()

        l1 = eval_loss()
        sd = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
        fp = os.path.join(save_dir, f"lora-smoke-{rep}.pt")
        torch.save(sd, fp)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data.zero_()
        l0 = eval_loss()
        loaded = torch.load(fp)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data.copy_(loaded[n])
        l2 = eval_loss()
        size_mb = os.path.getsize(fp) / 2 ** 20
        os.remove(fp)
        out["save_reload"] = {"loss_trained": l1, "loss_zeroed": l0, "loss_reloaded": l2, "identical": l1 == l2,
                              "file_mb": round(size_mb, 1), "deleted": not os.path.exists(fp)}
        log("save/reload:", out["save_reload"])
    except Exception as ex:  # noqa: BLE001
        out["save_reload"] = {"error": repr(ex)[:300]}
    out.update(gpu_mem())
    emit(out)
    return out


# =====================================================================================================================
# Main
# =====================================================================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", metavar="OUT")
    ap.add_argument("--tokens", metavar="OUT", help="tokenize the fixed public text (CNN/DailyMail test articles) -> OUT")
    ap.add_argument("--ref-logprobs", metavar="OUT")
    ap.add_argument("--token-file", default=os.path.expanduser("$HK_WORK/trainctx/corpus_ids.pt"))
    ap.add_argument("--probe-file", default=os.path.expanduser("$HK_WORK/trainctx/probe.json"))
    ap.add_argument("--ref-file", default=os.path.expanduser("$HK_WORK/trainctx/ref_logprobs.json"))
    ap.add_argument("--rep", choices=["r1", "r2"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lengths", default="")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--ce-chunk", type=int, default=4096)
    ap.add_argument("--offload-gb", type=float, default=24.0, help="pinned host budget for the offload fallback")
    ap.add_argument("--mlp-chunk", type=int, default=8192, help="MLP chunk for the chunked-MLP fallback")
    ap.add_argument("--deadline", type=float, default=0, help="unix time; no new step starts after it")
    ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/trainctx/results.jsonl"))
    ap.add_argument("--save-dir", default=os.path.expanduser("$HK_WORK/trainctx/tmp"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--shard-scales", choices=["exact", "served"], default="served",
                    help="served (default since 27-09-2026 15:15): vLLM's fused in_proj_qkv/z global-scale collapse, "
                         "i.e. the function production computes (the 27-09 ladder itself ran with exact; memory and "
                         "speed are the same); exact: each linear with its own global scale (transformers semantics)")
    ap.add_argument("--settings", default="", help="comma list of ladder settings to try, in order (default: all)")
    ap.add_argument("--check-logits", metavar="OUT", help="compare base-model prompt log-probs with --ref-file -> OUT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=32)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.tokens:
        return make_tokens(a.tokens)
    if a.ref_logprobs:
        return ref_logprobs(a.token_file, a.ref_logprobs)
    if a.probe:
        return probe(a.probe)
    if a.check_logits:
        return check_logits(a)
    assert a.rep, "--rep required"
    deadline = a.deadline or (time.time() + 10 * 3600)
    os.makedirs(a.save_dir, exist_ok=True)

    def emit(rec):
        rec = {"time": now(), "rep": a.rep, **rec}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with open(a.probe_file) as f:
        pr = json.load(f)
    import torch.nn.attention as tna
    if not pr.get("sdpa_choice"):
        raise RuntimeError("no working memory-efficient SDPA backend for head_dim 256 (probe)")
    Impl.sdpa_backends = [getattr(tna.SDPBackend, pr["sdpa_choice"])]
    Impl.conv = "cuda" if pr.get("causal_conv1d", {}).get("ok") else "torch"
    if not pr.get("fla_gdn", {}).get("ok"):
        raise RuntimeError("FLA chunk_gated_delta_rule failed the probe")
    Impl.gdn = "fla"
    _TRITON["ok"] = bool(pr.get("triton_dequant", {}).get("ok"))
    log(f"impl: sdpa {pr['sdpa_choice']}, conv {Impl.conv}, gdn {Impl.gdn}, triton dequant {_TRITON['ok']}, "
        f"alloc conf {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    hostmon = HostMonitor(min_avail_gb=3.0)
    hostmon.start()
    ids = torch.load(a.token_file)["ids"]
    ref = None
    if os.path.exists(a.ref_file):
        with open(a.ref_file) as f:
            ref = json.load(f)

    torch.cuda.reset_peak_memory_stats()
    log(f"loading {a.rep} ...", host_mem())
    model, t_load = build_model(MODEL, a.rep, "cuda", shard_scales=a.shard_scales)
    log(f"shard scales: {a.shard_scales}")
    params = model.lora_parameters()
    n_lora = sum(p.numel() for p in params)
    torch.cuda.synchronize()
    load = {"kind": "load", "load_s": round(t_load, 1), "static_alloc_gb": round(torch.cuda.memory_allocated() / GIB, 2),
            "lora_params_m": round(n_lora / 1e6, 2), "lora_modules": sum(1 for n in dict(model.named_parameters()) if n.endswith("lora_A")),
            "host_hwm_gb": round(host_mem()["hwm_gb"], 1), **gpu_mem()}
    log("loaded:", load)
    emit(load)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)

    if a.smoke:
        smoke(model, opt, params, ids, ref, a.rep, 512, emit, a.save_dir)
        # fresh adapter/optimizer for the ladder
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.lora_A is not None:
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5)); nn.init.zeros_(m.lora_B)
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)

    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    settings = [{"name": "gpu-ckpt"},
                {"name": "offload", "offload_gb": a.offload_gb},
                {"name": "offload+mlp-chunk", "offload_gb": a.offload_gb, "mlp_chunk": a.mlp_chunk}]
    if a.settings:
        want = a.settings.split(",")
        settings = [x for x in settings if x["name"] in want]
    si = 0
    last_step_s, last_T = None, None
    for T in lengths:
        if last_step_s is not None:
            est = last_step_s * (T / last_T) * 2.3 * (a.steps + 1)
            if time.time() + est > deadline:
                emit({"kind": "ladder", "T": T, "status": f"skipped: estimated {est / 60:.0f} min exceeds the deadline"})
                log(f"T={T}: skipped (estimated {est / 60:.0f} min > deadline)")
                break
        reached = False
        while si < len(settings):
            log(f"=== {a.rep} T={T} setting {settings[si]['name']}")
            r = run_length(model, opt, params, ids, T, a.steps, a.ce_chunk, settings[si], hostmon,
                           lambda rec: emit({"kind": "ladder", **rec}), deadline)
            if r["status"] == "ok" and len(r["steps"]) == a.steps:
                reached = True
                last_step_s, last_T = r.get("step_s", r["steps"][-1]["s"]), T
                break
            if r["status"].startswith("OOM"):
                si += 1
                continue
            break                                    # non-OOM failure / deadline: no fallback
        if not reached:
            log(f"{a.rep}: ladder stops at T={T}")
            break
    log("done")


if __name__ == "__main__":
    main()
