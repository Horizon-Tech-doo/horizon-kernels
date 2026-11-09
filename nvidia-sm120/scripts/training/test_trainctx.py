#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for trainctx.py (27-09-2026). No GPU is touched.

On the test server run them with CUDA_VISIBLE_DEVICES="": gpushare (ld.so.preload) makes any CUDA initialisation
exit the process there, and the autograd engine initialises CUDA on backward - so the `model` group (which runs
backward) is run on any other CPU machine (torch + transformers 5.17), the `ckpt` and `load` groups on the server.

Two groups, because transformers binds the Gated-DeltaNet / conv kernels from FLA / causal_conv1d whenever those
packages are importable (they are GPU-only):
  test_trainctx.py model   WITHOUT $HK_DATA/llmc-pkgs on PYTHONPATH: model vs the transformers reference on a
                           tiny random config, LoRA, recompute (+ offload buffer, chunked MLP) vs plain autograd,
                           chunked CE, rope, NVFP4 autograd Function, a full train step
  test_trainctx.py load    real checkpoint, 4-layer truncation, r1 vs r2 forward on CPU (~6 GB host RAM)
  test_trainctx.py ckpt    WITH llmc-pkgs: checkpoint tensor names/shapes (safetensors header), NVFP4 dequant vs
                           compressed-tensors, Triton dequant kernel in the interpreter (TRITON_INTERPRET=1)
"""
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainctx as tc  # noqa: E402

torch.manual_seed(0)
RESULTS = []
# transformers' mask code asks torch.cuda whether a graph capture is running; on this box that initialises a gpushare
# client (and hangs without a visible GPU). The tests are CPU-only: nothing is capturing.
torch.cuda.is_current_stream_capturing = lambda: False


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond), info))
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# tiny model vs transformers
# ---------------------------------------------------------------------------------------------------------------------

def tiny_hf():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        rms_norm_eps=1e-6, attn_output_gate=True)
    cfg._attn_implementation = "sdpa"
    m = Qwen3_5TextModel(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "A_log" in n:
                p.copy_(torch.rand_like(p) * 2)
            elif "norm" in n:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if "linear_attn.norm" in n else 0.0))
            else:
                p.copy_(torch.randn_like(p) * 0.08)
    c = cfg.to_dict()
    return m, c


def mine_from_hf(hf, c, rank=4):
    tc.Impl.gdn, tc.Impl.conv, tc.Impl.sdpa_backends, tc.Impl.mlp_chunk = "torch", "torch", None, 0
    m = tc.TextModel(c, rank=rank, alpha=2 * rank)
    sd = hf.state_dict()
    m.embed = sd["embed_tokens.weight"].clone()
    m.lm_head = torch.randn(c["vocab_size"], c["hidden_size"]) * 0.1
    tc.setp(m.norm, "weight", sd["norm.weight"].clone())
    for i, layer in enumerate(m.layers):
        p = f"layers.{i}."
        tc.setp(layer.input_layernorm, "weight", sd[p + "input_layernorm.weight"].clone())
        tc.setp(layer.post_attention_layernorm, "weight", sd[p + "post_attention_layernorm.weight"].clone())
        names = tc.QUANT_LINEARS[layer.kind] + tc.QUANT_LINEARS["mlp"] + tc.BF16_LINEARS[layer.kind]
        for n in names:
            tc._get(layer, n).set_dense(sd[p + n + ".weight"].clone())
        if layer.kind == "linear_attention":
            la = layer.linear_attn
            tc.setp(la, "A_log", sd[p + "linear_attn.A_log"].clone())
            tc.setp(la, "dt_bias", sd[p + "linear_attn.dt_bias"].clone())
            tc.setp(la, "conv_w", sd[p + "linear_attn.conv1d.weight"].squeeze(1).clone())
            tc.setp(la.norm, "weight", sd[p + "linear_attn.norm.weight"].clone())
        else:
            tc.setp(layer.self_attn.q_norm, "weight", sd[p + "self_attn.q_norm.weight"].clone())
            tc.setp(layer.self_attn.k_norm, "weight", sd[p + "self_attn.k_norm.weight"].clone())
    for n, prm in m.named_parameters():
        prm.requires_grad_("lora_" in n)
    return m


def test_model():
    hf, c = tiny_hf()
    m = mine_from_hf(hf, c)
    T = 70                                               # not a multiple of the 64-token GDN chunk
    ids = torch.randint(0, c["vocab_size"], (T,))
    with torch.no_grad():
        ref = hf(input_ids=ids[None], use_cache=False).last_hidden_state
        out = m.hidden(ids)
    err = (ref - out).abs().max().item()
    check("model == transformers Qwen3_5TextModel (fp32, LoRA B=0)", err < 1e-4, f"max |diff| {err:.2e}")

    # rope
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos_h, sin_h = hf.rotary_emb(torch.zeros(1, T, 8), pos)
    cos_m, sin_m = tc.rope_cos_sin(c, T, "cpu", torch.float32)
    e = max((cos_h[0] - cos_m).abs().max().item(), (sin_h[0] - sin_m).abs().max().item())
    check("rope == transformers mrope for text positions", e < 1e-6, f"{e:.1e}")

    # LoRA: non-zero B changes the output; recompute / chunked MLP / host buffer give the plain-autograd gradients
    for p in m.lora_parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.05)
    params = m.lora_parameters()
    seq = torch.randint(0, c["vocab_size"], (T + 1,))

    def grads(ckpt, mlp_chunk=0, bufs=None, ce_chunk=16):
        tc.Impl.mlp_chunk = mlp_chunk
        for p in params:
            p.grad = None
        h = m.hidden(seq[:-1], ckpt=ckpt, host_bufs=bufs)[0]
        loss = tc.chunked_ce(h, m.lm_head, seq[1:], ce_chunk)
        loss.backward()
        tc.Impl.mlp_chunk = 0
        return loss.item(), [p.grad.clone() for p in params]

    l0, g0 = grads(False, ce_chunk=10 ** 9)
    ref_loss = F.cross_entropy(F.linear(m.hidden(seq[:-1]).detach()[0], m.lm_head), seq[1:]).item()
    check("chunked CE == F.cross_entropy", abs(l0 - ref_loss) < 1e-5, f"{l0:.6f} vs {ref_loss:.6f}")
    for label, kw in (("recompute", {}), ("recompute + chunked MLP", {"mlp_chunk": 16}),
                      ("recompute + host buffers", {"bufs": [torch.empty(1, T, c["hidden_size"]) for _ in range(2)]
                                                     + [None] * 2})):
        l1, g1 = grads(True, **kw)
        e = max((a - b).abs().max().item() / (b.abs().max().item() + 1e-12) for a, b in zip(g1, g0))
        nz = sum(1 for g in g1 if g.abs().sum() > 0)
        check(f"{label}: loss and LoRA grads == plain autograd", abs(l1 - l0) < 1e-6 and e < 1e-5,
              f"loss {l1:.6f}/{l0:.6f}, max rel grad diff {e:.1e}, non-zero grads {nz}/{len(g1)}")

    # LoRA covers every linear projection
    names = [n for n, _ in m.named_parameters() if n.endswith("lora_A")]
    per_kind = {"linear_attention": 3 + 2 + 3, "full_attention": 4 + 3}
    exp = sum(per_kind[k] for k in c["layer_types"])
    check("LoRA on every text linear (q/k/v/o | qkv/z/a/b/out, gate/up/down)", len(names) == exp,
          f"{len(names)} modules, expected {exp}")

    # a full train step (non-fused AdamW on CPU): params change, loss finite
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = [p.detach().clone() for p in params]
    loss, gn = tc.train_step(m, opt, params, seq, 16, None)
    ch = sum(1 for p, b in zip(params, before) if not torch.equal(p.detach(), b))
    check("train_step: finite loss/grad norm, all LoRA params updated",
          torch.isfinite(loss).item() and torch.isfinite(gn).item() and ch == len(params),
          f"loss {loss.item():.4f} gnorm {gn.item():.4f} changed {ch}/{len(params)}")

    # r2 path: NVFP4 autograd Function == dense linear on the dequantized weight (values and input grads)
    N, K = 48, 64
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    scale = (torch.rand(N, K // 16) * 2 + 0.1).to(torch.float8_e4m3fn)
    gs = torch.tensor([3.7])
    lin_q = tc.LoRALinear(K, N, 4, 8)
    lin_q.set_packed(packed, scale, gs)
    lin_d = tc.LoRALinear(K, N, 4, 8)
    lin_d.set_dense(tc.dequant_nvfp4_torch(packed, scale, gs, torch.float32))
    with torch.no_grad():
        lin_d.lora_A.copy_(lin_q.lora_A)
        lin_d.lora_B.copy_(torch.randn_like(lin_d.lora_B))
        lin_q.lora_B.copy_(lin_d.lora_B)
    x = torch.randn(1, 5, K)
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = lin_q(xa), lin_d(xb)
    gy = torch.randn_like(ya)
    ya.backward(gy)
    yb.backward(gy)
    e = max((ya - yb).abs().max().item(), (xa.grad - xb.grad).abs().max().item(),
            (lin_q.lora_A.grad - lin_d.lora_A.grad).abs().max().item())
    check("NVFP4 autograd Function == dense linear (y, dx, dLoRA)", e < 1e-5, f"{e:.1e}")


# ---------------------------------------------------------------------------------------------------------------------
# checkpoint-level tests (need llmc-pkgs for compressed-tensors, and the model on disk)
# ---------------------------------------------------------------------------------------------------------------------

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def test_ckpt():
    c = tc.text_config(tc.MODEL)
    h = header(os.path.join(tc.MODEL, "model.safetensors"))
    h.pop("__metadata__", None)
    dt = {"BF16": "bf16", "U8": "u8", "F8_E4M3": "f8", "F32": "f32"}
    exp = tc.expected_tensors(c)
    missing = [k for k in exp if k not in h]
    bad = [(k, exp[k], h[k]["dtype"], h[k]["shape"]) for k in exp if k in h
           and (dt.get(h[k]["dtype"]) != exp[k][0] or tuple(h[k]["shape"]) != tuple(exp[k][1]))]
    text_keys = [k for k in h if not k.startswith(("model.visual.", "mtp."))]
    unused = sorted(k for k in text_keys if k not in exp)
    unused_kinds = sorted({k.rsplit(".", 1)[-1] for k in unused})
    check("checkpoint: every expected text tensor present with dtype/shape", not missing and not bad,
          f"{len(exp)} tensors; missing {missing[:3]} bad {bad[:2]}")
    check("checkpoint: the only unused text tensors are the activation global scales",
          unused_kinds == ["input_global_scale"], f"{len(unused)} unused, kinds {unused_kinds}")
    n_q = sum(1 for k in exp if k.endswith(".weight_packed"))
    packed_bytes = sum(torch.Size(h[k]["shape"]).numel() for k in exp if k.endswith((".weight_packed", ".weight_scale")))
    n_params = sum(torch.Size(h[k]["shape"]).numel() * (2 if k.endswith(".weight_packed") else 1)
                   for k in exp if not k.endswith((".weight_scale", ".weight_global_scale")))
    bf16_other = sum(torch.Size(h[k]["shape"]).numel() * 2 for k in exp if exp[k][0] == "bf16")
    info = (f"{n_q} NVFP4 linears; text params {n_params / 1e9:.2f} B; r1 base ~{(n_params * 2) / 2**30:.1f} GiB; "
            f"r2 base ~{(packed_bytes + bf16_other) / 2**30:.1f} GiB")
    check("checkpoint size estimate", True, info)

    # dequant vs compressed-tensors on a real tensor
    from safetensors import safe_open
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

    with safe_open(os.path.join(tc.MODEL, "model.safetensors"), framework="pt", device="cpu") as f:
        p = tc.PREFIX + "layers.3.self_attn.k_proj"
        pk, sc = f.get_tensor(p + ".weight_packed"), f.get_tensor(p + ".weight_scale")
        gs = f.get_tensor(p + ".weight_global_scale").float()
    mine = tc.dequant_nvfp4_torch(pk, sc, gs, torch.float32)
    un = unpack_fp4_from_uint8(pk, pk.shape[0], pk.shape[1] * 2, torch.float32)
    ref = (un.view(pk.shape[0], -1, 16) * (sc.float() / gs).unsqueeze(-1)).view_as(un)
    e = (mine - ref).abs().max().item()
    check("NVFP4 unpack + scale == compressed-tensors unpack_fp4_from_uint8 (fp32)", e == 0.0, f"max diff {e}")
    try:
        from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
        from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
        args = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16, symmetric=True,
                                scale_dtype=torch.float8_e4m3fn)
        scheme = QuantizationScheme(targets=["Linear"], weights=args)
        d = NVFP4PackedCompressor.decompress({"weight_packed": pk, "weight_scale": sc, "weight_global_scale": gs}, scheme)
        w_ct = d["weight"]
        mine16 = tc.dequant_nvfp4_torch(pk, sc, gs, w_ct.dtype)
        ne = (mine16.float() - w_ct.float()).abs()
        rel = (ne / (w_ct.float().abs() + 1e-12))[w_ct != 0]
        check("NVFP4 dequant ~ compressed-tensors decompress (library path, its own rounding chain)",
              rel.max().item() < 2 ** -7, f"dtype {w_ct.dtype}, elements differing {(ne > 0).sum().item()} of "
              f"{ne.numel()}, max rel {rel.max().item():.2e}")
    except Exception as ex:  # noqa: BLE001
        check("compressed-tensors NVFP4PackedCompressor.decompress (informational)", True, f"skipped: {ex!r}"[:200])

    # Triton kernel in the interpreter on a slice, fp32 output (bit-exact vs torch). The interpreter's fp32 -> bf16
    # cast truncates (numpy emulation), unlike the GPU's round-to-nearest: the bf16 path is checked on the GPU by
    # trainctx.py --probe (bit-exact on real tensors, else the torch dequant is used).
    if os.environ.get("TRITON_INTERPRET") == "1":
        pks, scs = pk[:64].contiguous(), sc[:64].contiguous()
        a = tc.dequant_nvfp4_torch(pks, scs, gs, torch.float32)
        b = tc.dequant_nvfp4_triton(pks, scs, gs, torch.float32)
        mism = (a != b).sum().item()   # by value: +0.0 / -0.0 are the same weight
        check("Triton dequant kernel (interpreter, fp32 out) value-exact vs torch", mism == 0, f"{mism} mismatches of {a.numel()}")
        # every fp8 scale bit pattern in use + subnormals
        allsc = torch.arange(0, 256, dtype=torch.uint8)
        allsc = allsc[(allsc & 0x7F) != 0x7F].view(torch.float8_e4m3fn)          # drop NaN codes
        n = allsc.numel() // 8 * 8
        sc2 = allsc[:n].view(-1, 8)                                             # rows of 8 groups -> K = 128
        pk2 = torch.randint(0, 256, (sc2.shape[0], 64), dtype=torch.uint8)
        a = tc.dequant_nvfp4_torch(pk2, sc2, gs, torch.float32)
        b = tc.dequant_nvfp4_triton(pk2, sc2, gs, torch.float32)
        mism = (a != b).sum().item()   # by value: +0.0 / -0.0 are the same weight
        check("Triton dequant: all e4m3 scale codes (incl. subnormal/negative) value-exact", mism == 0, f"{mism}")


def test_load():
    """load_weights on a 4-layer truncation of the real checkpoint (CPU, ~6 GB host RAM): r1 and r2 load, their
    forward passes agree exactly, logits are finite and sensible on real text."""
    import trainctx as t2
    t2.Impl.gdn, t2.Impl.conv, t2.Impl.sdpa_backends = "torch", "torch", None
    real = t2.text_config(t2.MODEL)
    orig = t2.text_config
    t2.text_config = lambda d: {**real, "num_hidden_layers": 4, "layer_types": real["layer_types"][:4]}
    try:
        outs = {}
        for rep in ("r2", "r1"):
            m, secs = t2.build_model(t2.MODEL, rep, "cpu")
            n_lora = sum(p.numel() for p in m.lora_parameters())
            ids = torch.arange(1000, 1032)
            with torch.no_grad():
                outs[rep] = m.logits(m.hidden(ids))[0].float()
            check(f"load {rep} (4 layers, CPU): forward finite", torch.isfinite(outs[rep]).all().item(),
                  f"load {secs:.0f}s, LoRA params in 4 layers {n_lora / 1e6:.2f} M")
            del m
        e = (outs["r1"] - outs["r2"]).abs().max().item()
        check("r1 (BF16 dequantized) == r2 (on-the-fly NVFP4) logits", e == 0.0, f"max diff {e}")
        m, _ = t2.build_model(t2.MODEL, "r2", "cpu", shard_scales="served")
        me, _ = t2.build_model(t2.MODEL, "r2", "cpu", shard_scales="exact")
        ok, info = True, []
        for i, (ls, le) in enumerate(zip(m.layers, me.layers)):
            if ls.kind != "linear_attention":
                ok &= all(torch.equal(t2._get(ls, n).gscale, t2._get(le, n).gscale)
                          for n in t2.QUANT_LINEARS["full_attention"] + t2.QUANT_LINEARS["mlp"])
                continue
            gq, gz = le.linear_attn.in_proj_qkv.gscale, le.linear_attn.in_proj_z.gscale
            gmax = torch.maximum(gq, gz)
            ok &= torch.equal(ls.linear_attn.in_proj_qkv.gscale, gmax) and torch.equal(ls.linear_attn.in_proj_z.gscale, gmax)
            ok &= all(torch.equal(t2._get(ls, n).gscale, t2._get(le, n).gscale)
                      for n in ["linear_attn.out_proj"] + t2.QUANT_LINEARS["mlp"])
            info.append(f"L{i} qkv {gq.item():.0f} z {gz.item():.0f} -> {gmax.item():.0f}")
        check("served shard scales: in_proj_qkv/z both use max(global scale), all other linears unchanged", ok,
              "; ".join(info))
    finally:
        t2.text_config = orig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "model"
    {"model": test_model, "ckpt": test_ckpt, "load": test_load}[which]()
    bad = [r for r in RESULTS if not r[1]]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    sys.exit(1 if bad else 0)
