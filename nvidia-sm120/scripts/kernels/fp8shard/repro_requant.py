# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU reproduction of vLLM 0.29's ModelOpt FP8 per-tensor weight loading for fused layers
# (MergedColumnParallelLinear in_proj_qkvz = [q, k, v | z], QKVParallelLinear qkv_proj = [q, k, v]):
#   loader: the in_proj_qkv scale is copied into slots 0-2, in_proj_z into slot 3 (linear.py weight_loader_v2),
#   rows are contiguous [q | k | v | z] (qwen3_5.py hf_to_vllm_mapper, qwen_gdn_linear_attn.create_qkvz_proj);
#   process_weights_after_loading: requantize_with_max_scale (w8a8_utils.py): per shard
#     w_dq = w_fp8.to(fp16) * s_shard ; w_new = fp8(clamp(w_dq / s_max)) ; effective weight = w_new * s_max.
# Compares the effective weight vLLM computes with against the true per-shard dequantised weight.
#   python repro_requant.py <layers.safetensors>
import sys
import torch
from safetensors import safe_open

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def vllm_requant(w_fp8, scales, widths):
    """Exact emulation of vllm/model_executor/layers/quantization/utils/w8a8_utils.requantize_with_max_scale."""
    s_max = scales.max()
    w = w_fp8.clone()
    start = 0
    for i, n in enumerate(widths):
        dq = w_fp8[start:start + n].to(torch.float16) * scales[i]            # per_tensor_dequantize (fp16!)
        q = (dq.float() / s_max).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)   # scaled_fp8_quant (static)
        w[start:start + n] = q
        start += n
    return w, s_max


def report(name, w_fp8, scales, widths, labels):
    w_new, s_max = vllm_requant(w_fp8, scales, widths)
    eff = w_new.float() * s_max
    true = torch.cat([w_fp8[a:a + n].float() * scales[i] for i, (a, n) in
                      enumerate(zip(torch.tensor([0] + widths[:-1]).cumsum(0).tolist(), widths))])
    start = 0
    rows = []
    for i, n in enumerate(widths):
        t, e = true[start:start + n], eff[start:start + n]
        rel = ((e - t).norm() / t.norm()).item()
        ratio = ((e * t).sum() / (t * t).sum()).item()            # least-squares gain of effective vs true
        zeroed = ((e == 0) & (t != 0)).float().mean().item()
        sat = (w_fp8[start:start + n].float().abs() * scales[i] / s_max >= FP8_MAX).float().mean().item()
        rows.append((labels[i], scales[i].item(), (scales[i] / s_max).item(), rel, ratio, zeroed, sat))
        start += n
    print(f"{name}: s_max={s_max.item():.6g}")
    for lab, s, r, rel, ratio, z, sat in rows:
        print(f"  {lab:3s} scale {s:.6g} ratio {r:.3f} | rel.err {rel:.4%} gain {ratio:.5f} "
              f"new-zeros {z:.3%} saturated {sat:.3%}")
    return rows


def main(path):
    f = safe_open(path, "pt")
    keys = set(f.keys())
    layers = sorted({int(k.split(".")[3]) for k in keys})
    out = {}
    for L in layers:
        p = f"model.language_model.layers.{L}."
        if p + "linear_attn.in_proj_qkv.weight" in keys:
            wq = f.get_tensor(p + "linear_attn.in_proj_qkv.weight")
            wz = f.get_tensor(p + "linear_attn.in_proj_z.weight")
            sq = f.get_tensor(p + "linear_attn.in_proj_qkv.weight_scale").float()
            sz = f.get_tensor(p + "linear_attn.in_proj_z.weight_scale").float()
            kd = 16 * 128
            vd = wq.shape[0] - 2 * kd
            out[L] = report(f"layer {L} GDN in_proj_qkvz", torch.cat([wq, wz]),
                            torch.stack([sq, sq, sq, sz]), [kd, kd, vd, wz.shape[0]], ["q", "k", "v", "z"])
        if p + "self_attn.q_proj.weight" in keys:
            ws = [f.get_tensor(p + f"self_attn.{n}_proj.weight") for n in "qkv"]
            ss = torch.stack([f.get_tensor(p + f"self_attn.{n}_proj.weight_scale").float() for n in "qkv"])
            out[L] = report(f"layer {L} attention qkv_proj", torch.cat(ws), ss, [w.shape[0] for w in ws],
                            ["q", "k", "v"])
    return out


if __name__ == "__main__":
    main(sys.argv[1])
