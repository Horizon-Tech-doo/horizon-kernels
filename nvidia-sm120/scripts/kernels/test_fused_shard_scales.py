# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Fused layer with different per-shard NVFP4 global scales (layer 21 in_proj_qkv|in_proj_z, 5408 vs 11584).
# Checks BOTH plugin paths against the exact per-shard reference:
#   fp4x2          nv.linear with per-column alpha (prefill path)
#   marlin + corr  Marlin with vLLM's collapsed 1/max(gs) plus the plugin's fp32 per-column correction (decode path)
# and reports what vLLM's plain max() collapse does to each shard.
import json, types, torch
from safetensors import safe_open
from tilelang.contrib import sm120_nvfp4 as nv
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def dq(wp, ws, gs):
    u = wp.view(torch.uint8); w = torch.empty(u.shape[0], u.shape[1] * 2, device="cuda")
    w[:, 0::2] = LUT[(u & 0xF).long()]; w[:, 1::2] = LUT[(u >> 4).long()]
    return w * ws.float().repeat_interleave(16, 1) / gs


pre = "model.language_model.layers.21.linear_attn."
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    parts = [(f.get_tensor(pre + p + ".weight_packed"), f.get_tensor(pre + p + ".weight_scale"),
              f.get_tensor(pre + p + ".weight_global_scale").float().reshape(1)) for p in ("in_proj_qkv", "in_proj_z")]
wp = torch.cat([p[0] for p in parts]); ws = torch.cat([p[1] for p in parts]); gs = torch.cat([p[2] for p in parts])
sizes = [p[0].shape[0] for p in parts]
N, K = wp.shape[0], wp.shape[1] * 2
ref_w = torch.cat([dq(*p) for p in parts])
x = torch.randn(1024, K, device="cuda").to(torch.bfloat16)
ref = x.float() @ ref_w.T
# prefill path
y4 = nv.linear(x, nv.prepare_weight(wp, ws, gs, shard_sizes=sizes), passes=2).float()
# decode path exactly as vllm_plugin.marlin_pw builds it
ml = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None,
                           weight=wp.clone(), weight_scale=ws.clone(), weight_global_scale=(1.0 / gs.max()).reshape(1).clone())
prepare_fp4_layer_for_marlin(ml)
corr = torch.cat([(gs.max() / g).expand(s) for g, s in zip(gs, sizes)]).float()
ym_raw = apply_fp4_marlin_linear(x, ml.weight, ml.weight_scale, ml.weight_global_scale, ml.workspace, N, K)
ym = (ym_raw.float() * corr).to(torch.bfloat16).float()
s0 = slice(0, sizes[0]); s1 = slice(sizes[0], None)
rel = lambda a, b: round(((a - b).norm() / b.norm()).item(), 5)
print(json.dumps({"global_scales": gs.tolist(), "shard_rows": sizes,
                  "fp4x2_rel_err": {"qkv": rel(y4[:, s0], ref[:, s0]), "z": rel(y4[:, s1], ref[:, s1])},
                  "marlin_plus_corr_rel_err": {"qkv": rel(ym[:, s0], ref[:, s0]), "z": rel(ym[:, s1], ref[:, s1])},
                  "vllm_collapsed_size_ratio": {"qkv": round((ym_raw.float()[:, s0].norm() / ref[:, s0].norm()).item(), 4),
                                                "z": round((ym_raw.float()[:, s1].norm() / ref[:, s1].norm()).item(), 4)},
                  "marlin_corr_vs_fp4x2_rel": rel(ym, y4)}, indent=1))
