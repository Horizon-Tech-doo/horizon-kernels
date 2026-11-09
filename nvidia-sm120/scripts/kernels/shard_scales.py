# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# How different are the per-shard NVFP4 weight_global_scales inside vLLM's fused layers?
import json, torch
from safetensors import safe_open
F = {"qkv (attn)": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
     "qkvz (gdn)": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]}
worst = {k: 0.0 for k in F}; ex = {}
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cpu") as f:
    keys = set(f.keys())
    for li in range(64):
        for fam, parts in F.items():
            names = [f"model.language_model.layers.{li}.{p}.weight_global_scale" for p in parts]
            if not all(n in keys for n in names): continue
            g = torch.stack([f.get_tensor(n).float().reshape(-1)[0] for n in names])
            # vLLM uses 1/max(g); shard i is then scaled by g_i/max(g) (1.0 = correct)
            ratio = (g / g.max()).min().item()
            if 1 - ratio > worst[fam]: worst[fam] = 1 - ratio; ex[fam] = (li, [round(v, 1) for v in g.tolist()])
print(json.dumps({fam: {"worst_shard_output_scaled_by": round(1 - worst[fam], 4), "example_layer_and_scales": ex.get(fam)} for fam in F}, indent=1))
