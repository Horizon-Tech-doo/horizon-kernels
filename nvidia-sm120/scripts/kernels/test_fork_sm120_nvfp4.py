# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Test horizon-tilelang's tilelang.contrib.sm120_nvfp4 (dynamic M, passes=1|2) on a real Qwen3.8 layer.
import json, time, types, torch
from safetensors import safe_open
from tilelang.contrib import sm120_nvfp4 as nv
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def dequant_weight(wp, ws, wgs):
    u = wp.view(torch.uint8)
    w = torch.empty(u.shape[0], u.shape[1] * 2, device="cuda")
    w[:, 0::2] = LUT[(u & 0xF).long()]; w[:, 1::2] = LUT[(u >> 4).long()]
    return w * ws.float().repeat_interleave(16, 1) / wgs


def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1e3


with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    for layer in ("layers.3.mlp.gate_proj", "layers.3.mlp.down_proj"):
        pre = f"model.language_model.{layer}."
        wp = f.get_tensor(pre + "weight_packed"); ws = f.get_tensor(pre + "weight_scale"); wgs = f.get_tensor(pre + "weight_global_scale").float()
        N, K = wp.shape[0], wp.shape[1] * 2
        w = nv.prepare_weight(wp, ws, wgs)
        Wdq = dequant_weight(wp, ws, wgs)
        ml = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None,
                                   weight=wp.clone(), weight_scale=ws.clone(), weight_global_scale=(1.0 / wgs).clone())
        prepare_fp4_layer_for_marlin(ml)
        for M in (300, 1000, 4097, 512, 2048, 8192):
            x = torch.randn(M, K, device="cuda"); x[:, torch.randint(0, K, (8,))] *= 20; x = x.to(torch.bfloat16)
            ref = x.float() @ Wdq.T
            r = {"layer": layer, "M": M}
            for passes in (1, 2):
                t0 = time.time(); y = nv.linear(x, w, passes=passes); torch.cuda.synchronize(); first = time.time() - t0
                r[f"p{passes}_rel_err"] = round(((y.float() - ref).norm() / ref.norm()).item(), 5)
                r[f"p{passes}_first_call_s"] = round(first, 2)
                if M in (512, 2048, 8192):
                    r[f"p{passes}_us"] = round(bench(lambda: nv.linear(x, w, passes=passes)), 1)
            if M in (512, 2048, 8192):
                r["marlin_us"] = round(bench(lambda: apply_fp4_marlin_linear(x, ml.weight, ml.weight_scale, ml.weight_global_scale, ml.workspace, N, K)), 1)
                r["p2_speedup"] = round(r["marlin_us"] / r["p2_us"], 2); r["p1_speedup"] = round(r["marlin_us"] / r["p1_us"], 2)
            print(json.dumps(r), flush=True)
