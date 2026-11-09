# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K2 v0 (fork tilelang.contrib.sm120_nvfp4.decode): correctness on a real layer vs exact reference and Marlin,
# plus timing with rotated weight copies (so L2 cannot flatter the memory-bound numbers).
import json, sys, types, torch
from safetensors import safe_open
from tilelang.contrib.sm120_nvfp4.decode import decode_linear, decode_linear_v1
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
CFG = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
TIME = "--time" in sys.argv


def dq(wp, ws, gs):
    u = wp.view(torch.uint8); w = torch.empty(u.shape[0], u.shape[1] * 2, device="cuda")
    w[:, 0::2] = LUT[(u & 0xF).long()]; w[:, 1::2] = LUT[(u >> 4).long()]
    return w * ws.float().repeat_interleave(16, 1) / gs


def marlin_layer(wp, ws, gs):
    l = types.SimpleNamespace(output_size_per_partition=wp.shape[0], input_size_per_partition=wp.shape[1] * 2, params_dtype=torch.bfloat16,
                              bias=None, weight=wp.clone(), weight_scale=ws.clone(), weight_global_scale=(1.0 / gs).clone())
    prepare_fp4_layer_for_marlin(l)
    return l


with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    pre = "model.language_model.layers.3.mlp.gate_proj."
    wp = f.get_tensor(pre + "weight_packed"); ws = f.get_tensor(pre + "weight_scale"); gs = f.get_tensor(pre + "weight_global_scale").float()
N, K = wp.shape[0], wp.shape[1] * 2
W = dq(wp, ws, gs); ml = marlin_layer(wp, ws, gs)
alpha = (1.0 / gs).expand(N).contiguous()
for M in (1, 5, 16, 33, 64):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ W.T
    y = decode_linear(x, wp.view(torch.int8), ws, alpha, **CFG).float()
    y1 = decode_linear_v1(x, wp.view(torch.int8), ws, alpha).float()
    y1b = decode_linear_v1(x, wp.view(torch.int8), ws, alpha).float()
    ym = apply_fp4_marlin_linear(x, ml.weight, ml.weight_scale, ml.weight_global_scale, ml.workspace, N, K).float()
    rel = lambda a: round(((a - ref).norm() / ref.norm()).item(), 5)
    print(json.dumps({"M": M, "k2v0_rel_err": rel(y), "k2v1_rel_err": rel(y1), "marlin_rel_err": rel(ym),
                      "k2v1_deterministic": bool(torch.equal(y1, y1b))}), flush=True)

if TIME:
    SH = {"qkvz": (16384, 5120), "o_proj": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
    for name, (N, K) in SH.items():
        copies = max(2, int(512e6 // (N * K // 2)) + 1)
        g = torch.Generator(device="cuda").manual_seed(0)
        Ws = [(torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device="cuda", generator=g),
               (torch.rand(N, K // 16, device="cuda", generator=g) + 0.5).to(torch.float8_e4m3fn)) for _ in range(copies)]
        Ms = [marlin_layer(w.view(torch.uint8), s, torch.tensor([2688.0], device="cuda")) for w, s in Ws]
        al = torch.full((N,), 1 / 2688.0, device="cuda")
        nbytes = N * K // 2 + N * K // 16
        for M in (1, 8, 16, 32, 64):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            def run_k2():
                for w, s in Ws: decode_linear(x, w, s, al, **CFG)
            def run_k2v1():
                for w, s in Ws: decode_linear_v1(x, w, s, al)
            def run_m():
                for l in Ms: apply_fp4_marlin_linear(x, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K)
            res = {}
            for nm, fn in (("k2", run_k2), ("k2v1", run_k2v1), ("marlin", run_m)):
                for _ in range(3): fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for _ in range(5): fn()
                s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                best = 1e9
                for _ in range(3):
                    s_.record(); gr.replay(); e_.record(); torch.cuda.synchronize(); best = min(best, s_.elapsed_time(e_) / (5 * copies))
                res[nm] = best * 1e3
            pr = lambda t: round(100 * nbytes / (t * 1e-6) / 1631e9, 1)
            print(json.dumps({"shape": name, "M": M, "k2v0_us": round(res["k2"], 1), "k2v1_us": round(res["k2v1"], 1),
                              "marlin_us": round(res["marlin"], 1), "roof_pct_v0": pr(res["k2"]), "roof_pct_v1": pr(res["k2v1"]),
                              "roof_pct_marlin": pr(res["marlin"]), "v1_vs_marlin": round(res["marlin"] / res["k2v1"], 3)}), flush=True)
        del Ws, Ms; torch.cuda.empty_cache()
