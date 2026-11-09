# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K2 race: every correct decode candidate vs Marlin (and the fork's v1) on all 5 layer shapes, M in 1..64.
#   k2_race.py <module1> <module2> ...        (modules follow decode_contract.py)
# Fair timing: CUDA-graph replay, >= 512 MB of rotated weight copies per shape (defeats the 128 MB L2), median of 5.
# A tiny ordinary kernel runs between consecutive calls (its own cost is measured and subtracted), so no candidate
# gains from cross-call PDL overlap: in production the neighbouring kernels are norms/activations, not the same GEMM.
# Refuses to run unless the GPU is idle (vLLM has no running requests). Writes $HK_WORK/k2_race.jsonl.
import importlib, json, statistics, sys, types, urllib.request, torch
from tilelang.contrib.sm120_nvfp4.decode_contract import SHAPES, synthetic_layer
from tilelang.contrib.sm120_nvfp4.decode import decode_linear_v1
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear

try:
    m = urllib.request.urlopen("http://localhost:8000/metrics", timeout=3).read().decode()
    busy = sum(float(v) for v in __import__("re").findall(r'vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.e+]+)', m))
    assert busy == 0, f"vLLM busy ({busy} requests): race refused"
except (urllib.error.URLError, ConnectionError):
    pass  # vLLM down: GPU is ours

MODS = {name: importlib.import_module(name) for name in sys.argv[1:]}
MS = (1, 4, 8, 16, 32, 64)
ROOF = 1631e9


SEP = torch.zeros(1, device="cuda")


def graph_time_us(fns, iters=5, repeats=5, sep=True):
    if sep:
        fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for fn in fns: fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            for fn in fns: fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(repeats):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (iters * len(fns)))
    return statistics.median(ts)


SEP_US = graph_time_us([lambda: None] * 8, sep=True)   # cost of the separator alone, subtracted below
print(json.dumps({"separator_us": round(SEP_US, 2)}), flush=True)
out = open("$HK_WORK/k2_race.jsonl", "a")
for sname, (N, K) in SHAPES.items():
    copies = max(2, int(512e6 // (N * K // 2)) + 1)
    layers = [synthetic_layer(N, K, seed=s) for s in range(copies)]
    nbytes = N * K // 2 + N * K // 16
    marl = []
    for packed, scale, alpha in layers:
        l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None,
                                  weight=packed.view(torch.uint8).clone(), weight_scale=scale.clone(), weight_global_scale=(1.0 / alpha[:1]).clone())
        prepare_fp4_layer_for_marlin(l); marl.append(l)
    preps = {name: [mod.prepare(*lay) for lay in layers] for name, mod in MODS.items()}
    for M in MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        row = {"shape": sname, "N": N, "K": K, "M": M}
        t = graph_time_us([lambda l=l: apply_fp4_marlin_linear(x, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K) for l in marl]) - SEP_US
        row["marlin_us"] = round(t, 2); row["marlin_roof"] = round(100 * nbytes / (t * 1e-6) / ROOF, 1)
        t = graph_time_us([lambda lay=lay: decode_linear_v1(x, lay[0], lay[1], lay[2]) for lay in layers]) - SEP_US
        row["v1_us"] = round(t, 2)
        for name, mod in MODS.items():
            if M > getattr(mod, "SUPPORTS_M", 64):
                continue
            try:
                t = graph_time_us([lambda p=p, mod=mod: mod.linear(x, p) for p in preps[name]]) - SEP_US
                row[name.split(".")[-1] + "_us"] = round(t, 2)
                row[name.split(".")[-1] + "_roof"] = round(100 * nbytes / (t * 1e-6) / ROOF, 1)
            except Exception as ex:  # noqa: BLE001
                row[name.split(".")[-1] + "_us"] = f"ERR {type(ex).__name__}"
        best = min((v, k) for k, v in row.items() if k.endswith("_us") and isinstance(v, float) and k != "v1_us")
        row["best_vs_marlin"] = round(row["marlin_us"] / best[0], 3)
        row["winner"] = best[1][:-3]
        print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
    del layers, marl, preps; torch.cuda.empty_cache()
