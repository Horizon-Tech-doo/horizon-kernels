# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Microbenchmark: achievable HBM read bandwidth vs vLLM Marlin NVFP4 GEMM at Qwen3.8-27B decode shapes.
import json, sys, time, torch, types
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear)

dev = "cuda"
torch.manual_seed(0)

def timeit(fn, iters=50, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1e-3  # seconds per call

# 1) roofline: pure read (sum) over 1 GB, and copy
x = torch.empty(512 * 2**20, dtype=torch.bfloat16, device=dev).normal_()
t = timeit(lambda: x.sum(dtype=torch.float32), iters=20)
read_bw = x.numel() * 2 / t
y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x), iters=20)
copy_bw = 2 * x.numel() * 2 / t
del x, y
print(json.dumps({"read_GBps": round(read_bw / 1e9), "copy_GBps": round(copy_bw / 1e9)}), flush=True)

# 2) Marlin NVFP4 at decode shapes (fused as vLLM runs them)
SHAPES = {  # name: (N, K, count per forward)
    "gdn.in_proj_qkvz": (16384, 5120, 48),
    "attn.qkv_proj":    (14336, 5120, 16),
    "o/out_proj":       (5120, 6144, 64),
    "mlp.gate_up":      (34816, 5120, 64),
    "mlp.down":         (5120, 17408, 64),
}
MS = [int(m) for m in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1,4,8,16,32,64".split(","))]

def make_layer(N, K):
    l = types.SimpleNamespace()
    l.output_size_per_partition = N; l.input_size_per_partition = K; l.params_dtype = torch.bfloat16
    l.weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    l.weight_scale = (torch.rand(N, K // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    l.weight_global_scale = torch.tensor(1.0 / 448, device=dev)
    l.bias = None
    prepare_fp4_layer_for_marlin(l)
    return l

res = []
for name, (N, K, cnt) in SHAPES.items():
    nbytes = N * K // 2 + N * K // 16  # fp4 weights + fp8 block scales
    ncopy = max(2, int(600e6 // nbytes) + 1)  # defeat the 128 MB L2
    layers = [make_layer(N, K) for _ in range(ncopy)]
    for M in MS:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        st = {"i": 0}
        def run():
            l = layers[st["i"] % ncopy]; st["i"] += 1
            apply_fp4_marlin_linear(a, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K)
        # graph captures a fixed sequence, so unroll over copies explicitly
        def run_all():
            for l in layers:
                apply_fp4_marlin_linear(a, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K)
        t = timeit(run_all, iters=max(1, 20 // ncopy + 1)) / ncopy
        r = {"shape": name, "N": N, "K": K, "M": M, "us": round(t * 1e6, 1),
             "GBps": round(nbytes / t / 1e9), "pct_read_roof": round(100 * nbytes / t / read_bw, 1),
             "ms_per_fwd": round(t * cnt * 1e3, 3)}
        res.append(r); print(json.dumps(r), flush=True)
    del layers; torch.cuda.empty_cache()

for M in MS:
    tot = sum(r["ms_per_fwd"] for r in res if r["M"] == M)
    print(json.dumps({"M": M, "marlin_linear_ms_per_forward": round(tot, 2)}), flush=True)
