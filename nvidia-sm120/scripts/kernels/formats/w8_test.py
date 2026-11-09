# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S4 gate (formats track, 27-09-2026): the FP8 W8A16 kernel (tilelang.contrib.sm120_fp8.decode_w8a16) on REAL FP8
# layers of nvidia/Qwen3.8-27B-NVFP4 and nvidia/Qwen3.6-35B-A3B-NVFP4 (ModelOpt per-tensor FP8 attention / GDN
# projections), fused exactly as vLLM fuses them (per-shard scales collapsed by requantize_with_max_scale).
#   correctness vs an exact FP32 reference (exact class: <= 0.0025), vs vLLM's W8A8 CUTLASS path (what runs today),
#   determinism, CUDA-graph capture (thread_local) + replay with new inputs, and speed (graph replay, > 512 MB of
#   rotated weight copies, separator kernel, 3 interleaved rounds, fastest round) vs CUTLASS (GEMM only and quant+GEMM).
#   run under $HK_WORK/gpu_lock.sh.  env: MODELS=nv27b,moe  MS=1,2,4,...  CFG_<N>x<K>_<M>="(..)" overrides; SPEED=0
import json
import os
import statistics
import sys
import time

import torch
from safetensors import safe_open

from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8

MODELS = {"nv27b": "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4",
          "moe": "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4"}
LAYERS = {"gdn0.qkvz": ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z"],
          "gdn0.out": ["layers.0.linear_attn.out_proj"],
          "attn3.qkv": ["layers.3.self_attn.q_proj", "layers.3.self_attn.k_proj", "layers.3.self_attn.v_proj"],
          "attn3.o": ["layers.3.self_attn.o_proj"]}
MS = [int(m) for m in os.environ.get("MS", "1,2,4,8,16,32,33,48,64,65,96,128,256,512,1024").split(",")]
CAP_MS = [1, 4, 32, 128]
PEAK = 1.8e12
OUT = os.environ.get("OUT", "$HK_WORK/formats/w8_test.jsonl")
dev = torch.device("cuda")
SEP = torch.zeros(1, device=dev)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_layer(root, shards):
    wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    ws, ss, ins = [], [], []
    for s in shards:
        name = "model.language_model." + s
        with safe_open(os.path.join(root, wm[name + ".weight"]), "pt", device="cpu") as f:
            ws.append(f.get_tensor(name + ".weight"))
            ss.append(f.get_tensor(name + ".weight_scale").float().reshape(()))
            ins.append(f.get_tensor(name + ".input_scale").float().reshape(()))
    w = torch.cat(ws).to(dev)
    scales = torch.stack(ss).to(dev)
    widths = [t.shape[0] for t in ws]
    return w, scales, widths, torch.stack(ins).max().to(dev)


def stock_fuse(w, scales, widths):
    """vLLM 0.29 ModelOptFp8LinearMethod.process_weights_after_loading: requantize to the max shard scale."""
    if len(widths) > 1 and not bool((scales == scales[0]).all()):
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale
        smax, w2 = requantize_with_max_scale(w, scales, widths)
        return w2.contiguous(), smax.reshape(())
    return w, scales.max().reshape(())


def cutlass_fns(w, ws, in_s):
    from vllm import _custom_ops as ops
    b = w.t()                                            # (K, N) column-major view, as vLLM stores it

    def gemm(xq, xs):
        return ops.cutlass_scaled_mm(xq, b, out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws.reshape(1))

    def full(x):
        xq, xs = ops.scaled_fp8_quant(x, in_s.reshape(1))
        return gemm(xq, xs)
    return gemm, full


def graph_us(fns, iters=4, rounds=3):
    fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for f in fns:
        f()
    torch.cuda.synchronize()
    for attempt in range(3):
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                for _ in range(iters):
                    for f in fns:
                        f()
            break
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt == 2:
                raise
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (iters * len(fns)))
    return min(ts)


def sep_us():
    return graph_us([lambda: None] * 8)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def capture_check(prep, K, m):
    xs = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
    w8.linear(xs, prep)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        yg = w8.linear(xs, prep)
    ok = True
    for _ in range(2):
        xs.copy_(torch.randn(m, K, device=dev, dtype=torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        ok &= bool(torch.equal(yg, w8.linear(xs, prep)))
    return ok


def main():
    out = open(OUT, "a")
    SEPU = sep_us()
    log("separator", round(SEPU, 2), "us")
    allpass = True
    seen_shapes = set()
    for mname in os.environ.get("MODELS", "nv27b,moe").split(","):
        root = MODELS[mname]
        for lname, shards in LAYERS.items():
            w0, scales, widths, in_s = load_layer(root, shards)
            w, ws = stock_fuse(w0, scales, widths)
            N, K = w.shape
            prep = w8.prepare(w, ws)
            ref_w = w.float() * ws                        # exact dequantised (collapsed) weight
            gemm, full = cutlass_fns(w, ws, in_s)
            xscale = float(in_s) * 448.0 / 6.0            # activations spanning the layer's static FP8 range
            log(mname, lname, "N=%d K=%d shards=%s scales=%s in_s=%.4g" % (N, K, widths, [round(float(s), 6) for s in scales], float(in_s)))
            speed = os.environ.get("SPEED", "1") == "1" and (N, K) not in seen_shapes
            seen_shapes.add((N, K))
            if speed:
                copies = max(2, int(512e6 // (N * K)) + 1)
                wcs = [w.clone() for _ in range(copies)]
                pcs = [w8.prepare(c, ws) for c in wcs]
                gcs = [cutlass_fns(c, ws, in_s) for c in wcs]
            for m in MS:
                x = torch.randn(m, K, device=dev, dtype=torch.bfloat16) * xscale
                cfg = os.environ.get(f"CFG_{N}x{K}_{m}")
                cfg = eval(cfg) if cfg else None
                y = w8.linear(x, prep, cfg)
                y2 = w8.linear(x, prep, cfg)
                ref = x.float() @ ref_w.T
                yc = full(x)
                row = {"model": mname, "layer": lname, "N": N, "K": K, "M": m,
                       "cfg": list(cfg or w8.config(N, K, m)),
                       "err_ours": round(rel(y, ref), 6), "err_cutlass_w8a8": round(rel(yc, ref), 6),
                       "err_bf16_floor": round(rel(ref.to(torch.bfloat16), ref), 6),
                       "deterministic": bool(torch.equal(y, y2))}
                if m in CAP_MS:
                    row["capture_ok"] = capture_check(prep, K, m)
                if speed:
                    xs_ = [x]
                    t_o = graph_us([lambda p=p: w8.linear(xs_[0], p, cfg) for p in pcs]) - SEPU
                    xq, xsc = __import__("vllm")._custom_ops.scaled_fp8_quant(x, in_s.reshape(1))
                    t_g = graph_us([lambda g=g: g[0](xq, xsc) for g in gcs]) - SEPU
                    t_f = graph_us([lambda g=g: g[1](x) for g in gcs]) - SEPU
                    row.update({"us_ours": round(t_o, 2), "us_cutlass_gemm": round(t_g, 2),
                                "us_cutlass_quant_gemm": round(t_f, 2),
                                "tbs_ours": round(N * K / (t_o * 1e-6) / 1e12, 3),
                                "pct_peak_ours": round(100 * N * K / (t_o * 1e-6) / PEAK, 1),
                                "pct_peak_cutlass": round(100 * N * K / (t_g * 1e-6) / PEAK, 1),
                                "speedup_vs_gemm": round(t_g / t_o, 3), "speedup_vs_quant_gemm": round(t_f / t_o, 3)})
                ok = row["err_ours"] <= 0.0025 and row["deterministic"] and row.get("capture_ok", True)
                row["pass"] = ok
                allpass &= ok
                print(json.dumps(row), flush=True)
                out.write(json.dumps(row) + "\n"); out.flush()
            del prep
            if speed:
                del wcs, pcs, gcs
            torch.cuda.empty_cache()
    print("W8_TEST", "ALL PASS" if allpass else "FAILURES", flush=True)


if __name__ == "__main__":
    main()
