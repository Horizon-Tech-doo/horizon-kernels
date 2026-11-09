# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Marlin-MoE microbenchmark (MoE P1 stage B, 27-09-2026). No vLLM server: calls vLLM 0.29's own fused_marlin_moe on
# the Qwen3.6-35B-A3B expert shapes (E=256, top-8, hidden 2048, expert width 512, NVFP4 W4A16) and reports the
# achieved weight bandwidth, which settles the "Marlin-MoE at ~45% of 1.8 TB/s" assumption in docs/moe-plan.md.
#
#   moe_marlin_bench.py [--m 1,2,4,8,16,32,64,128,256] [--copies 8] [--reps 50] [--out FILE] [--test]
#
# - Weights: random NVFP4 codes / e4m3 block scales in the checkpoint layout, converted with vLLM's own
#   prepare_nvfp4_moe_layer_for_marlin (the exact repack the server does). Speed does not depend on the values.
# - L2: `--copies` independent expert sets (8 x 453 MB) are used round-robin, so every call streams from HBM like the
#   40 layers of a real forward do.
# - Routing: uniform random top-8 per token (distinct experts per token). Bytes = distinct experts actually touched
#   (counted from topk_ids) x bytes per expert of the Marlin tensors (codes + scales), so GB/s is exact for the call.
# - Two timings per M:
#     op_us     the whole fused_marlin_moe call (align + w13 GEMM + SiLU*mul + w2 GEMM + sum), CUDA-graph replayed
#               (thread_local capture, gpushare-safe), CUDA events over reps x copies replays;
#     gemm_us   only the two Marlin-MoE GEMM kernels, from the torch profiler over the same replays.
# Run it as the ONLY GPU process (vLLM stopped), under $HK_WORK/gpu_lock.sh.
import argparse, json, os, time, types

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--m", default="1,2,4,8,16,32,64,128,256")
ap.add_argument("--copies", type=int, default=8)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/moe/marlin_moe_bench.jsonl"))
ap.add_argument("--tag", default="marlin-moe-27-09-2026")
ap.add_argument("--test", action="store_true", help="functional check only: 1 copy, M=4, few reps (timings invalid)")
A = ap.parse_args()
if A.test:
    A.m, A.copies, A.reps = "1,4,32", 1, 3

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E402
    prepare_nvfp4_moe_layer_for_marlin)
from vllm.scalar_type import scalar_types  # noqa: E402

E, H, I, TOPK = 256, 2048, 512, 8
DEV = "cuda"
PEAK = 1.8e12
torch.manual_seed(0)


def make_copy():
    layer = types.SimpleNamespace(num_experts=E, hidden_size=H, intermediate_size_per_partition=I,
                                  params_dtype=torch.bfloat16)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV)
    w13s = (torch.rand(E, 2 * I, H // 16, device=DEV) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV)
    w2s = (torch.rand(E, H, I // 16, device=DEV) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    g = torch.full((E,), 1e-3, device=DEV, dtype=torch.float32)
    w13m, w13sm, g1, w2m, w2sm, g2 = prepare_nvfp4_moe_layer_for_marlin(layer, w13, w13s, g.clone(), w2, w2s,
                                                                        g.clone(), is_act_and_mul=True)
    del w13, w13s, w2, w2s
    c = dict(w1=w13m, w2=w2m, w1_scale=w13sm, w2_scale=w2sm, global_scale1=g1, global_scale2=g2,
             workspace=layer.workspace)
    c["bytes_per_expert"] = sum(t.numel() * t.element_size() for t in (w13m, w13sm, w2m, w2sm)) / E
    return c


def call(c, x, tw, ti):
    return fused_marlin_moe(hidden_states=x, w1=c["w1"], w2=c["w2"], bias1=None, bias2=None,
                            w1_scale=c["w1_scale"], w2_scale=c["w2_scale"], topk_weights=tw, topk_ids=ti,
                            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E,
                            activation=MoEActivation.SILU, global_scale1=c["global_scale1"],
                            global_scale2=c["global_scale2"], workspace=c["workspace"])


def capture(fn, tries=5):
    for t in range(tries):
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn()
            torch.cuda.synchronize()
            return g
        except Exception as e:  # noqa: BLE001  (gpushare can break a capture at random: retry)
            print(f"capture retry {t}: {str(e)[:120]}", flush=True)
            torch.cuda.synchronize()
            time.sleep(1)
    raise RuntimeError("CUDA-graph capture failed")


def main():
    with set_current_vllm_config(VllmConfig()):
        t0 = time.time()
        copies = [make_copy() for _ in range(A.copies)]
        bpe = copies[0]["bytes_per_expert"]
        print(f"{A.copies} expert sets ready in {time.time() - t0:.1f}s, {bpe / 1e6:.3f} MB per expert "
              f"({bpe * E / 1e6:.0f} MB per layer)", flush=True)
        rows = []
        for M in [int(v) for v in A.m.split(",")]:
            x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 0.1
            ti = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
            tw = torch.softmax(torch.rand(M, TOPK, device=DEV), -1).float()
            distinct = int(ti.unique().numel())
            for c in copies:           # eager warm-up (module load / first launch outside capture)
                out = call(c, x, tw, ti)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(out).all())
            graphs = [capture(lambda c=c: call(c, x, tw, ti)) for c in copies]
            for g in graphs:
                g.replay()
            torch.cuda.synchronize()
            n = A.reps * len(graphs)
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for r in range(A.reps):
                for g in graphs:
                    g.replay()
            ev1.record(); torch.cuda.synchronize()
            op_us = ev0.elapsed_time(ev1) * 1e3 / n
            # kernel-level split from the profiler (fewer reps: trace size)
            pr = max(2, min(A.reps, 10))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for r in range(pr):
                    for g in graphs:
                        g.replay()
                torch.cuda.synchronize()
            ksum = {}
            for e in prof.key_averages():
                d = getattr(e, "self_device_time_total", None)
                if d is None:
                    d = getattr(e, "self_cuda_time_total", 0.0)
                if d and d > 0:
                    ksum[e.key] = ksum.get(e.key, 0.0) + d
            npr = pr * len(graphs)
            gemm = sum(v for k, v in ksum.items() if "marlin_moe" in k.lower() or "moe_wna16" in k.lower()) / npr
            other = {k[:70]: round(v / npr, 2) for k, v in sorted(ksum.items(), key=lambda t: -t[1])
                     if not ("marlin_moe" in k.lower() or "moe_wna16" in k.lower())}
            byts = distinct * bpe
            row = {"tag": A.tag, "M": M, "distinct_experts": distinct, "tokens_per_expert": round(M * TOPK / distinct, 2),
                   "bytes_mb": round(byts / 1e6, 2), "op_us": round(op_us, 2), "gemm_us": round(gemm, 2),
                   "op_gb_s": round(byts / op_us / 1e3, 1), "gemm_gb_s": round(byts / max(gemm, 1e-9) / 1e3, 1),
                   "gemm_frac_of_1800": round(byts / max(gemm, 1e-9) / 1e3 / 1800, 3),
                   "op_frac_of_1800": round(byts / op_us / 1e3 / 1800, 3),
                   "roofline_us": round(byts / PEAK * 1e6, 2), "other_kernels_us": other, "finite": finite,
                   "kernel_names": sorted({k[:90] for k in ksum if "marlin" in k.lower()})[:4],
                   "test": A.test}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("other_kernels_us", "kernel_names")}),
                  flush=True)
            print("   other kernels (us/call): " + json.dumps(other)[:400], flush=True)
            del graphs
        os.makedirs(os.path.dirname(A.out), exist_ok=True)
        with open(A.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
