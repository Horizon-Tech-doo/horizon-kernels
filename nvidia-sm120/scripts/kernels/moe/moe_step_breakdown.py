#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-decode-step time breakdown of a vLLM torch-profiler trace of the MoE model (MoE P1 stage B, 27-09-2026).
Unlike moe_trace.py it does not rely on vLLM's step annotations (with piecewise CUDA graphs they fire per graph
piece, ~40 per token): it sums every GPU kernel in the decode window by category and divides by the number of decode
steps (forward passes), counted as Marlin-MoE launches / 80 (40 MoE layers x 2 GEMMs; no speculative decoding). The
window runs from the first to the last Marlin-MoE launch, so idle time before/after the requests is excluded.
    moe_step_breakdown.py TRACE.json.gz USERS [--json OUT]"""
import argparse, collections, gzip, json, re

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("users", type=int); ap.add_argument("--json")
a = ap.parse_args()
ev = json.load(gzip.open(a.trace))["traceEvents"]
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
CATS = [
    ("routed experts: Marlin-MoE GEMMs", r"marlin_moe_wna16"),
    ("shared expert + lm_head: dense Marlin NVFP4", r"marlin::marlin"),
    ("FP8 dense GEMMs (GDN/attn projections, CUTLASS sm120)", r"enable_sm120_family|cutlass_scaled_mm_sm120|sm120.*gemm"),
    ("norms + FP8 activation quant (Inductor-fused)", r"^triton_.*(rms_norm|scaled_mm|layer_norm)"),
    ("MoE routing glue (top-k, align, sort, SiLU, sum, gates)",
     r"topkgating|moe_align|count_and_sort|moe_sum|act_and_mul|silu|sigmoid|reduce_kernel|fillfunctor|zeros"),
    ("BF16 GEMV/GEMM (router gate, shared gate, in_proj_ba)", r"dot_kernel|gemvx|cublas|splitkreduce|reduce_1block|cutlass_80|gemm"),
    ("full attention (10 layers) + KV write + RoPE", r"kernel_mha|flash|attn|reshape_and_cache|rope|rotary"),
    ("GDN core (recurrent, conv1d, gated norm)", r"gated_delta|recurrent|causal_conv1d|layer_norm_fwd|gdn|mamba"),
    ("other (copies, casts, sampling)", r"."),
]
def cat(n):
    n = n.lower()
    for c, p in CATS:
        if re.search(p, n):
            return c
    return "other (copies, casts, sampling)"
mo = [e for e in k if "marlin_moe_wna16" in e["name"]]
t0, t1 = mo[0]["ts"], mo[-1]["ts"] + mo[-1]["dur"]
steps = round(len(mo) / 80)
by, cnt = collections.Counter(), collections.Counter()
iv = []
for e in k:
    if e["ts"] < t0 or e["ts"] > t1:
        continue
    c = cat(e["name"]); by[c] += e["dur"]; cnt[c] += 1
    iv.append((e["ts"], min(e["ts"] + e["dur"], t1)))
iv.sort(); busy = 0; cur = None
for s, e in iv:
    if cur and s <= cur[1]:
        cur[1] = max(cur[1], e)
    else:
        if cur: busy += cur[1] - cur[0]
        cur = [s, e]
if cur: busy += cur[1] - cur[0]
win = t1 - t0
step_us = win / steps
out = {"trace": a.trace, "users": a.users, "steps": steps, "step_us": round(step_us, 1),
       "tok_s_equiv": round(a.users * 1e6 / step_us, 1), "kernels_per_step": round(sum(cnt.values()) / steps, 1),
       "idle_us_per_step": round((win - busy) / steps, 1), "categories": {}}
print(f"{a.users} users: {steps} decode steps, {step_us:.0f} us/step ({a.users * 1e6 / step_us:.0f} tok/s under the "
      f"profiler), {sum(cnt.values()) / steps:.0f} kernels/step, GPU idle {100 * (win - busy) / win:.1f}%")
for c, v in sorted(by.items(), key=lambda t: -t[1]):
    out["categories"][c] = {"us_step": round(v / steps, 1), "share": round(v / win, 4), "n_step": round(cnt[c] / steps, 1)}
    print(f"  {c:58s} {v / steps:8.1f} us  {100 * v / win:5.1f}%  {cnt[c] / steps:6.1f} launches")
print(f"  {'idle gaps':58s} {(win - busy) / steps:8.1f} us  {100 * (win - busy) / win:5.1f}%")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
