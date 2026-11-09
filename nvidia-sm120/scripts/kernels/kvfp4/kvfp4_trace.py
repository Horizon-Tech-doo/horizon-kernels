# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""kvfp4_trace.py TRACE.json.gz [--json OUT]  (29-09-2026)
Per-step GPU time breakdown of a vLLM torch-profiler trace, separating the FP4 KV kernels (all TileLang kernels are
named main_kernel; they are told apart by launch signature) from FlashInfer XQA / FA2, our GEMMs and the rest.
A step = one target forward annotation ("execute_context_0(0)...") to the next, so it includes the MTP draft steps.
Reports per steady-state step: GPU busy time per category (us), kernel count, GPU idle inside the step, and the
mean step wall time. Categories:
  fp4_decode   main_kernel, block 64 / 128, grid.y = KVH, >= 15 KB shared, >= 110 registers (ragged decode)
  fp4_combine  main_kernel, block 256, no shared memory, <= 48 registers, grid.y = rows (combine_ragged)
  fp4_write    main_kernel, no shared memory, 1-D grid: block 128 / 36..64 regs (TileLang write) or 256 / 512 (CUDA)
  fp4_dequant  main_kernel, block 128, no shared memory, grid.z == 2 (dequant_tokens / dequant_multi)
  flashinfer_prefill / xqa / other attention, kv write (FP8 reshape_and_cache), gemm_ours (other main_kernel),
  gemm_bf16, gdn, norm/act, sampling/spec glue, memcpy/memset, other
"""
import argparse
import collections
import gzip
import json
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--json", default="")
a = ap.parse_args()
ev = json.load(gzip.open(a.trace))["traceEvents"]
ann = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context_")],
             key=lambda e: e["ts"])
if not ann:
    print(json.dumps({"trace": a.trace, "error": "no step annotations"}))
    raise SystemExit
kern = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])


def classify(e):
    n = e["name"].lower()
    arg = e.get("args", {})
    blk = arg.get("block", [0, 0, 0])
    grid = arg.get("grid", [0, 0, 0])
    regs = arg.get("registers per thread", 0)
    if e.get("cat") != "kernel":
        return "memcpy/memset"
    if n == "main_kernel" or n.startswith("main_kernel"):
        smem = arg.get("shared memory", 0)
        gy = grid[1] if len(grid) > 1 else 0
        gz = grid[2] if len(grid) > 2 else 0
        # signatures measured in the 29-09-2026 traces (the baseline's own TileLang kernels never match these)
        if blk[0] in (64, 128) and gy in (2, 4) and smem >= 15000 and regs >= 110:
            return "fp4_decode"
        if blk[0] == 256 and smem == 0 and regs <= 48 and gy in (6, 8, 12, 16, 24, 32):
            return "fp4_combine"
        if blk[0] == 128 and smem == 0 and gz == 2:
            return "fp4_dequant"
        if smem == 0 and gy == 1 and gz == 1 and ((blk[0] == 128 and 36 <= regs <= 64) or blk[0] in (256, 512)):
            return "fp4_write"
        return "gemm_ours (TileLang)"
    if "hz4" in n and "write" in n:
        return "fp4_write"
    if "xqa" in n or "kernel_mha" in n:
        return "xqa_decode"
    if "prefill" in n or "flashinfer" in n or "batchprefill" in n or "singleprefill" in n:
        return "flashinfer_prefill"
    if "merge_state" in n or "mergestate" in n:
        return "flashinfer_merge"
    if "reshape_and_cache" in n or "concat_and_cache" in n:
        return "fp8_kv_write"
    if re.search(r"attn|attention|flash", n):
        return "other_attention"
    if re.search(r"regfrag|nvfp4", n):
        return "gemm_ours (TileLang)"
    if re.search(r"gemm|gemv|cutlass|nvjet|cublas|xmma|splitk", n):
        return "gemm_bf16"
    if re.search(r"gdn|gated_delta|recurrent|fla|chunk_|l2norm|causal_conv1d", n):
        return "gdn"
    if re.search(r"norm|silu|sigmoid|act_and_mul|rope|rotary", n):
        return "norm/act/rope"
    if re.search(r"argmax|sample|reject|topk|softmax|reduce|logits|spec|slot|prepare|expand|index|gather|scatter|"
                 r"cumsum|arange|compare|where|max", n):
        return "sampling/spec glue"
    return "other"


steps = []
ki = 0
for i in range(len(ann) - 1):
    t0, t1 = ann[i]["ts"], ann[i + 1]["ts"]
    while ki < len(kern) and kern[ki]["ts"] < t0:
        ki += 1
    j = ki
    cats = collections.Counter()
    cnt = collections.Counter()
    busy = 0.0
    last_end = t0
    while j < len(kern) and kern[j]["ts"] < t1:
        e = kern[j]
        c = classify(e)
        d = e.get("dur", 0)
        cats[c] += d
        cnt[c] += 1
        s, en = max(e["ts"], last_end), e["ts"] + d
        if en > s:
            busy += min(en, t1) - s
            last_end = max(last_end, en)
        j += 1
    steps.append({"wall": t1 - t0, "busy": busy, "cats": cats, "cnt": cnt, "gen": ann[i]["name"]})
if len(steps) > 6:
    steps = steps[1:-1]                  # drop the partial first / last step


def summary(sel):
    if not sel:
        return None
    allc = sorted({c for s in sel for c in s["cats"]}, key=lambda c: -sum(s["cats"][c] for s in sel))
    return {"steps": len(sel),
            "step_wall_us": round(statistics.mean(s["wall"] for s in sel), 1),
            "gpu_busy_us": round(statistics.mean(s["busy"] for s in sel), 1),
            "gpu_idle_us": round(statistics.mean(s["wall"] - s["busy"] for s in sel), 1),
            "kernels_per_step": round(statistics.mean(sum(s["cnt"].values()) for s in sel), 1),
            "per_step_us": {c: round(statistics.mean(s["cats"][c] for s in sel), 1) for c in allc},
            "launches_per_step": {c: round(statistics.mean(s["cnt"][c] for s in sel), 1) for c in allc}}


dec = [s for s in steps if s["gen"].startswith("execute_context_0(0)")]
mix = [s for s in steps if not s["gen"].startswith("execute_context_0(0)")]
out = {"trace": a.trace.split("/")[-1], "decode_steps": summary(dec), "mixed_prefill_steps": summary(mix),
       "annotations": dict(collections.Counter(s["gen"].split("_generation")[0] for s in steps).most_common(6))}
print(json.dumps(out, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
