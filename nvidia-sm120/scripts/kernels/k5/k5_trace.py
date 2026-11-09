# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Summarise a vLLM torch-profiler trace per decode step (K5 profiling, 26-09-2026).
# usage: k5_trace.py TRACE.json.gz [--gen N] [--top 40]
# A step = from one model-forward annotation ("execute_context_0(0)_generation_U(T)") to the next: main forward +
# MTP drafts + sampling/rejection + idle. Every number is µs per step, averaged over steady-state steps.
import argparse, collections, gzip, json, re, statistics
ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--top", type=int, default=45)
ap.add_argument("--json", default=""); a = ap.parse_args()
ev = json.load(gzip.open(a.trace))["traceEvents"]
ann = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context_0(0)")],
             key=lambda e: e["ts"])
gens = collections.Counter(e["name"] for e in ann)
name, _ = gens.most_common(1)[0]
ann = [e for e in ann if e["name"] == name]
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])

CATS = [  # first match wins; applied to the kernel name (lower case). Inductor (triton_*) names carry the op that
          # follows them (e.g. "..._rms_norm_marlin_gemm_0"), so they are classified before the GEMM patterns.
    ("rmsnorm+residual_add", r"^triton_.*(rms_norm|rmsnorm)"),
    ("activation (silu*mul)", r"^triton_.*silu"),
    ("attn output gate (sigmoid*)", r"^triton_.*sigmoid"),
    ("memset/zeros/copy", r"^triton_.*(zeros|fill)"),
    ("gemm_ours (regfrag W4A16 NVFP4)", r"nvfp4_regfrag|regfrag|^main_kernel"),
    ("draft_head K1a (FP8 argmax)", r"draft_head"),
    ("gemm_bf16 (lm_head, MTP, in_proj_ba)", r"gemm|gemv|cutlass|nvjet|cublas|sm80_xmma|sm90|splitkreduce"),
    ("full_attention", r"flash|attn|attention|kernel_mha|paged|mla|unified_attention"),
    ("kv_cache_write", r"reshape_and_cache|concat_and_cache"),
    ("rotary / qk_norm", r"rope|rotary"),
    ("gdn_conv1d", r"causal_conv1d"),
    ("gdn_recurrent+gating+norm", r"gdn|gated_delta|recurrent|fla|chunk_|l2norm|gating"),
    ("rmsnorm+residual_add", r"rms_norm|rmsnorm|layer_norm|_norm_"),
    ("activation (silu*mul)", r"silu|act_and_mul|gelu|sigmoid"),
    ("quant/convert", r"quant|fp8|to_copy|convert|cast|scaled"),
    ("sampling/argmax/spec glue", r"argmax|sample|reject|topk|softmax|reduce|logits|spec|slot_mapping|prepare|expand|"
                                  r"resample|eagle|index|gather|scatter|cumsum|arange|compare|where|max"),
    ("memset/zeros/copy", r"zeros|fill|memset|memcpy|copy|elementwise|vectorized|cat"),
]
def cat(n):
    n = n.lower()
    for c, p in CATS:
        if re.search(p, n): return c
    return "other"

def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out

periods = [b["ts"] - x["ts"] for x, b in zip(ann, ann[1:])]
med = statistics.median(periods)
steps = [(x, b) for x, b in zip(ann, ann[1:]) if b["ts"] - x["ts"] < 1.5 * med][2:-1]   # skip warm edges
S = len(steps)
by = collections.Counter(); byin = collections.Counter(); cnt = collections.Counter(); nm = collections.Counter()
nmc = collections.Counter(); geo = {}; idle = 0.0; idle_in = 0.0; win = 0.0; fwd = 0.0; nk = 0
ki = 0
for x, b in steps:
    t0, t1, f1 = x["ts"], b["ts"], x["ts"] + x["dur"]
    while ki < len(k) and k[ki]["ts"] < t0: ki += 1
    j = ki; iv = []; ivin = []
    while j < len(k) and k[j]["ts"] < t1:
        e = k[j]; c = cat(e["name"]); d = e["dur"]
        ar = e.get("args", {}); geo.setdefault(e["name"][:120], (ar.get("grid"), ar.get("block"), ar.get("registers per thread")))
        by[c] += d; cnt[c] += 1; nm[e["name"][:120]] += d; nmc[e["name"][:120]] += 1; nk += 1
        iv.append((e["ts"], min(e["ts"] + d, t1)))
        if e["ts"] < f1:
            byin[c] += d; ivin.append((e["ts"], min(e["ts"] + d, f1)))
        j += 1
    u = union(iv); busy = sum(e - s for s, e in u); idle += (t1 - t0) - busy; win += t1 - t0
    ui = union(ivin); idle_in += (f1 - t0) - sum(e - s for s, e in ui); fwd += f1 - t0
out = {"trace": a.trace, "annotation": name, "steps": S, "step_us": win / S, "main_forward_us": fwd / S,
       "kernels_per_step": nk / S}
print(f"{name}: {S} steady steps, step {win/S:.0f} us (main forward {fwd/S:.0f} us), {nk/S:.0f} kernels/step")
print(f"{'category':40s} {'us/step':>9s} {'share':>6s} {'in fwd':>8s} {'n/step':>7s}")
rows = []
for c, v in sorted(by.items(), key=lambda t: -t[1]):
    rows.append((c, v / S, 100 * v / win, byin[c] / S, cnt[c] / S))
rows.append(("idle gaps (GPU waiting)", idle / S, 100 * idle / win, idle_in / S, 0))
for r in rows: print(f"{r[0]:40s} {r[1]:9.1f} {r[2]:5.1f}% {r[3]:8.1f} {r[4]:7.1f}")
out["categories"] = [dict(zip(("cat", "us_step", "share", "us_in_fwd", "n_step"), r)) for r in rows]
print("--- top kernels (us/step, calls/step, avg us)")
for n, v in nm.most_common(a.top):
    print(f"{v/S:8.1f} {nmc[n]/S:6.1f} {v/nmc[n]:7.2f}  [{cat(n)[:14]}] g={geo[n][0]} b={geo[n][1]} {n[:90]}")
out["top"] = [(n, nm[n] / S, nmc[n] / S) for n, _ in nm.most_common(a.top)]
if a.json: json.dump(out, open(a.json, "w"), indent=1)
