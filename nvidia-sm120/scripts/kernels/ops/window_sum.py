# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Ops step 0: sum a whole profiler window by kernel category (for prefill windows, where k5_trace's decode-step
# logic does not apply). usage: window_sum.py TRACE.json.gz [--json OUT] [--top 30]
import argparse, collections, gzip, json, re
ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--json", default="")
ap.add_argument("--top", type=int, default=30); a = ap.parse_args()
ev = json.load(gzip.open(a.trace))["traceEvents"]
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
CATS = [
    ("rmsnorm (Inductor)", r"^triton_.*(rms_norm|rmsnorm)"),
    ("silu*mul (Inductor)", r"^triton_.*silu"),
    ("attn gate sigmoid (Inductor)", r"^triton_.*sigmoid"),
    ("inductor other", r"^triton_(poi|red|per)"),
    ("ours (TileLang main_kernel)", r"^main_kernel|regfrag"),
    ("gdn prefill: FLA chunk kernels", r"chunk_|solve_tril|recompute_w_u|kkt|cumsum|l2norm|wy_fast|merge_16x16"),
    ("gdn prefill: conv1d fwd", r"causal_conv1d_fwd|_causal_conv1d_fwd"),
    ("gdn decode: conv1d update", r"causal_conv1d_update"),
    ("gdn post-conv prep", r"post_conv|fused_post"),
    ("gdn decode core (CUDA/FLA recurrent)", r"gdn_decode|fused_recurrent|sigmoid_gating_delta|gated_delta"),
    ("gdn gated norm (Triton)", r"layer_norm_fwd|_layer_norm|rms_norm_gated|norm_gated"),
    ("attention prefill (FlashInfer)", r"BatchPrefill|prefill|fmha|flash_fwd|SinglePrefill"),
    ("attention decode (XQA)", r"kernel_mha|xqa|BatchDecode"),
    ("kv-cache write", r"reshape_and_cache|concat_and_cache"),
    ("rope + q/k norm", r"rope|rotary|qk_rmsnorm"),
    ("cublas / cutlass bf16", r"gemm|gemv|cutlass|nvjet|cublas|sm80_xmma|splitk"),
    ("sampling / spec glue", r"argmax|sample|reject|topk|softmax|logits|resample|index|gather|scatter|cumsum|arange"),
    ("copies / memsets", r"memcpy|memset|copy|fill|zeros|cat|elementwise|vectorized"),
]


def cat(n):
    n2 = n.lower()
    for c, p in CATS:
        if re.search(p, n if p.startswith("^") else n2):
            return c
    return "other"


if not k:
    raise SystemExit("no kernels")
t0, t1 = k[0]["ts"], max(e["ts"] + e["dur"] for e in k)
by = collections.Counter(); cnt = collections.Counter(); nm = collections.Counter(); nmc = collections.Counter()
for e in k:
    c = cat(e["name"]); by[c] += e["dur"]; cnt[c] += 1; nm[e["name"][:110]] += e["dur"]; nmc[e["name"][:110]] += 1
busy = sum(by.values())
print(f"window {1e-3 * (t1 - t0):.1f} ms, kernel time {1e-3 * busy:.1f} ms, {len(k)} kernels")
rows = [(c, v / 1e3, 100 * v / busy, cnt[c]) for c, v in by.most_common()]
for r in rows:
    print(f"{r[0]:42s} {r[1]:10.2f} ms {r[2]:5.1f}% n={r[3]}")
print("--- top kernels (ms, calls, avg us)")
for n, v in nm.most_common(a.top):
    print(f"{v / 1e3:9.2f} {nmc[n]:6d} {v / nmc[n]:9.1f}  [{cat(n)[:18]}] {n[:90]}")
if a.json:
    json.dump({"trace": a.trace, "window_ms": (t1 - t0) / 1e3, "kernel_ms": busy / 1e3,
               "categories": [dict(zip(("cat", "ms", "share", "n"), r)) for r in rows],
               "top": [(n, nm[n] / 1e3, nmc[n]) for n, _ in nm.most_common(a.top)]}, open(a.json, "w"), indent=1)
