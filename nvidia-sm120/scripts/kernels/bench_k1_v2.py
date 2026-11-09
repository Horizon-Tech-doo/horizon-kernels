# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K1a v1 (W8A8) vs v2 (W8A16) on the real lm_head: speed, plus argmax agreement with the BF16 reference on
# deliberately hard inputs (two near-tied candidate tokens), with and without LLM-style outlier channels in h.
import json, torch
from safetensors import safe_open
from horizon_kernels.draft_head import DraftHeadFP8


def bench(fn, iters=50):
    for _ in range(5): fn()
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
    W = f.get_tensor("lm_head.weight")
N, K = W.shape
heads = {"v1_w8a8": DraftHeadFP8(W, act="fp8"), "v2_w8a16": DraftHeadFP8(W, act="bf16")}
torch.manual_seed(0)
ROWS = 2048
a = torch.randint(0, N, (ROWS,), device="cuda"); b = torch.randint(0, N, (ROWS,), device="cuda")
base = (W[a].float() + 0.97 * W[b].float()) * 3 + torch.randn(ROWS, K, device="cuda") * 0.05
out_ch = torch.randint(0, K, (6,), device="cuda")
cases = {"no_outliers": base, "outliers_x40": base.clone()}
cases["outliers_x40"][:, out_ch] *= 40  # a few massive channels, as seen in real LLM hidden states
for cname, H in cases.items():
    H = H.to(torch.bfloat16)
    ref = torch.cat([(H[i:i + 256].float() @ W.float().T).argmax(-1) for i in range(0, ROWS, 256)])
    for hname, head in heads.items():
        got = torch.cat([head(H[i:i + 16]) for i in range(0, ROWS, 16)])
        print(json.dumps({"case": cname, "head": hname, "agree_pct": round(100 * (got == ref).float().mean().item(), 2)}), flush=True)
for M in (1, 8, 16):
    h = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    r = {"M": M, "bf16_ref_us": round(bench(lambda: (h @ W.T).argmax(-1)), 1)}
    for hname, head in heads.items():
        r[hname + "_us"] = round(bench(lambda: head(h)), 1)
    print(json.dumps(r), flush=True)
