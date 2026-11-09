# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Benchmark K1a (fused FP8 draft head) against BF16 GEMM + argmax on the real lm_head.
import sys, json, torch
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
    return best * 1e3  # us


if __name__ == "__main__":
    from safetensors import safe_open
    cfg = dict(BN=int(sys.argv[1]) if len(sys.argv) > 1 else 128, BK=int(sys.argv[2]) if len(sys.argv) > 2 else 128,
               stages=int(sys.argv[3]) if len(sys.argv) > 3 else 4, threads=int(sys.argv[4]) if len(sys.argv) > 4 else 128)
    with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
        W = f.get_tensor("lm_head.weight")
    head = DraftHeadFP8(W, act="fp8", **cfg)  # deployed K1a v1 (W8A8)
    torch.manual_seed(0)
    READ_ROOF = 1631e9
    for M in (1, 4, 8, 16):
        # realistic-ish hidden: a mix of real output-embedding directions plus noise, so argmax is peaked like a real LM
        ids = torch.randint(0, W.shape[0], (M,), device="cuda")
        h = (W[ids].float() * 3 + torch.randn(M, W.shape[1], device="cuda") * 0.02).to(torch.bfloat16)
        ref = (h @ W.T).float().argmax(-1)
        got = head(h)
        t_ref = bench(lambda: (h @ W.T).argmax(-1))
        t_new = bench(lambda: head(h))
        nbytes = head.W.numel() + head.SW.numel() * 4
        print(json.dumps({"M": M, "cfg": cfg, "agree": int((ref == got).sum()), "of": M,
                          "bf16_gemm_argmax_us": round(t_ref, 1), "fp8_fused_us": round(t_new, 1),
                          "speedup": round(t_ref / t_new, 2),
                          "fused_pct_roof": round(100 * nbytes / (t_new * 1e-6) / READ_ROOF, 1)}), flush=True)
