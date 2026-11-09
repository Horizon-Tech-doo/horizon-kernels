"""Baseline for moe_fused.py: vLLM 0.30's fused_experts (Triton on ROCm) on the same MoE shape, inside the vLLM ROCm SIF.
Same reference math (float32 per-expert loop) for correctness; same token counts; times the expert computation the
way moe_fused.py's "ms" does (vLLM's call includes its own moe_align_block_size)."""
import json, math, sys, torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts


def reference(X, W13, W2, ids, wts):
    Tn, topk = ids.shape
    I = W2.shape[2]
    out = torch.zeros(Tn, X.shape[1], dtype=torch.float32, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].float() @ W13[e].float().T
        a = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (a @ W2[e].float().T) * wts[tok, kk, None])
    return out


def bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


H, I, E, topk = 2048, 512, 256, 8
torch.manual_seed(0)
W13 = (torch.randn(E, 2 * I, H, device="cuda") * (1 / math.sqrt(H))).half()
W2 = (torch.randn(E, H, I, device="cuda") * (1 / math.sqrt(I))).half()
print(json.dumps({"vllm": __import__("vllm").__version__, "torch": torch.__version__}), flush=True)
for Tn in [1, 16, 64, 256, 1024, 4096]:
    X = torch.randn(Tn, H, device="cuda").half()
    wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, device="cuda"), -1), topk, dim=-1)
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    R = reference(X, W13, W2, ids.int(), wts)
    r = {"impl": "vllm_fused_experts", "T": Tn, "experts_hit": int(ids.unique().numel())}
    try:
        run = lambda: fused_experts(X, W13, W2, wts, ids.int())
        O = run().float()
        r["rel_vs_fp32"] = round(((O - R).norm() / R.norm()).item(), 6)
        ms = bench(run)
        wbytes = r["experts_hit"] * 3 * I * H * 2
        r.update(ms=round(ms, 4), tflops=round(2 * Tn * topk * 3 * I * H / ms * 1e-9, 2), weight_gbps=round(wbytes / ms / 1e6, 1))
    except Exception as e:  # noqa
        r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
    print(json.dumps(r), flush=True)
