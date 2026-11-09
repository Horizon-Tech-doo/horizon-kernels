"""TileLang AMD flash-attention fwd on gfx90a: correctness vs float64 (not the example's fp16 reference, which drifts
4% at 4k on ROCm) and speed vs torch SDPA, for d128 (non-causal) and d256 GQA causal (Qwen full-attention shape).
Tiles chosen to fit MI250X's 64 KB LDS: Q(bM x D) + K(bN x D) + V(bN x D) fp16."""
import json, os, re, runpy, sys, torch
import torch.nn.functional as F
src = sys.argv[1]
code = open(os.path.join(src, "examples/amd/example_amd_flash_attn_fwd.py")).read()
code, n = re.subn(r"^@tilelang\.autotune\(.*\)\n", "", code, flags=re.M); assert n == 1
dst = os.path.abspath("fa_fwd_noautotune.py"); open(dst, "w").write(code)
mod = runpy.run_path(dst, run_name="fa_example")
import tilelang
from tilelang.profiler import do_bench

def ref64(Q, K, V, causal, groups):
    q, k, v = Q.double(), K.double().repeat_interleave(groups, 2), V.double().repeat_interleave(groups, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    if causal:
        L = s.size(-1); s = s.masked_fill(torch.ones(L, L, device=s.device).tril() == 0, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(s, -1), v)

torch.manual_seed(0)
cases = [("d128", 1, 8, 4096, 128, False, 1), ("d256-gqa6-causal", 1, 24, 4096, 256, True, 6)]
for name, B, H, L, D, causal, g in cases:
    Q = torch.randn(B, L, H, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, L, H // g, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, L, H // g, D, device="cuda", dtype=torch.float16)
    R = ref64(Q, K, V, causal, g)
    flops = 4.0 * B * H * L * L * D * (0.5 if causal else 1.0)
    qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(g, 1), V.transpose(1, 2).repeat_interleave(g, 1)
    sd = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    so = sd().transpose(1, 2)
    sms = do_bench(sd, warmup=20, rep=100)
    print(json.dumps({"case": name, "impl": "torch_sdpa", "rel_vs_fp64": round(((so.double() - R).norm() / R.norm()).item(), 6),
                      "ms": round(sms, 3), "tflops": round(flops / sms * 1e-9, 1)}), flush=True)
    tiles = [(64, 64, 256, 2), (64, 32, 256, 2), (128, 32, 256, 2), (128, 64, 256, 2)] if D == 128 else \
            [(32, 32, 128, 2), (64, 16, 256, 2), (32, 32, 256, 2), (32, 16, 128, 2), (64, 16, 128, 2), (16, 32, 64, 2)]
    for bm, bn, th, kp in tiles:
        lds = (bm + 2 * bn) * D * 2
        r = {"case": name, "impl": "tilelang", "bM": bm, "bN": bn, "threads": th, "k_pack": kp, "lds_kb": lds // 1024}
        try:
            k = mod["fast_flashattn"](B, H, L, D, causal, g, block_M=bm, block_N=bn, num_split_q=128, threads=th, num_stages=0,
                                      enable_rasterization=True, k_pack=kp, panel_size=7, qk_coalesced_width=8, v_coalesced_width=4)
            O = k(Q, K, V)
            r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
            ms = do_bench(lambda: k(Q, K, V), warmup=20, rep=100)
            r.update(ms=round(ms, 3), tflops=round(flops / ms * 1e-9, 1), vs_sdpa=round(sms / ms, 3))
        except Exception as e:  # noqa
            r["err"] = (type(e).__name__ + ": " + str(e))[-160:]
        print(json.dumps(r), flush=True)
