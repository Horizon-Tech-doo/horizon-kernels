"""E3 CK flash-attn + SDPA + GEMM in the vLLM image. Usage: run_ck.py <case> <tag> [--check]"""
import json, sys
import torch
import torch.nn.functional as F
from bench_common import bench
from cfgs_cases import CASES
case, tag = sys.argv[1], sys.argv[2]
check = "--check" in sys.argv
s = CASES[case]
B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
import flash_attn
from flash_attn import flash_attn_func
info = dict(torch=torch.__version__, flash_attn=flash_attn.__version__, arch=torch.cuda.get_device_properties(0).gcnArchName, hip=getattr(torch.version, "hip", None))
print(json.dumps(info), flush=True)
torch.manual_seed(0)
Q = torch.randn(B, L, Hkv * G, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
r = dict(name=case + "_ck", case=case, tag=tag, stack="vllm")
O = flash_attn_func(Q, K, V, causal=causal)
if check:
    q, k, v = Q.double(), K.double().repeat_interleave(G, 2), V.double().repeat_interleave(G, 2)
    s_ = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    if causal:
        n = s_.size(-1)
        s_ = s_.masked_fill(torch.ones(n, n, device=s_.device).tril() == 0, float("-inf"))
    Rf = torch.einsum("bhqk,bkhd->bqhd", F.softmax(s_, -1), v)
    r["rel_vs_fp64"] = round(((O.double() - Rf).norm() / Rf.norm()).item(), 6)
    del q, k, v, s_, Rf
del O
ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=100)
r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
print(json.dumps(r), flush=True)
qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal))
print(json.dumps(dict(name=case + "_sdpa_rocm72", case=case, tag=tag, stack="vllm", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
ms = bench(lambda: a @ b.T, warmup=3, rep=20)
print(json.dumps(dict(name="gemm8192_rocm72", case=case, tag=tag, stack="vllm", ms=round(ms, 4), tflops=round(2 * 8192**3 / ms * 1e-9, 2))), flush=True)
