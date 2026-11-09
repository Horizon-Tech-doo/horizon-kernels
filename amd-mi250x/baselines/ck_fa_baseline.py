"""Track 1 baseline: flash-attention 2.8.3 (ROCm build; Composable Kernel backend unless FLASH_ATTENTION_TRITON_AMD_ENABLE)
and aiter (if it loads on gfx90a) inside the vLLM 0.30 ROCm SIF, same shapes as gqa_fa.py / fa_bench.py.
Native GQA (k/v with fewer heads), causal where gqa_fa.py is causal. Correct vs float64 on <= 8k tokens."""
import json, os, torch
import torch.nn.functional as F


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


def ref64(Q, K, V, causal, G):
    q, k, v = Q.double(), K.double().repeat_interleave(G, 2), V.double().repeat_interleave(G, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    if causal:
        n = s.size(-1)
        s = s.masked_fill(torch.ones(n, n, device=s.device).tril() == 0, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(s, -1), v)


info = {"torch": torch.__version__, "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "triton_amd_env": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE")}
impls = {}
try:
    import flash_attn
    from flash_attn import flash_attn_func
    info["flash_attn"] = flash_attn.__version__
    impls["flash_attn"] = lambda q, k, v, c: flash_attn_func(q, k, v, causal=c)
except Exception as e:  # noqa
    info["flash_attn_err"] = str(e)[:200]
if os.environ.get("TRY_AITER"):   # 5 Nov: importing aiter on gfx90a hung past a 20-min job limit (JIT build?) -> opt-in only
    try:
        import aiter
        impls["aiter"] = lambda q, k, v, c: aiter.flash_attn_func(q, k, v, causal=c)
        info["aiter"] = "loaded"
    except Exception as e:  # noqa
        info["aiter_err"] = str(e)[:200]
print(json.dumps(info), flush=True)
torch.manual_seed(0)
cases = [("d256-gqa6-causal", 1, L, 4, 6, 256, True) for L in (1024, 4096, 8192, 16384)] + \
        [("d128-gqa6-causal", 1, L, 4, 6, 128, True) for L in (1024, 4096, 8192, 16384)] + [("d128-mha", 1, 4096, 8, 1, 128, False)]
for name, B, L, Hkv, G, D, causal in cases:
    Q = torch.randn(B, L, Hkv * G, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    R = ref64(Q, K, V, causal, G) if L <= 8192 else None
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    for iname, f in impls.items():
        r = {"case": name, "impl": iname, "L": L}
        try:
            O = f(Q, K, V, causal)
            if R is not None:
                r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
            ms = bench(lambda: f(Q, K, V, causal))
            r.update(ms=round(ms, 3), tflops=round(flops / ms * 1e-9, 1))
        except Exception as e:  # noqa
            r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
        print(json.dumps(r), flush=True)
