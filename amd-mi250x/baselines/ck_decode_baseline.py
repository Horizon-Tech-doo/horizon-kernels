"""Track 2 baseline: CK decode (flash-attn 2.8.3 flash_attn_with_kvcache, split-KV) in the vLLM 0.30 ROCm SIF, same
shapes as gqa_decode.py: Qwen GQA (24 q / 4 kv heads, d256), one new token, contiguous cache [B, L, Hkv, D] with
per-sequence lengths (cache_seqlens), plus a paged variant (block_table) at the page sizes the build accepts.
Bandwidth = K+V bytes actually read / time vs ~1,600 GB/s HBM per GCD."""
import json, math, torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache


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


def ref64(Q, K, V, lens, G):
    B, H, D = Q.shape
    out = torch.empty(B, H, D, dtype=torch.float64, device=Q.device)
    for b in range(B):
        n = int(lens[b])
        k = K[b, :n].double().repeat_interleave(G, 1); v = V[b, :n].double().repeat_interleave(G, 1)
        s = torch.einsum("hd,nhd->hn", Q[b].double(), k) / math.sqrt(D)
        out[b] = torch.einsum("hn,nhd->hd", F.softmax(s, -1), v)
    return out


torch.manual_seed(0)
Hkv, G, D = 4, 6, 256
H = Hkv * G
for B, L in [(1, 4096), (1, 32768), (1, 131072), (8, 4096), (8, 32768), (32, 8192)]:
    lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    if B > 1:
        lens[1::2] = L * 3 // 4
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    R = ref64(Q, K, V, lens, G) if B * L <= 32 * 8192 else None
    nbytes = 2 * 2 * Hkv * D * int(lens.sum())
    q = Q[:, None]                                        # [B, 1, H, D]
    variants = {"contiguous": lambda: flash_attn_with_kvcache(q, K, V, cache_seqlens=lens)}
    for page in (16, 256):
        npg = L // page
        kp = K.view(B * npg, page, Hkv, D)
        vp = V.view(B * npg, page, Hkv, D)
        bt = torch.arange(B * npg, device="cuda", dtype=torch.int32).view(B, npg)
        variants[f"paged{page}"] = (lambda kp=kp, vp=vp, bt=bt: flash_attn_with_kvcache(q, kp, vp, cache_seqlens=lens, block_table=bt))
    for name, fn in variants.items():
        r = {"impl": f"ck_flash_attn_{name}", "B": B, "L": L}
        try:
            O = fn()[:, 0]
            if R is not None:
                r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
            ms = bench(fn)
            r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), pct_hbm=round(100 * nbytes / ms / 1e6 / 1600, 1))
        except Exception as e:  # noqa
            r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
        print(json.dumps(r), flush=True)
