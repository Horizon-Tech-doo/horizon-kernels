"""Track 2 baseline: vLLM 0.30's paged-attention decode ops on ROCm (paged_attention_v1/v2 from vllm._custom_ops, and
paged_attention_rocm if present), same data as gqa_decode.py's paged run: Qwen GQA (24 q / 4 kv heads, d256), page 16.
ROCM_ATTN's cache layout LHBNC, the default on gfx90a for a uniform model only (a hybrid model such as Qwen3.8 gets LBHNC with two head slots, page 784, BF16; see _kv_layout in gqa_decode.py): key [blocks, Hkv, D/x, page, x] (x = 8 for fp16), value [blocks, Hkv, D, page]. The custom op refuses head size 256,
so at this shape ROCM_ATTN decodes through chunked_prefill_paged_decode (evidence/N3/n3sk2.py); this script does not run that path."""
import inspect, json, math, torch
import torch.nn.functional as F
from vllm import _custom_ops as ops


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


names = [n for n in ("paged_attention_v1", "paged_attention_v2", "paged_attention_rocm") if hasattr(ops, n)]
unified = None
for mod in ("vllm.attention.ops.triton_unified_attention", "vllm.v1.attention.ops.triton_unified_attention",
            "vllm.v1.attention.backends.triton_unified_attention"):
    try:
        unified = __import__(mod, fromlist=["unified_attention"]).unified_attention
        print(json.dumps({"unified_attention": mod, "sig": str(inspect.signature(unified))}), flush=True)
        break
    except Exception as e:  # noqa
        pass
print(json.dumps({"ops": names, **{n: str(inspect.signature(getattr(ops, n))) for n in names}}), flush=True)
torch.manual_seed(0)
Hkv, G, D, PAGE = 4, 6, 256, 16
H, x = Hkv * G, 8
scale = 1.0 / math.sqrt(D)
one = torch.tensor(1.0, device="cuda", dtype=torch.float32)
for B, L in [(1, 4096), (1, 131072), (8, 4096), (8, 32768), (32, 8192), (64, 4096)]:
    lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    if B > 1:
        lens[1::2] = L * 3 // 4
    MAXP = L // PAGE
    NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    R = ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G) if B * L <= 32 * 8192 else None
    kcache = Kc.view(NBLK, PAGE, Hkv, D // x, x).permute(0, 2, 3, 1, 4).contiguous()
    vcache = Vc.permute(0, 2, 3, 1).contiguous()
    nbytes = 2 * 2 * Hkv * D * int(lens.sum())
    out = torch.empty_like(Q)
    part = 512
    nparts = (L + part - 1) // part
    exp_sums = torch.empty(B, H, nparts, device="cuda", dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(B, H, nparts, D, device="cuda", dtype=torch.float16)
    calls = {}
    if "paged_attention_v1" in names:
        calls["vllm_paged_v1"] = lambda: ops.paged_attention_v1(out, Q, kcache, vcache, Hkv, scale, BT, lens, PAGE, L, None, "auto", one, one)
    if "paged_attention_v2" in names:
        calls["vllm_paged_v2"] = lambda: ops.paged_attention_v2(out, exp_sums, max_logits, tmp_out, Q, kcache, vcache, Hkv, scale, BT, lens,
                                                                PAGE, L, None, "auto", one, one)
    if "paged_attention_rocm" in names:   # vLLM's custom ROCm decode kernel; partition 256 (vLLM's _PARTITION_SIZE_ROCM)
        np2 = (L + 255) // 256
        es2 = torch.empty(B, H, np2, device="cuda", dtype=torch.float32)
        ml2 = torch.empty_like(es2)
        to2 = torch.empty(B, H, np2, D, device="cuda", dtype=torch.float16)
        calls["vllm_paged_attention_rocm"] = lambda: ops.paged_attention_rocm(out, es2, ml2, to2, Q, kcache, vcache, Hkv, scale, BT, lens,
                                                                             None, PAGE, L, None, "auto", one, one)
    if unified is not None:   # vLLM's Triton unified attention (TRITON_ATTN, not the ROCm default), on [blocks, page, Hkv, D]
        q4 = Q.clone()                                       # [num_tokens=B, H, D]: one query token per sequence
        cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32)
        out_u = torch.empty_like(Q)
        calls["vllm_triton_unified"] = lambda: unified(q=q4, k=Kc, v=Vc, out=out_u, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=lens,
                                                       max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1),
                                                       block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
    for name, fn in calls.items():
        r = {"impl": name, "B": B, "L": L}
        try:
            fn()
            o_ = out_u if name == "vllm_triton_unified" else out
            if R is not None:
                r["rel_vs_fp64"] = round(((o_.double() - R).norm() / R.norm()).item(), 6)
            ms = bench(fn)
            r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), pct_hbm=round(100 * nbytes / ms / 1e6 / 1600, 1))
        except Exception as e:  # noqa
            r["err"] = (type(e).__name__ + ": " + str(e))[-240:]
        print(json.dumps(r), flush=True)
