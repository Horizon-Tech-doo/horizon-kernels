"""N3 round 2 skeptic. vLLM 0.30's default decode on MI250X vs the path round 2 benchmarked, same data, same job. Image stack only (vLLM exists only there).
Arms: landed (paged_kernels on [NBLK, Hkv, PAGE, 2D], LBHNC); vllm_p (unified_attention on the same tensor, the report's vendor arm);
vllm_rocm (ROCM_ATTN's path: chunked_prefill_paged_decode on split_kv_cache views of (2, NBLK, PAGE, Hkv*D), K [NBLK, Hkv, D/8, PAGE, 8], V [NBLK, Hkv, D, PAGE]);
vllm_tri (unified_attention on physical [NBLK, PAGE, Hkv, 2D], LBNHC, TRITON_ATTN's default); landed_s and vllm_s as in n3r2v.py (separate K, V, 64k gap).
usage: n3sk2.py ROT cycles. Ratio convention: vLLM ms / ours ms, higher is better."""
import json, math, sys, importlib.util, torch
ROT, cycles = int(sys.argv[1]), int(sys.argv[2]); sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as chunked, has_native_kv_cache_layout
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.platforms.rocm import use_rocm_custom_paged_attention
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
def P(**k): print(json.dumps({"stack": "image", **k}), flush=True)
def rel(o, R): return float(((o.double() - R).norm() / R.norm()).item())
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
MAIN = [(8, 32768), (16, 16384), (32, 8192), (64, 4096)]
SH = MAIN + [(64, 1024), (32, 1024)]
SEEDS = [(64, 1024), (32, 1024), (64, 2048)]
print("gfx", torch.cuda.get_device_properties(0).name, "custom_256", use_rocm_custom_paged_attention(torch.float16, 256, 16, G, 131072, 0, "auto"), "custom_128", use_rocm_custom_paged_attention(torch.float16, 128, 16, G, 131072, 0, "auto"), flush=True)
for B, L in SH + [s for s in SEEDS if s not in SH]:
    torch.manual_seed(5000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    KV = torch.randn(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    t = KV.transpose(1, 2); kv, vv = t[..., :D], t[..., D:]   # logical [NBLK, PAGE, Hkv, D]
    Kd = kv[BT.long()].reshape(B, L, Hkv, D); Vd = vv[BT.long()].reshape(B, L, Hkv, D)
    # ROCM_ATTN physical cache: K group then V group
    C2 = torch.empty(2, NBLK, PAGE, Hkv * D, device="cuda", dtype=torch.float16)
    kc2, vc2 = PagedAttention.split_kv_cache(C2, Hkv, D)
    kc2.copy_(kv.reshape(NBLK, PAGE, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); vc2.copy_(vv.permute(0, 2, 3, 1))
    # TRITON_ATTN default, LBNHC: physical [NBLK, PAGE, Hkv, 2D]
    NH = torch.empty(NBLK, PAGE, Hkv, 2 * D, device="cuda", dtype=torch.float16); NH[..., :D] = kv; NH[..., D:] = vv
    knh, vnh = NH[..., :D], NH[..., D:]
    sep = None
    if (B, L) in MAIN:
        N = kv.numel(); GAP = 65536; big = torch.empty(2 * N + GAP // 2, dtype=torch.float16, device="cuda")
        Ks = big[:N].view(kv.shape); Vs = big[N + GAP // 2:].view(kv.shape); Ks.copy_(kv); Vs.copy_(vv); sep = (Ks, Vs)
    P(t="meta", B=B, L=L, native_layout=has_native_kv_cache_layout(kc2, vc2), kshape=list(kc2.shape), vshape=list(vc2.shape))
    k1, k2 = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE)
    k1s, k2s = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, False) if sep else (None, None)
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    def uni(k, v, lens):
        def u():
            unified(q=Q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out
        return u
    def roc(lens):
        def r():
            chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True)
            return out
        return r
    base = torch.full((B,), L, dtype=torch.int32, device="cuda")
    if B > 1: base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    cases = [("base", base), ("full", full), ("part", part), ("skew", skew)] if (B, L) in SH else []
    for mode, lens in cases:
        A = {"landed": (lambda ll: (lambda: k2(*k1(Q, KV, KV, BT, ll))))(lens), "vllm_p": uni(kv, vv, lens), "vllm_rocm": roc(lens), "vllm_tri": uni(knh, vnh, lens)}
        if sep:
            A["landed_s"] = (lambda ll: (lambda: k2s(*k1s(Q, sep[0], sep[1], BT, ll))))(lens); A["vllm_s"] = uni(sep[0], sep[1], lens)
        names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    # length resampling at the disputed short shapes: same KV and table, new lengths
    if (B, L) in SEEDS:
        for sd in range(8):
            g = torch.Generator(device="cuda"); g.manual_seed(900 + sd)
            for mode, lo in [("part", L // 2), ("skew", 1)]:
                lens = torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g); lens[(lens % PAGE) == 0] -= 1
                A = {"landed": (lambda ll: (lambda: k2(*k1(Q, KV, KV, BT, ll))))(lens), "vllm_p": uni(kv, vv, lens), "vllm_rocm": roc(lens), "vllm_tri": uni(knh, vnh, lens)}
                names = list(A)
                for c in range(cycles):
                    for ai in range(len(names)):
                        n = names[(ai + c + ROT) % len(names)]; P(t="seed", B=B, L=L, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(bench(A[n]), 4), sum=int(lens.sum()))
    del KV, t, kv, vv, Kd, Vd, sep, k1, k2, C2, kc2, vc2, NH, knh, vnh; big = Ks = Vs = None; torch.cuda.empty_cache()
print("DONE", flush=True)
