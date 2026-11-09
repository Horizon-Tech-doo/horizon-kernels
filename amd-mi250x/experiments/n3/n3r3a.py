"""N3 round 3, step 1-4: landed on every vLLM layout against the vendor arm on the same tensors. Ratio: vLLM ms / ours ms, higher is better.
Image stack arms: land_b (LBHNC joint), land_bs (LBHNC K then V), land_n (LBNHC joint), land_ns (LBNHC K then V), land_r (LHBNC, K then V by shape), uni (unified_attention on LBHNC),
tri (unified_attention on LBNHC, TRITON_ATTN's default), roc (chunked_prefill_paged_decode on split_kv_cache views, ROCM_ATTN's default). LUMI stack: land_* only.
usage: n3r3a.py image|lumi ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
SH = [(8, 32768), (16, 16384), (32, 8192), (64, 4096), (64, 1024), (32, 1024)]
LANDS = {"land_b": (True, "joint"), "land_bs": (True, "k_v"), "land_n": ("lbnhc", "joint"), "land_ns": ("lbnhc", "k_v"), "land_r": ("lhbnc", "joint")}
if compile_only:
    i = 0
    for B, L in SH:
        for n, (lay, st) in LANDS.items():
            if i % CN == CI:
                MAXP = L // PAGE; gq.paged_kernels(B, B * MAXP, MAXP, Hkv, G, D, PAGE, lay, st)
            i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
if stack == "image":
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as chunked
    from vllm.v1.attention.ops.paged_attn import PagedAttention
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
def P(**k): print(json.dumps({"stack": stack, **k}), flush=True)
def rel(o, R): return float(((o.double() - R).norm() / R.norm()).item())
for B, L in SH:
    torch.manual_seed(5000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    KV = torch.randn(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    t = KV.transpose(1, 2); kv, vv = t[..., :D], t[..., D:]   # logical [NBLK, PAGE, Hkv, D]
    Kd = kv[BT.long()].reshape(B, L, Hkv, D); Vd = vv[BT.long()].reshape(B, L, Hkv, D)
    C2 = torch.empty(2, NBLK, PAGE, Hkv * D, device="cuda", dtype=torch.float16)
    if stack == "image":
        kc2, vc2 = PagedAttention.split_kv_cache(C2, Hkv, D)
    else:
        kc2 = C2[0].view(NBLK, Hkv, D // 8, PAGE, 8); vc2 = C2[1].view(NBLK, Hkv, D, PAGE)
    kc2.copy_(kv.reshape(NBLK, PAGE, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); vc2.copy_(vv.permute(0, 2, 3, 1))
    assert kc2.is_contiguous() and vc2.is_contiguous()
    NH = torch.empty(NBLK, PAGE, Hkv, 2 * D, device="cuda", dtype=torch.float16); NH[..., :D] = kv; NH[..., D:] = vv
    knh, vnh = NH[..., :D], NH[..., D:]
    P(t="meta", B=B, L=L, kshape=list(kc2.shape), vshape=list(vc2.shape), k_off=kc2.storage_offset(), v_off=vc2.storage_offset())
    KS = {}
    for n, (lay, st) in LANDS.items(): KS[n] = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, st)
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    def lm(n, lens):
        k1, k2 = KS[n]; a, b = {"land_b": (KV, KV), "land_bs": (KV, KV), "land_n": (NH, NH), "land_ns": (NH, NH), "land_r": (kc2, vc2)}[n]
        return lambda: k2(*k1(Q, a, b, BT, lens))
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
    base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        A = {n: lm(n, lens) for n in LANDS}
        if stack == "image": A.update(uni=uni(kv, vv, lens), tri=uni(knh, vnh, lens), roc=roc(lens))
        names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    # stale NaN past each sequence's end, in all three layouts: landed forms must not see it
    if (B, L) in [(32, 1024), (64, 4096)]:
        lens = skew; ok = torch.ones(NBLK, PAGE, device="cuda", dtype=torch.bool)
        for b in range(B):
            n = int(lens[b]); pg = n // PAGE
            if n % PAGE: ok[BT[b, pg].long(), n % PAGE:] = False
        ks, vs = KV.clone(), NH.clone(); kk, vv2 = kc2.clone(), vc2.clone()
        blk, sl = (~ok).nonzero(as_tuple=True)
        KV[blk, :, sl] = float("nan"); NH[blk, sl] = float("nan")
        kc2[blk, :, :, sl] = float("nan"); vc2[blk, :, :, sl] = float("nan")
        R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in LANDS: P(t="nan_tail", B=B, L=L, impl=n, rel=rel(lm(n, lens)(), R))
        KV.copy_(ks); NH.copy_(vs); kc2.copy_(kk); vc2.copy_(vv2); del ks, vs, kk, vv2
    del KV, t, kv, vv, Kd, Vd, KS, C2, kc2, vc2, NH, knh, vnh, A; torch.cuda.empty_cache()
print("DONE", flush=True)
