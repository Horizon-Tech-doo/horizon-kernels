"""N3 round 3, step 6, second grid, more uniform forms and the 1 GB shapes: split forms at short shapes on layout LBNHC (TRITON_ATTN's default), against tri and roc on the same data. Eight length draws per pattern.
Ratio: vLLM ms / ours ms, higher is better. Seeds 900-907 pick forms, 908-915 are held out to test a pick rule (recorded so the rule can be fit afterwards).
Also times k1 and k2 alone and a floor with all lengths 1. usage: n3r3c.py image|lumi ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G; LAY = "lbnhc"
def tp(S1, A, a, MAXP): return S1 + max(0, -(-(MAXP - S1 * A) // a))
FORMS = {64: [("t1_44_12", (1, 44, 12)), ("u16", 16), ("u22", 22), ("u32", 32), ("u64", 64), ("t2_24_8", (2, 24, 8))],
         128: [("t2_44_12", (2, 44, 12)), ("u32", 32), ("u43", 43), ("u64", 64), ("t4_24_8", (4, 24, 8))],
         256: [("v3", "v3"), ("u64", 64), ("u32", 32), ("u16", 16), ("t2_64_16", (2, 64, 16))],
         512: [("v3", "v3"), ("u128", 128), ("u64", 64), ("u32", 32), ("t2_128_32", (2, 128, 32))]}
SH = [(64, 1024), (64, 2048), (32, 1024), (32, 2048), (64, 4096), (32, 8192)]
def make(B, MAXP, spec_, noswap=False):
    NBLK = B * MAXP
    if spec_ == "v3":
        k1, k2 = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY); return k1, k2, 0
    if isinstance(spec_, tuple):
        S = tp(*spec_, MAXP); k1 = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", True, True, spec_, "kvg", LAY)
    else:
        S = -(-MAXP // spec_); k1 = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", True, True, None, "kvg", LAY)
    return k1, gq.decode_combine_v2(B, H, S, D), S
if compile_only:
    i = 0
    for B, L in SH:
        for n, f in FORMS[L // PAGE]:
            if i % CN == CI: make(B, L // PAGE, f)
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
for B, L in SH:
    torch.manual_seed(5000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    NH = torch.randn(NBLK, PAGE, Hkv, 2 * D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    knh, vnh = NH[..., :D], NH[..., D:]
    KS = {n: make(B, MAXP, f) for n, f in FORMS[MAXP]}
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    A0 = {}
    def fm(n, lens):
        k1, k2, S = KS[n]
        return lambda: k2(*k1(Q, NH, NH, BT, lens))
    Kd = knh[BT.long()].reshape(B, L, Hkv, D); Vd = vnh[BT.long()].reshape(B, L, Hkv, D)
    if stack == "image":
        C2 = torch.empty(2, NBLK, PAGE, Hkv * D, device="cuda", dtype=torch.float16); kc2, vc2 = PagedAttention.split_kv_cache(C2, Hkv, D)
        kc2.copy_(knh.reshape(NBLK, PAGE, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); vc2.copy_(vnh.permute(0, 2, 3, 1))
        def tri(lens):
            def u():
                unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
                return out
            return u
        def roc(lens):
            def r():
                chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True)
                return out
            return r
    one = torch.ones(B, dtype=torch.int32, device="cuda")
    A = {n: fm(n, one) for n in KS}
    if stack == "image": A["tri"] = tri(one)
    for n, f in A.items(): P(t="floor", B=B, L=L, impl=n, S=KS[n][2] if n in KS else 0, ms=round(bench(f), 4))
    for sd in range(8):
        for mode, lo in [("skew", 1), ("part", L // 2)]:
            g = torch.Generator(device="cuda"); g.manual_seed(900 + sd + (1000 if mode == "part" else 0))
            if sd < 8 and mode == "skew":   # same draws as the skeptic: seeds 900-907, skew drawn after a part draw
                g = torch.Generator(device="cuda"); g.manual_seed(900 + sd); torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32, generator=g)
            lens = torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g); lens[(lens % PAGE) == 0] -= 1
            A = {n: fm(n, lens) for n in KS}
            if stack == "image": A["tri"] = tri(lens); A["roc"] = roc(lens)
            names = list(A); pg = (lens + PAGE - 1) // PAGE
            if sd == 0:
                R = gq.ref64(Q, Kd, Vd, lens, G)
                for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
            for c in range(cycles):
                for ai in range(len(names)):
                    n = names[(ai + c + ROT) % len(names)]; P(t="seed", B=B, L=L, seed=900 + sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(bench(A[n]), 4), sum=int(lens.sum()), max=int(lens.max()), pages=int(pg.sum()))
            if sd == 0:   # k1 and k2 apart, one draw
                for n, (k1, k2, S) in KS.items():
                    Op, Lse = k1(Q, NH, NH, BT, lens)
                    P(t="split", B=B, L=L, mode=mode, impl=n, S=S, k1=round(bench(lambda: k1(Q, NH, NH, BT, lens)), 4), k2=round(bench(lambda: k2(Op, Lse)), 4))
    del NH, knh, vnh, KS, Kd, Vd, A
    if stack == "image": del C2, kc2, vc2
    torch.cuda.empty_cache()
print("DONE", flush=True)
