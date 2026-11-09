"""N3 round 3: load forms on ROCM_ATTN's own layout LHBNC. Arms: r_jx (K thread order j then x, K then V), r_xj (K in address order, K then V), r_xj_vk (K in address order, V then K), r_jx_vk; with roc and n (landed on LBNHC) on the same data for scale.
Ratio vLLM ms / ours ms. usage: n3r3d.py image|lumi ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
SH = [(8, 32768), (32, 8192), (64, 4096), (64, 1024)]
F = {"r_jx": ("joint", "jx"), "r_xj": ("joint", "xj"), "r_jx_vk": ("v_k", "jx"), "r_xj_vk": ("v_k", "xj")}
if compile_only:
    i = 0
    for B, L in SH:
        for n, (st, ko) in F.items():
            if i % CN == CI: MAXP = L // PAGE; gq.paged_kernels(B, B * MAXP, MAXP, Hkv, G, D, PAGE, "lhbnc", st, ko)
            i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
if stack == "image":
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as chunked
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
    C2 = torch.randn(2, NBLK, PAGE, Hkv * D, device="cuda", dtype=torch.float16)
    kc2 = C2[0].view(NBLK, Hkv, D // 8, PAGE, 8); vc2 = C2[1].view(NBLK, Hkv, D, PAGE)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    kv = kc2.permute(0, 3, 1, 2, 4).reshape(NBLK, PAGE, Hkv, D); vv = vc2.permute(0, 3, 1, 2)
    Kd = kv[BT.long()].reshape(B, L, Hkv, D); Vd = vv[BT.long()].reshape(B, L, Hkv, D)
    NH = torch.cat([kv, vv], -1)
    KS = {n: gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, "lhbnc", st, ko) for n, (st, ko) in F.items()}
    KN = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, "lbnhc", "joint")
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    for mode, lens in [("base", base), ("full", full), ("skew", skew)]:
        A = {n: (lambda n_: (lambda: KS[n_][1](*KS[n_][0](Q, kc2, vc2, BT, lens))))(n) for n in F}
        A["n"] = lambda: KN[1](*KN[0](Q, NH, NH, BT, lens))
        if stack == "image":
            A["roc"] = lambda: (chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True), out)[1]
        names = list(A); R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4))
    del C2, kc2, vc2, kv, vv, Kd, Vd, NH, KS, KN, A; torch.cuda.empty_cache()
print("DONE", flush=True)
