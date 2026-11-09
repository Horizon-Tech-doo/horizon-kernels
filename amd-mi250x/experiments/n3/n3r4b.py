"""N3 round 4: form grid on BF16, pc="bits", page 784, LBNHC, image stack, six shapes; six length patterns for the pick refit (seeds 940-947 are the fit set). Forms: kk is paged_kernels, Sn is uniform n splits with swap and rc.
Ratio vLLM ms / ours ms. usage: n3r4b.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE, N, DT, PC = 4, 6, 256, 16, 784, "bf16", "bits"; H = Hkv * G; LAY = "lbnhc"
GRID = {(32, 1024): [3, 4, 5], (32, 2048): [3, 4, 5, 6, 8], (64, 1024): [3, 4, 5, 6], (64, 2048): [3, 4, 5, 6, 8], (16, 4096): [6, 8, 10, 12, 16], (8, 8192): [10, 12, 16, 20, 24]}
def mk(B, L, f):
    MAXP = L // PAGE; NBLK = B * (-(-L // N))
    if f == "kk": return gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT, PC)
    return gq.paged_uniform(B, NBLK, MAXP, Hkv, G, D, f, PAGE, LAY, None, None, N, DT, PC)
if compile_only:
    jobs = [(B, L, f) for (B, L), Fs in GRID.items() for f in Fs + ["kk"]]
    for i, (B, L, f) in enumerate(jobs):
        if i % CN == CI: mk(B, L, f)
    print("COMPILED", CI, flush=True); sys.exit(0)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
def P(**k): print(json.dumps({"stack": stack, "dt": DT, **k}), flush=True)
for (B, L), Fs in GRID.items():
    torch.manual_seed(5000 + B + L); MAXB = -(-L // N); NBLK = B * MAXB
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(torch.bfloat16); knh, vnh = NH[..., :D], NH[..., D:]
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
    Kd = knh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; Vd = vnh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
    KK = {("S%d" % f if f != "kk" else "kk"): mk(B, L, f) for f in Fs + ["kk"]}
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    for sd in range(940, 948):
        g = torch.Generator(device="cuda"); g.manual_seed(sd); R = lambda lo: (lambda x: (x.__setitem__((x % PAGE) == 0, x[(x % PAGE) == 0] - 1), x)[1])(torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g))
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
        full = torch.full((B,), L, dtype=torch.int32, device="cuda")
        draws = ([("base", base), ("full", full)] if sd == 940 else []) + [("part", R(L // 2)), ("skew", R(1)), ("mid", R(L // 4)), ("near", R(L * 3 // 4))]
        for mode, lens in draws:
            ks, mx = int(lens.sum()), int(lens.max())
            A = {n: (lambda k=k, lens=lens: k[1](*k[0](Q, NH, NH, BT, lens))) for n, k in KK.items()}
            def tri(lens=lens):
                unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None); return out
            A["tri"] = tri; names = list(A)
            if sd == 940 and mode in ("base", "skew"):
                Rf = gq.ref64(Q, Kd, Vd, lens, G)
                for n in names: P(t="err", B=B, L=L, N=N, mode=mode, impl=n, rel=float(((A[n]().double() - Rf).norm() / Rf.norm()).item()))
            for c in range(cycles):
                for ai in range(len(names)):
                    n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="seed", B=B, L=L, N=N, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, r=round(ks / (B * mx), 4), pct=round(Hkv * D * ks / ms / 1e6 / 4, 1))
    del Q, NH, knh, vnh, Kd, Vd, BT, KK, A; torch.cuda.empty_cache()
print("DONE", flush=True)
