"""N3 round 4 last check on final bytes (scripts/gqa_decode.py), image stack, BF16, page 784, LBNHC, new seeds 964-971: the pick at Hkv 2 after its table entries and at Hkv 4 on the two headline cells; S5 on 16 x 4k Hkv 4, never timed. Arms kk (paged_kernels), pick, tri, plus S5 there. Ratio vLLM ms / ours ms. usage: n3r4d.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
G, D, PAGE, N, LAY = 6, 256, 16, 784, "lbnhc"
CELLS = [(2, 32, 1024), (2, 32, 2048), (2, 16, 4096), (2, 8, 8192), (2, 64, 1024), (2, 64, 2048), (2, 32, 4096), (2, 16, 8192), (4, 32, 1024), (4, 64, 1024), (4, 16, 4096)]
def arms(Hkv, B, L):
    a = (B, B * (-(-L // N)), L // PAGE, Hkv, G, D); kw = (PAGE, LAY, None, None, N, "bf16", "bits")
    return gq.paged_kernels(*a, *kw), gq.paged_picker(*a, *kw), (gq.paged_uniform(*a, 5, *kw) if (Hkv, B, L) == (4, 16, 4096) else None)
if compile_only:
    for i, c in enumerate(CELLS):
        if i % CN == CI: arms(*c)
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
def P(**k): print(json.dumps({"stack": stack, "dt": "bf16", **k}), flush=True)
for Hkv, B, L in CELLS:
    torch.manual_seed(5000 + B + L); MAXB = -(-L // N); NBLK = B * MAXB; H = Hkv * G
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(torch.bfloat16); knh, vnh = NH[..., :D], NH[..., D:]
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
    Kd = knh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; Vd = vnh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
    old, pick, s5 = arms(Hkv, B, L); scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    for sd in range(964, 972):
        g = torch.Generator(device="cuda"); g.manual_seed(sd); R = lambda lo: (lambda x: (x.__setitem__((x % PAGE) == 0, x[(x % PAGE) == 0] - 1), x)[1])(torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g))
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4; full = torch.full((B,), L, dtype=torch.int32, device="cuda")
        for mode, lens in ([("base", base), ("full", full)] if sd == 964 else []) + [("part", R(L // 2)), ("skew", R(1)), ("mid", R(L // 4)), ("near", R(L * 3 // 4))]:
            ks, mx = int(lens.sum()), int(lens.max()); kp = pick(ks, mx)
            A = dict(kk=lambda: old[1](*old[0](Q, NH, NH, BT, lens)), pick=lambda: kp[1](*kp[0](Q, NH, NH, BT, lens)))
            if s5: A["S5"] = lambda: s5[1](*s5[0](Q, NH, NH, BT, lens))
            def tri():
                unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None); return out
            A["tri"] = tri; names = list(A)
            if sd == 964 and mode in ("base", "skew"):
                Rf = gq.ref64(Q, Kd, Vd, lens, G)
                for n in names: P(t="err", part="D", Hkv=Hkv, B=B, L=L, N=N, mode=mode, impl=n, rel=float(((A[n]().double() - Rf).norm() / Rf.norm()).item()))
            for c in range(cycles):
                for ai in range(len(names)):
                    n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="seed", part="D", Hkv=Hkv, B=B, L=L, N=N, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, r=round(ks / (B * mx), 4), pct=round(Hkv * D * ks / ms / 1e6 / 4, 1))
    del Q, NH, knh, vnh, Kd, Vd, BT, old, pick, s5, A, kp; torch.cuda.empty_cache()
print("DONE", flush=True)
