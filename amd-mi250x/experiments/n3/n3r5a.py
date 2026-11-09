"""N3 round 5, working copy exp/N3/gqa_decode.py (dead-split skip added), image stack, page 784, LBNHC. Compiler output, no hand edits. Ratio vLLM ms / ours ms.
Part A: S (current uniform) vs X (dead="skip") vs tri, BF16. 64 x 1k S4 skew seeds 964-979; controls 32 x 1k S4, 64 x 2k S6 skew seeds 972-979; base, full on all three. err on every arm, stale-NaN tail, lens=1, split-boundary lengths.
Part F: FP16 32 x 1k base and full: kk, S5, tri.
Part T: held-out seeds 972-979, BF16: uniform forms, kk, tri per cell; table pick and block-rounds rule are read off the forms' timings, logged per draw.
usage: n3r5a.py image ROT cycles [compile I N]; env N3PARTS (default AFT), N3REV=1 reverses arm order."""
import json, math, os, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
PARTS = os.environ.get("N3PARTS", "AFT"); REV = os.environ.get("N3REV") == "1"
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
G, D, PAGE, N, LAY = 6, 256, 16, 784, "lbnhc"
ACELL = [(64, 1024, 4), (32, 1024, 4), (64, 2048, 6)]
TCELL = {(4, 8, 8192): [10, 12, 16, 20, 24], (4, 16, 4096): [5, 6, 8, 10], (4, 32, 1024): [3, 4, 5], (4, 32, 2048): [3, 4, 5, 6, 10], (4, 64, 1024): [3, 4, 5], (4, 64, 2048): [4, 5, 6],
         (2, 8, 32768): [20, 40], (2, 32, 8192): [5, 10], (2, 16, 8192): [10, 12, 20, 24]}
def nb(B, L): return B * (-(-L // N))
def un(Hkv, B, L, S, dt="bf16", dead="keep"): return gq.paged_uniform(B, nb(B, L), L // PAGE, Hkv, G, D, S, PAGE, LAY, None, None, N, dt, "bits", dead)
def kk(Hkv, B, L, dt="bf16"): return gq.paged_kernels(B, nb(B, L), L // PAGE, Hkv, G, D, PAGE, LAY, None, None, N, dt, "bits")
if compile_only:
    jobs = [("A", c) for c in ACELL] + [("F", None)] + [("T", c) for c in TCELL]
    for i, (w, c) in enumerate(jobs):
        if i % CN != CI: continue
        if w == "A": un(4, c[0], c[1], c[2]); un(4, c[0], c[1], c[2], "bf16", "skip")
        elif w == "F": kk(4, 32, 1024, "fp16"); un(4, 32, 1024, 5, "fp16")
        else:
            kk(*c)
            for S in TCELL[c]: un(*c, S)
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
def P(**k): print(json.dumps({"stack": stack, **k}), flush=True)
class Data:
    def __init__(s, B, L, Hkv, dt=torch.bfloat16):
        torch.manual_seed(5000 + B + L); s.B, s.L, s.Hkv = B, L, Hkv; s.MAXB = -(-L // N); NBLK = B * s.MAXB
        s.Q = torch.randn(B, Hkv * G, D, device="cuda", dtype=torch.float32).to(dt)
        s.NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(dt); s.knh, s.vnh = s.NH[..., :D], s.NH[..., D:]
        s.BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, s.MAXB)
        s.scale = 1.0 / math.sqrt(D); s.cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); s.out = torch.empty_like(s.Q)
    def ref(s, lens):
        Kd = s.knh[s.BT.long()].reshape(s.B, s.MAXB * N, s.Hkv, D)[:, :s.L]; Vd = s.vnh[s.BT.long()].reshape(s.B, s.MAXB * N, s.Hkv, D)[:, :s.L]; return gq.ref64(s.Q, Kd, Vd, lens, G)
    def lens(s, sd):
        B, L = s.B, s.L; g = torch.Generator(device="cuda"); g.manual_seed(sd); R = lambda lo: (lambda x: (x.__setitem__((x % PAGE) == 0, x[(x % PAGE) == 0] - 1), x)[1])(torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g))
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
        return dict(base=base, full=torch.full((B,), L, dtype=torch.int32, device="cuda"), part=R(L // 2), skew=R(1), mid=R(L // 4), near=R(L * 3 // 4))
    def tri(s, lens):
        return lambda: (unified(q=s.Q, k=s.knh, v=s.vnh, out=s.out, cu_seqlens_q=s.cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=s.L, softmax_scale=s.scale, causal=True, window_size=(-1, -1), block_table=s.BT, softcap=0, q_descale=None, k_descale=None, v_descale=None), s.out)[1]
    def ours(s, k, lens, nh=None): nh = s.NH if nh is None else nh; return lambda: k[1](*k[0](s.Q, nh, nh, s.BT, lens))
    def live(s, k, lens, nh=None): nh = s.NH if nh is None else nh; return lambda: k[1](*k[0](s.Q, nh, nh, s.BT, lens), lens)
    def poison(s, lens):
        C = s.NH.clone()
        for b in range(s.B):
            pos = torch.arange(int(lens[b]), s.MAXB * N, device="cuda"); C[s.BT[b, pos // N].long(), pos % N] = float("nan")
        return C
def blocks(d, lens, S):
    pps = -(-(d.L // PAGE) // S); pg = torch.div(lens + PAGE - 1, PAGE, rounding_mode="floor"); return d.Hkv * int(torch.div(pg + pps - 1, pps, rounding_mode="floor").sum()), pps
def rule(d, lens, forms, C=330):
    best, bt = None, None
    for S in forms:
        n, pps = blocks(d, lens, S); t = -(-n // C) * pps + 1e-3 * S
        if bt is None or t < bt: best, bt = S, t
    return best
def race(part, d, mode, sd, lens, A, cyc, check, dt="bf16", **extra):
    names = list(A); names = names[::-1] if REV else names; ks, mx = int(lens.sum()), int(lens.max())
    if check:
        R = d.ref(lens)
        for n in names: P(t="err", part=part, dt=dt, Hkv=d.Hkv, B=d.B, L=d.L, mode=mode, seed=sd, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
    for c in range(cyc):
        for ai in range(len(names)):
            n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n])
            P(t="seed", part=part, dt=dt, Hkv=d.Hkv, B=d.B, L=d.L, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, r=round(ks / (d.B * mx), 4), **extra)
if "A" in PARTS:
    for B, L, S in ACELL:
        d = Data(B, L, 4); k, x = un(4, B, L, S), un(4, B, L, S, "bf16", "skip")
        for sd in (range(964, 980) if (B, L) == (64, 1024) else range(972, 980)):
            Ls = d.lens(sd)
            for mode in (["base", "full"] if sd == 972 else []) + ["skew"]:
                lens = Ls[mode]; race("A", d, mode, sd, lens, dict(cur=d.ours(k, lens), skip=d.live(x, lens), tri=d.tri(lens)), cycles, sd == 972, S=S, blk=blocks(d, lens, S)[0])
        if ROT == 0 and not REV:   # correctness of the skip path on adversarial inputs
            one = torch.ones(B, dtype=torch.int32, device="cuda"); mix = torch.full((B,), L - 1, dtype=torch.int32, device="cuda"); mix[::2] = 1
            pps = -(-(L // PAGE) // S) * PAGE; bd = torch.tensor([pps, pps + 1, 2 * pps, 2 * pps + 1, pps - 1, 1, L, L - 1], device="cuda", dtype=torch.int32).repeat(B // 8 + 1)[:B]
            sk = d.lens(972)["skew"]
            for nm, lens in (("one", one), ("mix1", mix), ("bound", bd), ("skew", sk)):
                R = d.ref(lens); C = d.poison(lens)
                for imp, f in (("cur", d.ours(k, lens, C)), ("skip", d.live(x, lens, C)), ("skip_clean", d.live(x, lens))):
                    o = f(); P(t="err", part="N", dt="bf16", Hkv=4, B=B, L=L, mode=nm, impl=imp, nan=bool(o.isnan().any()), rel=float(((o.double() - R).norm() / R.norm()).item()))
            del C
        del d, k, x; torch.cuda.empty_cache()
if "F" in PARTS:
    d = Data(32, 1024, 4, torch.float16); a, b = kk(4, 32, 1024, "fp16"), un(4, 32, 1024, 5, "fp16"); Ls = d.lens(972)
    for mode in ("base", "full"):
        race("F", d, mode, 972, Ls[mode], dict(kk=d.ours(a, Ls[mode]), S5=d.ours(b, Ls[mode]), tri=d.tri(Ls[mode])), max(cycles, 5), True, "fp16")
    del d, a, b; torch.cuda.empty_cache()
if "T" in PARTS:
    for (Hkv, B, L), forms in TCELL.items():
        d = Data(B, L, Hkv); K = kk(Hkv, B, L); Fm = {S: un(Hkv, B, L, S) for S in forms}
        tb = gq._PICK_BF16_784.get((B * Hkv, L // PAGE))
        for sd in range(972, 980):
            Ls = d.lens(sd)
            for mode in (["base", "full"] if sd == 972 else []) + ["part", "skew", "mid", "near"]:
                lens = Ls[mode]; ks, mx = int(lens.sum()), int(lens.max()); tS = next(S for lo, S in tb if ks >= lo * B * mx) if tb else 0
                A = dict(kk=d.ours(K, lens), tri=d.tri(lens)); A.update({f"S{S}": d.ours(k, lens) for S, k in Fm.items()})
                race("T", d, mode, sd, lens, A, cycles, sd == 972 and mode in ("base", "skew"), table=tS, rule=rule(d, lens, forms), blk={str(S): blocks(d, lens, S)[0] for S in forms})
        del d, K, Fm, A; torch.cuda.empty_cache()
print("DONE", flush=True)
