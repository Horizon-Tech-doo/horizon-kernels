"""N3 round 2, part C, real vLLM layout only. (1) Short caches: paged_v3_config vs paged_v2_config vs uniform S, against vLLM. (2) Batch 16 uniform v3. (3) Slow mode in the
real layout: re-allocate the one KV tensor 10 times, then keep it and load new code objects. usage: n3r2c.py image|lumi TAIL cycles [compile I N]. Ratio convention: vLLM ms / ours ms."""
import json, math, os, shutil, sys, importlib.util, torch
stack, TAIL, cycles = sys.argv[1], sys.argv[2], int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
E = "/scratch/project_XXXXXXXXX/tilelang/exp/N3"
def load():
    spec = importlib.util.spec_from_file_location("gq", E + "/gqa_decode.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gq = load()
unified = __import__("vllm.v1.attention.ops.triton_unified_attention", fromlist=["unified_attention"]).unified_attention if stack == "image" and not compile_only else None
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
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
SH = [(32, 1024), (64, 1024), (16, 2048), (32, 2048), (64, 2048), (128, 2048), (8, 4096), (1, 131072), (16, 16384)]
def cfg(B, L, g=None):
    g = g or gq; MAXP = L // PAGE
    v2 = lambda S, taper, swap: ("v2", S, dict(taper=taper, swap=swap))
    v3 = lambda S1, S2, fr=86: ("v3", S1 + S2, dict(S1=S1, S2=S2, frac=fr))
    Sv, tp = g.paged_v2_config(MAXP); c = {"v2p": v2(Sv, tp, True)}
    a, b, fr = g.paged_v3_config(B, MAXP); c["v3c"] = v3(a, b, fr)
    if (B, L) == (16, 16384):
        c["u27p"] = v2(27, None, False); c["v3u27"] = v3(26, 1, 100); c["v3_24_4"] = v3(24, 4, 92)
    elif (B, L) in [(32, 1024), (64, 1024)]:
        for S in (4, 8, 16): c[f"u{S}"] = v2(S, None, True)
        c["v3a8"] = v3(7, 2, 86)
    elif (B, L) in [(16, 2048), (32, 2048), (64, 2048)]:
        for S in (8, 16): c[f"u{S}"] = v2(S, None, True)
    return c
def k1k2(B, L, kind, S, a, g=None):
    g = g or gq; MAXP = L // PAGE; NBLK = B * MAXP
    if kind == "v2": k1 = g.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", a["swap"], True, a["taper"], TAIL, True)
    else: k1 = g.decode_partial_paged_v3(B, NBLK, MAXP, Hkv, G, D, a["S1"], a["S2"], 128, PAGE, 32, a["frac"], 1, TAIL, True)
    return k1, g.decode_combine_v2(B, H, S, D)
if compile_only:
    i = 0
    for B, L in SH + [(32, 8192)]:
        for n, (kind, S, a) in cfg(B, L).items():
            if i % CN == CI:
                try: k1k2(B, L, kind, S, a)
                except Exception as ex: P(t="compile_err", B=B, L=L, impl=n, err=str(ex)[-300:])
            i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
def setup(B, L):
    torch.manual_seed(3000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    KV = torch.randn(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    return Q, KV, BT, MAXP, NBLK
def uni(Q, KV, BT, lens, L, out):
    t = KV.transpose(1, 2); k, v = t[..., :D], t[..., D:]; cu = torch.arange(Q.shape[0] + 1, device="cuda", dtype=torch.int32); sc = 1.0 / math.sqrt(D)
    def u():
        unified(q=Q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=sc, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
        return out
    return u
for B, L in SH:
    Q, KV, BT, MAXP, NBLK = setup(B, L); out = torch.empty_like(Q)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda")
    if B > 1: base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    C = cfg(B, L); KK = {n: k1k2(B, L, *C[n][:2], C[n][2]) for n in C}
    Kd = KV[BT.long()].transpose(2, 3)[..., :D].reshape(B, L, Hkv, D); Vd = KV[BT.long()].transpose(2, 3)[..., D:].reshape(B, L, Hkv, D)
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        R = gq.ref64(Q, Kd, Vd, lens, G) if B * L <= 32 * 8192 else None
        A = {n: (lambda kp: (lambda: kp[1](*kp[0](Q, KV, KV, BT, lens))))(KK[n]) for n in KK}
        if unified: A["vllm_p"] = uni(Q, KV, BT, lens, L, out)
        names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        if R is not None:
            for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    del KV, Kd, Vd, Q, BT, KK, A; torch.cuda.empty_cache()
# slow mode in the real layout, batch 32 x 8k. Cycle 0 after an allocation is a known transient, only later cycles count.
B, L = 32, 8192; Q, KV0, BT, MAXP, NBLK = setup(B, L); out = torch.empty_like(Q)
lens = torch.full((B,), L, dtype=torch.int32, device="cuda"); lens[1::2] = L * 3 // 4; nb = 2 * 2 * Hkv * D * int(lens.sum())
C = cfg(B, L)
def sets(tag, KV, kps, **meta):
    A = {n: (lambda kp: (lambda: kp[1](*kp[0](Q, KV, KV, BT, lens))))(kps[n]) for n in kps}
    if unified: A["vllm_p"] = uni(Q, KV, BT, lens, L, out)
    names = list(A)
    for c in range(3):
        for ai in range(len(names)):
            n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t=tag, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1), **meta)
KK = {n: k1k2(B, L, *C[n][:2], C[n][2]) for n in ("v2p", "v3c")}
K = None
for i in range(10):
    K = None; torch.cuda.empty_cache(); K = KV0.clone(); sets("realloc_p", K, KK, iter=i)
import tilelang.cache as _tc
root = os.environ["TILELANG_CACHE_DIR"]
for i in range(3):
    d = f"{root}-tilelang_reload{i}-{os.environ.get('SLURM_JOB_ID', 'x')}"; shutil.copytree(root, d, ignore=shutil.ignore_patterns("triton"))
    os.environ["TILELANG_CACHE_DIR"] = d
    for c in _tc._dispatch_map.values(): c._memory_cache.clear()
    g2 = load(); K2 = {n: k1k2(B, L, *C[n][:2], C[n][2], g2) for n in ("v2p", "v3c")}
    sets("reload_p", K, K2, iter=i, maps=sum(1 for l in open("/proc/self/maps") if "tilelang_reload" in l))
    shutil.rmtree(d, ignore_errors=True)
print("DONE", flush=True)
