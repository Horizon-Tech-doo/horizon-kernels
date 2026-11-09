"""N3 round 2, part B. Levers 1, 3, 4: length-aware splits (v3) and per-batch configs vs v2 and vLLM on length patterns base, full, part, skew, and unmeasured
shapes. Two memory layouts: "p" = vLLM 0.30's real cache, one tensor [NBLK, Hkv, PAGE, 2D] with K then V per row, read by vLLM through its own split views; and
"s" = the separate K [NBLK, PAGE, Hkv, D] and V tensors of round 1, in one block with a fixed gap so the K/V placement is the same in every run.
usage: n3r2b.py image|lumi TAIL GAP cycles [compile I N]. Compile mode only instantiates kernels I mod N, to fill the disk cache in parallel.
Arm order rotates per cycle. Ratio convention: vLLM ms / ours ms, higher is better."""
import json, math, sys, importlib.util, torch
stack, TAIL, GAP, cycles = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); compile_only = len(sys.argv) > 5
CI, CN = (int(sys.argv[6]), int(sys.argv[7])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
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
SHAPES = [(8, 32768), (16, 16384), (32, 8192), (64, 4096)]
EXTRA = [(128, 2048), (32, 1024), (64, 1024), (32, 2048), (64, 2048), (8, 4096), (16, 2048)]
def cfg(B, L):
    MAXP = L // PAGE
    Sv, tp = gq.paged_v2_config(MAXP); S1 = tp[0]; S2 = max(Sv - S1, 1)
    v2 = lambda S, taper, swap, tail, packed: ("v2", S, dict(taper=taper, swap=swap, tail=tail, packed=packed))
    v3 = lambda tail, packed, ml=1: ("v3", S1 + S2, dict(S1=S1, S2=S2, tail=tail, packed=packed, min_long=ml))
    c = {"v2p": v2(Sv, tp, True, TAIL, True), "v3p": v3(TAIL, True)}
    if (B, L) in SHAPES:
        c.update({"v2f": v2(Sv, tp, True, "v", False), "v2n": v2(Sv, tp, True, TAIL, False), "v3": v3(TAIL, False), "v3mp": v3(TAIL, True, 22)})
    if (B, L) == (16, 16384):
        c.update({"u27p": v2(27, None, False, TAIL, True), "t30p": v2(30, (23, 40, 16), True, TAIL, True), "t29p": v2(29, (25, 38, 19), True, TAIL, True)})
    if (B, L) == (8, 32768):
        c.update({"u55p": v2(55, None, False, TAIL, True), "t58p": v2(58, (45, 40, 20), True, TAIL, True)})
    return c
def k1k2(B, L, kind, S, a):
    MAXP = L // PAGE; NBLK = B * MAXP
    if kind == "v2": k1 = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", a["swap"], True, a["taper"], a["tail"], a["packed"])
    else: k1 = gq.decode_partial_paged_v3(B, NBLK, MAXP, Hkv, G, D, a["S1"], a["S2"], 128, PAGE, 32, 86, a["min_long"], a["tail"], a["packed"])
    return k1, gq.decode_combine_v2(B, H, S, D)
if compile_only:
    i = 0
    for B, L in SHAPES + EXTRA:
        for n, (kind, S, a) in cfg(B, L).items():
            if i % CN == CI:
                try: k1k2(B, L, kind, S, a)
                except Exception as ex: P(t="compile_err", B=B, L=L, impl=n, err=str(ex)[-300:])
            i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
for B, L in SHAPES + EXTRA:
    torch.manual_seed(2000 + B + L)
    MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    N = Kc.numel(); big = torch.empty(2 * N + GAP // 2, dtype=torch.float16, device="cuda")
    Ks = big[:N].view(Kc.shape); Vs = big[N + GAP // 2:].view(Kc.shape); Ks.copy_(Kc); Vs.copy_(Vc)
    KVp = torch.empty(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)
    KVp[..., :D] = Kc.transpose(1, 2); KVp[..., D:] = Vc.transpose(1, 2)
    KVt = KVp.transpose(1, 2); Kv, Vv = KVt[..., :D], KVt[..., D:]                 # the views vLLM's Triton backend makes
    Kd = Kc[BT.long()].view(B, L, Hkv, D); Vd = Vc[BT.long()].view(B, L, Hkv, D)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    scale = 1.0 / math.sqrt(D); cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32); out_u = torch.empty_like(Q)
    C = cfg(B, L); KK = {n: k1k2(B, L, *C[n][:2], C[n][2]) for n in C}
    def uni(k, v, lens):
        def u():
            unified(q=Q, k=k, v=v, out=out_u, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out_u
        return u
    def arms(lens, K, V, KV, kk=None):
        A = {}
        for n, kp in KK.items():
            packed = C[n][2]["packed"]
            A[n] = (lambda kp, a, b: (lambda: kp[1](*kp[0](Q, a, b, BT, lens))))(kp, KV if packed else K, KV if packed else V)
        if unified:
            A["vllm_p"] = uni(Kv, Vv, lens)
            if (B, L) in SHAPES: A["vllm_s"] = uni(K, V, lens)
        return A
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        R = gq.ref64(Q, Kd, Vd, lens, G) if B * L <= 32 * 8192 else None
        A = arms(lens, Ks, Vs, KVp); names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        if R is not None:
            for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    if B * L <= 32 * 8192:   # stale NaN and big values past n, on this shape, both layouts
        R = gq.ref64(Q, Kd, Vd, part, G)
        for pname, val in [("stale_big", 3e4), ("stale_nan", float("nan"))]:
            Kp, Vp, KVq = Ks.clone(), Vs.clone(), KVp.clone()
            for b in range(B):
                n = int(part[b]); pid = int(BT[b, (n - 1) // PAGE]); off = n % PAGE
                Kp[pid, off:] = val; Vp[pid, off:] = val; KVq[pid, :, off:] = val
            A = arms(part, Kp, Vp, KVq)
            if unified: A["vllm_p"] = uni(KVq.transpose(1, 2)[..., :D], KVq.transpose(1, 2)[..., D:], part)
            for n in A:
                o = A[n](); P(t="err", B=B, L=L, mode=pname, impl=n, rel=rel(o, R), nan=int(torch.isnan(o).sum().item()))
            del Kp, Vp, KVq
    del Kc, Vc, Ks, Vs, big, KVp, KVt, Kv, Vv, Kd, Vd, Q, BT, KK, A
    torch.cuda.empty_cache()
print("DONE", flush=True)
