"""N3 round 2, verify the landed selection (paged_kernels), real vLLM layout, vs vLLM's unified_attention on the same tensor, on all shapes and four length patterns, plus stale data.
Also separate K, V tensors in one block with a 64k gap (round 1's layout, fixed placement). usage: n3r2v.py image|lumi X cycles [compile I N]. Ratio: vLLM ms / ours ms, higher is better."""
import json, math, sys, importlib.util, torch
stack, _, cycles = sys.argv[1], sys.argv[2], int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
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
MAIN = [(8, 32768), (16, 16384), (32, 8192), (64, 4096), (1, 131072)]
SH = MAIN + [(128, 2048), (64, 2048), (32, 2048), (16, 2048), (64, 1024), (32, 1024), (8, 4096), (8, 2048), (16, 1024), (4, 4096), (32, 512), (1, 4096), (2, 8192), (4, 16384), (1, 16384)]
if compile_only:
    for i, (B, L) in enumerate(SH):
        if i % CN == CI:
            MAXP = L // PAGE; gq.paged_kernels(B, B * MAXP, MAXP, Hkv, G, D, PAGE)
            if (B, L) in MAIN: gq.paged_kernels(B, B * MAXP, MAXP, Hkv, G, D, PAGE, False)
    print("COMPILED", CI, flush=True); sys.exit(0)
for B, L in SH:
    torch.manual_seed(5000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    KV = torch.randn(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    t = KV.transpose(1, 2); kv, vv = t[..., :D], t[..., D:]
    Kd = kv[BT.long()].reshape(B, L, Hkv, D); Vd = vv[BT.long()].reshape(B, L, Hkv, D)
    sep = None
    if (B, L) in MAIN:
        N = kv.numel(); GAP = 65536; big = torch.empty(2 * N + GAP // 2, dtype=torch.float16, device="cuda")
        Ks = big[:N].view(kv.shape); Vs = big[N + GAP // 2:].view(kv.shape); Ks.copy_(kv); Vs.copy_(vv); sep = (Ks, Vs)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda")
    if B > 1: base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    k1, k2 = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE)
    k1s, k2s = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, False) if sep else (None, None)
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    def uni(k, v, lens):
        def u():
            unified(q=Q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out
        return u
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        small = B * L <= 32 * 8192
        R = gq.ref64(Q, Kd, Vd, lens, G) if small else None
        A = {"landed": (lambda ll: (lambda: k2(*k1(Q, KV, KV, BT, ll))))(lens)}
        if sep: A["landed_s"] = (lambda ll: (lambda: k2s(*k1s(Q, sep[0], sep[1], BT, ll))))(lens)
        if unified:
            A["vllm_p"] = uni(kv, vv, lens)
            if sep: A["vllm_s"] = uni(sep[0], sep[1], lens)
        names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        if R is not None:
            for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    if B * L <= 32 * 8192:
        R = gq.ref64(Q, Kd, Vd, part, G)
        for pname, val in [("stale_big", 3e4), ("stale_nan", float("nan"))]:
            KVq = KV.clone()
            for b in range(B):
                n = int(part[b]); pid = int(BT[b, (n - 1) // PAGE]); off = n % PAGE
                KVq[pid, :, off:] = val
            o = k2(*k1(Q, KVq, KVq, BT, part)); P(t="err", B=B, L=L, mode=pname, impl="landed", rel=rel(o, R), nan=int(torch.isnan(o).sum().item()))
            if unified:
                t2 = KVq.transpose(1, 2); o = uni(t2[..., :D], t2[..., D:], part)(); P(t="err", B=B, L=L, mode=pname, impl="vllm_p", rel=rel(o, R), nan=int(torch.isnan(o).sum().item()))
            del KVq
    del KV, t, kv, vv, Kd, Vd, sep, k1, k2; big = Ks = Vs = None; torch.cuda.empty_cache()
print("DONE", flush=True)
