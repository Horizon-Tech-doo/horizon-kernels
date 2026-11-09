"""N3 round 2, part A, batch 32 only. (1) tail forms for the NaN fix, on four length patterns, stale_nan correctness. (2) slow mode test:
re-allocate K and V with the same data, reload the code object with the allocation kept, and contiguous or offset K/V placements.
usage: n3r2a.py image|lumi. Arm order rotates per cycle. Ratio convention in the summary: vLLM ms / ours ms, higher is better."""
import json, math, os, shutil, sys, importlib.util, torch
stack = sys.argv[1]; sys.argv = ["x"]; reps = 50
E = "/scratch/project_XXXXXXXXX/tilelang/exp/N3"
def load():
    spec = importlib.util.spec_from_file_location("gq", E + "/gqa_decode.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gq = load()
unified = __import__("vllm.v1.attention.ops.triton_unified_attention", fromlist=["unified_attention"]).unified_attention if stack == "image" else None
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
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G; B, L = 32, 8192; MAXP = L // PAGE; NBLK = B * MAXP
torch.manual_seed(32032)
Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
full = torch.full((B,), L, dtype=torch.int32, device="cuda")
part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
Sv, tp = gq.paged_v2_config(MAXP)
scale = 1.0 / math.sqrt(D); cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32); out_u = torch.empty_like(Q)
k2v = gq.decode_combine_v2(B, H, Sv, D)
k1o = gq.decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, 13, 128, PAGE, 32); k2o = gq.decode_combine(B, H, 13, D)
def vk(tail): return gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, Sv, 128, PAGE, 32, "if", True, True, tp, tail)
K1 = {}
for tail in ["v", "kv", "after", "kvg"]:
    try:
        K1[tail] = vk(tail); src = K1[tail].get_kernel_source()
        P(t="kern", tail=tail, sync=src.count("__syncthreads"))
    except Exception as ex:
        P(t="kern", tail=tail, err=str(ex)[-400:])
def uni(K, V, lens):
    def u():
        unified(q=Q, k=K, v=V, out=out_u, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
        return out_u
    return u
def arms(K, V, lens, tails):
    A = {"old": lambda: k2o(*k1o(Q, K, V, BT, lens))}
    for t in tails:
        if t in K1: A["v2_" + t] = (lambda kk: (lambda: k2v(*kk(Q, K, V, BT, lens))))(K1[t])
    if unified: A["vllm_triton"] = uni(K, V, lens)
    return A
# (1) tail forms
for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
    R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G)
    A = arms(Kc, Vc, lens, list(K1)); names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
    for n in names: P(t="err", mode=mode, impl=n, rel=rel(A[n](), R))
    for c in range(4):
        for ai in range(len(names)):
            n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t="tail", mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), part, G)
for pname, val in [("stale_big", 3e4), ("stale_nan", float("nan"))]:
    Kp, Vp = Kc.clone(), Vc.clone()
    for b in range(B):
        n = int(part[b]); pid = int(BT[b, (n - 1) // PAGE]); off = n % PAGE
        Kp[pid, off:] = val; Vp[pid, off:] = val
    A = arms(Kp, Vp, part, list(K1))
    for n in A:
        o = A[n](); P(t="err", mode=pname, impl=n, rel=rel(o, R), nan=int(torch.isnan(o).sum().item()))
    del Kp, Vp
# (2) slow mode: fixed code, base lengths
kF = K1["v"]; nb = 2 * 2 * Hkv * D * int(base.sum())
def timeset(tag, K, V, k1v, k1old, k2vv, k2oo, cyc=2, **meta):
    A = {"v2f": lambda: k2vv(*k1v(Q, K, V, BT, base)), "old": lambda: k2oo(*k1old(Q, K, V, BT, base))}
    if unified: A["vllm_triton"] = uni(K, V, base)
    names = list(A)
    for c in range(cyc):
        for ai in range(len(names)):
            n = names[(ai + c) % len(names)]; ms = bench(A[n]); P(t=tag, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1), **meta)
keep = None
for i in range(10):
    K = V = pad = None; torch.cuda.empty_cache()
    if i >= 5: pad = torch.empty(int(torch.randint(1, 64, (1,))) * (1 << 21) + 4096 * int(torch.randint(0, 512, (1,))), dtype=torch.uint8, device="cuda")
    K = Kc.clone(); V = Vc.clone()
    timeset("realloc", K, V, kF, k1o, k2v, k2o, iter=i, pad=(pad is not None), kp=K.data_ptr(), vp=V.data_ptr(), delta=V.data_ptr() - K.data_ptr())
keep = (K, V)
# same allocation, new code objects: a copy of the kernel cache gives a new library path, so a new dlopen and a new fat binary
root = os.environ["TILELANG_CACHE_DIR"]
def libs():
    return sum(1 for l in open("/proc/self/maps") if "tilelang_reload" in l)
for i in range(5):
    d = f"{root}-tilelang_reload{i}-{os.environ.get('SLURM_JOB_ID','x')}"; shutil.copytree(root, d, ignore=shutil.ignore_patterns("triton"))
    os.environ["TILELANG_CACHE_DIR"] = d
    import tilelang; 
    pass
    import tilelang.cache as _tc
    from tilelang import env as _e
    before = sum(len(c._memory_cache) for c in _tc._dispatch_map.values())
    for c in _tc._dispatch_map.values(): c._memory_cache.clear()
    g2 = load()
    kF2 = g2.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, Sv, 128, PAGE, 32, "if", True, True, tp, "v"); k2v2 = g2.decode_combine_v2(B, H, Sv, D)
    k1o2 = g2.decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, 13, 128, PAGE, 32); k2o2 = g2.decode_combine(B, H, 13, D)
    timeset("reload", keep[0], keep[1], kF2, k1o2, k2v2, k2o2, iter=i, maps=libs(), newobj=(kF2 is not kF), mem_before=before, mem_after=sum(len(c._memory_cache) for c in _tc._dispatch_map.values()), cache_dir=str(_e.TILELANG_CACHE_DIR)[-40:], so_maps=sum(1 for l in open("/proc/self/maps") if ".so" in l and "cache" in l))
    shutil.rmtree(d, ignore_errors=True)
os.environ["TILELANG_CACHE_DIR"] = root

keep = K = V = None; torch.cuda.empty_cache()
# placements
N = Kc.numel(); shape = Kc.shape
def contig(order, gap_bytes):
    big = torch.empty(2 * N + gap_bytes // 2, dtype=torch.float16, device="cuda")
    a = big[:N].view(shape); b = big[N + gap_bytes // 2:].view(shape)
    a2, b2 = (Kc, Vc) if order == "KV" else (Vc, Kc)
    a.copy_(a2); b.copy_(b2)
    return (a, b) if order == "KV" else (b, a)
R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), base, G)
for rep in range(3):
    for name, order, gap in [("sep", "S", 0), ("con_KV", "KV", 0), ("con_VK", "VK", 0)] + [("gap" + str(g // 1024) + "k", "KV", g) for g in (1024, 2048, 4096, 8192, 16384, 32768, 48 << 10, 64 << 10, 80 << 10, 96 << 10, 128 << 10, 192 << 10, 256 << 10, 384 << 10, 512 << 10, 1 << 20, 3 << 20, 7 << 20)]:
        torch.cuda.empty_cache()
        if order == "S": K = Kc.clone(); V = Vc.clone()
        else: K, V = contig(order, gap)
        if rep == 0: P(t="err", mode="place_" + name, impl="v2f", rel=rel(k2v(*kF(Q, K, V, BT, base)), R))
        timeset("place", K, V, kF, k1o, k2v, k2o, place=name, rep=rep, delta=V.data_ptr() - K.data_ptr())
        K = V = None
print("DONE", flush=True)
