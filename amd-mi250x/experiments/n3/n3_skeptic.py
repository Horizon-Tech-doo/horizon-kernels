"""N3 skeptic check: landed kernels from the real scripts/ copy vs vLLM on length patterns the headline never used.
The headline lens are L and L*3/4, always whole pages, so the partial last page path never ran. Runs in the vLLM image, one process.
Arm order rotates per cycle. Ratio printed by summ: vLLM ms / ours ms, higher is better."""
import json, math, sys, importlib.util, torch
cycles = int(sys.argv[1]); reps = 50
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py")
gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
unified = __import__("vllm.v1.attention.ops.triton_unified_attention", fromlist=["unified_attention"]).unified_attention
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
OLDS = {32: 13, 64: 5}
def rel(o, R): return float(((o.double() - R).norm() / R.norm()).item())
for B, L in [(32, 8192), (64, 4096)]:
    torch.manual_seed(1000 + B)
    MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32)
    part[(part % PAGE) == 0] -= 1                                  # every sequence ends inside a page
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32)
    skew[(skew % PAGE) == 0] -= 1
    So = OLDS[B]; Sv, tp = gq.paged_v2_config(MAXP)
    k1o = gq.decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, So, 128, PAGE, 32); k2o = gq.decode_combine(B, H, So, D)
    k1v = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, Sv, 128, PAGE, 32, "if", True, True, tp); k2v = gq.decode_combine_v2(B, H, Sv, D)
    scale = 1.0 / math.sqrt(D); cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32); out_u = torch.empty_like(Q)
    def arms(K, V, lens):
        def u():
            unified(q=Q, k=K, v=V, out=out_u, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True,
                    window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out_u
        return {"old": lambda: k2o(*k1o(Q, K, V, BT, lens)), "v2f": lambda: k2v(*k1v(Q, K, V, BT, lens)), "vllm_triton": u}
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G)
        A = arms(Kc, Vc, lens); names = list(A); nbytes = 2 * 2 * Hkv * D * int(lens.sum())
        for n in names:
            print(json.dumps({"t": "err", "B": B, "L": L, "mode": mode, "impl": n, "rel_vs_fp64": rel(A[n](), R)}), flush=True)
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c) % len(names)]; ms = bench(A[n])
                print(json.dumps({"t": "time", "B": B, "L": L, "mode": mode, "cycle": c, "pos": ai, "impl": n, "ms": round(ms, 4), "gbps": round(nbytes / ms / 1e6, 1), "pct": round(nbytes / ms / 1e6 / 16, 1)}), flush=True)
    # stale data past n in the last page: the same reference as the clean data must come back
    R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), part, G)
    for pname, val in [("stale_big", 3e4), ("stale_nan", float("nan"))]:
        Kp, Vp = Kc.clone(), Vc.clone()
        for b in range(B):
            n = int(part[b]); pid = int(BT[b, (n - 1) // PAGE]); off = n % PAGE
            Kp[pid, off:] = val; Vp[pid, off:] = val
        A = arms(Kp, Vp, part)
        for n in A:
            o = A[n]()
            print(json.dumps({"t": "err", "B": B, "L": L, "mode": pname, "impl": n, "rel_vs_fp64": rel(o, R), "nan_elems": int(torch.isnan(o).sum().item())}), flush=True)
        del Kp, Vp
    del Kc, Vc, Q, BT, k1o, k1v, k2o, k2v
    torch.cuda.empty_cache()
print("DONE", flush=True)
