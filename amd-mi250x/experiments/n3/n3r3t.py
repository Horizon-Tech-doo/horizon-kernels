"""N3 round 3: tile size on rocm_hybrid. V is D-major, so a 16-key tile reads 32 bytes per row at a 1568-byte stride; a 32-key tile reads 64. Page 784 is not a multiple of 32, so this runs at page 800, as a probe and not as a deployable shape. Hypothesis: longer runs per row raise bandwidth on this layout. Arms: rh16 (landed), rh32 (32-key tile, 128 threads, 16 rows), rh32t (256 threads, 64 rows), roc. 32 rows fails layout inference (job 23508678).
Ratio vLLM ms / ours ms. usage: n3r3t.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, N = 4, 6, 256, 800; H = Hkv * G
SH = [(8, 32768), (32, 8192), (64, 4096)]
def forms(B, L):
    NBLK = B * (-(-L // N)); out = {"rh16": gq.paged_kernels(B, NBLK, L // 16, Hkv, G, D, 16, "rocm_hybrid", None, None, N)}
    for name, th, bm in (("rh32", 128, 16), ("rh32t", 256, 64)):
        MP = L // 32; S1 = max(1, MP * 86 // (100 * 22)); S2 = max(-(-max(MP - S1 * 22, 0) // 6), 1)
        try:
            out[name] = (gq.decode_partial_paged_v3(B, NBLK, MP, Hkv, G, D, S1, S2, th, 32, bm, 86, "kvg", "rocm_hybrid", "joint", "xj", N), gq.decode_combine_v2(B, H, S1 + S2, D))
        except Exception as e:
            print("ERR", name, str(e)[:200], flush=True)
    return out
if compile_only:
    for i, (B, L) in enumerate(SH):
        if i % CN == CI:
            try: forms(B, L)
            except Exception as e: print("ERR", str(e)[-500:], flush=True)
    print("COMPILED", CI, flush=True); sys.exit(0)
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
    torch.manual_seed(5000 + B + L); MAXB = -(-L // N); NBLK = B * MAXB
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    kv = torch.randn(NBLK, N, Hkv, D, device="cuda", dtype=torch.float16); vv = torch.randn(NBLK, N, Hkv, D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
    Kd = kv[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; Vd = vv[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
    C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=torch.float16); K6 = C.view(NBLK, 2, Hkv, D // 8, N, 8); V5 = C.view(NBLK, 2, Hkv, D, N)
    K6[:, 0].copy_(kv.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); V5[:, 1].copy_(vv.permute(0, 2, 3, 1))
    kc2, vc2 = PagedAttention.split_kv_cache(C.transpose(0, 1), Hkv, D); F = forms(B, L)
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    for mode, lens in [("base", base), ("full", full)]:
        A = {n: (lambda k=k: (lambda: k[1](*k[0](Q, K6, V5, BT, lens))))() for n, k in F.items()}
        A["roc"] = lambda: (chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True), out)[1]
        names = list(A); R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in names: P(t="err", B=B, L=L, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c + ROT) % len(names)]; P(t="time", B=B, L=L, mode=mode, cycle=c, pos=ai, impl=n, ms=round(bench(A[n]), 4))
    del C, K6, V5, kv, vv, Kd, Vd, kc2, vc2, F, A; torch.cuda.empty_cache()
print("DONE", flush=True)
