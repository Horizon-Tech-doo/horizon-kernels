"""N3 round 3: the real frame for Qwen3.8 on vLLM 0.30, read from live starts (jobs 23508507, 23508528): page 784 tokens, default backend ROCM_ATTN on physical [NBLK, 2, 784, Hkv * D] (K then V in each page, vLLM calls it LBHNC), cache in the model's dtype.
Arms: rh (landed on that memory, K address order, K then V), rh_vk (V then K), roc (chunked_prefill_paged_decode on split_kv_cache views of the same memory, as the backend calls it, not native),
n (landed on [NBLK, 784, Hkv, 2D], TRITON_ATTN's default), ns (K then V), tri (unified_attention on it). fp16 and bf16. Ratio vLLM ms / ours ms. usage: n3r3h.py image|lumi ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE, N = 4, 6, 256, 16, 784; H = Hkv * G
SH = [(8, 32768), (16, 16384), (32, 8192), (64, 4096), (64, 1024), (32, 1024)]
LANDS = {"rh": ("rocm_hybrid", "joint", "xj"), "rh_vk": ("rocm_hybrid", "v_k", "xj"), "n": ("lbnhc", "joint", "jx"), "ns": ("lbnhc", "k_v", "jx")}
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
if compile_only:
    i = 0
    for B, L in SH:
        for dt in DT:
            for n, (lay, st, ko) in LANDS.items():
                if i % CN == CI:
                    MAXP = L // PAGE; NBLK = B * (-(-L // N)); gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, st, ko, N, dt)
                i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
if stack == "image":
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as chunked, has_native_kv_cache_layout
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
def rel(o, R): return float(((o.double() - R).norm() / R.norm()).item())
for B, L in SH:
  for dt, dty in DT.items():
    torch.manual_seed(5000 + B + L); MAXB = -(-L // N); MAXP = L // PAGE; NBLK = B * MAXB
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(dty)
    kv = torch.randn(NBLK, N, Hkv, D, device="cuda", dtype=torch.float32).to(dty); vv = torch.randn(NBLK, N, Hkv, D, device="cuda", dtype=torch.float32).to(dty)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
    Kd = kv[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; Vd = vv[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
    C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=dty)
    K6 = C.view(NBLK, 2, Hkv, D // 8, N, 8); V5 = C.view(NBLK, 2, Hkv, D, N)
    K6[:, 0].copy_(kv.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); V5[:, 1].copy_(vv.permute(0, 2, 3, 1))
    NH = torch.cat([kv, vv], -1); knh, vnh = NH[..., :D], NH[..., D:]
    KS = {n: gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, st, ko, N, dt) for n, (lay, st, ko) in LANDS.items()}
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    if stack == "image":
        kc2, vc2 = PagedAttention.split_kv_cache(C.transpose(0, 1), Hkv, D)
        P(t="meta", B=B, L=L, dt=dt, k_stride=list(kc2.stride()), v_stride=list(vc2.stride()), native=has_native_kv_cache_layout(kc2, vc2), v_off=vc2.storage_offset())
    def lm(n, lens):
        k1, k2 = KS[n]; a, b = (K6, V5) if n.startswith("rh") else (NH, NH)
        return lambda: k2(*k1(Q, a, b, BT, lens))
    def tri(lens):
        def u():
            unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out
        return u
    def roc(lens):
        def r():
            chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True)
            return out
        return r
    base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
    full = torch.full((B,), L, dtype=torch.int32, device="cuda")
    part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32); part[(part % PAGE) == 0] -= 1
    skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32); skew[(skew % PAGE) == 0] -= 1
    for mode, lens in [("base", base), ("full", full), ("part", part), ("skew", skew)]:
        A = {n: lm(n, lens) for n in LANDS}
        if stack == "image": A.update(tri=tri(lens), roc=roc(lens))
        names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
        R = gq.ref64(Q, Kd, Vd, lens, G)
        for n in names: P(t="err", B=B, L=L, dt=dt, mode=mode, impl=n, rel=rel(A[n](), R))
        for c in range(cycles):
            for ai in range(len(names)):
                n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="time", B=B, L=L, dt=dt, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
    del kv, vv, Kd, Vd, C, K6, V5, NH, knh, vnh, KS, Q, A
    if stack == "image": del kc2, vc2
    torch.cuda.empty_cache()
print("DONE", flush=True)
