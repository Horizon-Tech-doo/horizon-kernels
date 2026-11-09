"""N3 round 3 skeptic: the held-out pick test (n3r3p.py) repeated on BF16, the dtype the model is served in; n3r3p.py ran FP16 only. Same seeds 924-931, page 784, layout LBNHC, final bytes of gqa_decode.py. Adds (16,4096) and (8,8192), which fit the 111-page live cache and were never raced on random data.
Arms: old, pick, tri, roc. Ratio vLLM ms / ours ms. usage: n3sk3.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G; LAY = "lbnhc"
SH = [(64, 1024), (64, 2048), (32, 1024), (32, 2048), (16, 4096), (8, 8192)]; NS = (784,); DT = "bf16"
def mk(B, L, N):
    MAXP = L // PAGE; NBLK = B * (-(-L // N))
    return gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT), gq.paged_picker(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT)
if compile_only:
    i = 0
    for B, L in SH:
        for N in NS:
            if i % CN == CI: mk(B, L, N)
            i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
if stack == "image":
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
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
  for N in NS:
    torch.manual_seed(5000 + B + L); MAXB = -(-L // N); NBLK = B * MAXB
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(torch.bfloat16); knh, vnh = NH[..., :D], NH[..., D:]
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
    Kd = knh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; Vd = vnh[BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
    old, pick = mk(B, L, N)
    scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
    if stack == "image":
        C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=torch.bfloat16); kc2, vc2 = PagedAttention.split_kv_cache(C.transpose(0, 1), Hkv, D)
        kc2.copy_(knh.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); vc2.copy_(vnh.permute(0, 2, 3, 1))
    for sd in range(924, 932):
        g = torch.Generator(device="cuda"); g.manual_seed(sd)
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
        full = torch.full((B,), L, dtype=torch.int32, device="cuda")
        part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32, generator=g); part[(part % PAGE) == 0] -= 1
        skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32, generator=g); skew[(skew % PAGE) == 0] -= 1
        for mode, lens in ([("base", base), ("full", full)] if sd == 924 else []) + [("part", part), ("skew", skew)]:
            ks, mx = int(lens.sum()), int(lens.max()); kp = pick(ks, mx); which = "S4" if ks < 0.62 * B * mx else ("S3" if ks < 0.8 * B * mx else "old")
            A = {"old": lambda: old[1](*old[0](Q, NH, NH, BT, lens)), "pick": lambda: kp[1](*kp[0](Q, NH, NH, BT, lens))}
            if stack == "image":
                def tri():
                    unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None); return out
                def roc():
                    chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True); return out
                A.update(tri=tri, roc=roc)
            names = list(A)
            if sd == 924:
                R = gq.ref64(Q, Kd, Vd, lens, G)
                for n in names: P(t="err", B=B, L=L, N=N, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
            for c in range(cycles):
                for ai in range(len(names)):
                    n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="seed", dt=DT, B=B, L=L, N=N, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), which=which, sum=ks, max=mx, pct=round(16 * Hkv * D * ks / ms / 1e6 / 16 / 4, 1))
    del NH, knh, vnh, Kd, Vd, old, pick, kp, A
    if stack == "image": del C, kc2, vc2
    torch.cuda.empty_cache()
print("DONE", flush=True)
