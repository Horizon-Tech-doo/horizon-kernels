"""N3 round 3, step 5: fast/slow flip diagnostic. One kept allocation, full lengths, 32x8k and 64x4k, layout LBHNC then LBNHC. Each iteration: landed (joint), landed_s (K then V), vLLM unified_attention
on the same tensor, 20 reps each, 100 iterations, then 5 s idle, three sets. Every record has wall time so the clock log (clocks-<job>.log) lines up. Ratio: vLLM ms / ours ms. usage: n3r3b.py [skip_compile]"""
import json, math, sys, time, importlib.util, torch
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
def bench(fn, rep=20):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
for B, L in [(32, 8192), (64, 4096)]:
    for lay, lname in [(True, "lbhnc"), ("lbnhc", "lbnhc")]:
        torch.manual_seed(5000 + B + L); MAXP = L // PAGE; NBLK = B * MAXP
        Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        shape = (NBLK, Hkv, PAGE, 2 * D) if lname == "lbhnc" else (NBLK, PAGE, Hkv, 2 * D)
        C = torch.randn(*shape, device="cuda", dtype=torch.float16)
        BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
        k = C[..., :D] if lname == "lbnhc" else C[..., :D].transpose(1, 2); v = C[..., D:] if lname == "lbnhc" else C[..., D:].transpose(1, 2)
        lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
        a1, a2 = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, "joint"); s1, s2 = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, "k_v")
        scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
        def u():
            unified(q=Q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return out
        A = {"landed": lambda: a2(*a1(Q, C, C, BT, lens)), "landed_s": lambda: s2(*s1(Q, C, C, BT, lens)), "vllm": u}
        names = list(A)
        for n in names:
            for _ in range(3): A[n]()
        for st in range(3):
            for it in range(100):
                for ai in range(3):
                    n = names[(ai + it) % 3]; ms = bench(A[n]); print(json.dumps({"t": "flip", "B": B, "L": L, "lay": lname, "set": st, "it": it, "pos": ai, "impl": n, "ms": round(ms, 4), "wall": round(time.time(), 3)}), flush=True)
            time.sleep(5)
        del C, k, v, Q, BT, A; a1 = a2 = s1 = s2 = None; torch.cuda.empty_cache()
print("DONE", flush=True)
