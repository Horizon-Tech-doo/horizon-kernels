"""N3 round 4: remove the two BF16 losses to Triton. Image stack, BF16, page 784, layout LBNHC, compiled by TileLang (no hand edits), exp/N3/gqa_decode.py working copy.
Part A: 32 x 1k base and full, forms S4 (control), S5, S6, tapers t1_24_8, t1_24_12, t2_16_8, t2_24_8, all with pc="bits"; S4h is the current hip conversion. Part B: 64 x 1k skew, seeds 924-939, S4 swap/rc variants.
Part C: paged_kernels hip against bits on six shapes and four patterns. Part D: k1 and k2 alone, hip against bits. Ratio vLLM ms / ours ms. usage: n3r4a.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE, N, DT = 4, 6, 256, 16, 784, "bf16"; H = Hkv * G; LAY = "lbnhc"
def uni(B, L, S, pc, swap=True, rc=True, taper=None):
    MAXP = L // PAGE; NBLK = B * (-(-L // N))
    k1 = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", swap, rc, taper, "kvg", LAY, "joint", "jx", N, DT, pc)
    return k1, gq.decode_combine_v2(B, H, S, D, 256, DT, pc)
def tp(S1, A, a, MAXP=64): return S1 + max(0, -(-(MAXP - S1 * A) // a))
FA = {"S4h": (4, "hip", None), "S4b": (4, "bits", None), "S5b": (5, "bits", None), "S6b": (6, "bits", None),
      "t1_24_8b": (tp(1, 24, 8), "bits", (1, 24, 8)), "t1_24_12b": (tp(1, 24, 12), "bits", (1, 24, 12)), "t2_16_8b": (tp(2, 16, 8), "bits", (2, 16, 8)), "t2_24_8b": (tp(2, 24, 8), "bits", (2, 24, 8))}
FB = {"TTh": (True, True, "hip"), "TTb": (True, True, "bits"), "FTb": (False, True, "bits"), "TFb": (True, False, "bits"), "FFb": (False, False, "bits")}
SC = [(64, 1024), (64, 2048), (32, 1024), (32, 2048), (16, 4096), (8, 8192)]
def mkall():
    a = {n: uni(32, 1024, S, pc, True, True, t) for n, (S, pc, t) in FA.items()}
    return a
if compile_only:
    jobs = [("A", n) for n in FA] + [("B", n) for n in FB] + [("C", s, pc) for s in SC for pc in ("hip", "bits")]
    for i, j in enumerate(jobs):
        if i % CN != CI: continue
        if j[0] == "A": S, pc, t = FA[j[1]]; uni(32, 1024, S, pc, True, True, t)
        elif j[0] == "B": sw, rc, pc = FB[j[1]]; uni(64, 1024, 4, pc, sw, rc)
        else: (B, L), pc = j[1], j[2]; MAXP = L // PAGE; gq.paged_kernels(B, B * (-(-L // N)), MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT, pc)
    print("COMPILED", CI, flush=True); sys.exit(0)
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
def P(**k): print(json.dumps({"stack": stack, "dt": DT, **k}), flush=True)
class Data:
    def __init__(s, B, L):
        torch.manual_seed(5000 + B + L); s.B, s.L = B, L; MAXB = -(-L // N); NBLK = B * MAXB
        s.Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        s.NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(torch.bfloat16); s.knh, s.vnh = s.NH[..., :D], s.NH[..., D:]
        s.BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
        s.Kd = s.knh[s.BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]; s.Vd = s.vnh[s.BT.long()].reshape(B, MAXB * N, Hkv, D)[:, :L]
        s.scale = 1.0 / math.sqrt(D); s.cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); s.out = torch.empty_like(s.Q)
        C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=torch.bfloat16); s.C = C; s.kc2, s.vc2 = PagedAttention.split_kv_cache(C.transpose(0, 1), Hkv, D)
        s.kc2.copy_(s.knh.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); s.vc2.copy_(s.vnh.permute(0, 2, 3, 1))
    def lens(s, sd):
        B, L = s.B, s.L; g = torch.Generator(device="cuda"); g.manual_seed(sd)
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
        full = torch.full((B,), L, dtype=torch.int32, device="cuda")
        part = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int32, generator=g); part[(part % PAGE) == 0] -= 1
        skew = torch.randint(1, L, (B,), device="cuda", dtype=torch.int32, generator=g); skew[(skew % PAGE) == 0] -= 1
        return dict(base=base, full=full, part=part, skew=skew)
    def tri(s, lens):
        return lambda: (unified(q=s.Q, k=s.knh, v=s.vnh, out=s.out, cu_seqlens_q=s.cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=s.L, softmax_scale=s.scale, causal=True, window_size=(-1, -1), block_table=s.BT, softcap=0, q_descale=None, k_descale=None, v_descale=None), s.out)[1]
    def roc(s, lens):
        return lambda: (chunked(query=s.Q, key=None, value=None, output=s.out, kv_cache_dtype="auto", key_cache=s.kc2, value_cache=s.vc2, block_table=s.BT, query_start_loc=s.cu, seq_lens=lens, max_seq_len=s.L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=s.scale, output_scale=None, sinks=None, causal=True), s.out)[1]
    def ours(s, kk, lens): return lambda: kk[1](*kk[0](s.Q, s.NH, s.NH, s.BT, lens))
def race(part, d, mode, sd, lens, A, cyc, check):
    names = list(A); ks, mx = int(lens.sum()), int(lens.max())
    if check:
        R = gq.ref64(d.Q, d.Kd, d.Vd, lens, G)
        for n in names: P(t="err", part=part, B=d.B, L=d.L, N=N, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
    for c in range(cyc):
        for ai in range(len(names)):
            n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n])
            P(t="seed", part=part, B=d.B, L=d.L, N=N, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, pct=round(Hkv * D * ks / ms / 1e6 / 4, 1))
# Part A
d = Data(32, 1024); ls = d.lens(924); KA = {n: uni(32, 1024, S, pc, True, True, t) for n, (S, pc, t) in FA.items()}
for mode in ("base", "full"):
    A = {n: d.ours(k, ls[mode]) for n, k in KA.items()}; A.update(tri=d.tri(ls[mode]), roc=d.roc(ls[mode]))
    race("A", d, mode, 924, ls[mode], A, 3 * cycles, True)
# Part D: k1 and k2 alone at 32 x 1k base
lens = ls["base"]
for pcn, kk in (("hip", KA["S4h"]), ("bits", KA["S4b"])):
    Op, Lse = kk[0](d.Q, d.NH, d.NH, d.BT, lens)
    for c in range(cycles):
        P(t="part", impl="k1_" + pcn, cycle=c, ms=round(bench(lambda: kk[0](d.Q, d.NH, d.NH, d.BT, lens)), 4)); P(t="part", impl="k2_" + pcn, cycle=c, ms=round(bench(lambda: kk[1](Op, Lse)), 4))
del d, KA; torch.cuda.empty_cache()
# Part B
d = Data(64, 1024); KB = {n: uni(64, 1024, 4, pc, sw, rc) for n, (sw, rc, pc) in FB.items()}
for sd in range(924, 940):
    lens = d.lens(sd)["skew"]; A = {n: d.ours(k, lens) for n, k in KB.items()}; A.update(tri=d.tri(lens)); race("B", d, "skew", sd, lens, A, cycles, sd == 924)
del d, KB; torch.cuda.empty_cache()
# Part C
for B, L in SC:
    d = Data(B, L); MAXP = L // PAGE; NBLK = B * (-(-L // N))
    kh = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT, "hip"); kb = gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, LAY, None, None, N, DT, "bits")
    for sd in range(924, 928):
        L_ = d.lens(sd)
        for mode in (["base", "full"] if sd == 924 else []) + ["part", "skew"]:
            A = dict(hip=d.ours(kh, L_[mode]), bits=d.ours(kb, L_[mode]), tri=d.tri(L_[mode])); race("C", d, mode, sd, L_[mode], A, cycles, sd == 924 and mode == "full")
    del d, kh, kb, A; torch.cuda.empty_cache()
print("DONE", flush=True)
