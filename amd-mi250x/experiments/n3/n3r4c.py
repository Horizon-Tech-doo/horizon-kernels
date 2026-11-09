"""N3 round 4 final bytes (scripts/gqa_decode.py), image stack, page 784. Part H: pick on held-out seeds 948-963, BF16, LBNHC, Hkv 4. Part R: rocm_hybrid, BF16. Part K: Hkv 2 as read from the tensor-parallel 2 start (23508969; 12 q heads, 2 kv heads, page 784), BF16: paged_kernels, pick, uniform forms at 640 and 320 blocks, tri, roc. Part F: FP16 control.
kk is paged_kernels with pc bits, kkh with pc hip. Ratio vLLM ms / ours ms. usage: n3r4c.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
G, D, PAGE, N = 6, 256, 16, 784; LAY = "lbnhc"
SH4 = [(64, 1024), (64, 2048), (32, 1024), (32, 2048), (16, 4096), (8, 8192)]
SH2 = [(64, 1024), (32, 1024), (64, 2048), (32, 2048), (16, 4096), (8, 8192), (32, 4096), (16, 8192), (64, 4096), (32, 8192), (8, 32768)]
def kk(B, L, Hkv, pc="bits", lay=LAY, dt="bf16"): return gq.paged_kernels(B, B * (-(-L // N)), L // PAGE, Hkv, G, D, PAGE, lay, None, None, N, dt, pc)
def pk(B, L, Hkv): return gq.paged_picker(B, B * (-(-L // N)), L // PAGE, Hkv, G, D, PAGE, LAY, None, None, N, "bf16", "bits")
def un(B, L, Hkv, S): return gq.paged_uniform(B, B * (-(-L // N)), L // PAGE, Hkv, G, D, S, PAGE, LAY, None, None, N, "bf16", "bits")
def forms2(B, L): 
    R = B * 2; return {"S640": 640 // R, **({"S320": 320 // R} if R <= 64 else {})}
if compile_only:
    jobs = [("h", B, L) for B, L in SH4] + [("k", B, L) for B, L in SH2] + [("f", 32, 1024)]
    for i, (w, B, L) in enumerate(jobs):
        if i % CN != CI: continue
        if w == "h": kk(B, L, 4); kk(B, L, 4, "hip"); pk(B, L, 4); kk(B, L, 4, "bits", "rocm_hybrid"); kk(B, L, 4, "hip", "rocm_hybrid")
        elif w == "k":
            kk(B, L, 2); pk(B, L, 2); kk(B, L, 2, "bits", "rocm_hybrid")
            for S in forms2(B, L).values(): un(B, L, 2, S)
        else: kk(B, L, 4, "bits", LAY, "fp16")
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
def P(**k): print(json.dumps({"stack": stack, **k}), flush=True)
class Data:
    def __init__(s, B, L, Hkv, dt=torch.bfloat16):
        torch.manual_seed(5000 + B + L); s.B, s.L, s.Hkv = B, L, Hkv; MAXB = -(-L // N); s.MAXB = MAXB; NBLK = B * MAXB; H = Hkv * G
        s.Q = torch.randn(B, H, D, device="cuda", dtype=torch.float32).to(dt)
        s.NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(dt); s.knh, s.vnh = s.NH[..., :D], s.NH[..., D:]
        s.BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXB)
        s.scale = 1.0 / math.sqrt(D); s.cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); s.out = torch.empty_like(s.Q)
        s.C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=dt); s.kc2, s.vc2 = PagedAttention.split_kv_cache(s.C.transpose(0, 1), Hkv, D)
        s.kc2.copy_(s.knh.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); s.vc2.copy_(s.vnh.permute(0, 2, 3, 1))
        s.K6 = s.C.view(NBLK, 2, Hkv, D // 8, N, 8); s.V5 = s.C.view(NBLK, 2, Hkv, D, N)
    def ref(s, lens):
        Kd = s.knh[s.BT.long()].reshape(s.B, s.MAXB * N, s.Hkv, D)[:, :s.L]; Vd = s.vnh[s.BT.long()].reshape(s.B, s.MAXB * N, s.Hkv, D)[:, :s.L]; return gq.ref64(s.Q, Kd, Vd, lens, G)
    def lens(s, sd):
        B, L = s.B, s.L; g = torch.Generator(device="cuda"); g.manual_seed(sd); R = lambda lo: (lambda x: (x.__setitem__((x % PAGE) == 0, x[(x % PAGE) == 0] - 1), x)[1])(torch.randint(lo, L, (B,), device="cuda", dtype=torch.int32, generator=g))
        base = torch.full((B,), L, dtype=torch.int32, device="cuda"); base[1::2] = L * 3 // 4
        return dict(base=base, full=torch.full((B,), L, dtype=torch.int32, device="cuda"), part=R(L // 2), skew=R(1), mid=R(L // 4), near=R(L * 3 // 4))
    def tri(s, lens):
        return lambda: (unified(q=s.Q, k=s.knh, v=s.vnh, out=s.out, cu_seqlens_q=s.cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=s.L, softmax_scale=s.scale, causal=True, window_size=(-1, -1), block_table=s.BT, softcap=0, q_descale=None, k_descale=None, v_descale=None), s.out)[1]
    def roc(s, lens):
        return lambda: (chunked(query=s.Q, key=None, value=None, output=s.out, kv_cache_dtype="auto", key_cache=s.kc2, value_cache=s.vc2, block_table=s.BT, query_start_loc=s.cu, seq_lens=lens, max_seq_len=s.L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=s.scale, output_scale=None, sinks=None, causal=True), s.out)[1]
    def ours(s, k, lens): return lambda: k[1](*k[0](s.Q, s.NH, s.NH, s.BT, lens))
    def hyb(s, k, lens): return lambda: k[1](*k[0](s.Q, s.K6, s.V5, s.BT, lens))
def race(part, d, mode, sd, lens, A, cyc, check, dt="bf16"):
    names = list(A); ks, mx = int(lens.sum()), int(lens.max())
    if check:
        R = d.ref(lens)
        for n in names: P(t="err", part=part, dt=dt, Hkv=d.Hkv, B=d.B, L=d.L, N=N, mode=mode, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
    for c in range(cyc):
        for ai in range(len(names)):
            n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n])
            P(t="seed", part=part, dt=dt, Hkv=d.Hkv, B=d.B, L=d.L, N=N, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, r=round(ks / (d.B * mx), 4), pct=round(d.Hkv * D * ks / ms / 1e6 / 4, 1))
# Part H: held-out pick, Hkv 4
for B, L in SH4:
    d = Data(B, L, 4); old, oldh, pick = kk(B, L, 4), kk(B, L, 4, "hip"), pk(B, L, 4)
    for sd in range(948, 956):
        Ls = d.lens(sd)
        for mode in (["base", "full"] if sd == 948 else []) + ["part", "skew", "mid", "near"]:
            lens = Ls[mode]; ks, mx = int(lens.sum()), int(lens.max()); A = dict(kkh=d.ours(oldh, lens), kk=d.ours(old, lens), pick=d.ours(pick(ks, mx), lens), tri=d.tri(lens))
            if sd == 948: A["roc"] = d.roc(lens)
            race("H", d, mode, sd, lens, A, cycles, sd == 948 and mode in ("base", "skew"))
    del d, old, oldh, pick, A; torch.cuda.empty_cache()
# Part R: rocm_hybrid on final bytes, one seed
for B, L in SH4:
    d = Data(B, L, 4); rh, rhh = kk(B, L, 4, "bits", "rocm_hybrid"), kk(B, L, 4, "hip", "rocm_hybrid"); Ls = d.lens(948)
    for mode in ("base", "full", "part", "skew"):
        lens = Ls[mode]; race("R", d, mode, 948, lens, dict(rhh=d.hyb(rhh, lens), rh=d.hyb(rh, lens), tri=d.tri(lens), roc=d.roc(lens)), cycles, mode in ("base", "skew"))
    del d, rh, rhh; torch.cuda.empty_cache()
# Part K: Hkv 2
for B, L in SH2:
    d = Data(B, L, 2); old, pick, rh = kk(B, L, 2), pk(B, L, 2), kk(B, L, 2, "bits", "rocm_hybrid"); Fm = {n: un(B, L, 2, S) for n, S in forms2(B, L).items()}
    for sd in range(948, 951):
        Ls = d.lens(sd)
        for mode in (["base", "full"] if sd == 948 else []) + ["part", "skew"]:
            lens = Ls[mode]; ks, mx = int(lens.sum()), int(lens.max()); A = dict(kk=d.ours(old, lens), pick=d.ours(pick(ks, mx), lens), rh=d.hyb(rh, lens), tri=d.tri(lens), roc=d.roc(lens))
            A.update({n: d.ours(k, lens) for n, k in Fm.items()}); race("K", d, mode, sd, lens, A, cycles, sd == 948 and mode in ("base", "skew"))
    del d, old, pick, rh, Fm, A; torch.cuda.empty_cache()
# Part F: FP16 control on final bytes, S4 path unchanged by pc
d = Data(32, 1024, 4, torch.float16); k = kk(32, 1024, 4, "bits", LAY, "fp16"); Ls = d.lens(948)
for mode in ("base", "full"):
    race("F", d, mode, 948, Ls[mode], dict(kk=d.ours(k, Ls[mode]), tri=d.tri(Ls[mode])), cycles, True, "fp16")
print("DONE", flush=True)
