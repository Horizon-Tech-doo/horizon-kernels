"""N3 round 5 confirmation after the skip gate (_SKIP_MIN_BH): same as n3r5z.py on seeds 988-991, and at Hkv 2 only 8 x 8k, 16 x 4k, 8 x 32k, 32 x 8k, 16 x 8k, 32 x 1k on seeds 988-989; no rocm_hybrid. n3r5z.py header follows. N3 round 5 final bytes: real script scripts/gqa_decode.py, image stack, BF16, page 784. New seeds 980-987. Part Z: Hkv 4, all six shapes: pick (with lens, so dead skip under r 0.72), pick0 (same split count, no skip, where it differs), kk, tri, roc (ROCM_ATTN through vLLM's chunked_prefill_paged_decode) on every draw. Part R: rocm_hybrid (no vLLM config change) seeds 980-981. Part K: Hkv 2 shapes, seeds 980-983: pick, tri, roc.
Ratio vLLM ms / ours ms. usage: n3r5z.py image ROT cycles [compile I N]; env N3REV=1 reverses arm order."""
import json, math, os, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
REV = os.environ.get("N3REV") == "1"
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
G, D, PAGE, N, LAY = 6, 256, 16, 784, "lbnhc"
SH4 = [(8, 8192), (16, 4096), (32, 2048), (32, 1024), (64, 2048), (64, 1024)]
SH2 = [(8, 8192), (16, 4096), (32, 2048), (32, 1024), (64, 1024), (64, 2048), (32, 4096), (16, 8192), (8, 32768), (32, 8192)]
SH2Y = [(8, 8192), (16, 4096), (8, 32768), (32, 8192), (16, 8192), (32, 1024)]
def nb(B, L): return B * (-(-L // N))
def kk(Hkv, B, L, lay=LAY): return gq.paged_kernels(B, nb(B, L), L // PAGE, Hkv, G, D, PAGE, lay, None, None, N, "bf16", "bits")
def pk(Hkv, B, L): return gq.paged_picker(B, nb(B, L), L // PAGE, Hkv, G, D, PAGE, LAY, None, None, N, "bf16", "bits")
if compile_only:
    jobs = [(4, c) for c in SH4] + [(2, c) for c in SH2Y]
    for i, (h, (B, L)) in enumerate(jobs):
        if i % CN != CI: continue
        kk(h, B, L); pk(h, B, L)
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
def P(**k): print(json.dumps({"stack": stack, "dt": "bf16", **k}), flush=True)
class Data:
    def __init__(s, B, L, Hkv):
        dt = torch.bfloat16; torch.manual_seed(5000 + B + L); s.B, s.L, s.Hkv = B, L, Hkv; s.MAXB = -(-L // N); NBLK = B * s.MAXB
        s.Q = torch.randn(B, Hkv * G, D, device="cuda", dtype=torch.float32).to(dt)
        s.NH = torch.randn(NBLK, N, Hkv, 2 * D, device="cuda", dtype=torch.float32).to(dt); s.knh, s.vnh = s.NH[..., :D], s.NH[..., D:]
        s.BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, s.MAXB)
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
def race(part, d, mode, sd, lens, A, cyc, check):
    names = list(A); names = names[::-1] if REV else names; ks, mx = int(lens.sum()), int(lens.max())
    if check:
        R = d.ref(lens)
        for n in names: P(t="err", part=part, Hkv=d.Hkv, B=d.B, L=d.L, mode=mode, seed=sd, impl=n, rel=float(((A[n]().double() - R).norm() / R.norm()).item()))
    for c in range(cyc):
        for ai in range(len(names)):
            n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n])
            P(t="seed", part=part, Hkv=d.Hkv, B=d.B, L=d.L, seed=sd, mode=mode, cycle=c, pos=ai, impl=n, ms=round(ms, 4), sum=ks, max=mx, r=round(ks / (d.B * mx), 4))
for Hkv, shapes, seeds, part in ((4, SH4, range(988, 992), "Z"), (2, SH2Y, range(988, 990), "K")):
    for B, L in shapes:
        d = Data(B, L, Hkv); K, pick = kk(Hkv, B, L), pk(Hkv, B, L); rh = None
        for sd in seeds:
            Ls = d.lens(sd)
            for mode in (["base", "full"] if sd == seeds[0] else []) + ["part", "skew", "mid", "near"]:
                lens = Ls[mode]; ks, mx = int(lens.sum()), int(lens.max()); kp, kp0 = pick(ks, mx, lens), pick(ks, mx)
                A = dict(pick=d.ours(kp, lens), tri=d.tri(lens), roc=d.roc(lens))
                if kp[0] is not kp0[0]: A["pick0"] = d.ours(kp0, lens)   # same split count, dead splits kept
                if Hkv == 4: A["kk"] = d.ours(K, lens)
                race(part, d, mode, sd, lens, A, cycles, sd == seeds[0] and mode in ("base", "skew"))
                if False and rh is not None and sd < 982 and mode in ("base", "full", "part", "skew"):
                    race("R", d, mode, sd, lens, dict(rh=d.hyb(rh, lens), tri=d.tri(lens), roc=d.roc(lens)), cycles, sd == 980 and mode in ("base", "skew"))
        del d, K, pick, rh, A, kp, kp0; torch.cuda.empty_cache()
print("DONE", flush=True)
