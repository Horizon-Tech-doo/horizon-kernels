"""C2P: MoE and decode on the image stack (ROCm 7.2.3, clang 22), one process per compiler tree, vLLM's Triton kernels in the same process.
Ratio: ours / vLLM, time, lower is better. --compile fills the cache on a CPU only.
Kernel files: copies of the landed tilelang/moe_fused.py and tilelang/gqa_decode.py (md5 logged). C2P_ARM names the tree; C2P_FASTEXP=1 sets T.exp = T.__exp before any kernel is built (fork-main lowers T.exp to the accurate form, the 0.1.15 wheel to the fast one)."""
import argparse, hashlib, json, math, os, random, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="x"); ap.add_argument("--seed", default="0")
ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--rep", type=int, default=40); ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--only", default="moe,decode")
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
if os.environ.get("C2P_FASTEXP"):
    T.exp = T.__exp
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/" + os.environ.get("C2P_M", "m"))   # m2: decode k1 with 1 KB of unused LDS, so 3 blocks per CU as on the wheel
import moe_fused as M
import gqa_decode as gq
ARM = os.environ.get("C2P_ARM", "?")
if os.environ.get("C2P_POST"):      # diagnosis only, text edit on generated source, not a TileLang form: P = #pragma unroll on GEMM's ki loop, O = no RegVec on arrays read as GEMM operands
    import re, tvm_ffi
    def _post(code, target):
        if os.environ["C2P_POST"] == "P":
            return code.replace("for (int ki = 0;", "#pragma unroll\n      for (int ki = 0;")
        for n in set(re.findall(r"\(\(float16x\d\*\)(\w+)\)", code)):
            code = re.sub(r"\*\(tl::RegVec<(uint\d)>\*\)\(" + n + r" \+", r"*(\1*)(" + n + " +", code)
        return code
    tvm_ffi.register_global_func("tilelang_callback_hip_postproc", _post, override=True)
meta = dict(arm_tree=ARM, tag=a.tag)
def emit(**k):
    print(json.dumps({**meta, **k}), flush=True)
emit(kind="info", tilelang=tilelang.__file__, ver=tilelang.__version__, torch=torch.__version__, fastexp=bool(os.environ.get("C2P_FASTEXP")),
     md5={f: hashlib.md5(open(p, "rb").read()).hexdigest() for f, p in (("moe", M.__file__), ("dec", gq.__file__))})
CASES = [(16, "uniform"), (16, "zipf1"), (16, "zipf2"), (64, "uniform"), (256, "uniform"), (1024, "uniform"), (4096, "uniform"), (4096, "zipf2")]
DEC = [(8, 32768), (32, 8192), (64, 4096), (64, 1024)]
Hkv, G, D_, PAGE = 4, 6, 256, 16
H, I, E, topk = 2048, 512, 256, 8

def dec_kernels(B, L):
    MAXP = L // PAGE; NBLK = B * MAXP
    return gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D_, PAGE, packed="lbnhc")

if a.compile:
    for Tn in sorted({t for t, _ in CASES}):
        t0 = time.time(); m = M.FusedMoE(Tn, device="cpu"); emit(kind="built", T=Tn, cfg=str(m.cfg), gu_flags=m.gu_flags, t=round(time.time() - t0, 1))
    for B, L in DEC:
        t0 = time.time(); k1, k2 = dec_kernels(B, L)
        emit(kind="built", B=B, L=L, t=round(time.time() - t0, 1), lazy=str(type(k1)))
    sys.exit(0)

EV = lambda: torch.cuda.Event(enable_timing=True)
def plain(fn):
    for _ in range(a.warm):
        fn(None)
    ps = []
    for _ in range(a.rep):
        s, e = EV(), EV(); s.record(); fn(None); e.record(); ps.append((s, e))
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) for s, e in ps], dtype=torch.float64)
    return dict(tot_med=round(t.median().item(), 4), tot_min=round(t.min().item(), 4))
def stages(fn, n):
    for _ in range(a.warm):
        fn(None)
    es = []
    for _ in range(a.rep):
        ev = [EV() for _ in range(n + 1)]; fn(lambda i, ev=ev: ev[i].record()); es.append(ev)
    torch.cuda.synchronize()
    t = torch.tensor([[e[i].elapsed_time(e[i + 1]) for i in range(n)] for e in es], dtype=torch.float64)
    return [round(x, 4) for x in t.median(0).values.tolist()]
def h(O):
    return hashlib.sha1(O.detach().cpu().numpy().tobytes()).hexdigest()[:12]

if "moe" in a.only:
    from vllm.model_executor.layers.fused_moe import fused_moe as FM
    Dt = torch.load("/scratch/project_XXXXXXXXX/tilelang/exp/C2P/md_data.pt")
    W13, W2 = Dt["W13"].cuda(), Dt["W2"].cuda()
    MEMO = {}
    for Tn, route in CASES:
        d = Dt[(Tn, route)]
        X, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
        out = torch.zeros(Tn, H, dtype=torch.float32, device=X.device)
        for e_ in ids.unique().tolist():
            tok, kk = (ids == e_).nonzero(as_tuple=True)
            hh = X[tok].float() @ W13[e_].float().T
            out.index_add_(0, tok, ((F.silu(hh[:, :I]) * hh[:, I:]) @ W2[e_].float().T) * wts[tok, kk, None])
        if Tn not in MEMO:
            MEMO[Tn] = M.FusedMoE(Tn); emit(kind="ship_cfg", T=Tn, cfg=str(MEMO[Tn].cfg), gu_flags=MEMO[Tn].gu_flags, unroll=MEMO[Tn].unroll)
        m = MEMO[Tn]; idi = ids.int()
        arms = {"ours": (lambda rec, m=m: m(X, W13, W2, ids, wts, rec)), "vllm": (lambda rec: FM.fused_experts(X, W13, W2, wts, idi))}
        for n, fn in arms.items():
            O = fn(None); dd = O.float() - out
            emit(kind="err", T=Tn, route=route, arm=n, rel=round((dd.norm() / out.norm()).item(), 7), nonfinite=int((~torch.isfinite(O.float())).sum()), sha=h(O))
        rng = random.Random(f"{a.seed}-{a.tag}-{Tn}-{route}")
        for r in range(a.rounds):
            order = list(arms); rng.shuffle(order)
            for n in order:
                res = plain(arms[n])
                if n == "ours":
                    res["stages"] = stages(arms[n], 4)
                emit(kind="time", wl="moe", T=Tn, route=route, arm=n, round=r, pos=order.index(n), **res)
        del X, ids, wts, out
    del W13, W2, Dt; torch.cuda.empty_cache()

if "decode" in a.only:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
    H_ = Hkv * G
    for B, L in DEC:
        torch.manual_seed(B)
        MAXP = L // PAGE; NBLK = B * MAXP
        lens = torch.full((B,), L, dtype=torch.int32, device="cuda"); lens[1::2] = L * 3 // 4
        Q = torch.randn(B, H_, D_, device="cuda", dtype=torch.float16)
        NH = torch.randn(NBLK, PAGE, Hkv, 2 * D_, device="cuda", dtype=torch.float16)   # LBNHC, TRITON_ATTN's default layout; same tensor for both arms
        kk, vv = NH[..., :D_], NH[..., D_:]
        BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
        R = gq.ref64(Q, kk[BT.long()].reshape(B, L, Hkv, D_), vv[BT.long()].reshape(B, L, Hkv, D_), lens, G) if B * L <= 32 * 8192 else None
        k1, k2 = dec_kernels(B, L)
        scale = 1.0 / math.sqrt(D_); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); uo = torch.empty_like(Q)
        def tri(rec=None):
            unified(q=Q, k=kk, v=vv, out=uo, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
            return uo
        arms = {"ours": (lambda rec=None: k2(*k1(Q, NH, NH, BT, lens))), "vllm_tri": tri}
        for n, fn in arms.items():
            O = fn()
            emit(kind="err", B=B, L=L, arm=n, rel_vs_fp64=round(((O.double() - R).norm() / R.norm()).item(), 7) if R is not None else None, nonfinite=int((~torch.isfinite(O.float())).sum()), sha=h(O))
        rng = random.Random(f"{a.seed}-{a.tag}-{B}-{L}")
        for r in range(a.rounds):
            order = list(arms); rng.shuffle(order)
            for n in order:
                emit(kind="time", wl="decode", B=B, L=L, arm=n, round=r, pos=order.index(n), **plain(arms[n]))
        del Q, NH, kk, vv, BT, R, uo; torch.cuda.empty_cache()
emit(kind="done")
