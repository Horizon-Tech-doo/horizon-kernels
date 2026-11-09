"""Correctness of the grid kernels (n1grid.GridMoE, T-row sum) at every T = 1 to 4096 on three routings. One set of inputs per routing at T = 4096;
input T is its prefix. Each output row depends only on that token's X, ids and weights, so the fp32 reference for T is the first T rows of the
reference at 4096; checked directly at a sample of T. Each G serves T in (G/2, G]. Grid kernels are a prototype, not the landed per-T import.
args: --data file --seed n --gs 1,2,4,.. [--build-only] [--rev] [--vllm]"""
import argparse, json, os, random, sys, time
import torch
import torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--seed", type=int, default=21); ap.add_argument("--gs", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
ap.add_argument("--build-only", action="store_true"); ap.add_argument("--rev", action="store_true"); ap.add_argument("--vllm", action="store_true")
a = ap.parse_args()
H, I, E, topk, TM = 2048, 512, 256, 8, 4096
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_fused as M, n1grid
GS = [int(x) for x in a.gs.split(",")]
meta = {"jobid": os.environ.get("SLURM_JOB_ID"), "seed": a.seed}
def emit(**k): print(json.dumps({**meta, **k}), flush=True)
if a.build_only:
    for G in GS:
        n1grid.GridMoE(G, sum_mode="gridt"); emit(kind="built", G=G)
    sys.exit(0)
D = torch.load(a.data)
W13, W2 = D["W13"].cuda(), D["W2"].cuda()
g = torch.Generator().manual_seed(a.seed)
rank = torch.randperm(E, generator=g).float()
X = torch.randn(TM, H, generator=g).half()
wts, ids = torch.topk(torch.softmax(torch.randn(TM, E, generator=g), -1), topk, dim=-1)
R_ = {"uniform": (ids.int(), (wts / wts.sum(-1, keepdim=True)).float())}
for name, s in (("zipf1", 1.0), ("zipf2", 2.0)):
    gum = -torch.log(-torch.log(torch.rand(TM, E, generator=g).clamp_min(1e-20)))
    v, i2 = torch.topk(-s * torch.log(rank + 1) + gum, topk, dim=-1)
    R_[name] = (i2.int(), torch.softmax(v, -1).float())
X = X.cuda()
def reference(X, ids, wts):
    Tn = ids.shape[0]
    out = torch.zeros(Tn, H, dtype=torch.float32, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].float() @ W13[e].float().T
        ac = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (ac @ W2[e].float().T) * wts[tok, kk, None])
    return out
grids = {G: n1grid.GridMoE(G, sum_mode="gridt") for G in GS}
def gof(Tn):
    return next(G for G in GS if G >= Tn)
Tmax = max(GS)
rng = random.Random(a.seed)
SAMPLE = [t for t in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1023, 1025, 2047, 2049, 2800, 4095, 4096) if t <= Tmax]
for route, (idc, wc) in R_.items():
    idc, wc = idc.cuda(), wc.cuda()
    t0 = time.time()
    R = reference(X, idc, wc)
    pre = [t for t in SAMPLE]
    pchk = max(((reference(X[:t], idc[:t], wc[:t]) - R[:t]).norm() / R[:t].norm()).item() for t in pre)
    emit(kind="prefix_check", route=route, max_rel_prefix_vs_direct=round(pchk, 9), n=len(pre))
    Ts = list(range(1, Tmax + 1))
    if route == "uniform":
        if a.rev: Ts.reverse()
    else:
        rng.shuffle(Ts)
    rel = [0.0] * (Tmax + 1); tok = [0.0] * (Tmax + 1); nf = 0; bad = []
    for Tn in Ts:
        O = grids[gof(Tn)](X[:Tn], W13, W2, idc[:Tn], wc[:Tn])
        Rt = R[:Tn]
        Of = O.float()
        d = Of - Rt
        n = int((~torch.isfinite(Of)).sum().item())
        nf += n
        rel[Tn] = (d.norm() / Rt.norm()).item()
        tok[Tn] = (d.norm(dim=1) / Rt.norm(dim=1)).max().item()
        if n: bad.append(Tn)
    emit(kind="err_all_T", route=route, n=len(Ts), order=("rev" if a.rev else "asc") if route == "uniform" else "shuffled", worst_rel=round(max(rel), 7), worst_T=rel.index(max(rel)),
         worst_token_rel=round(max(tok), 7), worst_token_T=tok.index(max(tok)), nonfinite=nf, nonfinite_T=bad[:20], over_1e3=sum(1 for x in rel if x > 1e-3), rel_T_ge_16=round(max(rel[16:] or [0]), 7), wall_s=round(time.time() - t0, 1),
         first_T=Ts[0])
    emit(kind="err_per_T", route=route, rel=[round(x, 7) for x in rel[1:]], tok=[round(x, 7) for x in tok[1:]])
    if a.vllm:
        from vllm.model_executor.layers.fused_moe import fused_moe as FM
        for Tn in SAMPLE:
            O = FM.fused_experts(X[:Tn], W13, W2, wc[:Tn], idc[:Tn])
            d = O.float() - R[:Tn]
            emit(kind="vllm_err", route=route, T=Tn, rel=round((d.norm() / R[:Tn].norm()).item(), 7), tok=round((d.norm(dim=1) / R[:Tn].norm(dim=1)).max().item(), 7))
