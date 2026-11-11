"""MOE-MAX tuned-baseline race: the landed FusedMoE (moe_fused.py, table dispatch) against vLLM's Triton fused_experts with
vLLM's default config AND with the config tuned for this device (tune_moe.py -> vLLM-format JSON), in one process.
Reduced from round 6's mx6.py (same timing, graph capture, routing data and reference code). Arms:
  ship   landed module, FusedMoE(T)                         same   a second, separately built instance (identical-code noise)
  vllm   fused_experts, vLLM's own config lookup (no file for this device -> get_default_config)
  vllm2  the same call captured a second time (identical-code noise on the vLLM side)
  vllmt  fused_experts under override_config(tuned[T]); tuned[T] is picked from the JSON by vLLM's nearest-key rule
  vllmz  the same with a second config file (--tuned2: tuned on zipf2 routing); only at T that are keys of that file
'g:<arm>' captures one whole call in a HIP graph and times replays. Arm order rotates per round; --reverse flips it.
Correctness per case and arm: rel L2 against the fp32 per-expert reference (the established gate) and against an fp64 one;
tuned against default: max abs difference, rel L2 and the share of bitwise-equal outputs."""
import argparse, hashlib, json, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--cases"); ap.add_argument("--arms"); ap.add_argument("--tuned", default=""); ap.add_argument("--tuned2", default="")
ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--rep", type=int, default=60); ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--reverse", action="store_true"); ap.add_argument("--build-only", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
H, I, E, topk = 2048, 512, 256, 8
D = torch.load(a.data)
W13, W2 = D["W13"].cuda(), D["W2"].cuda()
meta = {"jobid": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__, "tag": a.tag}
EV = lambda: torch.cuda.Event(enable_timing=True)
def emit(**k):
    print(json.dumps({**meta, **k}), flush=True)

import moe_fused as M
emit(kind="module", md5=hashlib.md5(open(M.__file__, "rb").read()).hexdigest(), clang=M.clang_major())
TUNED = {}
if a.tuned:
    TUNED = {int(k): v for k, v in json.load(open(a.tuned)).items() if k != "triton_version"}
    emit(kind="tuned_file", md5=hashlib.md5(open(a.tuned, "rb").read()).hexdigest(), keys=sorted(TUNED))
TUNED2 = {}
if a.tuned2:
    TUNED2 = {int(k): v for k, v in json.load(open(a.tuned2)).items() if k != "triton_version"}
    emit(kind="tuned_file2", md5=hashlib.md5(open(a.tuned2, "rb").read()).hexdigest(), keys=sorted(TUNED2))

def reference(X, ids, wts, dt):
    out = torch.zeros(ids.shape[0], H, dtype=dt, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].to(dt) @ W13[e].to(dt).T
        ac = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (ac @ W2[e].to(dt).T) * wts[tok, kk, None].to(dt))
    return out

def bench_plain(fn):
    for _ in range(a.warm):
        fn(None)
    pairs = []
    for _ in range(a.rep):
        s, e = EV(), EV()
        s.record(); fn(None); e.record()
        pairs.append((s, e))
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) for s, e in pairs], dtype=torch.float64)
    return dict(tot_med=round(t.median().item(), 5), tot_mean=round(t.mean().item(), 5), tot_min=round(t.min().item(), 5))

def graphed(fn):
    """Capture one whole call; replay times the kernels with no host work between them."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(None)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(None)
    def run(rec, g=g, out=out):
        g.replay()
        return out
    return run

def vllm_fn(Tn, X, ids, wts, arm):
    from vllm.model_executor.layers import fused_moe as FP
    from vllm.model_executor.layers.fused_moe import fused_moe as FM
    idi = ids.int()
    dt = FM._get_config_dtype_str(use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, dtype=X.dtype)
    if arm in ("vllmt", "vllmz"):
        TD = TUNED if arm == "vllmt" else TUNED2
        kn = min(TD, key=lambda k: abs(k - Tn))
        cfg = TD[kn]
        emit(T=Tn, kind="vllm_config", arm=arm, key=kn, cfg=dict(cfg))
        def fn(rec):
            with FP.override_config(cfg):
                return FM.fused_experts(X, W13, W2, wts, idi)
        return fn
    assert FM.get_moe_configs(E, I, dt) is None, "vLLM found a config file for this device: the default arm would not be the default"
    cfg = FM.try_get_optimal_moe_config(W13.size(), W2.size(), topk, dt, Tn, block_shape=None)
    emit(T=Tn, kind="vllm_config", arm=arm, cfg=dict(cfg))
    return lambda rec: FM.fused_experts(X, W13, W2, wts, idi)

MEMO = {}
def ours(Tn, name):
    if (Tn, name) not in MEMO:
        m = M.FusedMoE(Tn)
        emit(T=Tn, kind="cfg", arm=name, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in m.cfg.items()}, P=m.P, path=getattr(m, "path", None))
        MEMO[(Tn, name)] = m
    return MEMO[(Tn, name)]

cases = [(int(c.split(":")[0]), c.split(":")[1]) for c in a.cases.split(",")]
for Tn, route in cases:
    d = D[(Tn, route)]
    X, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
    hits = ids.flatten().bincount(minlength=E)
    emit(T=Tn, route=route, kind="routing", experts_hit=int((hits > 0).sum()), max_rows=int(hits.max()), mean_rows=round(hits.float().mean().item(), 2))
    arms = {}
    for spec in a.arms.split(","):
        gr = spec.startswith("g:")
        base = spec[2:] if gr else spec
        if base == "vllmz" and Tn not in TUNED2:
            continue
        try:
            if base.startswith("vllm"):
                if a.build_only:
                    continue
                fn = vllm_fn(Tn, X, ids, wts, base)
            else:
                m = ours(Tn, base)
                fn = lambda rec, m=m: m(X, W13, W2, ids, wts, rec)
            if a.build_only:
                continue
            if gr:
                fn = graphed(fn)
            arms[spec] = fn
        except Exception as ex:
            emit(T=Tn, route=route, kind="arm_failed", arm=spec, err=(type(ex).__name__ + ": " + str(ex))[-500:])
    if a.build_only:
        emit(T=Tn, route=route, kind="prebuilt")
        continue
    R32, R64 = reference(X, ids, wts, torch.float32), reference(X, ids, wts, torch.float64)
    outs = {}
    for n, fn in arms.items():
        O = fn(None); torch.cuda.synchronize()
        O = O.clone()
        outs[n] = O
        emit(T=Tn, route=route, kind="err", arm=n, dtype=str(O.dtype), rel_vs_fp32=round(((O.float() - R32).norm() / R32.norm()).item(), 7),
             rel_vs_fp64=round(((O.double() - R64).norm() / R64.norm()).item(), 7), nonfinite=int((~torch.isfinite(O)).sum()))
    for p in ("g:", ""):
        if p + "vllmt" in outs and p + "vllm" in outs:
            A, B = outs[p + "vllmt"].double(), outs[p + "vllm"].double()
            emit(T=Tn, route=route, kind="tuned_vs_default", graph=bool(p), maxabs=round((A - B).abs().max().item(), 7), rel=round(((A - B).norm() / B.norm()).item(), 7),
                 bitwise_equal_share=round((A == B).double().mean().item(), 5), maxabs_out=round(B.abs().max().item(), 5))
    order = list(arms)[::-1] if a.reverse else list(arms)
    for r in range(a.rounds):
        sh = r * len(order) // max(a.rounds, 1)
        for n in order[sh:] + order[:sh]:
            emit(T=Tn, route=route, kind="time", arm=n, round=r, **bench_plain(arms[n]))
