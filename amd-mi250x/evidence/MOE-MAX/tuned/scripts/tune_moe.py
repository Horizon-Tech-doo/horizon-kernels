"""Tune vLLM's Triton fused_moe config for one batch size T on the MoE shape of the MOE-MAX races (E=256, N=512, H=2048,
top-k 8, fp16), the way vLLM's benchmarks/kernels/benchmark_moe.py --tune does, without Ray (not in the vLLM 0.30.0 image).

What is vLLM's own code here: the ROCm search space (get_configs_compute_bound -> get_rocm_tuning_space: BLOCK_SIZE_M/N in
16..256, BLOCK_SIZE_K in 16..256, GROUP_SIZE_M 1/4/8/16/32, num_warps 1/2/4/8, num_stages 2, waves_per_eu 0/1/2/4,
matrix_instr_nonkdim 16/32, kpack 1/2 = 40,000 configs) and its pruning (prune_rocm_configs for both GEMMs), imported from the
image's benchmark_moe.py through a stub 'ray' module. The measurement mirrors benchmark_config: override_config(cfg), fused_topk
on random gating, fused_experts, a HIP graph of several calls, new gating before every replay, mean time per call.

Two modes. Exhaustive (stages full, fin), as vLLM's tuner: every pruned config is timed (used where the pruned space is small
enough, T <= 8: 704 to 4,992 configs). Staged (stages s1, s2, fin), for T >= 16 where the pruned space is 7,808 to 25,760 configs:
  full every pruned config, random order, plus vLLM's default config (exhaustive mode only).
  s1  every pruned config with GROUP_SIZE_M 1, waves_per_eu 0, matrix_instr_nonkdim 16, kpack 1 (all block shapes x num_warps),
      random order, plus vLLM's default config; a config whose eager call is > 3x the best eager call so far is not graphed.
  s2  for the --tops (default 8) fastest s1 block/warp choices: every pruned combination of GROUP_SIZE_M, waves_per_eu, matrix_instr_nonkdim, kpack.
  fin the 5 fastest of full or s1+s2 and vLLM's default, 3 interleaved rounds at vLLM's tuner settings (graph of 10 calls, 20 replays);
      the fastest by median of round means is the tuned config (the default itself if nothing beats it).
Weights are scaled (1/sqrt(fan-in)) so the fp16 output stays finite; vLLM's tuner uses unscaled randn.
--prebuild compiles a shard of a stage's configs (several processes fill the Triton cache in parallel; nothing is timed).
Some configs abort the process inside the compiler ("LLVM ERROR: Found 1 machine code errors"). Every config is logged before
it is tried (prebuild: <out>.pb.<stage>.<shard>; timing: a 'start' record), the job script reruns the process, and a config
that was started and never finished is recorded as failed and not tried again.
In full/s1/s2 a config whose eager call is > 3x the best eager call + 1 ms is recorded as skipped and not graphed.
--route zipf1|zipf2 replaces vLLM's uniform random gating by the skewed routing recipe of the race data (ms_mk.py: logits
-s*log(rank+1) + Gumbel noise, s = 1 or 2, fixed random expert rank); not something vLLM's tuner offers. Default: uniform.
argv: --T n --stage full|s1|s2|fin --out file.jsonl [--prebuild --shard i/n] [--budget seconds]"""
import argparse, gc, json, math, os, random, statistics as st, sys, time, types
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int); ap.add_argument("--stage", choices=["full", "s1", "s2", "fin"]); ap.add_argument("--out")
ap.add_argument("--prebuild", action="store_true"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--budget", type=float, default=600.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tops", type=int, default=8); ap.add_argument("--route", default="uniform", choices=["uniform", "zipf1", "zipf2"])
a = ap.parse_args()
_ray = types.ModuleType("ray"); _ray.remote = lambda *x, **k: (lambda c: c)
_exp = types.ModuleType("ray.experimental"); _tq = types.ModuleType("ray.experimental.tqdm_ray"); _tq.tqdm = lambda x, *p, **k: x
sys.modules.update({"ray": _ray, "ray.experimental": _exp, "ray.experimental.tqdm_ray": _tq})
import torch
sys.path.append("/app/vllm/benchmarks/kernels")
import benchmark_moe as BM
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk, override_config
from vllm.model_executor.layers.fused_moe import fused_moe as FM
from vllm.triton_utils import triton

H, I, E, topk = 2048, 512, 256, 8
T = a.T
key = lambda c: json.dumps(c, sort_keys=True)
OUT = open(a.out, "a") if (a.out and not a.prebuild) else None
def emit(**k):
    s = json.dumps({"T": T, "stage": a.stage, "route": a.route, "jobid": os.environ.get("SLURM_JOB_ID"), **k})
    if OUT:
        OUT.write(s + "\n"); OUT.flush()
    if k.get("kind") != "start":
        print(s[:400], flush=True)

def prior():
    recs = []
    if a.out and os.path.exists(a.out):
        for l in open(a.out):
            try:
                recs.append(json.loads(l))
            except ValueError:
                pass
    return recs

DT = FM._get_config_dtype_str(dtype=torch.float16)
assert FM.get_moe_configs(E, I, DT) is None, "a tuned file is already visible to vLLM"
DEFAULT = dict(FM.try_get_optimal_moe_config((E, 2 * I, H), (E, H, I), topk, DT, T, block_shape=None))

def pruned():
    space = BM.get_configs_compute_bound(True, None)
    seen, out = set(), []
    for n, k in ((2 * I, H), (H, I)):                      # as prune_rocm_search_space, merged by key (its list merge is quadratic)
        for c in BM.prune_rocm_configs(T * topk, n, k, space, True):
            if key(c) not in seen:
                seen.add(key(c)); out.append(c)
    return len(space), out

def stage_list():
    nspace, P = pruned()
    base = lambda c: (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], c["num_warps"])
    s1 = [c for c in P if c["GROUP_SIZE_M"] == 1 and c["waves_per_eu"] == 0 and c["matrix_instr_nonkdim"] == 16 and c["kpack"] == 1]
    random.Random(1000 + T).shuffle(s1)
    if a.stage == "full":
        full = list(P); random.Random(3000 + T).shuffle(full)
        return nspace, len(P), [DEFAULT] + full
    if a.stage == "s1":
        return nspace, len(P), [DEFAULT] + s1
    R = [r for r in prior() if r.get("kind") == "time" and not r.get("skipped")]
    r1 = sorted([r for r in R if r["stage"] == "s1" and key(r["cfg"]) != key(DEFAULT)], key=lambda r: r["us"])
    if a.stage == "s2":
        tops = [base(r["cfg"]) for r in r1[:a.tops]]
        s1k = {key(c) for c in s1}
        s2 = [c for c in P if base(c) in tops and key(c) not in s1k]
        random.Random(2000 + T).shuffle(s2)
        return nspace, len(P), s2
    best = {}
    for r in R:
        if r["stage"] in ("full", "s1", "s2") and key(r["cfg"]) != key(DEFAULT):
            best[key(r["cfg"])] = min(best.get(key(r["cfg"]), 1e30), r["us"])
    top = [json.loads(k) for k, _ in sorted(best.items(), key=lambda kv: kv[1])[:5]]
    return nspace, len(P), [DEFAULT] + top

g = torch.Generator(device="cuda").manual_seed(a.seed)
x = torch.randn(T, H, generator=g, device="cuda").half()
w1 = (torch.randn(E, 2 * I, H, generator=g, device="cuda") / math.sqrt(H)).half()
w2 = (torch.randn(E, H, I, generator=g, device="cuda") / math.sqrt(I)).half()
NG = 20
gating = torch.randn(NG, T, E, generator=g, device="cuda")
if a.route != "uniform":
    rank = torch.randperm(E, generator=g, device="cuda").float()
    gum = -torch.log(-torch.log(torch.rand(NG, T, E, generator=g, device="cuda").clamp_min(1e-20)))
    gating = -float(a.route[-1]) * torch.log(rank + 1) + gum
input_gating = torch.empty(T, E, device="cuda")
input_gating.copy_(gating[0])

def run(cfg):
    with override_config(cfg):
        r = fused_topk(x, input_gating, topk, renormalize=True)
        return fused_experts(x, w1, w2, r[0], r[1])

def measure(cfg, ncall, iters, eager_best=None):
    input_gating.copy_(gating[0])
    o = run(cfg); torch.cuda.synchronize()                 # JIT
    t0 = time.perf_counter(); o = run(cfg); torch.cuda.synchronize(); eager = (time.perf_counter() - t0) * 1e3
    res = dict(eager_ms=round(eager, 4))
    if eager_best is not None and eager > 3 * eager_best + 1.0:
        return dict(res, skipped=True), o
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(ncall):
            run(cfg)
    torch.cuda.synchronize()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    lat = []
    for i in range(iters):
        input_gating.copy_(gating[i % NG]); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); gr.replay(); e.record(); e.synchronize()
        lat.append(s.elapsed_time(e))
    gr.reset()
    return dict(res, us=round(sum(lat) / (iters * ncall) * 1000, 3), us_med=round(st.median(lat) / ncall * 1000, 3)), o

nspace, npr, L = stage_list()
def pb_logs():
    import glob
    st_, dn = [], set()
    for p in glob.glob(f"{a.out}.pb.{a.stage}.*"):
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            (st_.append(r["k"]) if r["e"] == "s" else dn.add(r["k"]))
    return st_, dn
if a.prebuild:
    i, n = (int(v) for v in a.shard.split("/"))
    pf = f"{a.out}.pb.{a.stage}.{i}"
    seen = set()
    if os.path.exists(pf):
        for l in open(pf):
            try:
                seen.add(json.loads(l)["k"])
            except ValueError:
                pass
    ok = bad = 0
    with open(pf, "a") as f:
        for c in L[i::n]:
            if key(c) in seen:                              # done, or it killed an earlier attempt
                continue
            f.write(json.dumps({"e": "s", "k": key(c)}) + "\n"); f.flush()
            try:
                run(c); torch.cuda.synchronize(); ok += 1
            except Exception:
                bad += 1
            f.write(json.dumps({"e": "d", "k": key(c)}) + "\n"); f.flush()
    print(json.dumps({"kind": "prebuilt", "T": T, "stage": a.stage, "shard": a.shard, "ok": ok, "failed": bad, "resumed_past": len(seen)}), flush=True)
    sys.exit(0)

P0 = [r for r in prior() if r.get("stage") == a.stage]
done = {key(r["cfg"]) for r in P0 if r.get("kind") in ("time", "fail")}
for r in P0:                                                # a config that was started and never finished killed the process
    if r.get("kind") == "start" and key(r["cfg"]) not in done:
        emit(kind="fail", cfg=r["cfg"], err="process died in this config"); done.add(key(r["cfg"]))
pst, pdn = pb_logs() if a.stage != "fin" else ([], set())
for k in pst:                                               # aborted the compiler in the prebuild
    if k not in pdn and k not in done:
        emit(kind="fail", cfg=json.loads(k), err="process died in this config (prebuild)"); done.add(k)
emit(kind="meta", space=nspace, pruned=npr, n=len(L), default=DEFAULT, triton=triton.__version__, torch=torch.__version__,
     vllm=__import__("vllm").__version__, device=torch.cuda.get_device_name(0), cfg_file=FM.get_config_file_name(E, I, DT), resumed=len(done))
t_start = time.time()
if a.stage in ("full", "s1", "s2"):
    ncall, iters = (10, 5) if T <= 24 else (4, 4)
    eb = min([r["eager_ms"] for r in prior() if r.get("kind") == "time" and "eager_ms" in r and not r.get("skipped")], default=None)
    n = 0
    for c in L:
        if key(c) in done:
            continue
        if time.time() - t_start > a.budget:
            emit(kind="budget_stop", timed=n, left=sum(1 for c2 in L if key(c2) not in done)); break
        emit(kind="start", cfg=c)
        try:
            r, _ = measure(c, ncall, iters, eb)
            emit(kind="time", cfg=c, **r)
            if not r.get("skipped"):
                eb = r["eager_ms"] if eb is None else min(eb, r["eager_ms"])
        except Exception as ex:
            emit(kind="fail", cfg=c, err=(type(ex).__name__ + ": " + str(ex))[-300:])
        done.add(key(c)); n += 1
        if n % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
    emit(kind="stage_done", timed=n, secs=round(time.time() - t_start, 1))
else:
    per, outs = {}, {}
    for rnd in range(3):
        for c in (L if rnd % 2 == 0 else L[::-1]):
            try:
                r, o = measure(c, 10, 20)
                per.setdefault(key(c), []).append(r["us"])
                if rnd == 0:
                    outs[key(c)] = o.float().clone()
                emit(kind="time", cfg=c, round=rnd, **r)
            except Exception as ex:
                emit(kind="fail", cfg=c, err=(type(ex).__name__ + ": " + str(ex))[-300:])
    med = {k: st.median(v) for k, v in per.items() if len(v) == 3}
    kb = min(med, key=med.get)
    kd = key(DEFAULT)
    od = outs[kd]
    for k, o in outs.items():                               # same inputs and routing for every config
        emit(kind="out", cfg=json.loads(k), nonfinite=int((~torch.isfinite(o)).sum()), rel_vs_default=round(((o - od).norm() / od.norm()).item(), 7),
             maxabs_vs_default=round((o - od).abs().max().item(), 6))
    R = [r for r in prior() if r.get("kind") == "time"]
    emit(kind="best", cfg=BM.sort_config(json.loads(kb)) if kb != kd else DEFAULT, is_default=kb == kd, us=round(med[kb], 3), default_us=round(med[kd], 3),
         ratio=round(med[kb] / med[kd], 4), fin={k: [round(x, 3) for x in per[k]] for k in per},
         n_s1=sum(1 for r in R if r["stage"] == "s1"), n_s2=sum(1 for r in R if r["stage"] == "s2"), n_full=sum(1 for r in R if r["stage"] == "full"),
         n_skipped=sum(1 for r in R if r.get("skipped")), n_fail=sum(1 for r in prior() if r.get("kind") == "fail"),
         space=nspace, pruned=npr, triton=triton.__version__, cfg_file=FM.get_config_file_name(E, I, DT))
    emit(kind="stage_done", secs=round(time.time() - t_start, 1))
