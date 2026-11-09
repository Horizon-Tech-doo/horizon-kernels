"""JPORT race, one GPU, one process, wheel 0.1.15 on the vLLM image (clang 22): the candidate module's dispatch (form J at 2816-3584) against the landed tile (rows 128 bN32, d128_tile's pick in a6e9732c) and CK, same process.
Ratio: ours / same-process vendor, throughput, higher is better; vendor read = mean of pre and post.
Arms: new = gqa_flash_fast(**d128_tile(L)) of the candidate module (J inside 2816-3584, the old tile at 2560 and 4096); old = gqa_flash_fast with the landed tile; jx = gqa_flash_j forced at the boundary lengths only (information, not dispatched).
Usage: jport.py --compile        CPU; fills the cache
       jport.py --tag T --order fwd|rev [--inputs ...] [--rep 50]
Order: fwd is plan order; rev reverses lengths, inputs and arms. No shuffle."""
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="c"); ap.add_argument("--order", default="fwd")
ap.add_argument("--inputs", default="randn0,randn1,self,big,ar,ramp"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
import tilelang
from tilelang.contrib import hipcc
_orig = hipcc.compile_hip; COMPILES = [0]


def _cnt(code, *x, **k):
    COMPILES[0] += 1
    return _orig(code, *x, **k)


hipcc.compile_hip = _cnt          # gqa_fa's hook wraps this one
import gqa_fa
from bench_common import bench
from n5lib import inputs, ref
LS = (2560, 2816, 3072, 3328, 3584, 4096)
OLD = dict(P=0, bN=32, threads=512, rows=128, rev=True, swap=True, recip=True)     # landed d128_tile pick at all six lengths on clang 22
K = {}


def tile(L, arm):
    if arm == "new":
        return gqa_fa.d128_tile(L)
    if arm == "old":
        return dict(OLD)
    return dict(OLD, bN=64, form="J")


def build(L, arm):
    if (L, arm) not in K:
        if arm == "jx":
            K[(L, arm)] = gqa_fa.gqa_flash_j(L)
        else:
            t = tile(L, arm)
            K[(L, arm)] = gqa_fa.gqa_flash_fast(1, L, 4, 6, 128, True, t["P"], t["bN"], t["threads"], 2, True, t["rev"], t["recip"], t["swap"], t.get("rows", 0), t.get("lb", 0), t.get("sched", ""), t.get("form", ""))
    return K[(L, arm)]


def arms(L):
    return ["new", "old"] + (["jx"] if L in (2560, 4096) else [])


if a.compile:
    for L in LS:
        for arm in arms(L):
            n0 = COMPILES[0]; t0 = time.time()
            build(L, arm)
            print(json.dumps(dict(L=L, arm=arm, tile=tile(L, arm) if arm != "jx" else "J", compiles=COMPILES[0] - n0, t=round(time.time() - t0, 1))), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, order=a.order, ver=tilelang.__version__, clang=gqa_fa.clang_major(),
                      gqa_fa=hashlib.md5(open(gqa_fa.__file__, "rb").read()).hexdigest())), flush=True)
from flash_attn import flash_attn_func
rv = a.order == "rev"
for L in sorted(LS, reverse=rv):
    s = dict(B=1, L=L, Hkv=4, G=6, D=128)
    flops = 4.0 * 24 * L * L * 128 * 0.5
    for inp in (a.inputs.split(",")[::-1] if rv else a.inputs.split(",")):
        Q, K_, V = inputs(inp, s)
        R = ref(Q, K_, V, True, 6)
        qt, kt, vt = Q.transpose(1, 2), K_.transpose(1, 2).repeat_interleave(6, 1), V.transpose(1, 2).repeat_interleave(6, 1)
        base = dict(case=f"d128_{L}", L=L, input=inp, tag=a.tag, order=a.order)
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True).transpose(1, 2)

        def err(O):
            return {"rel_vs_fp64": round(((O.double() - R).norm() / R.norm()).item(), 7), "rel_vs_sdpa": round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)}

        def vendor(impl, slot):
            f = (lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)) if impl == "sdpa" else (lambda: flash_attn_func(Q, K_, V, causal=True))
            x = {}
            if slot == "pre" and impl == "ck":
                O = f(); x = err(O); del O
            ms = bench(f, warmup=5 if impl == "sdpa" else 10, rep=25 if impl == "sdpa" else a.rep)
            print(json.dumps(dict(base, impl=impl, slot=slot, ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), **x)), flush=True)

        vendor("sdpa", "pre"); vendor("ck", "pre")
        ar_ = arms(L)
        for pos, arm in enumerate(ar_[::-1] if rv else ar_):
            r = dict(base, impl="tl", arm=arm, pos=pos, form=(tile(L, arm).get("form", "") if arm != "jx" else "J"))
            try:
                n0 = COMPILES[0]
                k = build(L, arm)
                r["compiles"] = COMPILES[0] - n0
                O = k(Q, K_, V)
                r.update(err(O))
                r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K_, V), O)) for _ in range(30))}/30"
                r["nonfinite"] = int((~torch.isfinite(O)).sum().item())
                del O
                ms = bench(lambda: k(Q, K_, V), warmup=10, rep=a.rep)
                r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            print(json.dumps(r), flush=True)
        vendor("ck", "post"); vendor("sdpa", "post")
        del Q, K_, V, R, So, qt, kt, vt
        torch.cuda.empty_cache()
print(json.dumps(dict(done=1, compiles=COMPILES[0])), flush=True)
