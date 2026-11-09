"""Wheel process, one GPU, native gqa_flash_fast only (its own wrapper, no byte swap): hand rows 128 at bN 32, 48, 64 through the landed path. Ratio: ours / same-process vendor, TF, higher is better. Usage: wport2.py --compile | --tag T --order fwd|rev"""
import argparse, hashlib, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="c"); ap.add_argument("--order", default="fwd"); ap.add_argument("--inputs", default="randn0,big"); ap.add_argument("--rep", type=int, default=50)
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
hipcc.compile_hip = _cnt
import gqa_fa
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r5 import extend
extend(CASES)
ARMS = [(32, 512, 128), (48, 512, 128), (64, 512, 128)]
LS = (3072, 8192)
def build(L, t):
    return gqa_fa.gqa_flash_fast(1, L, 4, 6, 128, True, 0, t[0], t[1], 2, True, True, True, True, t[2])
if a.compile:
    for L in LS:
        for t in ARMS:
            n0 = COMPILES[0]; build(L, t); print(json.dumps(dict(L=L, t=t, compiles=COMPILES[0] - n0)), flush=True)
    sys.exit(0)
print(json.dumps(dict(info=1, tilelang=tilelang.__file__, ver=tilelang.__version__, tag=a.tag, order=a.order, gqa_fa=hashlib.md5(open(gqa_fa.__file__, "rb").read()).hexdigest())), flush=True)
from flash_attn import flash_attn_func
rv = a.order == "rev"
for L in sorted(LS, reverse=rv):
    s = CASES[f"d128_{L}"]; flops = 4.0 * 24 * L * L * 128 * 0.5
    for inp in (a.inputs.split(",")[::-1] if rv else a.inputs.split(",")):
        Q, K_, V = inputs(inp, s); R = ref(Q, K_, V, True, 6)
        qt, kt, vt = Q.transpose(1, 2), K_.transpose(1, 2).repeat_interleave(6, 1), V.transpose(1, 2).repeat_interleave(6, 1)
        base = dict(case=f"d128_{L}", input=inp, tag=a.tag, order=a.order)
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True).transpose(1, 2)
        def vendor(impl, slot):
            f = (lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)) if impl == "sdpa" else (lambda: flash_attn_func(Q, K_, V, causal=True))
            ms = bench(f, warmup=5 if impl == "sdpa" else 10, rep=25 if impl == "sdpa" else a.rep)
            print(json.dumps(dict(base, impl=impl, slot=slot, name=f"d128_{L}_{impl}", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        vendor("sdpa", "pre"); vendor("ck", "pre")
        for pos, t in enumerate(ARMS[::-1] if rv else ARMS):
            r = dict(base, impl="tl", kind="native", name=f"d128_{L}_r{t[2]}_b{t[0]}_t{t[1]}_native", pos=pos, bN=t[0])
            try:
                n0 = COMPILES[0]; k = build(L, t); r["compiles"] = COMPILES[0] - n0
                O = k(Q, K_, V)
                r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7); r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K_, V), O)) for _ in range(30))}/30"
                del O
                ms = bench(lambda: k(Q, K_, V), warmup=10, rep=a.rep); r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            print(json.dumps(r), flush=True)
        vendor("ck", "post"); vendor("sdpa", "post")
        del Q, K_, V, R, So, qt, kt, vt
        torch.cuda.empty_cache()
print(json.dumps(dict(done=1, compiles=COMPILES[0])), flush=True)
