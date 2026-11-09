"""Wheel process, one GPU: is a port to the stock wheel possible with no TileLang change? Arm stub: frozen bytes (fork-main generated source + RegVec definition, compiled on the wheel's own headers, bytes3/, same instruction stream as the fork form) launched through a wheel wrapper from a stub prim_func that declares only K_s and V_s, so the launch asks 32768 B LDS. Arm native: landed gqa_flash_fast, d128_tile's pick. Experiment only, nothing lands. Ratio: ours / same-process vendor, TF, higher is better. Usage: wstub.py --compile | --tag T --order fwd|rev"""
import argparse, glob, hashlib, json, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="c"); ap.add_argument("--order", default="fwd"); ap.add_argument("--inputs", default="randn0,big,ar"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
from tilelang.contrib import hipcc
_orig = hipcc.compile_hip; HAND = [None]; COMPILES = [0]
def _cnt(code, *x, **k):
    COMPILES[0] += 1
    if HAND[0] is not None:
        return bytearray(HAND[0])
    return _orig(code, *x, **k)
hipcc.compile_hip = _cnt
import gqa_fa
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r5 import extend
extend(CASES)
BY = "/scratch/project_XXXXXXXXX/tilelang/exp/D128/bytes3"
PICK = {3072: dict(bN=32, th=512, rows=128), 8192: dict(bN=32, th=1024, rows=256)}
def stub(L, bN=64, threads=512, rows=128, B=1, Hkv=4, G=6, D=128):
    NB = -(-L * G // rows); dt = T.float16
    @T.prim_func
    def main(Q: T.Tensor((B, L, Hkv * G, D), dt), K: T.Tensor((B, L, Hkv, D), dt), V: T.Tensor((B, L, Hkv, D), dt), O: T.Tensor((B, L, Hkv * G, D), dt)):
        with T.Kernel(B * Hkv, NB, threads=threads) as (b0, b1):
            K_s = T.alloc_shared((bN, D), dt); V_s = T.alloc_shared((bN, D), dt)     # device code is discarded; both arrays must be read so the wrapper counts them
            T.copy(K[0, 0:bN, 0, :], K_s); T.copy(V[0, 0:bN, 0, :], V_s)
            T.sync_threads()
            for i, d in T.Parallel(bN, D):
                O[0, i, 0, d] = K_s[i, d] + V_s[i, d] + Q[0, i, 0, d]
    return main
K = {}
def build(L, arm):
    if (L, arm) in K:
        return K[(L, arm)]
    if arm == "native":
        p = PICK[L]; k = gqa_fa.gqa_flash_fast(1, L, 4, 6, 128, True, 0, p["bN"], p["th"], 2, True, True, True, True, p["rows"])
    else:
        h = open(f"{BY}/d128_{L}_r128_b64_t512_JDRF_frozen.hsaco", "rb").read(); HAND[0] = h
        try:
            k = tilelang.compile(stub(L), out_idx=[3], compile_flags=["-DD128_STUB_" + hashlib.md5(h).hexdigest()[:12]])
        finally:
            HAND[0] = None
    K[(L, arm)] = k
    return k
ARMS = ["stub", "native"]
if a.compile:
    for L in PICK:
        for arm in ARMS:
            n0 = COMPILES[0]; build(L, arm); print(json.dumps(dict(L=L, arm=arm, compiles=COMPILES[0] - n0, ver=tilelang.__version__)), flush=True)
    for p in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/host_kernel.cu", recursive=True):
        s = open(p).read(); d = open(os.path.dirname(p) + "/device_kernel.cu").read()
        if "K_s" in d and "acc_s" not in d:
            print(json.dumps(dict(stub_lds=re.findall(r"\[9\]\.v_int64\) = \(\(int64_t\)(\d+)\)", s), stub_dir=os.path.basename(os.path.dirname(p))[:8])), flush=True)
    sys.exit(0)
print(json.dumps(dict(info=1, tilelang=tilelang.__file__, ver=tilelang.__version__, tag=a.tag, order=a.order, gqa_fa=hashlib.md5(open(gqa_fa.__file__, "rb").read()).hexdigest())), flush=True)
from flash_attn import flash_attn_func
rv = a.order == "rev"
for L in sorted(PICK, reverse=rv):
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
        for pos, arm in enumerate(ARMS[::-1] if rv else ARMS):
            r = dict(base, impl="tl", kind=arm, name=f"d128_{L}_{arm}", pos=pos)
            try:
                n0 = COMPILES[0]; k = build(L, arm); r["compiles"] = COMPILES[0] - n0
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
