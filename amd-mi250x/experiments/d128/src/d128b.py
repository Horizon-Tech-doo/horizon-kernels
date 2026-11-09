"""D128 race, one GPU, one process: SDPA and CK before and after the arms, form arms (fork-main TileLang source) and hand arms (exact wheel bytes). Ratio: ours / same-process vendor, throughput, higher is better.
Usage: d128.py --compile BYTESDIR [--re R]   CPU; fills the cache, keeps form bytes, wraps hand bytes
       d128.py --tag T --order fwd|rev [--re R] [--inputs randn0,...] [--rep 50]
Order: fwd is plan order; rev reverses lengths, inputs and arms, so two jobs run in opposite order. No shuffle."""
import argparse, glob, hashlib, json, os, re, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--compile"); ap.add_argument("--re", default="."); ap.add_argument("--tag", default="c"); ap.add_argument("--order", default="fwd")
ap.add_argument("--inputs", default="randn0,randn1,self,big,ar,ramp"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r5 import extend
extend(CASES)
import tilelang
from tilelang.contrib import hipcc
_orig = hipcc.compile_hip
COMPILES = [0]; HAND = [None]; KEEP = [None]
BYTES = os.environ.get("D128_BYTES", os.path.join(os.path.dirname(os.path.abspath("plan.json")), "..", "bytes"))


def _cnt(code, *x, **k):
    COMPILES[0] += 1
    if HAND[0] is not None:         # the fork wrapper launches the wheel's code object; no hipcc
        return bytearray(HAND[0])
    b = _orig(code, *x, **k)
    if KEEP[0]:
        open(f"{KEEP[0]}.cc", "w").write(code); open(f"{KEEP[0]}.hsaco", "wb").write(bytes(b))
    return b


hipcc.compile_hip = _cnt
import tlfa
ARMS = [c for c in json.load(open("plan2.json")) if re.search(a.re, c["name"])]
K = {}


def build(c, bytes_dir=BYTES):
    if c["name"] in K:
        return K[c["name"]]
    pos = (1, c["L"], 4, 6, 128, True, 0, c["bN"], c["th"])
    if c["kind"] == "form":
        k = tlfa.fwd(*pos, c["opts"], 2, True, True, True, True, c["rows"])
    else:
        h = open(f"{bytes_dir}/{c['name']}.hsaco", "rb").read()
        HAND[0] = h
        try:     # opts only shape the wrapper's prim_func signature and grid; generated device code is discarded
            k = tilelang.compile(tlfa.make(*pos, "SDRF", 2, True, True, True, True, c["rows"]), out_idx=[3], compile_flags=["-DD128_HAND_" + hashlib.md5(h).hexdigest()[:12]])
        finally:
            HAND[0] = None
    K[c["name"]] = k
    return k


def libs():
    """md5 of every cached kernel library this process can load."""
    return {os.path.basename(os.path.dirname(p))[:12]: hashlib.md5(open(p, "rb").read()).hexdigest() for p in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/kernel_lib.so", recursive=True)}


if a.compile:
    os.makedirs(a.compile, exist_ok=True)
    for c in ARMS:
        t0 = time.time(); n0 = COMPILES[0]; r = dict(name=c["name"], kind=c["kind"])
        KEEP[0] = f"{a.compile}/{c['name']}" if c["kind"] == "form" else None
        try:
            build(c, a.compile)
            if c["kind"] == "form" and COMPILES[0] > n0:
                r["hsaco_md5"] = hashlib.md5(open(KEEP[0] + ".hsaco", "rb").read()).hexdigest(); r["cc_md5"] = hashlib.md5(open(KEEP[0] + ".cc", "rb").read()).hexdigest()
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-400:]
        r.update(compiles=COMPILES[0] - n0, t=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, order=a.order, n=len(ARMS), tilelang=tilelang.__file__, ver=tilelang.__version__, libs=libs(),
                      gcd=dict(rocr=os.environ.get("ROCR_VISIBLE_DEVICES"), uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))))), flush=True)
from flash_attn import flash_attn_func
rv = a.order == "rev"
cases = sorted({c["case"] for c in ARMS}, key=lambda x: int(x.split("_")[1]), reverse=rv)
for case in cases:
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if L <= 8192 else max(20, a.rep // 4)
    for inp in (a.inputs.split(",")[::-1] if rv else a.inputs.split(",")):
        Q, K_, V = inputs(inp, s)
        R = ref(Q, K_, V, causal, G) if L <= 8192 else None
        qt, kt, vt = Q.transpose(1, 2), K_.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, order=a.order)
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)

        def err(O):
            x = {"rel_vs_sdpa": round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)}
            if R is not None:
                x["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
            return x

        def vendor(impl, slot):
            f = (lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)) if impl == "sdpa" else (lambda: flash_attn_func(Q, K_, V, causal=causal))
            x = {}
            if slot == "pre":
                O = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2) if impl == "sdpa" else f()
                x = err(O); del O
            ms = bench(f, warmup=5 if impl == "sdpa" else 10, rep=max(10, rep // 2) if impl == "sdpa" else rep)
            print(json.dumps(dict(base, name=f"{case}_{impl}", impl=impl, slot=slot, ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), **x)), flush=True)

        vendor("sdpa", "pre"); vendor("ck", "pre")
        cs = [c for c in ARMS if c["case"] == case]
        for pos, c in enumerate(cs[::-1] if rv else cs):
            r = dict(base, name=c["name"], impl="tl", kind=c["kind"], opts=c["opts"], bN=c["bN"], th=c["th"], rows=c["rows"], pos=pos)
            t0 = time.time()
            try:
                n0 = COMPILES[0]
                k = build(c)
                r["compiles"] = COMPILES[0] - n0
                O = k(Q, K_, V)
                r.update(err(O))
                nrep = 30 if L <= 8192 else 10
                r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K_, V), O)) for _ in range(nrep))}/{nrep}"
                r["nonfinite"] = int((~torch.isfinite(O)).sum().item())
                del O
                ms = bench(lambda: k(Q, K_, V), warmup=10, rep=rep)
                r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
        vendor("ck", "post"); vendor("sdpa", "post")
        del Q, K_, V, R, So, qt, kt, vt
        torch.cuda.empty_cache()
print(json.dumps(dict(done=1, compiles=COMPILES[0])), flush=True)
