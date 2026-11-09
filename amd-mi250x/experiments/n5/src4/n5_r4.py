"""N5 round 4. Landed module only (tilelang/gqa_fa.py signature). Arm order is a real shuffle, seeded by tag and case; position saved in each row. SDPA and CK each run before the first arm and after the last, slot saved in the row.
Plan files: JSON list of dicts: case, P, bN, th, rows, rev, swap, lb.
Usage: n5_r4.py --plan X --compile
       n5_r4.py --plan X --tag r1 --check [--ck] [--inputs randn0,ar] [--rep 50]"""
import argparse, glob, hashlib, json, os, random, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="c")
ap.add_argument("--inputs", default="randn0"); ap.add_argument("--check", action="store_true"); ap.add_argument("--ck", action="store_true"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
from tilelang.contrib import hipcc
_orig = hipcc.compile_hip
COMPILES = [0]


def _cnt(code, *x, **k):
    COMPILES[0] += 1
    return _orig(code, *x, **k)


hipcc.compile_hip = _cnt
import gqa_fa
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r4 import extend
extend(CASES)

ARMS = []
for c in json.load(open(a.plan)):
    c["name"] = f"{c['case']}_{'r' + str(c['rows']) if c.get('rows') else 'P' + str(c['P'])}_b{c['bN']}_t{c['th']}{'_lb' + str(c['lb']) if c.get('lb') else ''}"
    ARMS.append(c)


def build(c):
    s = CASES[c["case"]]
    return gqa_fa.gqa_flash_fast(s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"], c.get("P", 0), c["bN"], c["th"], 2, True, c.get("rev", True), True, c.get("swap", True), c.get("rows", 0), c.get("lb", 0))


def kernel_meta(t0):
    """Registers, occupancy and code-object hash of the kernel this build just wrote, so a flag set can be shown to change the code. Newest cache entry written since t0."""
    cdir = os.environ["TILELANG_CACHE_DIR"] + "/0.1.15/linux-x86_64/kernels"
    fs = sorted(glob.glob(cdir + "/*/resource_usage.json"), key=os.path.getmtime)
    if not fs or os.path.getmtime(fs[-1]) < t0 - 1:
        return {}
    d = os.path.dirname(fs[-1])
    ru = json.load(open(fs[-1]))["main_kernel"]
    h = hashlib.md5()
    names = []
    for f in sorted(glob.glob(d + "/*")):
        if f.endswith((".hsaco", ".so", ".o", ".bin")):
            names.append(os.path.basename(f)); h.update(open(f, "rb").read())
    return dict(vgprs=ru["extra"].get("VGPRs"), occ=ru["extra"].get("Occupancy [waves/SIMD]"), scratch=ru["scratch_bytes"], obj=names, obj_md5=h.hexdigest())


if a.compile:
    for c in ARMS:
        t0 = time.time(); n0 = COMPILES[0]; r = dict(name=c["name"])
        try:
            build(c)
            if COMPILES[0] > n0:
                r.update(kernel_meta(t0))
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-400:]
        r.update(compiles=COMPILES[0] - n0, t=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
    sys.exit(0)


def gcd_id():
    """Hardware identifier of the visible GCD, no node name."""
    out = dict(rocr=os.environ.get("ROCR_VISIBLE_DEVICES"), ordinal=os.environ.get("GPU_DEVICE_ORDINAL"))
    try:
        out["uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    except Exception as e:  # noqa: BLE001
        out["uuid_err"] = type(e).__name__
    return out


print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, n=len(ARMS), stack=os.environ.get("N5_STACK"), flags=os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", ""), gcd=gcd_id())), flush=True)
for case in sorted({c["case"] for c in ARMS}):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if B * L <= 8192 else max(20, a.rep // 4)
    for inp in a.inputs.split(","):
        Q, K, V = inputs(inp, s)
        R = ref(Q, K, V, causal, G) if a.check and B * L <= 8192 else None
        qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, stack=os.environ.get("N5_STACK"))
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)

        def run_sdpa(slot):
            t0_ = time.time()
            ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
            vr = {}
            if R is not None and slot == "pre":
                vr["rel_vs_fp64"] = round(((So.double() - R).norm() / R.norm()).item(), 7)
            print(json.dumps(dict(base, name=case + "_sdpa", impl="sdpa", slot=slot, ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), t0=t0_, t1=time.time(), **vr)), flush=True)

        def run_ck(slot):
            from flash_attn import flash_attn_func
            vr = {}
            if R is not None and slot == "pre":
                Oc = flash_attn_func(Q, K, V, causal=causal)
                vr["rel_vs_fp64"] = round(((Oc.double() - R).norm() / R.norm()).item(), 7)
                del Oc
            t0_ = time.time()
            ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=rep)
            print(json.dumps(dict(base, name=case + "_ck", impl="ck", slot=slot, ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), t0=t0_, t1=time.time(), **vr)), flush=True)

        run_sdpa("pre")
        if a.ck:
            run_ck("pre")
        cs = [c for c in ARMS if c["case"] == case]
        random.Random(f"{a.tag}/{case}/{inp}").shuffle(cs)
        first = {}
        for pos, c in enumerate(cs):
            r = dict(base, name=c["name"], impl="tl", pos=pos, P=c.get("P", 0), bN=c["bN"], th=c["th"], rows=c.get("rows", 0), lb=c.get("lb", 0))
            n0 = COMPILES[0]; t0 = time.time()
            try:
                k = build(c)
                r["compiles"] = COMPILES[0] - n0
                O = k(Q, K, V)
                if a.check:
                    if R is not None:
                        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
                    r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                    nrep = 30 if B * L <= 8192 else 10
                    r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K, V), O)) for _ in range(nrep))}/{nrep}"
                    key = (c.get("P", 0), c["bN"], c["th"], c.get("rows", 0))     # lb must not change a single bit
                    if key in first:
                        r["bitwise_same_as_first"] = bool(torch.equal(first[key], O))
                    else:
                        first[key] = O.clone()
                del O
                t0_ = time.time()
                ms = bench(lambda: k(Q, K, V), warmup=10, rep=rep)
                r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), t0=t0_, t1=time.time())
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
        if a.ck:
            run_ck("post")
        run_sdpa("post")
        del Q, K, V, R, So, qt, kt, vt, first
        torch.cuda.empty_cache()
