"""N5 round 3. Landed module only. Arm order is a real shuffle, seeded by tag and case; position saved in each row. SDPA and CK run in the same process before the first arm and CK again after the last.
Plan files: JSON list of dicts: case, P, bN, th, rows, edge, rev, recip, swap, stride.
Usage: n5_r3.py --plan X --compile [--re R]
       n5_r3.py --plan X --tag r1 --check [--ck] [--re R] [--inputs randn0] [--rep 50]"""
import argparse, json, os, random, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--compile", action="store_true"); ap.add_argument("--re", default="."); ap.add_argument("--tag", default="c")
ap.add_argument("--inputs", default="randn0"); ap.add_argument("--check", action="store_true"); ap.add_argument("--ck", action="store_true"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import re
import torch
import torch.nn.functional as F
import tilelang
from tilelang.contrib import hipcc
_orig = hipcc.compile_hip
COMPILES = [0]


NOX = os.environ.get("N5_NOX") == "1"


def _nox(code):
    """Layout without TileLang's row-into-column xor: column bits alone pick the place in a row. Harness only: tests what the xor buys at each stride."""
    tid = "((int)threadIdx.x)"
    rep = [(f"((((({tid} & 7) >> 2) + (ki & 1)) & 1) * 32)", "((ki & 1) * 32)"),
           (f"((((({tid} & 63) >> 5) + (({tid} & 3) >> 1)) & 1) * 16)", f"((({tid} & 63) >> 5) * 16)"),
           (f"((((({tid} & 31) >> 4) + ({tid} & 1)) & 1) * 8)", f"((({tid} & 31) >> 4) * 8)"),
           (f"((((({tid} & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)", "(((j_2 & 3) >> 1) * 32)"),
           ("(((local_id >> 1) + (j_2 & 1)) & 1) * 16)", "(j_2 & 1) * 16)"),
           (f"((({tid} & 15) >> 3) + (local_id & 1)) & 1) * 8)", f"({tid} & 15) >> 3) * 8)")]
    for a_, b_ in rep:
        if a_ not in code:
            raise RuntimeError("nox: pattern not found: " + a_)
        code = code.replace(a_, b_)
    code, n = re.subn(r"\(\(\(\(\(\(d0_v\) >> (\d)\) \^ \(\(kk_v \+ \d\) >> \d\)\) & 1\)\) \* (\d+)\)", lambda m: f"((((d0_v) >> {m.group(1)}) & 1) * {m.group(2)})", code)
    if n != 12:
        raise RuntimeError(f"nox: {n} store terms")
    return code


def _cnt(code, *x, **k):
    COMPILES[0] += 1
    return _orig(_nox(code) if NOX else code, *x, **k)


hipcc.compile_hip = _cnt
import gqa_fa
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r3 import extend, real_inputs
extend(CASES)
_inputs = inputs
inputs = lambda kind, s: real_inputs(kind, s) if kind.startswith("real") else _inputs(kind, s)  # noqa: E731

pat = re.compile(a.re)
ARMS = []
for c in json.load(open(a.plan)):
    fl = ("E" if c.get("edge", True) else "") + ("R" if c.get("rev", True) else "") + ("D" if c.get("recip", True) else "") + ("W" if c.get("swap", True) else "")
    c["name"] = f"{c['case']}_{'r' + str(c['rows']) if c.get('rows') else 'P' + str(c['P'])}_b{c['bN']}_t{c['th']}_{fl}_s{c.get('stride', 0)}{'_lb' + str(c['lb']) if c.get('lb') else ''}{'_ring' if c.get('ring') else ''}"
    if pat.search(c["name"]):
        ARMS.append(c)


def build(c):
    s = CASES[c["case"]]
    return gqa_fa.gqa_flash_fast(s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"], c.get("P", 0), c["bN"], c["th"], 2, c.get("edge", True), c.get("rev", True), c.get("recip", True), c.get("swap", True), c.get("rows", 0), c.get("stride", 0), c.get("lb", 0), c.get("ring", False))


if a.compile:
    for c in ARMS:
        t0 = time.time(); n0 = COMPILES[0]; r = dict(name=c["name"])
        try:
            build(c)
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-400:]
        r.update(compiles=COMPILES[0] - n0, t=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, nox=NOX, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, n=len(ARMS), stack=os.environ.get("N5_STACK"))), flush=True)
for case in sorted({c["case"] for c in ARMS}):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if B * L <= 8192 else max(20, a.rep // 4)
    for inp in a.inputs.split(","):
        Q, K, V = inputs(inp, s)
        R = ref(Q, K, V, causal, G) if a.check and B * L <= 8192 else None
        qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, stack=os.environ.get("N5_STACK"), nox=NOX)
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
        t0_ = time.time()
        ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
        vr = {}
        if R is not None:
            vr["rel_vs_fp64"] = round(((So.double() - R).norm() / R.norm()).item(), 7)
            vr["rel_input_rounding_note"] = "Q, K, V are fp16 in every arm; R is computed from those same fp16 values"
        print(json.dumps(dict(base, name=case + "_sdpa", impl="sdpa", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2), t0=t0_, t1=time.time(), **vr)), flush=True)

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

        if a.ck:
            run_ck("pre")
        cs = [c for c in ARMS if c["case"] == case]
        random.Random(f"{a.tag}/{case}/{inp}").shuffle(cs)
        first = {}
        for pos, c in enumerate(cs):
            r = dict(base, name=c["name"], impl="tl", pos=pos, P=c.get("P", 0), bN=c["bN"], th=c["th"], rows=c.get("rows", 0), stride=c.get("stride", 0), lb=c.get("lb", 0), ring=c.get("ring", False))
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
                    key = (c.get("P", 0), c["bN"], c["th"], c.get("rows", 0), c.get("recip", True), c.get("swap", True))     # stride, lb must not change a single bit
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
        del Q, K, V, R, So, qt, kt, vt, first
        torch.cuda.empty_cache()
