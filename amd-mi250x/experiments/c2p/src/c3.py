"""C2P prefill race on one GPU, one process: CK, SDPA and TileLang forms, same inputs (N5's). Ratio convention: ours / CK or SDPA, throughput, higher is better.
Usage: c3.py --compile [--re R]  (CPU, fills the cache)   |  c3.py --tag r1 [--re R] [--inputs randn0] [--rep 50] [--land]
--land runs gqa_fa.gqa_flash_fast (the hand-edited N5 kernels, C edits on generated source) on the stock wheel instead of tlfa forms."""
import argparse, json, os, re, sys, time
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/exp/N5/src")
ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true"); ap.add_argument("--land", action="store_true"); ap.add_argument("--re", default="."); ap.add_argument("--tag", default="c")
ap.add_argument("--inputs", default="randn0"); ap.add_argument("--rep", type=int, default=50); ap.add_argument("--no-check", action="store_true")
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
from bench_common import bench
from n5lib import CASES, inputs, ref
import tilelang
if a.land:
    sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
    import gqa_fa
else:
    import tlfa

D256 = dict(edge=True, rev=True, recip=True, swap=True)
MHA = dict(edge=True, rev=False, recip=True, swap=False)
TILES = []
for L in (1024, 4096, 8192, 16384):
    TILES += [(f"d256_{L}", 32, 32, 768, 0, D256), (f"d256_{L}", 32, 48, 768, 0, D256), (f"d128_{L}", 0, 32, 512, 128, D256), (f"d128_{L}", 32, 96, 768, 0, D256)]
TILES += [("mha_4096", 160, 32, 640, 0, MHA), ("mha_4096", 192, 96, 768, 0, MHA)]
if os.environ.get("C3_TILES"):     # sweep file: JSON list of [case, P, bN, threads, rows, "c" or "m"]
    TILES = [(c, P, b, t, r, D256 if f == "c" else MHA) for c, P, b, t, r, f in json.load(open(os.environ["C3_TILES"]))]
FORMS = os.environ.get("C3_FORMS", "-,F,DF,JDF,SF,SDF").split(",")
ARMS = []
for case, P, bN, th, rows, fl in TILES:
    for f in (["land"] if a.land else FORMS):
        f = "" if f == "-" else f
        ARMS.append(dict(name=f"{case}_{'r' + str(rows) if rows else 'P' + str(P)}_b{bN}_t{th}_{f or 'plain'}", case=case, P=P, bN=bN, th=th, rows=rows, opts=f, **fl))
if os.environ.get("C3_ARMS"):      # explicit arm list: JSON list of [case, P, bN, threads, rows, "c" or "m", form]
    ARMS = [dict(name=f"{c}_{'r' + str(r) if r else 'P' + str(P)}_b{b}_t{t}_{f}", case=c, P=P, bN=b, th=t, rows=r, opts=f, **(D256 if fl == "c" else MHA)) for c, P, b, t, r, fl, f in json.load(open(os.environ["C3_ARMS"]))]
pat = re.compile(a.re)
ARMS = [c for c in ARMS if pat.search(c["name"])]


def build(c):
    s = CASES[c["case"]]
    pos = (s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"], c["P"], c["bN"], c["th"])
    if a.land:
        return gqa_fa.gqa_flash_fast(*pos, 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
    if "W" in c["opts"]:
        return tlfa.fwd_w(*pos, c["opts"], 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
    if "H" in c["opts"]:
        return tlfa.fwd_h(*pos, c["opts"], 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
    return tlfa.fwd(*pos, c["opts"], 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])


if a.compile:
    for c in ARMS:
        t0 = time.time(); r = dict(name=c["name"])
        try:
            k = build(c)
            u = getattr(k, "resource_usage", None) or {}
            for n, v in dict(u).items():
                r.update(vgpr=v.extra.get("VGPRs"), agpr=v.extra.get("AGPRs"), scratch=v.scratch_bytes, occ=v.extra.get("Occupancy [waves/SIMD]"), lds=v.extra.get("LDS Size [bytes/block]"))
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-400:]
        r["t"] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, n=len(ARMS), tilelang=tilelang.__file__, ver=tilelang.__version__, land=a.land)), flush=True)
rot = sum(map(ord, a.tag))
for case in sorted({c["case"] for c in ARMS}):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if L <= 8192 else max(20, a.rep // 4)
    for inp in a.inputs.split(","):
        Q, K, V = inputs(inp, s)
        check = not a.no_check
        R = ref(Q, K, V, causal, G) if check and L <= 8192 else None
        qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, stack=os.environ.get("C3_STACK", "image"), land=a.land)
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
        ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
        print(json.dumps(dict(base, name=case + "_sdpa", impl="sdpa", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        if os.environ.get("C3_STACK", "image") == "image":
            from flash_attn import flash_attn_func
            ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=rep)
            print(json.dumps(dict(base, name=case + "_ck", impl="ck", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        cs = [c for c in ARMS if c["case"] == case]
        rr = rot % max(1, len(cs))
        seq = cs[rr:] + cs[:rr]
        if os.environ.get("C3_SHUFFLE"):   # shuffled, not rotated; seed logged, so the order can be recomputed. Rows also come out in run order.
            import random
            sd = f'{os.environ["C3_SHUFFLE"]}-{a.tag}-{case}-{inp}'
            random.Random(sd).shuffle(seq)
            print(json.dumps(dict(base, kind="order", seed=sd, order=[c["name"] for c in seq])), flush=True)
        for c in seq:
            r = dict(base, name=c["name"], impl="tl", opts=c["opts"], P=c["P"], bN=c["bN"], th=c["th"], rows=c["rows"])
            t0 = time.time()
            try:
                k = build(c)
                O = k(Q, K, V)
                if check:
                    if R is not None:
                        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
                    r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                    nrep = 30 if L <= 8192 else 10
                    r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K, V), O)) for _ in range(nrep))}/{nrep}"
                del O
                ms = bench(lambda: k(Q, K, V), warmup=10, rep=rep)
                r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
        if os.environ.get("C3_SDPA_END"):   # second SDPA read after the arms, to bound drift inside the process
            ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
            print(json.dumps(dict(base, name=case + "_sdpa_end", impl="sdpa_end", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        del Q, K, V, R, So, qt, kt, vt
        torch.cuda.empty_cache()
