"""CPU-only compile of arms; records register use. Usage: n5_compile.py <regex> ; cache from TILELANG_CACHE_DIR, flavour from N5_PATCH."""
import glob, json, os, re, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tilelang
import patch
patch.install()
from tvm.target import Target
from tilelang.rocm.target import ROCM_MTRIPLE
import gqa_e3b, gqa_fa
from n5lib import CASES
from n5_arms import ARMS, SEL
TGT = Target({"kind": "hip", "mcpu": "gfx90a", "mtriple": ROCM_MTRIPLE, "thread_warp_size": 64})
pat = re.compile(SEL.get(sys.argv[1], sys.argv[1]))
dump = os.environ.get("N5_DUMP")
FL = os.environ.get("N5_PATCH", "")
for c in ARMS:
    if not pat.search(c["name"]):
        continue
    s = CASES[c["case"]]
    t0 = time.time()
    r = dict(name=c["name"], patch=FL)
    patch.CTX.update(swap=c.get("swap", False), rev=c.get("rev", False), rows=c.get("rows", 0), bN=c["bN"], D=s["D"], th=c["th"], L=s["L"], Hkv=s["Hkv"], P=c["P"], G=s["G"], B=s["B"])
    try:
        B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
        if c["form"] == "land":
            f = gqa_fa.make(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], True, 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
        else:
            f = gqa_e3b.make(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], 2, c["edge"], c["recip"], c["mode"], c["vt"], c["vp"], c["rev"])
        k = tilelang.compile(f, out_idx=[3], target=TGT)
        src = k.get_kernel_source(kernel_only=True)
        ok, bad, nb = patch.race_free(patch.LAST.get("src", src))
        r.update(race_free=ok, race=bad, nbar=nb)
        if dump:
            open(f"{dump}/{c['name']}{'_p' + FL if FL else ''}.cu", "w").write(patch.LAST.get("src", src))
        fs = sorted(glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/resource_usage.json", recursive=True), key=os.path.getmtime)
        e = list(json.load(open(fs[-1])).values())[0]["extra"]
        r.update(sync_src=src.count("__syncthreads"), V=int(e["VGPRs"]), A=int(e["AGPRs"]), occ=int(e["Occupancy [waves/SIMD]"]), scr=int(e["ScratchSize [bytes/lane]"]), lds=e.get("LDS Size [bytes/block]", e.get("LDS", None)), t=round(time.time() - t0, 1))
    except Exception as ex:  # noqa: BLE001
        r.update(err=type(ex).__name__, msg=str(ex)[-300:])
    print(json.dumps(r), flush=True)
