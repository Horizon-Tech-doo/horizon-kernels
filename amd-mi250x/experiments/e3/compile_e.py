import glob, json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tilelang
import patch
patch.install()
from tvm.target import Target
from tilelang.rocm.target import ROCM_MTRIPLE
from gqa_e3b import make
from cfgs_e import CASES, CONFIGS
TGT = Target({"kind": "hip", "mcpu": "gfx90a", "mtriple": ROCM_MTRIPLE, "thread_warp_size": 64})
for c in CONFIGS:
    s = CASES[c["case"]]
    t0 = time.time()
    r = dict(name=c["name"], patch=os.environ.get("E3_PATCH", ""))
    patch.CTX.update(bN=c["bN"], D=s["D"], th=c["th"], L=s["L"], Hkv=s["Hkv"])
    try:
        f = make(s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"], c["P"], c["bN"], c["th"], 2, c["edge"], c["recip"], c["mode"], c["vt"], c["v_pack"], c["rev"])
        k = tilelang.compile(f, out_idx=[3], target=TGT)
        src = k.get_kernel_source(kernel_only=True)
        fs = sorted(glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/resource_usage.json", recursive=True), key=os.path.getmtime)
        e = list(json.load(open(fs[-1])).values())[0]["extra"]
        r.update(sync_src=src.count("__syncthreads"), V=int(e["VGPRs"]), A=int(e["AGPRs"]), occ=int(e["Occupancy [waves/SIMD]"]), scr=int(e["ScratchSize [bytes/lane]"]), t=round(time.time() - t0, 1))
    except Exception as ex:  # noqa: BLE001
        r.update(err=type(ex).__name__, msg=str(ex)[-300:])
    print(json.dumps(r), flush=True)
