"""Wheel process, CPU. Compiles the hand arms through the landed gqa_flash_fast path and keeps code object and edited source. Usage: handcap.py OUTDIR [--re R] [--second] (second: name suffix for the determinism pass)."""
import hashlib, json, os, re, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
out = sys.argv[1]; pat = re.compile(sys.argv[sys.argv.index("--re") + 1]) if "--re" in sys.argv else re.compile("."); os.makedirs(out, exist_ok=True)
import tilelang
from tilelang.contrib import hipcc
orig = hipcc.compile_hip; cur = [None]
def cap(code, *a, **k):
    b = orig(code, *a, **k)
    open(f"{out}/{cur[0]}.cc", "w").write(code); open(f"{out}/{cur[0]}.hsaco", "wb").write(bytes(b))
    print(json.dumps(dict(name=cur[0], hsaco_md5=hashlib.md5(bytes(b)).hexdigest(), hsaco_bytes=len(b), cc_md5=hashlib.md5(code.encode()).hexdigest(), a=[str(x)[:300] for x in a], k={n: str(v)[:300] for n, v in k.items()}, tilelang=tilelang.__file__, ver=tilelang.__version__)), flush=True)
    return b
hipcc.compile_hip = cap     # gqa_fa's hook wraps this, so `code` here is after its C edits
import gqa_fa
print(json.dumps(dict(gqa_fa_md5=hashlib.md5(open(gqa_fa.__file__, "rb").read()).hexdigest())), flush=True)
for c in json.load(open("plan.json")):
    if c["kind"] != "hand" or not pat.search(c["name"]):
        continue
    cur[0] = c["name"]
    gqa_fa.gqa_flash_fast(1, c["L"], 4, 6, 128, True, 0, c["bN"], c["th"], 2, True, True, True, True, c["rows"])
