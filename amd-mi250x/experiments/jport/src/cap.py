"""CPU, wheel process. Compiles through a given gqa_fa module in a fresh cache and keeps each code object. Usage: cap.py MODDIR OUTDIR old|j. old: landed tile rows 128 bN32 at the six lengths; j: gqa_flash_j at 2560-4096."""
import hashlib, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
mod, out, what = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
sys.path.insert(0, mod)
import tilelang
from tilelang.contrib import hipcc
orig = hipcc.compile_hip; cur = [None]
def cap(code, *a, **k):
    b = orig(code, *a, **k)
    open(f"{out}/{cur[0]}.cc", "w").write(code); open(f"{out}/{cur[0]}.hsaco", "wb").write(bytes(b))
    print(json.dumps(dict(name=cur[0], hsaco_md5=hashlib.md5(bytes(b)).hexdigest(), cc_md5=hashlib.md5(code.encode()).hexdigest(), opts=[str(x)[:200] for x in a] + [f"{n}={str(v)[:200]}" for n, v in k.items()])), flush=True)
    return b
hipcc.compile_hip = cap
import gqa_fa
print(json.dumps(dict(mod=gqa_fa.__file__, md5=hashlib.md5(open(gqa_fa.__file__, "rb").read()).hexdigest(), ver=tilelang.__version__, clang=gqa_fa.clang_major())), flush=True)
for L in (2560, 2816, 3072, 3328, 3584, 4096):
    if what == "old":
        cur[0] = f"old_{L}"; gqa_fa.gqa_flash_fast(1, L, 4, 6, 128, True, 0, 32, 512, 2, True, True, True, True, 128)
    else:
        cur[0] = f"j_{L}"; gqa_fa.gqa_flash_j(L)
