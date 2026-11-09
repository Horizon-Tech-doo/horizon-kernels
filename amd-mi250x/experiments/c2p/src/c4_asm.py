"""Dump clang-22 assembly of one form. Usage: c4_asm.py land|tl case P bN th rows opts recip swap rev out"""
import json, os, re, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
form, case, P, bN, th, rows, opts, out = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7]), sys.argv[7], sys.argv[8]
kind, L = case.split("_"); L = int(L)
Hkv, G, D, causal, fl = (8, 1, 128, False, (True, False, True, False)) if kind == "mha" else (4, 6, int(kind[1:]), True, (True, True, True, True))
import tilelang
from tilelang.contrib import hipcc
from tilelang.contrib.rocm import find_hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
if form == "land":
    import gqa_fa
    k = gqa_fa.gqa_flash_fast(1, L, Hkv, G, D, causal, P, bN, th, 2, *fl, rows)
    if os.path.exists(out): pass
    src = k.get_kernel_source()
    src = gqa_fa._fast_edit(src, dict(L=L, Hkv=Hkv, B=1, D=D, bN=bN, threads=th, swap=fl[3]))
else:
    import tlfa
    k = tlfa.fwd(1, L, Hkv, G, D, causal, P, bN, th, opts, 2, *fl, rows)
    src = k.get_kernel_source()
src = re.sub(r'^#line.*\n', '', src, flags=re.M)
open(out + ".cc", "w").write(src)
r = subprocess.run([find_hipcc(), "-O3", "-S", "--cuda-device-only", "--offload-arch=gfx90a", "-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-o", out + ".s", out + ".cc"], capture_output=True, text=True)
print(r.returncode, r.stderr[-300:])
