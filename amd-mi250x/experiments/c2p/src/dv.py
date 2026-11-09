"""dv.py: compile-only variants of decode k1 (32 x 8192) on fork-main. Text edits on generated source to find what makes ki non-unrolled. CPU."""
import json, os, re, subprocess, sys, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/m")
import tilelang, tilelang.rocm.language as T
T.exp = T.__exp
import gqa_decode as gq
from tilelang.contrib.rocm import find_hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
OUT = "/scratch/project_XXXXXXXXX/tilelang/exp/C2P/asm3/v"; os.makedirs(OUT, exist_ok=True)
B, L = int(sys.argv[1]), int(sys.argv[2]); PAGE = 16; MAXP = L // PAGE; NBLK = B * MAXP
k1, k2 = gq.paged_kernels(B, NBLK, MAXP, 4, 6, 256, PAGE, packed="lbnhc")
src = re.sub(r'^#line.*\n', '', k1.get_kernel_source(), flags=re.M)
open(f"{OUT}/src_{B}_{L}.cc", "w").write(src)
V = {
 "base": src,
 "pragma_ki": src.replace("for (int ki = 0;", "#pragma unroll\n      for (int ki = 0;"),
 "q_plain": re.sub(r"\*\(tl::RegVec<(uint\d)>\*\)\((q|p) \+", r"*(\1*)(\2 +", src),
 "no_regvec": re.sub(r"tl::RegVec<(uint\d)>", r"\1", src),
 "blocal_plain": re.sub(r"\*\(tl::RegVec<(uint\d)>\*\)\((B_local\w*) \+", r"*(\1*)(\2 +", src),
}
for n, s in V.items():
    f = f"{OUT}/{n}_{B}_{L}"
    open(f + ".cc", "w").write(s)
    r = subprocess.run([find_hipcc(), "-O3", "-S", "--cuda-device-only", "--offload-arch=gfx90a", "-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-o", f + ".s", f + ".cc"], capture_output=True, text=True)
    if r.returncode: print(n, "ERR", r.stderr[-300:]); continue
    a = open(f + ".s").read()
    ops = collections.Counter(l.split()[0] for l in a.splitlines() if l.startswith("\t") and not l.startswith("\t."))
    g = lambda k: (re.findall(k + r":\s*(\d+)", a) or ["?"])[0]
    print(n, "ins", sum(ops.values()), "mfma", ops["v_mfma_f32_16x16x16f16"], "cnd", ops["v_cndmask_b32_e32"], "barr", ops["s_barrier"], "VGPR", g("; NumVgprs"), "AGPR", g("; NumAgprs"), "scratch", g("; ScratchSize"), "occ", g("; Occupancy"), flush=True)
