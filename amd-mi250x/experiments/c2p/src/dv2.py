"""dv2.py: compile-only, tlfa finalists on fork-main under text edits: pragma on ki loop; GEMM operands not RegVec. Prints counts. CPU. Stack comes from the launcher."""
import json, os, re, subprocess, sys, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/exp/N5/src")
import tilelang, tlfa
from n5lib import CASES
from tilelang.contrib.rocm import find_hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
OUT = "/scratch/project_XXXXXXXXX/tilelang/exp/C2P/asm3/t2"; os.makedirs(OUT, exist_ok=True)
D256 = (True, True, True, True); MHA = (True, False, True, False)
ARMS = [("d128_16384", 0, 64, 512, 128, D256, "JDRF"), ("d128_8192", 0, 32, 1024, 256, D256, "SDRF"), ("d256_8192", 32, 32, 768, 0, D256, "SDRF"), ("mha_4096", 160, 32, 640, 0, MHA, "JDRF"), ("d128_8192", 32, 96, 768, 0, D256, "SDRF")]
def oper(s):
    names = set(re.findall(r"\(\(float16x\d\*\)(\w+)\)", s))
    for n in names:
        s = re.sub(r"\*\(tl::RegVec<(uint\d)>\*\)\(" + n + r" \+", r"*(\1*)(" + n + " +", s)
    return s
for c, P, bN, th, rows, fl, f in ARMS:
    S = CASES[c]
    k = tlfa.fwd(S["B"], S["L"], S["Hkv"], S["G"], S["D"], S["causal"], P, bN, th, f, 2, *fl, rows)
    src = re.sub(r'^#line.*\n', '', k.get_kernel_source(), flags=re.M)
    V = {"base": src, "pragma_ki": src.replace("for (int ki = 0;", "#pragma unroll\n      for (int ki = 0;"), "oper": oper(src)}
    for n, s in V.items():
        p = f"{OUT}/{c}_{P}_{bN}_{rows}_{f}_{n}"
        open(p + ".cc", "w").write(s)
        r = subprocess.run([find_hipcc(), "-O3", "-S", "--cuda-device-only", "--offload-arch=gfx90a", "-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-o", p + ".s", p + ".cc"], capture_output=True, text=True)
        if r.returncode: print(c, n, "ERR", r.stderr[-200:]); continue
        a = open(p + ".s").read()
        ops = collections.Counter(l.split()[0] for l in a.splitlines() if l.startswith("\t") and not l.startswith("\t."))
        g = lambda k: (re.findall(k + r":\s*(\d+)", a) or ["?"])[0]
        print(c, P, bN, rows, f, n, "ins", sum(ops.values()), "mfma", sum(v for k, v in ops.items() if "mfma" in k), "cnd", ops["v_cndmask_b32_e32"], "barr", ops["s_barrier"], "VGPR", g("; NumVgprs"), "AGPR", g("; NumAgprs"), "scratch", g("; ScratchSize"), "occ", g("; Occupancy"), flush=True)
